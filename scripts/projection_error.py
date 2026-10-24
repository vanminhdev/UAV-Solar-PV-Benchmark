#!/usr/bin/env python
"""Consistency check of the frame -> map projection of single thermal frames (EPSG:32649).

    python -X utf8 code/scripts/projection_error.py [--dataset PATH_TO_dataset] [--out DIR] [--no-figures]

Question: if a detection at pixel (x_px, y_px) of one raw thermal frame is projected onto the map with the
nominal footprint model (flat terrain, RTK pose, gimbal angles, nominal H20T optics: 640x512, 12 um, f from XMP),
how far is it from the centre of the same box on the thermal ortho (thermal_boxes.csv)?
There are NO field checkpoints. The ortho box centre is the reference, so everything below is a *consistency*
measurement between two products of the same flight, not an absolute accuracy.

Inputs (read only): <dataset>/farm_*/annotations/model2_thermal_anomaly_obb/{thermal_box_frame_detections,thermal_boxes}.csv,
  metadata/inventory/{image_inventory,speed_xmp}.csv, metadata/association/thermal_box_to_cell.csv,
  farm_01_thuan_nam_19/module_reference/TN19_modules_master.gpkg (layer 'modules').
Outputs (new files only): <out>/projection_residuals.csv, projection_summary.csv, projection_dependence.csv,
  projection_cell_hit.csv, figures/*.png.

Projection model (same conventions as build_footprints.py): yaw = heading of image "up" clockwise from north,
heading h=(sin psi, cos psi), right=(cos psi, -sin psi); optical axis = cos(p) h + sin(p) z (pitch -90 = nadir);
roll = rotation about h; the pixel ray is intersected with the horizontal plane at RelativeAltitude below the camera.

Variants (residual = projected - ortho box centre; "OOF" = out-of-fold, calibration fitted on other boxes only,
5 folds grouped by box):
  V0_nominal      nominal model
  V1_shift        V0 minus a per-flight constant (E,N) shift
  V2_shift_lag    V1 + exposure/position time offset: shift along the XMP ground velocity, res = b_flight + k*v
  V3_shift_lag_rs V2 + global scale a and rotation c about the nadir point (focal/altitude and yaw errors)
  V4_lrf_shift_lag V2 but the altitude is the laser range finder distance where plausible (|LRF-alt|<=6 m)
  V5_alt_shift_lag V2 with the projection height reduced by one global constant Delta (the ortho boxes sit on the module
                  plane ~2 m above the take-off datum, RelativeAltitude is above take-off); Delta chosen by a 1-D grid search
  V6_alt_local_field  V5 base (no per-flight shift, no lag) minus a local drift field = mean residual of the 15 nearest
                  OTHER boxes of the same flight (leave-box-out; the same device as the box->grid association). Diagnostic:
                  it needs reference boxes, so it is not a deployable calibration, but it separates a smooth ortho/pose
                  drift from the frame-to-frame scatter.
Calibrations are Huber-weighted least squares (scale 0.5 m).
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
EPSG = 32649
W_PX, H_PX, PITCH_MM = 640, 512, 0.012
MODULE_W = 1.9
VARIANTS = ["V0_nominal", "V1_shift", "V2_shift_lag", "V3_shift_lag_rs", "V4_lrf_shift_lag", "V5_alt_shift_lag",
            "V6_alt_local_field"]
FLIGHTS_ALL = None


# ------------------------------------------------------------------------------------------ projection
def project(x_px, y_px, E0, N0, alt, yaw, pitch, roll, f_mm):
    """Vectorised pixel -> ground (E, N). Returns E, N, ground offset (E-E0, N-N0)."""
    psi, p, rr = np.radians(yaw), np.radians(pitch), np.radians(roll)
    z = np.array([0.0, 0.0, 1.0])
    h = np.stack([np.sin(psi), np.cos(psi), np.zeros_like(psi)], 1)
    right = np.stack([np.cos(psi), -np.sin(psi), np.zeros_like(psi)], 1)
    axis = np.cos(p)[:, None] * h + np.sin(p)[:, None] * z
    up = -np.sin(p)[:, None] * h + np.cos(p)[:, None] * z
    k = h
    def rot(v):
        c, s = np.cos(rr)[:, None], np.sin(rr)[:, None]
        return v * c + np.cross(k, v) * s + k * np.sum(k * v, 1, keepdims=True) * (1 - c)
    axis, up, right = rot(axis), rot(up), rot(right)
    xs = (x_px - W_PX / 2) * PITCH_MM          # mm, to the right
    ys = (H_PX / 2 - y_px) * PITCH_MM          # mm, up
    d = f_mm[:, None] * axis + xs[:, None] * right + ys[:, None] * up
    t = alt / -d[:, 2]
    dE, dN = t * d[:, 0], t * d[:, 1]
    return E0 + dE, N0 + dN, dE, dN


def huber_lstsq(X, y, scale=0.5, it=8):
    w = np.ones(len(y))
    for _ in range(it):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(y - X @ beta)
        w = np.where(r <= scale, 1.0, scale / np.maximum(r, 1e-9))
    return beta


def design(d, flights, with_lag, with_rs):
    """Rows: first all E equations then all N equations. Unknowns: b_E,b_N per flight, [k], [a, c]."""
    n = len(d)
    cols = []
    for f in flights:
        m = (d.flight_id.values == f).astype(float)
        cols.append(np.concatenate([m, np.zeros(n)]))
        cols.append(np.concatenate([np.zeros(n), m]))
    if with_lag:
        cols.append(np.concatenate([d.vE.values, d.vN.values]))
    if with_rs:
        oE, oN = d.offE.values, d.offN.values
        cols.append(np.concatenate([oE, oN]))      # scale a
        cols.append(np.concatenate([-oN, oE]))     # rotation c (small angle, ccw)
    return np.stack(cols, 1)


def fit_apply(train, test, flights, with_lag, with_rs, key):
    """Fit on train (residual of key 'resE_'+key), return corrected residuals of test + params."""
    X = design(train, flights, with_lag, with_rs)
    y = np.concatenate([train[f"rE_{key}"].values, train[f"rN_{key}"].values])
    beta = huber_lstsq(X, y)
    Xt = design(test, flights, with_lag, with_rs)
    corr = Xt @ beta
    n = len(test)
    return test[f"rE_{key}"].values - corr[:n], test[f"rN_{key}"].values - corr[n:], beta


def calibrate(d, key, with_shift_only, with_lag, with_rs, nfold=5):
    """Return in-sample and out-of-fold corrected residuals (E,N) and global parameter estimates."""
    flights = sorted(d.flight_id.unique())
    boxes = np.array(sorted(d.box_id.unique()))
    rng = np.random.default_rng(20260420)
    fold_of = dict(zip(boxes, rng.permutation(len(boxes)) % nfold))
    fold = d.box_id.map(fold_of).values
    eI, nI, beta = fit_apply(d, d, flights, with_lag, with_rs, key)
    eO, nO = np.zeros(len(d)), np.zeros(len(d))
    for k in range(nfold):
        te = fold == k
        e, n_, _ = fit_apply(d[~te], d[te], flights, with_lag, with_rs, key)
        eO[te], nO[te] = e, n_
    par = {}
    j = 2 * len(flights)
    if with_lag:
        par["k_s_per_ms"] = beta[j]; j += 1
    if with_rs:
        par["scale_a"] = beta[j]; par["rot_c_deg"] = np.degrees(beta[j + 1])
    par["shift"] = {f: (beta[2 * i], beta[2 * i + 1]) for i, f in enumerate(flights)}
    return (eI, nI), (eO, nO), par


# ------------------------------------------------------------------------------------------ statistics
def stats(rE, rN, **kw):
    r = np.hypot(rE, rN)
    return dict(**kw, n=len(r), median_m=np.median(r), p90_m=np.percentile(r, 90), p95_m=np.percentile(r, 95),
                mean_dE=np.mean(rE), mean_dN=np.mean(rN), sd_E=np.std(rE), sd_N=np.std(rN),
                frac_lt_0p5module=np.mean(r < 0.5 * MODULE_W), frac_lt_1module=np.mean(r < MODULE_W))


def local_field(d, key, k=15):
    """Residual of key minus the mean residual of the k nearest OTHER boxes (box means) of the same flight."""
    eo, no = np.zeros(len(d)), np.zeros(len(d))
    for f, s in d.groupby("flight_id"):
        bm = s.groupby("box_id").agg(E=(f"rE_{key}", "mean"), N=(f"rN_{key}", "mean"), x=("centre_E", "first"),
                                     y=("centre_N", "first"))
        xy = bm[["x", "y"]].values
        for bid, ix in s.groupby("box_id").groups.items():
            dist = np.hypot(xy[:, 0] - bm.loc[bid, "x"], xy[:, 1] - bm.loc[bid, "y"])
            dist[bm.index.get_loc(bid)] = np.inf
            nn = np.argsort(dist)[:k]
            nn = nn[np.isfinite(dist[nn])]
            if len(nn) == 0:
                fe, fn = 0.0, 0.0
            else:
                fe, fn = np.median(bm.E.values[nn]), np.median(bm.N.values[nn])
            pos = d.index.get_indexer(ix)
            eo[pos] = d[f"rE_{key}"].values[pos] - fe
            no[pos] = d[f"rN_{key}"].values[pos] - fn
    return eo, no


def altitude_class(fid):
    m = re.search(r"_(\d+)M_", fid)
    return int(m.group(1)) if m else -1


# ------------------------------------------------------------------------------------------ main
def load(dataset: Path):
    det = pd.concat([pd.read_csv(f) for f in sorted(dataset.glob(
        "farm_*/annotations/model2_thermal_anomaly_obb/thermal_box_frame_detections.csv"))], ignore_index=True)
    bx = pd.concat([pd.read_csv(f) for f in sorted(dataset.glob(
        "farm_*/annotations/model2_thermal_anomaly_obb/thermal_boxes.csv"))], ignore_index=True)
    inv = pd.read_csv(dataset / "metadata/inventory/image_inventory.csv")
    inv = inv[inv.sensor == "T"].copy()
    spd = pd.read_csv(dataset / "metadata/inventory/speed_xmp.csv")
    spd = spd[spd.sensor == "T"][["flight_id", "file", "speed_xmp_m_s", "vx", "vy"]]
    # pass id from the heading of all thermal frames of a flight
    inv = inv.sort_values(["flight_id", "seq"]).reset_index(drop=True)
    dyaw = inv.groupby("flight_id").GimbalYawDegree.diff()
    dyaw = (dyaw + 180) % 360 - 180
    inv["pass_id"] = (dyaw.abs() > 45).astype(int).groupby(inv.flight_id).cumsum()
    d = det.merge(bx[["box_id", "centre_E", "centre_N", "n_frames", "in_eval_zone", "class_final"]], on="box_id")
    d = d.merge(inv[["flight_id", "file", "GpsLatitude", "GpsLongitude", "RelativeAltitude", "GimbalYawDegree",
                     "GimbalPitchDegree", "GimbalRollDegree", "focal_mm", "LRFTargetDistance", "pass_id", "RtkFlag"]],
                left_on=["flight_id", "thermal_file"], right_on=["flight_id", "file"], how="left").drop(columns="file")
    d = d.merge(spd, left_on=["flight_id", "thermal_file"], right_on=["flight_id", "file"], how="left").drop(columns="file")
    d["dup_box_frame"] = d.duplicated(["box_id", "thermal_file"], keep=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    ds = Path(a.dataset)
    out = Path(a.out) if a.out else ds / "metadata/projection_check"
    out.mkdir(parents=True, exist_ok=True)
    d = load(ds)
    d = d.dropna(subset=["GpsLatitude", "speed_xmp_m_s"]).reset_index(drop=True)
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    d["camE"], d["camN"] = tr.transform(d.GpsLongitude.values, d.GpsLatitude.values)
    d["vE"], d["vN"] = d.vy.values, d.vx.values                  # XMP: X north, Y east (README_speed_xmp)
    alt = d.RelativeAltitude.values
    f_mm = d.focal_mm.fillna(13.5).values
    pE, pN, oE, oN = project(d.x_px.values, d.y_px.values, d.camE.values, d.camN.values, alt,
                             d.GimbalYawDegree.values, d.GimbalPitchDegree.values, d.GimbalRollDegree.values, f_mm)
    d["offE"], d["offN"] = oE, oN
    d["rE_V0"], d["rN_V0"] = pE - d.centre_E.values, pN - d.centre_N.values
    # LRF altitude variant
    lrf = d.LRFTargetDistance.values
    ok = np.isfinite(lrf) & (np.abs(lrf - alt) <= 6.0)
    alt4 = np.where(ok, lrf, alt)
    d["lrf_used"] = ok
    pE4, pN4, oE4, oN4 = project(d.x_px.values, d.y_px.values, d.camE.values, d.camN.values, alt4,
                                 d.GimbalYawDegree.values, d.GimbalPitchDegree.values, d.GimbalRollDegree.values, f_mm)
    d["rE_V4raw"], d["rN_V4raw"] = pE4 - d.centre_E.values, pN4 - d.centre_N.values
    d["nomE"], d["nomN"] = pE, pN
    # V5: global projection-height offset Delta (grid search on median |res| after the local drift field is removed)
    best = None
    for delta in np.arange(0, 4.01, 0.25):
        e5, n5, _, _ = project(d.x_px.values, d.y_px.values, d.camE.values, d.camN.values, alt - delta,
                               d.GimbalYawDegree.values, d.GimbalPitchDegree.values, d.GimbalRollDegree.values, f_mm)
        re_, rn_ = e5 - d.centre_E.values, n5 - d.centre_N.values
        d["rE_tmp"], d["rN_tmp"] = re_, rn_
        re_, rn_ = local_field(d, "tmp")        # criterion insensitive to the smooth ortho drift
        sc = np.median(np.hypot(re_, rn_))
        if best is None or sc < best[0]:
            best = (sc, delta)
    delta5 = best[1]
    p5 = project(d.x_px.values, d.y_px.values, d.camE.values, d.camN.values, alt - delta5,
                 d.GimbalYawDegree.values, d.GimbalPitchDegree.values, d.GimbalRollDegree.values, f_mm)
    d["rE_V5raw"], d["rN_V5raw"] = p5[0] - d.centre_E.values, p5[1] - d.centre_N.values

    psi = np.radians(d.GimbalYawDegree.values)
    d["hE"], d["hN"] = np.sin(psi), np.cos(psi)
    d["tilt_deg"] = np.hypot(d.GimbalPitchDegree + 90, d.GimbalRollDegree)
    d["radius_px"] = np.hypot(d.x_px - W_PX / 2, d.y_px - H_PX / 2)
    d["x_off_px"], d["y_off_px"] = d.x_px - W_PX / 2, H_PX / 2 - d.y_px
    d["alt_class_m"] = d.flight_id.map(altitude_class)
    d["v_along_ms"] = d.vE * d.hE + d.vN * d.hN                   # signed speed along the image-up axis

    res = {"V0_nominal": ((d.rE_V0.values, d.rN_V0.values), (d.rE_V0.values, d.rN_V0.values), {})}
    params = {}
    for name, key, lag, rs in (("V1_shift", "V0", False, False), ("V2_shift_lag", "V0", True, False),
                               ("V3_shift_lag_rs", "V0", True, True), ("V4_lrf_shift_lag", "V4raw", True, False),
                               ("V5_alt_shift_lag", "V5raw", True, False)):
        ins, oof, par = calibrate(d, key, True, lag, rs)
        res[name] = (ins, oof, par)
        params[name] = par
    # V6: local drift field from the 15 nearest other boxes of the same flight (leave-box-out)
    e6, n6 = local_field(d, "V5raw")
    res["V6_alt_local_field"] = ((e6, n6), (e6, n6), {})
    params["V5_alt_shift_lag"]["delta_alt_m"] = delta5
    for v in VARIANTS:
        ins, oof = res[v][0], res[v][1]
        d[f"rE_{v}"], d[f"rN_{v}"] = ins            # in-sample
        d[f"rE_{v}_oof"], d[f"rN_{v}_oof"] = oof     # out-of-fold
    # corrected map position of the best-looking variant is rebuilt in the grid step

    # along / across (heading frame) components for the nominal and the V1 residual
    for v in ("V0_nominal", "V1_shift", "V2_shift_lag", "V5_alt_shift_lag"):
        rE, rN = d[f"rE_{v}"], d[f"rN_{v}"]
        d[f"along_{v}"] = rE * d.hE + rN * d.hN
        d[f"across_{v}"] = rE * d.hN - rN * d.hE                  # positive to the right of the heading

    # ---------------------------------------------------------------------------- summary
    rows = []
    d["_all"] = "all"
    for v in VARIANTS:
        for mode, sfx in (("in_sample", ""), ("out_of_fold", "_oof")):
            if v == "V0_nominal" and mode == "out_of_fold":
                continue
            for lvl, col in (("all", "_all"), ("flight", "flight_id"), ("altitude_m", "alt_class_m")):
                for g, s in d.groupby(col):
                    rows.append(stats(s[f"rE_{v}{sfx}"].values, s[f"rN_{v}{sfx}"].values, level=lvl,
                                      group=str(g), variant=v, mode=mode))
    summ = pd.DataFrame(rows)
    summ = summ[["level", "group", "variant", "mode", "n", "median_m", "p90_m", "p95_m", "mean_dE", "mean_dN", "sd_E", "sd_N",
                 "frac_lt_0p5module", "frac_lt_1module"]].round(4)
    summ.to_csv(out / "projection_summary.csv", index=False)

    # ---------------------------------------------------------------------------- dependence
    dep = []
    def add(section, **kw):
        dep.append(dict(section=section, **kw))
    # (a) per pass bias within flight (V0): constant per flight? and per heading
    for (f, p), s in d.groupby(["flight_id", "pass_id"]):
        if len(s) >= 5:
            add("per_pass_V0", flight=f, group=f"pass{p}", n=len(s), yaw_deg=s.GimbalYawDegree.median(),
                speed=s.speed_xmp_m_s.mean(), E=s.rE_V0_nominal.median(), N=s.rN_V0_nominal.median(),
                along=s.along_V0_nominal.median(), across=s.across_V0_nominal.median())
    pp = pd.DataFrame([x for x in dep if x["section"] == "per_pass_V0"])
    for f, s in pp.groupby("flight"):
        add("pass_median_spread_V0", flight=f, n=len(s), E_sd=s.E.std(), N_sd=s.N.std(), along_sd=s.along.std(),
            across_sd=s.across.std())
    # (b) binned tables, residual after the per-flight shift (V1 in-sample), by factor
    bins = {"radius_px": [0, 100, 200, 300, 450], "x_off_px": [-330, -200, -100, 0, 100, 200, 330],
            "y_off_px": [-260, -150, -50, 50, 150, 260], "speed_xmp_m_s": [0, 1, 1.5, 2, 2.5, 3, 4],
            "v_along_ms": [-4, -2.5, -1, 1, 2.5, 4], "tilt_deg": [0, 0.5, 1, 1.5, 2, 6]}
    for col, edges in bins.items():
        c = pd.cut(d[col], edges)
        for g, s in d.groupby(c):
            if len(s) >= 10:
                add(f"bin_{col}", variant="V1_shift", group=str(g), n=len(s), E=s.rE_V1_shift.median(), N=s.rN_V1_shift.median(),
                    along=s.along_V1_shift.median(), across=s.across_V1_shift.median(),
                    abs_median=np.hypot(s.rE_V1_shift, s.rN_V1_shift).median(),
                    abs_p95=np.percentile(np.hypot(s.rE_V1_shift, s.rN_V1_shift), 95))
    # yaw (heading) classes
    ycls = pd.cut(((d.GimbalYawDegree + 360 + 45) % 360), [0, 90, 180, 270, 360], labels=["N", "E", "S", "W"], right=False)
    for (f, g), s in d.groupby([d.flight_id, ycls]):
        if len(s) >= 10:
            add("by_heading_V0", flight=f, group=str(g), n=len(s), E=s.rE_V0_nominal.median(), N=s.rN_V0_nominal.median(),
                along=s.along_V0_nominal.median(), across=s.across_V0_nominal.median(), speed=s.speed_xmp_m_s.mean())
    # (c) regressions on the V1 (flight-demeaned) residual components in the heading frame
    def ols(y, X, names, label, vname="V1_shift"):
        X = np.column_stack([np.ones(len(y))] + X)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        cov = np.linalg.inv(X.T @ X) * r.var(ddof=X.shape[1])
        se = np.sqrt(np.diag(cov))
        for nm, bi, si in zip(["const"] + names, b, se):
            add("regression", variant=vname, group=label, term=nm, coef=bi, se=si, n=len(y), r2=1 - r.var() / y.var())
    for vn in ("V1_shift", "V5_alt_shift_lag"):
        for comp in ("along", "across"):
            y = d[f"{comp}_{vn}"].values
            ols(y, [d.v_along_ms.values, d.x_off_px.values / 100, d.y_off_px.values / 100, d.tilt_deg.values],
                ["v_along_ms", "x_off_per100px", "y_off_per100px", "tilt_deg"], f"{comp}~speed+pos+tilt", vn)
    # lag factor: along residual vs along speed per flight (slope = k, seconds), from the V0 residual with per-flight shift
    for f, s in d.groupby("flight_id"):
        X = np.column_stack([np.ones(len(s)), s.v_along_ms.values])
        for comp in ("along",):
            y = s[f"{comp}_V1_shift"].values
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ b
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * r.var(ddof=2)))
            add("lag_per_flight", flight=f, group="along~v_along", n=len(s), k_s=b[1], se=se[1],
                implied_px_at_mean_speed=abs(b[1]) * s.speed_xmp_m_s.mean() / (s.RelativeAltitude.mean() * PITCH_MM / 13.5),
                r_speed_range=f"{s.speed_xmp_m_s.min():.1f}-{s.speed_xmp_m_s.max():.1f}")
    # within-box scatter of the projected points (independent of the ortho)
    g = d.groupby("box_id")
    for v, (c1, c2) in (("V0_nominal", ("rE_V0_nominal", "rN_V0_nominal")), ("V2_shift_lag", ("rE_V2_shift_lag", "rN_V2_shift_lag")),
                           ("V5_alt_shift_lag", ("rE_V5_alt_shift_lag", "rN_V5_alt_shift_lag"))):
        dm1 = d[c1] - g[c1].transform("mean")
        dm2 = d[c2] - g[c2].transform("mean")
        cnt = g[c1].transform("size")
        m = cnt >= 2
        corr = np.sqrt(cnt[m] / (cnt[m] - 1))
        wr = np.hypot(dm1[m], dm2[m]) * corr
        add("within_box_scatter", variant=v, group="boxes with >=2 frames", n=int(m.sum()), abs_median=wr.median(),
            abs_p95=np.percentile(wr, 95), n_boxes=int(d[m].box_id.nunique()))
    for p, v in params.items():
        for k, val in v.items():
            if k != "shift":
                add("fitted_param", variant=p, group="global", term=k, coef=val)
            else:
                for f, (be, bn) in val.items():
                    add("fitted_shift", variant=p, flight=f, E=be, N=bn)
    dep = pd.DataFrame(dep)
    dep.round(4).to_csv(out / "projection_dependence.csv", index=False)

    # ---------------------------------------------------------------------------- cell hit (TN19)
    cell = cell_hit(ds, d, res, out)

    keep = ["flight_id", "box_id", "seq", "thermal_file", "x_px", "y_px", "dup_box_frame",
            "alt_class_m", "RelativeAltitude", "GimbalYawDegree", "GimbalPitchDegree", "GimbalRollDegree", "tilt_deg",
            "speed_xmp_m_s", "v_along_ms", "radius_px", "x_off_px", "y_off_px", "camE", "camN", "nomE", "nomN", "centre_E", "centre_N",
            "lrf_used", "LRFTargetDistance", "pass_id"]
    o = d[keep].copy()
    for v in VARIANTS:
        o[f"resE_{v}"], o[f"resN_{v}"] = d[f"rE_{v}"], d[f"rN_{v}"]
        o[f"abs_{v}"] = np.hypot(d[f"rE_{v}"], d[f"rN_{v}"])
        if v != "V0_nominal":
            o[f"abs_{v}_oof"] = np.hypot(d[f"rE_{v}_oof"], d[f"rN_{v}_oof"])
    for v in ("V0_nominal", "V1_shift", "V2_shift_lag", "V5_alt_shift_lag"):
        o[f"along_{v}"], o[f"across_{v}"] = d[f"along_{v}"], d[f"across_{v}"]
    o.round(4).to_csv(out / "projection_residuals.csv", index=False)
    print(summ[(summ.level.isin(["all"]))].to_string())
    print(summ[(summ.level == "altitude_m")].to_string())
    print(summ[(summ.level == "flight") & summ.variant.isin(["V0_nominal", "V2_shift_lag"])].to_string())
    print(dep[dep.section.isin(["lag_per_flight", "regression", "fitted_param", "pass_median_spread_V0", "within_box_scatter"])]
          .to_string())
    print(cell.to_string())
    if not a.no_figures:
        figures(d, out)


# ------------------------------------------------------------------------------------------ grid
def cell_hit(ds: Path, d: pd.DataFrame, res: dict, out: Path) -> pd.DataFrame:
    from shapely import STRtree
    from shapely.geometry import Point
    from build_footprints import read_gpkg_layer
    assoc = pd.read_csv(ds / "metadata/association/thermal_box_to_cell.csv")
    assoc = assoc[assoc.status == "unique"][["box_id", "module_id", "reg_dE", "reg_dN"]]
    g = read_gpkg_layer(ds / "farm_01_thuan_nam_19/module_reference/TN19_modules_master.gpkg", "modules")
    geoms = list(g.geometry)
    ids = g.module_id.values
    tree = STRtree(geoms)
    idx_of = {m: i for i, m in enumerate(ids)}
    row_of = dict(zip(ids, zip(g.String_ID, g.Row_ID)))
    t = d.merge(assoc, on="box_id")
    t = t[t.flight_id.str.startswith("TN19")].reset_index(drop=True)
    # corrected position = nominal position - estimated correction = box centre + residual
    nb_cache = {}
    def neigh(m):
        if m not in nb_cache:
            gi = idx_of[m]
            q = tree.query(geoms[gi].buffer(0.12))
            nb_cache[m] = {ids[j] for j in q}
        return nb_cache[m]
    def hit(E, N):
        out_ = []
        for e, n_ in zip(E, N):
            pt = Point(e, n_)
            j = int(tree.nearest(pt))                 # nearest module polygon (0 m if the point is inside one)
            out_.append((ids[j], geoms[j].distance(pt)))
        return out_
    rows = []
    cases = {"ceiling_box_centre": (t.centre_E.values, t.centre_N.values)}
    for v in VARIANTS:
        for mode, sfx in (("in_sample", ""), ("out_of_fold", "_oof")):
            if v == "V0_nominal" and mode == "out_of_fold":
                continue
            cases[f"{v}|{mode}"] = (t.centre_E.values + t[f"rE_{v}{sfx}"].values, t.centre_N.values + t[f"rN_{v}{sfx}"].values)
    for name, (E, N) in cases.items():
        for reg, tag in ((True, "with_box_registration"), (False, "no_registration")):
            if not reg and not (name.startswith("V0") or name.startswith("ceiling")):
                continue
            pe = E + (t.reg_dE.values if reg else 0)
            pn = N + (t.reg_dN.values if reg else 0)
            hh = hit(pe, pn)
            h = np.array([x[0] for x in hh], dtype=object)
            dist = np.array([x[1] for x in hh])
            same = np.array([x == m for x, m in zip(h, t.module_id)])
            near = np.array([x in neigh(m) for x, m in zip(h, t.module_id)])
            samerow = np.array([row_of[x] == row_of[m] for x, m in zip(h, t.module_id)])
            var, _, mode = name.partition("|")
            for lvl, grp in (("all", pd.Series("all", index=t.index)), ("flight", t.flight_id), ("altitude_m", t.alt_class_m.astype(str))):
                for gv, ix in grp.groupby(grp).groups.items():
                    ix = np.asarray(ix)
                    rows.append(dict(variant=var, mode=mode or "-", registration=tag, level=lvl, group=gv, n=len(ix),
                                     frac_same_cell=same[ix].mean(), frac_within_one_cell=near[ix].mean(), frac_same_row=samerow[ix].mean(),
                                     frac_point_off_module_gt0p5m=np.mean(dist[ix] > 0.5)))
    c = pd.DataFrame(rows).round(4)
    c.to_csv(out / "projection_cell_hit.csv", index=False)
    return c[(c.level.isin(["all", "altitude_m"]))]


# ------------------------------------------------------------------------------------------ figures
def figures(d: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    cols = {f: c for f, c in zip(sorted(d.flight_id.unique()), plt.cm.tab10.colors)}
    # 1 residual vectors + scatter
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    f0 = "TN19_20260411_31M_T3"
    s = d[d.flight_id == f0]
    q = ax[0].quiver(s.nomE, s.nomN, s.rE_V0_nominal, s.rN_V0_nominal, np.hypot(s.rE_V0_nominal, s.rN_V0_nominal),
                     angles="xy", scale_units="xy", scale=0.5, cmap="viridis", width=0.002)
    ax[0].set_aspect("equal"); ax[0].set_title(f"{f0}: nominal residual vectors\n(arrow length x2 for visibility)")
    plt.colorbar(q, ax=ax[0], label="|res| (m)")
    for f, ss in d.groupby("flight_id"):
        ax[1].scatter(ss.rE_V0_nominal, ss.rN_V0_nominal, s=3, alpha=.35, color=cols[f], label=f.split("_", 2)[2] if f.startswith("TN19") else f.split("_")[0] + "_" + f.split("_")[2])
        ax[2].scatter(ss.rE_V5_alt_shift_lag_oof, ss.rN_V5_alt_shift_lag_oof, s=3, alpha=.35, color=cols[f])
    for a_, t_ in ((ax[1], "V0 nominal"), (ax[2], "V5 height offset+shift+lag (out of fold)")):
        a_.set_aspect("equal"); a_.axhline(0, c="k", lw=.5); a_.axvline(0, c="k", lw=.5)
        a_.add_patch(plt.Circle((0, 0), 0.95, fill=False, ls="--", color="r")); a_.set_xlim(-6, 6); a_.set_ylim(-6, 6)
        a_.set_xlabel("residual E (m)"); a_.set_ylabel("residual N (m)"); a_.set_title(t_ + "  (red = 0.5 module)")
    ax[1].legend(markerscale=4, fontsize=7)
    fig.tight_layout(); fig.savefig(fd / "fig1_residual_vectors_scatter.png", dpi=130); plt.close(fig)
    # 2 vs speed
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for f, ss in d.groupby("flight_id"):
        ax[0].scatter(ss.v_along_ms, ss.along_V1_shift, s=3, alpha=.3, color=cols[f], label=f.split("_", 2)[2])
    ax[0].set_xlabel("signed XMP speed along heading (m/s)"); ax[0].set_ylabel("along-track residual after flight shift (m)")
    ax[0].set_title("along-track residual vs speed"); ax[0].legend(markerscale=4, fontsize=7)
    xs = np.linspace(-3.5, 3.5, 2)
    k = d.groupby("flight_id").apply(lambda s: np.polyfit(s.v_along_ms, s.along_V1_shift, 1)[0]).median()
    ax[0].plot(xs, k * xs, "k-", lw=1)
    ax[1].scatter(d.speed_xmp_m_s, np.hypot(d.rE_V1_shift, d.rN_V1_shift), s=3, alpha=.25, color="C0")
    ax[1].set_xlabel("speed (m/s)"); ax[1].set_ylabel("|residual| after flight shift (m)"); ax[1].set_title("|residual| vs speed")
    ax[2].scatter(d.radius_px, np.hypot(d.rE_V1_shift, d.rN_V1_shift), s=3, alpha=.25, color="C1")
    ax[2].set_xlabel("pixel distance from frame centre (px)"); ax[2].set_ylabel("|residual| after flight shift (m)")
    ax[2].set_title("|residual| vs image position")
    fig.tight_layout(); fig.savefig(fd / "fig2_residual_vs_speed_position.png", dpi=130); plt.close(fig)
    # 3 histograms
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 6, 61)
    for v, c in (("V0_nominal", "C3"), ("V2_shift_lag", "C0"), ("V5_alt_shift_lag", "C2"), ("V6_alt_local_field", "C4")):
        r = np.hypot(d[f"rE_{v}" + ("" if v == "V0_nominal" else "_oof")], d[f"rN_{v}" + ("" if v == "V0_nominal" else "_oof")])
        ax[0].hist(r, bins, histtype="step", color=c, lw=1.6, label=f"{v}  med {r.median():.2f}  p95 {np.percentile(r, 95):.2f}")
    ax[0].axvline(0.95, c="k", ls="--"); ax[0].axvline(1.9, c="k", ls=":")
    ax[0].set_xlabel("|residual| (m); dashed 0.5 module, dotted 1 module"); ax[0].set_ylabel("detections"); ax[0].legend(fontsize=8)
    ax[0].set_title("all flights (calibrated variants out of fold)")
    for ac, c in zip(sorted(d.alt_class_m.unique()), ("C0", "C1", "C2")):
        s = d[d.alt_class_m == ac]
        r = np.hypot(s.rE_V5_alt_shift_lag_oof, s.rN_V5_alt_shift_lag_oof)
        ax[1].hist(r, bins, histtype="step", color=c, lw=1.6, label=f"{ac} m (n={len(s)}) med {r.median():.2f}")
    ax[1].axvline(0.95, c="k", ls="--"); ax[1].set_xlabel("|residual| (m)"); ax[1].legend(fontsize=8)
    ax[1].set_title("V5 out of fold, by altitude")
    fig.tight_layout(); fig.savefig(fd / "fig3_residual_histograms.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
