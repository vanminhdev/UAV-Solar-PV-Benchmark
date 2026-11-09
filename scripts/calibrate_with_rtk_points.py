#!/usr/bin/env python
"""Evaluate and correct a base orthomosaic with RTK GNSS points, and PROPOSE the propagation of the correction.

Nothing here modifies a dataset table or raster. All outputs go to a directory you give (default
``.local/gcp_calibration/results`` and ``.../proposed``).

Sub-commands (``python -X utf8 code/scripts/calibrate_with_rtk_points.py <cmd> ...``)

  calibrate  --measured FILLED.csv [--points points_all.csv] --farm TN19 [--out DIR]
             residuals ortho-pick -> RTK, then translation / similarity / affine fitted on the CONTROL points
             and evaluated on the CHECK points, leave-one-out over all points, model choice, JSON with the
             chosen transform (refitted on all accepted points) and its uncertainty.
  propagate  --calibration RESULT.json [--out-dir DIR]
             composes the base correction with ortho_registration_transforms.csv (primary rows) and writes
             PROPOSED tables: flight transforms, thermal box coordinates, frame footprints (GPKG).
  simulate   [--farm TN19] [--out DIR]     synthetic test, labelled SIMULATION, NOT RESULTS
  selftest

Conventions
  * Map frame EPSG:32649 (WGS 84 / UTM 49N), metres. Receivers give WGS 84 lat/lon + ELLIPSOIDAL height;
    only lat/lon are used (planar correction). Heights are neither used nor corrected.
  * Direction: the transform maps ORTHO coordinates -> RTK (reference) coordinates:
        E' = oE + tE + m11 (E - oE) + m12 (N - oN)      N' = oN + tN + m21 (E - oE) + m22 (N - oN)
    (same convention as dataset/metadata/ortho_registration/ortho_registration_transforms.csv).
    ``origin`` (oE, oN) = centroid of the fitting points, so tE, tN are the shift at the centre of the points.
  * Models: translation (2 params), similarity (4: shift, rotation, uniform scale), affine (6).
    Minimum fitting points: translation 1, similarity 3 (2 is the algebraic minimum: no redundancy),
    affine 6 (refused below that; with only 5 points an affine has no redundancy to be tested).
Requires numpy, pandas, pyproj (rasterio/geopandas are NOT needed; shapely only for footprint propagation).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
KIT = ROOT / ".local" / "gcp_calibration"
MIN_POINTS = {"translation": 1, "similarity": 3, "affine": 6}
MODELS = ["translation", "similarity", "affine"]
PARAMS = {"translation": 2, "similarity": 4, "affine": 6}
BANNER_SIM = "SIMULATION, NOT RESULTS"


def need_pyproj():
    if Transformer is None:
        sys.exit("pyproj is not installed: cannot convert RTK lat/lon (WGS 84) to EPSG:32649. `pip install pyproj`.")


def latlon_to_utm(lat, lon):
    need_pyproj()
    t = Transformer.from_crs(4326, 32649, always_xy=True)
    e, n = t.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(e), np.asarray(n)


# ------------------------------------------------------------------ model algebra
def _design(model, x, y):
    """Matrix A (2n x p): displacement (dE,dN) = A @ params, x,y = coordinates relative to the origin."""
    n = len(x)
    z, o = np.zeros(n), np.ones(n)
    if model == "translation":
        return np.vstack([np.column_stack([o, z]), np.column_stack([z, o])])
    if model == "similarity":  # params tE,tN,da,db ; a=1+da, b=db
        return np.vstack([np.column_stack([o, z, x, -y]), np.column_stack([z, o, y, x])])
    if model == "affine":  # tE,tN,d11,d12,d21,d22
        return np.vstack([np.column_stack([o, z, x, y, z, z]), np.column_stack([z, o, z, z, x, y])])
    raise ValueError(model)


def _matrix(model, p):
    if model == "translation":
        return np.eye(2)
    if model == "similarity":
        return np.array([[1 + p[2], -p[3]], [p[3], 1 + p[2]]])
    return np.eye(2) + np.array([[p[2], p[3]], [p[4], p[5]]])


def fit(model, E, N, dE, dN, origin=None, prior_sigma=0.05):
    """Least-squares fit of the displacement field. E,N ortho picks; dE,dN = RTK - ortho. Returns a dict."""
    E, N, dE, dN = (np.asarray(v, float) for v in (E, N, dE, dN))
    n = len(E)
    if n < MIN_POINTS[model]:
        raise ValueError(f"{model} needs >= {MIN_POINTS[model]} points, got {n}")
    o = (E.mean(), N.mean()) if origin is None else origin
    x, y = E - o[0], N - o[1]
    A = _design(model, x, y)
    b = np.concatenate([dE, dN])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ p
    dof = 2 * n - PARAMS[model]
    rss = float(res @ res)
    sig_hat = float(np.sqrt(rss / dof)) if dof > 0 else float("nan")
    sigma = max(sig_hat, prior_sigma) if dof > 0 else prior_sigma  # conservative: never below the picking prior
    AtA = A.T @ A
    cov = sigma**2 * np.linalg.pinv(AtA)
    M = _matrix(model, p)
    return dict(model=model, origin_E=float(o[0]), origin_N=float(o[1]), params=p, M=M, tE=float(p[0]),
                tN=float(p[1]), n_points=n, dof=dof, sigma_hat_m=sig_hat, sigma_used_m=sigma, cov=cov,
                rank_ok=bool(np.linalg.matrix_rank(AtA) == PARAMS[model]))


def apply(f, E, N):
    E, N = np.asarray(E, float), np.asarray(N, float)
    x, y = E - f["origin_E"], N - f["origin_N"]
    M = f["M"]
    return (f["origin_E"] + f["tE"] + M[0, 0] * x + M[0, 1] * y, f["origin_N"] + f["tN"] + M[1, 0] * x + M[1, 1] * y)


def pred_sd(f, E, N):
    """1-sigma horizontal standard error of the corrected position at (E,N) from the parameter covariance."""
    E, N = np.atleast_1d(np.asarray(E, float)), np.atleast_1d(np.asarray(N, float))
    A = _design(f["model"], E - f["origin_E"], N - f["origin_N"])
    n = len(E)
    out = []
    for i in range(n):
        J = A[[i, n + i], :]
        out.append(float(np.sqrt(np.trace(J @ f["cov"] @ J.T))))
    return np.array(out)


def scale_rot(f):
    """(scale, rotation_deg ccw, sd_scale, sd_rot_deg) of the linear part (similarity: exact; affine: polar approx)."""
    M = f["M"]
    if f["model"] == "translation":
        return 1.0, 0.0, 0.0, 0.0
    if f["model"] == "similarity":
        a, b = M[0, 0], M[1, 0]
        s = np.hypot(a, b)
        C = f["cov"][2:4, 2:4]
        g_s = np.array([a / s, b / s])
        g_r = np.array([-b / s**2, a / s**2])
        return float(s), float(np.degrees(np.arctan2(b, a))), float(np.sqrt(g_s @ C @ g_s)), \
            float(np.degrees(np.sqrt(g_r @ C @ g_r)))
    s = float(np.sqrt(abs(np.linalg.det(M))))
    return s, float(np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1]))), float("nan"), float("nan")


def rmse(dE, dN):
    dE, dN = np.asarray(dE, float), np.asarray(dN, float)
    if len(dE) == 0:
        return dict(n=0, rmse_E=np.nan, rmse_N=np.nan, rmse_r=np.nan, max_r=np.nan)
    r = np.hypot(dE, dN)
    return dict(n=len(dE), rmse_E=float(np.sqrt(np.mean(dE**2))), rmse_N=float(np.sqrt(np.mean(dN**2))),
                rmse_r=float(np.sqrt(np.mean(dE**2 + dN**2))), max_r=float(r.max()))


def loo(model, d, prior_sigma):
    """Leave-one-out over all points of d (DataFrame with E,N,dE,dN). Returns residual arrays or None."""
    n = len(d)
    if n - 1 < MIN_POINTS[model]:
        return None
    rE, rN = [], []
    for i in range(n):
        m = np.arange(n) != i
        f = fit(model, d.E.values[m], d.N.values[m], d.dE.values[m], d.dN.values[m], prior_sigma=prior_sigma)
        e, nn = apply(f, d.E.values[i:i + 1], d.N.values[i:i + 1])
        rE.append(d.E.values[i] + d.dE.values[i] - e[0])
        rN.append(d.N.values[i] + d.dN.values[i] - nn[0])
    return np.array(rE), np.array(rN)


# ------------------------------------------------------------------ calibrate
def load_measured(measured_csv, points_csv, farm, allow_float, spare_as, rtk_sigma):
    pts = pd.read_csv(points_csv, encoding="utf-8-sig")
    pts = pts[pts.farm == farm].copy()
    m = pd.read_csv(measured_csv, encoding="utf-8-sig", dtype=str)
    m.columns = [c.strip() for c in m.columns]
    for c in ("measured_lat", "measured_lon"):
        m[c] = pd.to_numeric(m[c].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    d = pts.merge(m[[c for c in m.columns if c not in pts.columns or c == "point_id"]], on="point_id", how="left")
    notes = []
    d["measured"] = d.measured_lat.notna() & d.measured_lon.notna()
    if (~d.measured).any():
        notes.append("not measured: " + ", ".join(d.loc[~d.measured, "point_id"]))
    if "fix_type" in d:
        fx = d.fix_type.fillna("").str.lower().str.replace(" ", "").str.replace("_", "")
        ok = fx.str.contains("fixed") & ~fx.str.contains("float")
        bad = d.measured & ~ok
        if not allow_float and bad.any():
            notes.append("excluded (fix_type is not RTK fixed): " + ", ".join(d.loc[bad, "point_id"]))
            d.loc[bad, "measured"] = False
        elif bad.any():
            notes.append("WARNING float/other fixes ACCEPTED (--allow-float): " + ", ".join(d.loc[bad, "point_id"]))
    else:
        notes.append("WARNING: no fix_type column; fix quality not checked")
    if "use" in d:
        off = d.measured & d.use.fillna("1").str.strip().str.lower().isin(["0", "no", "n", "false"])
        d.loc[off, "measured"] = False
        if off.any():
            notes.append("excluded by use=0: " + ", ".join(d.loc[off, "point_id"]))
    d = d[d.measured].copy()
    e, n = latlon_to_utm(d.measured_lat.values, d.measured_lon.values)
    d["E"], d["N"] = d.E_ortho_m.values, d.N_ortho_m.values
    d["E_rtk"], d["N_rtk"] = e, n
    d["dE"], d["dN"] = d.E_rtk - d.E, d.N_rtk - d.N
    role = d.role_suggested.copy()
    if spare_as == "ignore":
        d = d[role != "spare"].copy()
        role = role[role != "spare"]
    else:
        role = role.replace({"spare": spare_as})
    d["role"] = role.values
    d["prior_sigma_m"] = np.hypot(d.pick_uncertainty_m, rtk_sigma)
    return d.reset_index(drop=True), notes


def calibrate(measured_csv, points_csv, farm, out_dir, allow_float=False, spare_as="check", rtk_sigma=0.03,
              tag="", sim=False, quiet=False):
    d, notes = load_measured(measured_csv, points_csv, farm, allow_float, spare_as, rtk_sigma)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl, chk = d[d.role == "control"], d[d.role == "check"]
    prior = float(np.median(d.prior_sigma_m)) if len(d) else 0.1
    pick = float(np.median(d.pick_uncertainty_m)) if len(d) else 0.1
    summary = dict(simulation=BANNER_SIM if sim else None, farm=farm,
                   base_ortho_flight_id=d.ortho_flight_id.iloc[0] if len(d) else None,
                   created=dt.datetime.now().isoformat(timespec="seconds"), notes=notes,
                   n_measured=len(d), n_control=len(ctrl), n_check=len(chk),
                   median_pick_uncertainty_m=pick, rtk_sigma_assumed_m=rtk_sigma,
                   vertical_note="ellipsoidal heights were NOT used; planar correction only; DSM heights untouched")
    if len(d) == 0:
        summary["status"] = "no usable measured points"
        (out_dir / f"calibration_{farm}.json").write_text(json.dumps(summary, indent=2))
        return summary
    summary["raw_residuals"] = dict(mean_dE_m=float(d.dE.mean()), mean_dN_m=float(d.dN.mean()),
                                    sd_dE_m=float(d.dE.std(ddof=1)) if len(d) > 1 else None,
                                    sd_dN_m=float(d.dN.std(ddof=1)) if len(d) > 1 else None,
                                    **{k: v for k, v in rmse(d.dE, d.dN).items()})
    rows, fits = [], {}
    base_chk = rmse(chk.dE, chk.dN) if len(chk) else rmse([], [])
    rows.append(dict(model="none (ortho as is)", n_params=0, n_control=len(ctrl), n_check=len(chk),
                     check_rmse_r_m=base_chk["rmse_r"], check_max_m=base_chk["max_r"],
                     loo_rmse_r_m=rmse(d.dE, d.dN)["rmse_r"], admissible=True, note="reference: no correction"))
    for mname in MODELS:
        r = dict(model=mname, n_params=PARAMS[mname], n_control=len(ctrl), n_check=len(chk),
                 min_points_required=MIN_POINTS[mname])
        if len(ctrl) < MIN_POINTS[mname]:
            r.update(admissible=False, note=f"refused: needs >= {MIN_POINTS[mname]} control points, have {len(ctrl)}")
            if mname == "affine":
                r["note"] = f"affine refused: fewer than 6 points in the fitting set ({len(ctrl)} control)"
            rows.append(r)
            continue
        f = fit(mname, ctrl.E.values, ctrl.N.values, ctrl.dE.values, ctrl.dN.values, prior_sigma=prior)
        fits[mname] = f
        ce, cn = apply(f, ctrl.E.values, ctrl.N.values)
        fit_r = rmse(ctrl.E_rtk.values - ce, ctrl.N_rtk.values - cn)
        if len(chk):
            ke, kn = apply(f, chk.E.values, chk.N.values)
            kr = rmse(chk.E_rtk.values - ke, chk.N_rtk.values - kn)
        else:
            kr = rmse([], [])
        lo = loo(mname, d, prior)
        lr = rmse(*lo) if lo else rmse([], [])
        r.update(admissible=True, fit_rmse_r_m=fit_r["rmse_r"], dof=f["dof"], check_rmse_r_m=kr["rmse_r"],
                 check_rmse_E_m=kr["rmse_E"], check_rmse_N_m=kr["rmse_N"], check_max_m=kr["max_r"],
                 loo_rmse_r_m=lr["rmse_r"], loo_max_m=lr["max_r"], note="")
        if f["dof"] == 0:
            r["note"] = "no redundancy (dof=0): fit residual meaningless, rely on check/LOO"
        if not f["rank_ok"]:
            r["note"] = "rank deficient design (points too close/collinear)"
            r["admissible"] = False
        rows.append(r)
    cmp = pd.DataFrame(rows)
    # ---- choice: simplest model not clearly worse than the best (check points, else LOO)
    adm = cmp[(cmp.admissible) & (cmp.model.isin(MODELS))]
    basis = "check_rmse_r_m" if len(chk) >= 2 else "loo_rmse_r_m"
    summary["selection_basis"] = ("RMSE on check points (fit on control points only)" if basis == "check_rmse_r_m"
                                  else "leave-one-out RMSE over all points (fewer than 2 check points)")
    chosen = None
    if len(adm) and adm[basis].notna().any():
        best = adm[basis].min()
        tol = max(0.25 * best, 0.5 * pick, 0.03)
        summary["selection_rule"] = (f"first model in the order translation < similarity < affine whose {basis} <= best + "
                                     f"tolerance; tolerance = max(25% of best, half the picking uncertainty, 3 cm) = {tol:.3f} m")
        for mname in MODELS:
            row = adm[adm.model == mname]
            if len(row) and row[basis].iloc[0] <= best + tol:
                chosen = mname
                break
    cmp.to_csv(out_dir / f"model_comparison_{farm}.csv", index=False)
    summary["model_comparison"] = json.loads(cmp.to_json(orient="records"))
    if chosen is None:
        summary["status"] = "no model could be fitted (too few control points)"
        (out_dir / f"calibration_{farm}.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary
    # ---- final transform refit on ALL accepted points with the chosen model
    n_all = len(d)
    use = chosen
    if n_all < MIN_POINTS[use]:
        use = "translation"
    ff = fit(use, d.E.values, d.N.values, d.dE.values, d.dN.values, prior_sigma=prior)
    s, rot, ds, drot = scale_rot(ff)
    ext = dict(E=[d.E.min(), d.E.max()], N=[d.N.min(), d.N.max()])
    corners = [(a, b) for a in ext["E"] for b in ext["N"]]
    sd_c = float(pred_sd(ff, ff["origin_E"], ff["origin_N"])[0])
    sd_k = float(max(pred_sd(ff, a, b)[0] for a, b in corners))
    ffe, ffn = apply(ff, d.E.values, d.N.values)
    d["resid_after_E"], d["resid_after_N"] = d.E_rtk.values - ffe, d.N_rtk.values - ffn
    d["resid_before_r"] = np.hypot(d.dE, d.dN)
    d["resid_after_r"] = np.hypot(d.resid_after_E, d.resid_after_N)
    d[["point_id", "role", "E", "N", "E_rtk", "N_rtk", "dE", "dN", "resid_before_r", "resid_after_E", "resid_after_N",
       "resid_after_r"]].round(4).to_csv(out_dir / f"residuals_{farm}.csv", index=False)
    summary.update(
        status="ok", chosen_model=chosen, final_model=use, n_points_final_fit=n_all,
        chosen_transform=dict(
            model=use, direction="ortho EPSG:32649 -> RTK/WGS84-derived EPSG:32649", units="metres, degrees",
            formula="E' = origin_E + tE + m11 (E-origin_E) + m12 (N-origin_N); N' = origin_N + tN + m21 (E-origin_E) + m22 (N-origin_N)",
            origin_E=ff["origin_E"], origin_N=ff["origin_N"], tE_m=ff["tE"], tN_m=ff["tN"],
            m11=ff["M"][0, 0], m12=ff["M"][0, 1], m21=ff["M"][1, 0], m22=ff["M"][1, 1],
            scale=s, rotation_deg_ccw=rot, sd_scale=ds, sd_rotation_deg=drot,
            parameter_names=["tE", "tN"] + {"translation": [], "similarity": ["d(m11)=d(m22)", "m21"],
                                             "affine": ["d m11", "d m12", "d m21", "d m22"]}[use],
            parameter_covariance=ff["cov"].tolist(),
            sigma_hat_m=ff["sigma_hat_m"], sigma_used_m=ff["sigma_used_m"], dof=ff["dof"],
            position_sd_at_origin_m=sd_c, position_sd_worst_corner_of_points_bbox_m=sd_k),
        rmse_before_all_points_m=rmse(d.dE, d.dN)["rmse_r"], rmse_after_all_points_m=rmse(d.resid_after_E, d.resid_after_N)["rmse_r"],
        validation_check_points=dict(model=chosen, **{k: float(cmp[cmp.model == chosen][c].iloc[0]) for k, c in
                                                      [("check_rmse_r_m", "check_rmse_r_m"), ("check_max_m", "check_max_m"),
                                                       ("loo_rmse_r_m", "loo_rmse_r_m")]}),
        points_extent_m=dict(E=float(np.ptp(d.E)), N=float(np.ptp(d.N))),
        warnings=[w for w in [
            "fewer than 2 check points: validation relies on leave-one-out" if len(chk) < 2 else None,
            "similarity rotation/scale are weakly determined: points span < 100 m in one direction" if use != "translation" and min(np.ptp(d.E), np.ptp(d.N)) < 100 else None,
            "residual after correction is large compared with the picking uncertainty: local distortions that a global model cannot remove"
            if rmse(d.resid_after_E, d.resid_after_N)["rmse_r"] > 3 * max(pick, 0.05) else None,
            "final model refitted with fewer points than the chosen model needs: translation used" if use != chosen else None,
        ] if w])
    (out_dir / f"calibration_{farm}.json").write_text(json.dumps(summary, indent=2, default=float))
    if not quiet:
        print(cmp.round(3).to_string())
        print(f"chosen: {chosen} -> final {use}; shift at origin dE={ff['tE']:.3f} dN={ff['tN']:.3f} m, "
              f"rot={rot:.4f} deg, scale={s:.6f}; position sd {sd_c:.3f}/{sd_k:.3f} m")
    return summary


# ------------------------------------------------------------------ propagate
def _compose(cal, o_src, t1, M1, base_origin_ok=True):
    """Composite (flight -> base -> RTK). Flight frame origin o_src, flight->base: p_b = o + t1 + M1 (p - o)."""
    c = np.array([cal["origin_E"], cal["origin_N"]])
    M2 = np.array([[cal["m11"], cal["m12"]], [cal["m21"], cal["m22"]]])
    t2 = np.array([cal["tE_m"], cal["tN_m"]])
    o = np.asarray(o_src, float)
    M = M2 @ M1
    t = c + t2 + M2 @ (o + np.asarray(t1) - c) - o
    return o, t, M


def _row_apply(o, t, M, E, N):
    x, y = np.asarray(E) - o[0], np.asarray(N) - o[1]
    return o[0] + t[0] + M[0, 0] * x + M[0, 1] * y, o[1] + t[1] + M[1, 0] * x + M[1, 1] * y


def propagate(cal_json, out_dir, dataset=DS, do_footprints=True):
    cal_all = json.loads(Path(cal_json).read_text())
    if cal_all.get("status") != "ok":
        sys.exit("calibration status is not ok: " + str(cal_all.get("status")))
    sim = bool(cal_all.get("simulation"))
    cal = cal_all["chosen_transform"]
    base = cal_all["base_ortho_flight_id"]
    farm = cal_all["farm"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = (BANNER_SIM + " - ") if sim else ""
    tr = pd.read_csv(Path(dataset) / "metadata/ortho_registration/ortho_registration_transforms.csv")
    tr = tr[(tr.is_primary == 1) & (tr.model != "none")]
    mine = tr[tr.base_ortho_flight_id == base].copy()
    comp = {}
    rows = []
    # base itself: identity flight->base
    o_b = np.array([cal["origin_E"], cal["origin_N"]])
    comp[base] = _compose(cal, o_b, (0, 0), np.eye(2))
    rows.append(dict(flight_id=base, base_ortho_flight_id="RTK-calibrated (EPSG:32649)", model="base_correction_only",
                     source_registration_model="identity (is the base)", quality="(base)", caveats=""))
    for r in mine.itertuples():
        M1 = np.array([[r.m11, r.m12], [r.m21, r.m22]])
        comp[r.flight_id] = _compose(cal, (r.origin_E, r.origin_N), (r.tE, r.tN), M1)
        rows.append(dict(flight_id=r.flight_id, base_ortho_flight_id="RTK-calibrated (EPSG:32649)",
                         model="composed_" + r.model, source_registration_model=r.model, quality=r.quality,
                         caveats=r.caveats if isinstance(r.caveats, str) else ""))
    tabs = []
    for row in rows:
        o, t, M = comp[row["flight_id"]]
        s = float(np.sqrt(abs(np.linalg.det(M))))
        tabs.append(dict(row, is_primary=1, origin_E=o[0], origin_N=o[1], m11=M[0, 0], m12=M[0, 1], m21=M[1, 0], m22=M[1, 1],
                         tE=t[0], tN=t[1], scale=s, rotation_deg=float(np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1]))),
                         shift_E_at_origin_m=t[0], shift_N_at_origin_m=t[1], status="PROPOSED" if not sim else BANNER_SIM,
                         base_correction_model=cal["model"], base_correction_position_sd_m=cal["position_sd_at_origin_m"]))
    T = pd.DataFrame(tabs)
    skipped = tr[~tr.flight_id.isin(comp)]
    T.to_csv(out / f"PROPOSED_{farm}_flight_transforms_to_RTK.csv", index=False, encoding="utf-8-sig")
    written = [out / f"PROPOSED_{farm}_flight_transforms_to_RTK.csv"]
    # (iii) ortho box coordinates (centre_E/N, E1..N4) in the flight ortho frame
    farm_dirs = sorted(Path(dataset).glob("farm_*"))
    nbox = 0
    for fd in farm_dirs:
        bx = fd / "annotations/thermal_anomaly_obb/thermal_boxes.csv"
        if not bx.exists():
            continue
        b = pd.read_csv(bx)
        b = b[b.flight_id.isin(comp.keys())].copy()
        if b.empty:
            continue
        pieces = []
        for fid, g in b.groupby("flight_id"):
            o, t, M = comp[fid]
            g = g.copy()
            for ce, cn in [("centre_E", "centre_N")] + [(f"E{i}", f"N{i}") for i in range(1, 5)]:
                g[ce + "_rtk"], g[cn + "_rtk"] = _row_apply(o, t, M, g[ce].values, g[cn].values)
            pieces.append(g)
        g = pd.concat(pieces)
        g["status"] = "PROPOSED" if not sim else BANNER_SIM
        p = out / f"PROPOSED_thermal_boxes_{fd.name}_rtk.csv"
        g.to_csv(p, index=False)
        written.append(p)
        nbox += len(g)
    # (ii) thermal frame footprints
    nfp = 0
    if do_footprints:
        sys.path.insert(0, str(Path(__file__).parent))
        import build_footprints as bf
        from shapely import affinity
        fp = bf.read_gpkg_layer(Path(dataset) / "metadata/footprints/image_footprints.gpkg", "image_footprints")
        fp = fp[fp.flight_id.isin(comp.keys())].copy().drop(columns=["fid"], errors="ignore")
        geoms = []
        for fid, geom in zip(fp.flight_id, fp.geometry):
            o, t, M = comp[fid]
            # affine_transform matrix [a,b,d,e,xoff,yoff]: x' = a x + b y + xoff
            xoff = o[0] + t[0] - M[0, 0] * o[0] - M[0, 1] * o[1]
            yoff = o[1] + t[1] - M[1, 0] * o[0] - M[1, 1] * o[1]
            geoms.append(affinity.affine_transform(geom, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], xoff, yoff]))
        fp["geometry"] = geoms
        fp["status"] = "PROPOSED" if not sim else BANNER_SIM
        if len(fp):
            p = out / f"PROPOSED_{farm}_image_footprints_rtk.gpkg"
            bf.write_gpkg(p, {"image_footprints_rtk": (fp.reset_index(drop=True), "POLYGON")})
            written.append(p)
            nfp = len(fp)
    note = out / "README_PROPOSED.txt"
    note.write_text(
        f"{tag}PROPOSED tables only. Nothing in dataset/ was changed.\n"
        f"farm {farm}, base ortho {base}; base correction model {cal['model']}, position sd "
        f"{cal['position_sd_at_origin_m']:.3f} m at the point centroid ({cal['position_sd_worst_corner_of_points_bbox_m']:.3f} m worst corner).\n"
        f"{len(T)} flight transforms (composed = registration-to-base followed by the base correction), {nbox} boxes, {nfp} frame footprints.\n"
        "Flights whose registration to the base is 'weak' keep their registration error on top of the base correction; "
        "flights without a primary transform (e.g. TT 39M_T6) or registered to another base ortho are NOT included.\n"
        "Frame footprints are shifted with the transform of their flight, i.e. they are assumed to be expressed in the same "
        "frame as that flight's orthomosaic (ortho_registration/projection_chain.csv step 1); this is an assumption.\n"
        "Heights/DSM are not corrected. The correction is planar.\n", encoding="utf-8")
    written.append(note)
    return dict(written=[str(w) for w in written], n_transforms=len(T), n_boxes=nbox, n_footprints=nfp,
                not_included=sorted(set(skipped.flight_id))[:50])


# ------------------------------------------------------------------ simulation
def simulate(farm, out_dir, points_csv=None, shift=(2.0, -1.2), rot_deg=0.01, noise=0.03, seed=7, scale=1.0):
    need_pyproj()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(points_csv or KIT / "points_all.csv", encoding="utf-8-sig")
    pts = pts[pts.farm == farm].copy()
    rng = np.random.default_rng(seed)
    c = np.array([pts.E_ortho_m.mean(), pts.N_ortho_m.mean()])
    th = np.radians(rot_deg)
    R = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = pts[["E_ortho_m", "N_ortho_m"]].values
    T = c + np.array(shift) + (P - c) @ R.T + rng.normal(0, noise, P.shape)
    t = Transformer.from_crs(32649, 4326, always_xy=True)
    lon, lat = t.transform(T[:, 0], T[:, 1])
    m = pd.DataFrame(dict(point_id=pts.point_id.values, measured_lat=lat, measured_lon=lon,
                          ellipsoidal_height_m=100.0, fix_type="RTK fixed", notes=BANNER_SIM))
    mp = out / f"SIMULATION_fake_measurements_{farm}.csv"
    m.to_csv(mp, index=False)
    summ = calibrate(mp, points_csv or KIT / "points_all.csv", farm, out, sim=True, quiet=True)
    # recovery: compare the recovered transform with the truth over the points' extent
    ct = summ["chosen_transform"]
    f = dict(model=ct["model"], origin_E=ct["origin_E"], origin_N=ct["origin_N"], tE=ct["tE_m"], tN=ct["tN_m"],
             M=np.array([[ct["m11"], ct["m12"]], [ct["m21"], ct["m22"]]]))
    re_, rn_ = apply(f, P[:, 0], P[:, 1])
    tt = c + np.array(shift) + (P - c) @ R.T  # noise-free truth
    dev = np.hypot(re_ - tt[:, 0], rn_ - tt[:, 1])
    true_shift_at_origin = (c + np.array(shift) + (np.array([ct["origin_E"], ct["origin_N"]]) - c) @ R.T) - np.array([ct["origin_E"], ct["origin_N"]])
    res = dict(simulation=BANNER_SIM, farm=farm, truth=dict(shift_E_m=shift[0], shift_N_m=shift[1], rotation_deg=rot_deg, scale=scale,
                                                         noise_sd_m=noise, rotation_centre=c.tolist()),
               chosen_model=summ["chosen_model"], final_model=summ["final_model"],
               recovered=dict(shift_at_origin_E_m=ct["tE_m"], shift_at_origin_N_m=ct["tN_m"], rotation_deg=ct["rotation_deg_ccw"],
                              sd_rotation_deg=ct["sd_rotation_deg"], scale=ct["scale"], sd_scale=ct["sd_scale"]),
               true_shift_at_origin_m=true_shift_at_origin.tolist(),
               shift_error_at_origin_m=[ct["tE_m"] - true_shift_at_origin[0], ct["tN_m"] - true_shift_at_origin[1]],
               max_deviation_of_recovered_vs_true_transform_over_points_m=float(dev.max()),
               rmse_check_points_m=summ["validation_check_points"]["check_rmse_r_m"],
               position_sd_at_origin_m=ct["position_sd_at_origin_m"])
    (out / f"SIMULATION_recovery_{farm}.json").write_text(json.dumps(res, indent=2, default=float))
    return res, summ


def selftest():
    rng = np.random.default_rng(1)
    E = rng.uniform(0, 400, 9) + 265000
    N = rng.uniform(0, 300, 9) + 1263000
    dE = 2 + 0.0002 * (E - E.mean()) - 0.0001 * (N - N.mean())
    dN = -1 + 0.0001 * (E - E.mean()) + 0.0002 * (N - N.mean())
    f = fit("similarity", E, N, dE, dN)
    e, n = apply(f, E, N)
    assert np.allclose(e, E + dE, atol=1e-6) and np.allclose(n, N + dN, atol=1e-6)
    try:
        fit("affine", E[:5], N[:5], dE[:5], dN[:5])
        raise AssertionError("affine with 5 points must be refused")
    except ValueError:
        pass
    fa = fit("affine", E, N, dE, dN)
    assert np.allclose(apply(fa, E, N)[0], E + dE, atol=1e-6)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calibrate")
    c.add_argument("--measured", required=True)
    c.add_argument("--points", default=str(KIT / "points_all.csv"))
    c.add_argument("--farm", required=True)
    c.add_argument("--out", default=str(KIT / "results"))
    c.add_argument("--allow-float", action="store_true")
    c.add_argument("--spare-as", choices=["check", "control", "ignore"], default="check")
    c.add_argument("--rtk-sigma", type=float, default=0.03)
    p = sp.add_parser("propagate")
    p.add_argument("--calibration", required=True)
    p.add_argument("--out-dir", default=str(KIT / "proposed"))
    p.add_argument("--no-footprints", action="store_true")
    s = sp.add_parser("simulate")
    s.add_argument("--farm", default="TN19")
    s.add_argument("--out", default=str(KIT / "results" / "_simulation_test"))
    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.measured, a.points, a.farm, a.out, a.allow_float, a.spare_as, a.rtk_sigma)
    elif a.cmd == "propagate":
        print(json.dumps(propagate(a.calibration, a.out_dir, do_footprints=not a.no_footprints), indent=2))
    elif a.cmd == "simulate":
        print(BANNER_SIM)
        res, _ = simulate(a.farm, a.out)
        print(json.dumps(res, indent=2, default=float))
    else:
        selftest()


if __name__ == "__main__":
    main()
