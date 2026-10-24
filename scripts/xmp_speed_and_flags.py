#!/usr/bin/env python
"""XMP-speed cross-check of thermal ground speed + per-image quality flags.

Run (from the project root):
  python -X utf8 code/scripts/xmp_speed_and_flags.py [compare|flags|all]
    compare  read DJI XMP of all JPGs (first 128 KB, 16 threads) -> dataset/metadata/inventory/speed_xmp.csv
             + comparison tables on stdout (see README_speed_xmp.md)
    flags    -> image_flags.csv, image_flags_summary.csv (see README_flags.md)
    all      both (default)
  python -X utf8 code/scripts/xmp_speed_and_flags.py xmp   force a re-read of the XMP (cache lives in the OS temp dir)
"""
from __future__ import annotations

import re
import sqlite3
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
INV = DS / "metadata/inventory"
CACHE = Path(tempfile.gettempdir()) / "solar_uav_xmp_cache.csv"  # outside the project; delete to force a re-read
TAGS = ["FlightXSpeed", "FlightYSpeed", "FlightZSpeed", "GpsStatus", "RtkFlag", "RelativeAltitude"]
RX = {t: re.compile(rb'drone-dji:' + t.encode() + rb'="([^"]*)"') for t in TAGS}
SPEED_THR = 3.0
ALT_THR = 3.0
TILT_THR = 2.0
JUMP_MARGIN_RTK_M = 1.0     # position-noise margin, RTK flights (RTK std ~1 cm, generous)
JUMP_MARGIN_NORTK_M = 3.0   # BIM2 (no RTK; 1-3 m absolute scatter)
JUMP_MAX_DT = 10            # links with a timestamp gap > 10 s are not tested (turns / pauses / dropped frames)
TEMP_MARGIN_C = 5.0
FOOT = DS / "metadata/footprints/image_footprints.gpkg"


def read_one(p: Path) -> dict:
    with open(p, "rb") as f:
        b = f.read(131072)
    r = {"file": p.name, "flight_id": p.parent.name}
    for t in TAGS:
        m = RX[t].search(b)
        r[t] = m.group(1).decode() if m else None
    return r


def run_xmp() -> pd.DataFrame:
    paths = sorted(DS.glob("farm_0*/raw_images/**/*.JPG"))
    print("reading XMP of", len(paths), "images")
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(read_one, paths))
    d = pd.DataFrame(rows)
    d.to_csv(CACHE, index=False)
    return d


def load_xmp() -> pd.DataFrame:
    d = pd.read_csv(CACHE) if CACHE.exists() else run_xmp()
    for c in ["FlightXSpeed", "FlightYSpeed", "FlightZSpeed", "RelativeAltitude", "RtkFlag"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load_base() -> pd.DataFrame:
    """Inventory + XMP speeds, sorted per flight/sensor in time order, with UTM positions.

    DJI convention verified against the RTK positions (corr of consecutive-position velocity with the XMP values:
    FlightYSpeed vs d(Easting)/dt = 0.97, FlightXSpeed vs d(Northing)/dt = 0.83): X = north, Y = east, Z = down (NED).
    The ground speed hypot(X, Y) does not depend on that choice.
    """
    inv = pd.read_csv(INV / "image_inventory.csv")
    x = load_xmp()
    d = inv.merge(x[["flight_id", "file", "FlightXSpeed", "FlightYSpeed", "FlightZSpeed"]],
                  on=["flight_id", "file"], how="left", validate="1:1")
    d["speed_xmp"] = np.hypot(d.FlightXSpeed, d.FlightYSpeed)
    tr = Transformer.from_crs(4326, 32649, always_xy=True)
    d["E"], d["N"] = tr.transform(d.GpsLongitude.values, d.GpsLatitude.values)
    d["ts"] = pd.to_datetime(d.timestamp.astype(str), format="%Y%m%d%H%M%S")
    return d.sort_values(["flight_id", "sensor", "ts", "seq"], kind="stable").reset_index(drop=True)


def pct(s) -> float:
    return float(np.mean(s) * 100) if len(s) else np.nan


# ---------------------------------------------------------------- (A) speed comparison
def compare() -> pd.DataFrame:
    base = load_base()
    out = base[["flight_id", "file", "sensor", "speed_xmp", "FlightXSpeed", "FlightYSpeed", "FlightZSpeed"]].rename(
        columns={"speed_xmp": "speed_xmp_m_s", "FlightXSpeed": "vx", "FlightYSpeed": "vy", "FlightZSpeed": "vz"})
    out = out.sort_values(["flight_id", "file"]).round(3)
    out.to_csv(INV / "speed_xmp.csv", index=False)
    print("speed_xmp.csv rows", len(out), "with XMP", int(out.speed_xmp_m_s.notna().sum()))

    ts = pd.read_csv(INV / "thermal_speed.csv")
    t = base[base.sensor == "T"].merge(ts, on=["flight_id", "file"], how="left", validate="1:1")
    t = t.sort_values(["flight_id", "ts", "seq"])
    # XMP speed averaged over the same centred 5-image window as speed_smoothed_m_s
    t["xmp_win5"] = t.groupby("flight_id").speed_xmp.transform(lambda s: s.rolling(5, center=True, min_periods=5).mean())
    rows = []
    for f, g in t[t.speed_xmp.notna()].groupby("flight_id"):
        a = g.dropna(subset=["speed_smoothed_m_s"])
        b = g.dropna(subset=["speed_m_s"])
        fx = (a.speed_xmp > SPEED_THR).values
        fg = (a.exceeds_3ms == 1).values
        rows.append(dict(
            flight_id=f, n_thermal=len(g), n_gps_valid=len(a),
            r_raw=b.speed_xmp.corr(b.speed_m_s), r_smooth=a.speed_xmp.corr(a.speed_smoothed_m_s),
            r_sm_vs_xmp5=a.xmp_win5.corr(a.speed_smoothed_m_s),
            bias_sm=(a.speed_smoothed_m_s - a.speed_xmp).mean(),
            sd_raw_diff=(b.speed_m_s - b.speed_xmp).std(), sd_sm_diff=(a.speed_smoothed_m_s - a.speed_xmp).std(),
            xmp_med=g.speed_xmp.median(), xmp_p95=g.speed_xmp.quantile(.95), gps_sm_p95=a.speed_smoothed_m_s.quantile(.95),
            gt3_xmp_all=pct(g.speed_xmp > SPEED_THR), gt3_xmp_same=pct(fx), gt3_gps=pct(fg), differs=pct(fx != fg),
            n_xmp_gt3=int((g.speed_xmp > SPEED_THR).sum()), n_xmp_gt5=int((g.speed_xmp > 5).sum()),
            n_gps_nan=int(g.speed_smoothed_m_s.isna().sum())))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(r.round(2).to_string())
    t = t[t.speed_xmp.notna()]
    a = t.dropna(subset=["speed_smoothed_m_s"])
    b = t.dropna(subset=["speed_m_s"])
    print("\nPOOLED thermal with XMP:", len(t), "| GPS-smoothed valid:", len(a), "| GPS raw valid:", len(b))
    print("r_raw", b.speed_xmp.corr(b.speed_m_s), "r_smooth", a.speed_xmp.corr(a.speed_smoothed_m_s),
          "r_smooth_vs_xmp_win5", a.xmp_win5.corr(a.speed_smoothed_m_s))
    d = a.speed_smoothed_m_s - a.speed_xmp
    print("bias(smoothed-xmp)", d.mean(), "MAE", d.abs().mean(), "sd", d.std())
    print("raw: bias", (b.speed_m_s - b.speed_xmp).mean(), "sd", (b.speed_m_s - b.speed_xmp).std())
    print("share>3: xmp(all)", pct(t.speed_xmp > 3), "| same images: xmp", pct(a.speed_xmp > 3), "gps", pct(a.exceeds_3ms == 1),
          "| flags differ", pct((a.speed_xmp > 3) != (a.exceeds_3ms == 1)))
    a = a.assign(fx=a.speed_xmp > 3, fg=a.exceeds_3ms == 1)
    print(a.groupby("farm").apply(lambda z: pd.Series(dict(
        n=len(z), xmp=pct(z.fx), gps=pct(z.fg), differs=pct(z.fx != z.fg),
        gps1_xmp0=int((z.fg & ~z.fx).sum()), gps0_xmp1=int((~z.fg & z.fx).sum())))).round(1))
    for thr in (3.05, 3.15):
        print("share xmp >", thr, round(pct(t.speed_xmp > thr), 2))
    vc = t.speed_xmp.round(1).value_counts()
    print("XMP values near 3:", vc[(vc.index >= 2.8) & (vc.index <= 3.3)].sort_index().to_dict())
    first = t.groupby("flight_id").head(1).set_index("flight_id").speed_xmp.round(1)
    print("XMP speed of first thermal frame per flight:", first.to_dict())
    print("thermal images where GPS smoothed is NaN:", int(t.speed_smoothed_m_s.isna().sum()), "of", len(t))
    w = base[(base.sensor == "W") & base.speed_xmp.notna()]
    print("RGB XMP share>3 by flight:", w.groupby("flight_id").speed_xmp.apply(lambda s: round(pct(s > 3), 1)).to_dict())
    print("XMP missing:", base[base.speed_xmp.isna()].groupby(["farm", "sensor"]).size().to_dict())
    return r


# ---------------------------------------------------------------- (B) flags
def read_gpkg_attrs() -> pd.DataFrame:
    c = sqlite3.connect(FOOT)
    try:
        return pd.read_sql("select flight_id,file,tilt_deg,alt_outlier,centre_in_ortho,frac_outside_ortho,footprint_mode "
                           "from image_footprints", c)
    finally:
        c.close()


def jump_flags(b: pd.DataFrame) -> pd.DataFrame:
    """GPS-jump rule (also documented in README_flags.md).

    Link test: consecutive images of one flight+sensor (time order) with 1 <= dt <= 10 s; d = horizontal distance.
    Timestamps have 1 s resolution so the true interval is at most dt+1 s and the possible distance is bounded by
        d_max = v_ref * (dt + 1) + margin
    v_ref = max(XMP ground speed of the two images) for XMP flights; for BIM2 (no XMP) the p95 of d/dt of that
    flight+sensor. margin = 1 m (RTK) / 3 m (BIM2). A link with d > d_max is 'bad' and the later image is flagged,
    unless the previous link was also bad and the track returns (|P_i - P_{i-2}| <= both d_max): then image i-1 is
    the isolated spike and only i-1 stays flagged.
    Pair test: every exact/nearest RGB-thermal pair with delta_pos_m > max(v of both) * (|delta_t|+1) + margin
    flags both images.
    """
    b = b.copy()
    g = b.groupby(["flight_id", "sensor"])
    b["dt"] = g.ts.diff().dt.total_seconds()
    b["d"] = np.hypot(g.E.diff(), g.N.diff())
    b["v_prev"] = g.speed_xmp.shift()
    ok = (b.dt >= 1) & (b.dt <= JUMP_MAX_DT)
    implied = (b.d / b.dt).where(ok)
    p95 = implied.groupby([b.flight_id, b.sensor]).transform(lambda s: s.quantile(.95))
    vref = np.fmax(b.speed_xmp, b.v_prev)
    vref = vref.where(vref.notna(), p95)
    margin = np.where(b.RtkFlag.isna(), JUMP_MARGIN_NORTK_M, JUMP_MARGIN_RTK_M)
    b["dmax"] = vref * (b.dt + 1) + margin
    bad = ok & (b.d > b.dmax)
    b["bad"] = bad
    g = b.groupby(["flight_id", "sensor"])
    prev_bad = g.bad.shift(fill_value=False).astype(bool)
    ret = np.hypot(b.E - g.E.shift(2), b.N - g.N.shift(2)) <= (b.dmax + g.dmax.shift())
    flag = bad & ~(prev_bad & ret)
    key = b.flight_id + "/" + b.file
    jump = pd.Series(flag.values, index=key.values)
    pr = pd.read_csv(DS / "metadata/pairing/rgb_thermal_pairing.csv")
    pr = pr[pr.pairing_status.isin(["exact_pair", "nearest_pair"])]
    sp = b.set_index(["flight_id", "file"]).speed_xmp
    sv = lambda fl, fi: sp.reindex(pd.MultiIndex.from_arrays([fl, fi])).values
    p95f = implied.groupby(b.flight_id).quantile(.95)
    vp = np.fmax(sv(pr.flight_id, pr.rgb_file), sv(pr.flight_id, pr.thermal_file))
    vp = np.where(np.isnan(vp), pr.flight_id.map(p95f).values, vp)
    mp = np.where(pr.flight_id.str.startswith("BIM2"), JUMP_MARGIN_NORTK_M, JUMP_MARGIN_RTK_M)
    pbad = pr.delta_pos_m.values > vp * (pr.delta_t_s.abs().values + 1) + mp
    print("tested links:", int(ok.sum()), "bad links:", int(bad.sum()), "| tested pairs:", len(pr), "bad pairs:", int(pbad.sum()))
    for _, r in pr[pbad].iterrows():
        jump.loc[r.flight_id + "/" + r.rgb_file] = True
        jump.loc[r.flight_id + "/" + r.thermal_file] = True
    b["flag_gps_jump"] = jump.reindex(key.values).values.astype(int)
    b["slack"] = (b.d - b.dmax).where(ok)
    return b[["flight_id", "file", "flag_gps_jump", "slack"]]


def flags() -> None:
    b = load_base()
    ts = pd.read_csv(INV / "thermal_speed.csv")
    tp = pd.read_csv(DS / "metadata/thermal/thermal_parameters.csv")
    pr = pd.read_csv(DS / "metadata/pairing/rgb_thermal_pairing.csv")
    fp = read_gpkg_attrs()
    assert len(b) == 19934 and not b.duplicated(["flight_id", "file"]).any()
    # 1 altitude (XMP RelativeAltitude == inventory RelativeAltitude)
    med = b.groupby("flight_id").RelativeAltitude.transform("median")
    b["flag_altitude_outlier"] = ((b.RelativeAltitude - med).abs() > ALT_THR).astype(int)
    # 2 tilt and 4 ortho (footprint gpkg)
    b = b.merge(fp, on=["flight_id", "file"], how="left", validate="1:1")
    assert b.tilt_deg.notna().all() and b.centre_in_ortho.notna().all()
    b["flag_tilt_gt2deg"] = (b.tilt_deg > TILT_THR).astype(int)
    b["flag_outside_ortho"] = (b.centre_in_ortho == 0).astype(int)
    print("alt flag vs gpkg alt_outlier mismatches:", int((b.flag_altitude_outlier != b.alt_outlier).sum()))
    # 5 resolution: size differs from the modal size of (model, sensor)
    size = b.width.astype(str) + "x" + b.height.astype(str)
    modal = size.groupby([b.model, b.sensor]).transform(lambda s: s.mode().iloc[0])
    b["flag_unusual_resolution"] = (size != modal).astype(int)
    # 6 speed: XMP; BIM2 thermal (no XMP): GPS smoothed speed. Thermal images only (3 m/s is the thermal screening threshold)
    b = b.merge(ts[["flight_id", "file", "speed_smoothed_m_s"]], on=["flight_id", "file"], how="left")
    b["speed_m_s_used"] = b.speed_xmp
    b["speed_source"] = np.where(b.speed_xmp.notna(), "xmp", "none")
    gps = b.speed_xmp.isna() & (b.sensor == "T") & b.speed_smoothed_m_s.notna()
    b.loc[gps, "speed_m_s_used"] = b.speed_smoothed_m_s[gps]
    b.loc[gps, "speed_source"] = "gps_smoothed"
    b["flag_speed_gt3"] = ((b.sensor == "T") & (b.speed_m_s_used > SPEED_THR)).astype(int)
    # 7 unpaired
    m = pd.concat([pr[["flight_id", "rgb_file", "pairing_status", "pair_method"]].rename(columns={"rgb_file": "file"}),
                   pr[["flight_id", "thermal_file", "pairing_status", "pair_method"]].rename(columns={"thermal_file": "file"})])
    m = m.dropna(subset=["file"]).drop_duplicates(["flight_id", "file"])
    b = b.merge(m, on=["flight_id", "file"], how="left", validate="1:1")
    b["flag_unpaired"] = (b.pairing_status.isna() | b.pairing_status.astype(str).str.startswith("unpaired")).astype(int)
    b["unpaired_reason"] = np.where(b.flag_unpaired == 1, b.pair_method.fillna("not_in_pairing_csv"), "")
    # 8 thermal extreme
    tp = tp[tp.decode_ok.astype(str) == "True"]
    p999 = tp.groupby("flight_id").temp_max_c.quantile(.999).rename("t_p999")
    b = b.merge(tp[["flight_id", "file", "temp_min_c", "temp_max_c"]], on=["flight_id", "file"], how="left", validate="1:1")
    b = b.merge(p999, on="flight_id", how="left")
    hi = b.temp_max_c > b.t_p999 + TEMP_MARGIN_C
    lo = b.temp_min_c < 0
    b["flag_thermal_extreme"] = ((hi | lo) & (b.sensor == "T")).astype(int)
    print("thermal images without decoded temperatures:", int(((b.sensor == "T") & b.temp_max_c.isna()).sum()),
          "| above p99.9+margin:", int(hi.sum()), "| min < 0 C:", int(lo.sum()))
    # 3 gps jump
    j = jump_flags(b)
    b = b.merge(j, on=["flight_id", "file"], how="left", validate="1:1")
    print("largest link slack d - d_max (negative = consistent):", round(float(b.slack.max()), 2))

    cols = ["flag_altitude_outlier", "flag_tilt_gt2deg", "flag_gps_jump", "flag_outside_ortho", "flag_unusual_resolution",
            "flag_speed_gt3", "flag_unpaired", "flag_thermal_extreme"]
    b["n_flags"] = b[cols].sum(axis=1)
    out = b[["farm", "flight_id", "file", "sensor"] + cols + ["n_flags", "speed_m_s_used", "speed_source", "unpaired_reason"]]
    out = out.sort_values(["farm", "flight_id", "file"]).round({"speed_m_s_used": 2})
    assert len(out) == 19934
    out.to_csv(INV / "image_flags.csv", index=False)

    def summ(df, level, fid):
        r = {"level": level, "farm": df.farm.iloc[0] if level != "all" else "ALL", "flight_id": fid, "n_images": len(df),
             "n_thermal": int((df.sensor == "T").sum()), "n_rgb": int((df.sensor == "W").sum())}
        r.update({c: int(df[c].sum()) for c in cols})
        r["n_images_any_flag"] = int((df.n_flags > 0).sum())
        return r
    rows = [summ(g, "flight", f) for f, g in out.groupby("flight_id")]
    rows += [summ(g, "farm", "ALL_" + f) for f, g in out.groupby("farm")]
    rows.append(summ(out, "all", "ALL"))
    sm = pd.DataFrame(rows)
    sm.to_csv(INV / "image_flags_summary.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(sm.drop(columns=["level", "farm"]).rename(columns=lambda c: c.replace("flag_", "")).to_string())
    print("\nunpaired by sensor/reason:\n", out[out.flag_unpaired == 1].groupby(["sensor", "unpaired_reason"]).size())
    print("speed source:\n", out.groupby(["farm", "sensor", "speed_source"]).size())
    print("n_flags distribution:", out.n_flags.value_counts().sort_index().to_dict())
    for c in ["flag_unusual_resolution", "flag_thermal_extreme", "flag_gps_jump", "flag_outside_ortho"]:
        print(c, out[out[c] == 1][["flight_id", "file"]].head(25).values.tolist())
    print("altitude outliers by flight:", out[out.flag_altitude_outlier == 1].groupby("flight_id").size().to_dict())
    print("tilt flags by flight:", out[out.flag_tilt_gt2deg == 1].groupby("flight_id").size().to_dict())
    print("speed flag by farm/sensor:", out.groupby(["farm", "sensor"]).flag_speed_gt3.sum().to_dict())
    print("co-occurrence:\n", out[cols].T.dot(out[cols]).rename(columns=lambda c: c[5:12], index=lambda c: c[5:]).to_string())
    drop_train = ["flag_altitude_outlier", "flag_tilt_gt2deg", "flag_gps_jump", "flag_outside_ortho",
                  "flag_unusual_resolution", "flag_speed_gt3", "flag_thermal_extreme"]
    drop_alt = ["flag_altitude_outlier", "flag_tilt_gt2deg", "flag_gps_jump", "flag_outside_ortho",
                "flag_unusual_resolution", "flag_thermal_extreme"]
    for name, ex in (("detection training", drop_train), ("cross-altitude comparison", drop_alt)):
        keep = out[ex].sum(axis=1) == 0
        print(name, "kept", int(keep.sum()), "thermal", int((keep & (out.sensor == "T")).sum()),
              "rgb", int((keep & (out.sensor == "W")).sum()), "| per farm kept:", out[keep].groupby("farm").size().to_dict())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "xmp":
        run_xmp()
    if cmd in ("compare", "all"):
        compare()
    if cmd in ("flags", "all"):
        flags()
