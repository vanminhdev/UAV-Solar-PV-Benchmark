#!/usr/bin/env python
"""Build dataset/metadata/inventory/flight_manifest.csv and thermal_speed.csv.

Inputs : dataset/metadata/inventory/image_inventory.csv, dataset/farm_*/photogrammetry/<flight_id>/*.tif
Outputs: flight_manifest.csv (one row per flight_id), thermal_speed.csv (one row per thermal image)
Usage  : python -X utf8 build_flight_manifest.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2] / "dataset"
INV = ROOT / "metadata" / "inventory"
GAP_S = 10          # drop dt larger than this (turns, pauses)
WIN = 5             # smoothing window (images)
THRESH = 3.0        # m/s smearing criterion
TAU_S = 0.010       # assumed microbolometer time constant for informative smear estimate

# (pitch_um, focal_mm, nominal?) -- H20T: RGB 1.55 um f=4.5; thermal 12 um f=13.5. Mavic 2 EA nominal.
CAM = {
    "ZH20T": dict(platform="DJI Zenmuse H20T payload (airframe not recorded)", rgb=(1.55, 4.5), th=(12.0, 13.5), nominal=False),
    "MAVIC2-ENTERPRISE-ADVANCED": dict(platform="DJI Mavic 2 Enterprise Advanced (nominal specs)",
                                       rgb=(6.4e3 / 8000, 4.5), th=(12.0, 9.0), nominal=True),
}
tr = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)


def speeds(g):
    """g: images of one flight+sensor. Returns frame with speed, speed_smoothed (m/s)."""
    g = g.sort_values(["timestamp", "file"]).copy()
    t = pd.to_datetime(g.timestamp.astype(str), format="%Y%m%d%H%M%S").astype("int64").to_numpy() / 1e9
    x, y = tr.transform(g.GpsLongitude.to_numpy(), g.GpsLatitude.to_numpy())
    d = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    dt = np.r_[np.nan, np.diff(t)]
    ok = (dt > 0) & (dt <= GAP_S)
    v = np.where(ok, d / np.where(dt > 0, dt, np.nan), np.nan)
    # smoothed: path length / elapsed time over a centred window of WIN images, only across valid links
    n, h = len(g), WIN // 2
    vs = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - h), min(n - 1, i + h)
        links = np.arange(lo + 1, hi + 1)
        links = links[ok[links]]
        if len(links) >= 2 and len(links) == hi - lo:        # complete, gap-free window (>=3 images)
            vs[i] = d[links].sum() / dt[links].sum()
    g["speed_m_s"], g["speed_smoothed_m_s"] = v, vs
    return g


def tif_info(fid, farm_dir):
    d = farm_dir / "photogrammetry" / fid
    tifs = sorted(d.glob("*.tif")) if d.is_dir() else []
    good = []
    for p in tifs:
        try:
            with open(p, "rb") as f:
                m = f.read(4)
            if p.stat().st_size > 0 and m[:2] in (b"II", b"MM"):
                good.append(p.name)
        except OSError:
            pass
    return (any("orthomosaic" in n for n in good), any("dsm" in n for n in good), len(good))


def main():
    inv = pd.read_csv(INV / "image_inventory.csv")
    farm_dirs = {p.name.split("_")[1]: p for p in ROOT.glob("farm_*")}
    farm_code = {"TN19": "farm_01_thuan_nam_19", "BIM2": "farm_02_bim_2",
                 "TNDL": "farm_03_thuan_nam_duc_long", "TT": "farm_04_thien_tan"}
    inv["dt"] = pd.to_datetime(inv.timestamp.astype(str), format="%Y%m%d%H%M%S")

    thermal_rows, speed_all = [], {}
    for (fid, s), g in inv.groupby(["flight_id", "sensor"]):
        sp = speeds(g)
        speed_all[(fid, s)] = sp
        if s == "T":
            thermal_rows.append(sp[["flight_id", "file", "speed_m_s", "speed_smoothed_m_s"]])
    th = pd.concat(thermal_rows, ignore_index=True)
    th["exceeds_3ms"] = np.where(th.speed_smoothed_m_s.notna(), (th.speed_smoothed_m_s > THRESH).astype(int), np.nan)
    th["exceeds_3ms"] = th.exceeds_3ms.astype("Int64")
    th.round({"speed_m_s": 3, "speed_smoothed_m_s": 3}).to_csv(INV / "thermal_speed.csv", index=False)

    rows = []
    for fid, g in inv.groupby("flight_id", sort=False):
        farm = g.farm.iloc[0]
        cam = CAM[g.model.iloc[0]]
        ra = g.RelativeAltitude
        alt = float(ra.median())
        w = g[g.sensor == "W"]
        wmode = int(w.width.mode().iloc[0]) if len(w) else None
        rgb_pitch, rgb_f = cam["rgb"]
        if wmode and not cam["nominal"] and wmode != 4056:      # reduced-resolution RGB: scale pitch
            rgb_pitch = rgb_pitch * 4056 / wmode
        gsd_rgb = alt * rgb_pitch / rgb_f / 10 if len(w) else np.nan
        has_t = (g.sensor == "T").any()
        gsd_t = alt * cam["th"][0] / cam["th"][1] / 10 if has_t else np.nan
        prim = "T" if has_t else "W"
        sp = speed_all[(fid, prim)]
        tsp = speed_all[(fid, "T")] if has_t else None
        ortho, dsm, ntif = tif_info(fid, ROOT / farm_code[farm])
        r = dict(
            farm=farm, flight_id=fid, date=g.dt.min().strftime("%Y-%m-%d"),
            start_time=g.dt.min().strftime("%H:%M:%S"), end_time=g.dt.max().strftime("%H:%M:%S"),
            duration_min=round((g.dt.max() - g.dt.min()).total_seconds() / 60, 1),
            platform_camera=cam["platform"], camera_model_exif=g.model.iloc[0],
            specs_nominal=cam["nominal"], rgb_width_px=wmode,
            nominal_altitude_m=int(round(alt)), rel_alt_median=round(alt, 2),
            rel_alt_iqr=round(ra.quantile(.75) - ra.quantile(.25), 2),
            rel_alt_min=round(ra.min(), 2), rel_alt_max=round(ra.max(), 2),
            rgb_count=int((g.sensor == "W").sum()), thermal_count=int((g.sensor == "T").sum()),
            gsd_rgb_cm=round(gsd_rgb, 3), gsd_thermal_cm=round(gsd_t, 3),
            rtk_fix_pct=round(100 * (g.RtkFlag == 50).mean(), 1) if g.RtkFlag.notna().any() else np.nan,
            gimbal_pitch_median=round(g.GimbalPitchDegree.median(), 1),
            n_pitch_off_nadir_gt2deg=int((np.abs(g.GimbalPitchDegree + 90) > 2).sum()),
            has_thermal=bool(has_t), has_photogrammetry_products=ntif > 0,
            has_orthomosaic=ortho, has_dsm=dsm, n_tif=ntif,
            mean_ground_speed_m_s=round(sp.speed_m_s.mean(), 2),
            mean_ground_speed_sensor=prim,
        )
        if has_t:
            v = tsp.speed_m_s.dropna(); vs = tsp.speed_smoothed_m_s.dropna()
            ex = tsp.speed_smoothed_m_s   # exceedance judged on the smoothed speed only
            nval = ex.notna().sum()
            r.update(thermal_speed_median_m_s=round(v.median(), 2), thermal_speed_p95_m_s=round(v.quantile(.95), 2),
                     thermal_speed_smoothed_median_m_s=round(vs.median(), 2),
                     thermal_speed_smoothed_p95_m_s=round(vs.quantile(.95), 2),
                     n_thermal_speed_valid=int(nval), n_thermal_exceeds_3ms=int((ex > THRESH).sum()),
                     pct_thermal_exceeds_3ms=round(100 * (ex > THRESH).sum() / max(nval, 1), 1),
                     smear_px_p95_tau10ms=round(vs.quantile(.95) * TAU_S / (gsd_t / 100), 3),
                     speed_threshold_m_s=THRESH)
        rows.append(r)
    man = pd.DataFrame(rows).sort_values(["farm", "flight_id"])
    man.to_csv(INV / "flight_manifest.csv", index=False)
    print(man[["flight_id", "nominal_altitude_m", "rgb_count", "thermal_count", "gsd_rgb_cm", "gsd_thermal_cm",
               "thermal_speed_p95_m_s", "n_thermal_exceeds_3ms", "n_thermal_speed_valid",
               "has_photogrammetry_products", "mean_ground_speed_m_s"]].to_string(index=False))
    print("thermal images:", len(th), "exceed:", int((th.exceeds_3ms == 1).sum()))


if __name__ == "__main__":
    main()
