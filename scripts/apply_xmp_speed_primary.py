#!/usr/bin/env python
"""Make the DJI XMP flight-controller speed the primary speed standard in the release metadata.

Run after build_flight_manifest.py and xmp_speed_and_flags.py compare (which writes speed_xmp.csv):
  python -X utf8 code/scripts/apply_xmp_speed_primary.py

Policy: H20T flights (TN19, TNDL, TT) -> speed_source = 'xmp' (horizontal speed hypot(FlightXSpeed, FlightYSpeed)).
        BIM2 (Mavic 2 EA, no speed fields in XMP) -> speed_source = 'gps_unverified' (GPS-derived, 1 s timestamps).
Edits (idempotent):
  thermal_speed.csv      + speed_xmp_m_s, speed_source, exceeds_3ms_xmp, exceeds_3ms_gps (= legacy exceeds_3ms, kept as is)
  flight_manifest.csv    n_/pct_thermal_exceeds_3ms now follow speed_source; + pct_exceeds_3ms_source, *_gps, *_xmp columns
  environment/flight_environment_context.csv  pct_thermal_exceeds_3ms + pct_exceeds_3ms_source merged from the manifest
Legacy GPS columns (speed_m_s, speed_smoothed_m_s, exceeds_3ms) are NOT renamed because register_rgb_thermal.py reads them.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "dataset/metadata/inventory"
THR = 3.0


def main():
    ts = pd.read_csv(INV / "thermal_speed.csv")
    ts = ts[["flight_id", "file", "speed_m_s", "speed_smoothed_m_s", "exceeds_3ms"]]       # drop previously added columns
    sx = pd.read_csv(INV / "speed_xmp.csv")[["flight_id", "file", "speed_xmp_m_s"]]
    ts = ts.merge(sx, on=["flight_id", "file"], how="left", validate="1:1")
    has = ts.speed_xmp_m_s.notna()
    ts["speed_source"] = np.where(has, "xmp", "gps_unverified")
    ts["exceeds_3ms_xmp"] = pd.array(np.where(has, (ts.speed_xmp_m_s > THR).astype(float), np.nan), dtype="Int64")
    ts["exceeds_3ms_gps"] = ts.exceeds_3ms.astype("Int64")
    ts = ts[["flight_id", "file", "speed_xmp_m_s", "speed_source", "exceeds_3ms_xmp",
             "speed_m_s", "speed_smoothed_m_s", "exceeds_3ms_gps", "exceeds_3ms"]]
    ts["exceeds_3ms"] = ts.exceeds_3ms.astype("Int64")
    ts.to_csv(INV / "thermal_speed.csv", index=False)

    man = pd.read_csv(INV / "flight_manifest.csv")
    new = {}
    for fid, g in ts.groupby("flight_id"):
        gv = g.exceeds_3ms_gps.notna()          # same rounding as the legacy flag (thermal_speed.csv stores 3 decimals)
        xv = g.speed_xmp_m_s.notna()
        n_gps = int((g.exceeds_3ms_gps == 1).sum())
        d = dict(n_thermal_exceeds_3ms_gps=n_gps, pct_thermal_exceeds_3ms_gps=round(100 * n_gps / max(gv.sum(), 1), 1))
        if xv.any():
            n_x = int((g.speed_xmp_m_s > THR).sum())
            d.update(n_thermal_xmp_valid=int(xv.sum()), n_thermal_exceeds_3ms_xmp=n_x,
                     pct_thermal_exceeds_3ms_xmp=round(100 * n_x / xv.sum(), 1),
                     thermal_speed_xmp_p95_m_s=round(float(g.speed_xmp_m_s.quantile(.95)), 2),
                     pct_exceeds_3ms_source="xmp", n_thermal_exceeds_3ms=n_x,
                     pct_thermal_exceeds_3ms=round(100 * n_x / xv.sum(), 1))
        else:
            d.update(pct_exceeds_3ms_source="gps_unverified", n_thermal_exceeds_3ms=n_gps,
                     pct_thermal_exceeds_3ms=d["pct_thermal_exceeds_3ms_gps"])
        new[fid] = d
    add = pd.DataFrame.from_dict(new, orient="index").rename_axis("flight_id").reset_index()
    man = man.drop(columns=[c for c in add.columns if c != "flight_id" and c in man.columns]).merge(add, on="flight_id", how="left")
    man["n_thermal_xmp_valid"] = man.n_thermal_xmp_valid.astype("Int64")
    man["n_thermal_exceeds_3ms_xmp"] = man.n_thermal_exceeds_3ms_xmp.astype("Int64")
    man.to_csv(INV / "flight_manifest.csv", index=False)

    ep = ROOT / "dataset/metadata/environment/flight_environment_context.csv"
    env = pd.read_csv(ep)
    env = env.drop(columns=[c for c in ("pct_thermal_exceeds_3ms", "pct_exceeds_3ms_source") if c in env.columns])
    env = env.merge(man[["flight_id", "pct_thermal_exceeds_3ms", "pct_exceeds_3ms_source"]], on="flight_id", how="left")
    env.to_csv(ep, index=False)

    t = ts.merge(man[["flight_id"]].assign(farm=man.flight_id.str.split("_").str[0]), on="flight_id")
    print(t.groupby(["farm", "speed_source"]).apply(lambda z: pd.Series(dict(
        n=len(z), xmp_gt3=(z.speed_xmp_m_s > THR).sum(), gps_gt3=(z.exceeds_3ms_gps == 1).sum(),
        gps_valid=z.exceeds_3ms_gps.notna().sum()))))
    h = t[t.speed_source == "xmp"].groupby("farm").speed_xmp_m_s.apply(lambda s: round(100 * (s > THR).mean(), 1))
    print("XMP %>3 per farm (all thermal with XMP):", h.to_dict())
    print(man[["flight_id", "pct_thermal_exceeds_3ms", "pct_exceeds_3ms_source", "pct_thermal_exceeds_3ms_gps"]].to_string())


if __name__ == "__main__":
    main()
