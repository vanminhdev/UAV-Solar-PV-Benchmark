"""Fetch hourly reanalysis weather/irradiance context for each farm and flight (NOT in-situ measurements).

    python fetch_environment_context.py <dataset_root>

Source: Open-Meteo Historical Weather API (archive-api.open-meteo.com; ERA5/ERA5-Land reanalysis based, ~0.25 deg
grid, hourly; CC BY 4.0, https://open-meteo.com). One request per farm (centroid of the image GPS positions rounded
to 0.001 deg) and day. Writes dataset/metadata/environment/{reanalysis_hourly.csv, flight_environment_context.csv}.
The values are model/reanalysis context for the flight window, not plane-of-array irradiance and not site
measurements; irradiance is horizontal (GHI), the module tilt is unknown.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_gusts_10m", "cloud_cover"]


def fetch(lat, lon, d0, d1):
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=d0, end_date=d1, hourly=",".join(VARS),
                                    timezone="Asia/Bangkok", wind_speed_unit="ms"))
    with urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=60) as r:
        return json.loads(r.read())


def main():
    ds = Path(sys.argv[1])
    out = ds / "metadata" / "environment"
    out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(ds / "metadata/inventory/image_inventory.csv", usecols=["farm", "GpsLatitude", "GpsLongitude"], low_memory=False)
    cen = inv.groupby("farm")[["GpsLatitude", "GpsLongitude"]].mean().round(3)
    man = pd.read_csv(ds / "metadata/inventory/flight_manifest.csv")
    rows = []
    for farm, g in man.groupby("farm"):
        lat, lon = cen.loc[farm]
        j = fetch(lat, lon, g.date.min(), g.date.max())
        h = pd.DataFrame(j["hourly"])
        h.insert(0, "farm", farm)
        h["grid_lat"], h["grid_lon"], h["grid_elevation_m"] = j["latitude"], j["longitude"], j.get("elevation")
        rows.append(h)
    hr = pd.concat(rows, ignore_index=True)
    hr["time"] = pd.to_datetime(hr["time"])
    hr.to_csv(out / "reanalysis_hourly.csv", index=False)
    ctx = []
    for r in man.itertuples():
        t0 = pd.Timestamp(f"{r.date} {r.start_time}")
        t1 = pd.Timestamp(f"{r.date} {r.end_time}")
        h = hr[hr.farm == r.farm].set_index("time")
        grid = pd.date_range(t0, t1, freq="1min")
        rec = dict(farm=r.farm, flight_id=r.flight_id, date=r.date, start_local=r.start_time, end_local=r.end_time)
        for v in VARS:
            s = h[v].reindex(h.index.union(grid)).interpolate("time").reindex(grid)
            rec[v + "_mean"] = round(float(s.mean()), 1)
        rec["shortwave_radiation_min"] = round(float(h["shortwave_radiation"].reindex(h.index.union(grid)).interpolate("time").reindex(grid).min()), 1)
        ctx.append(rec)
    ctx = pd.DataFrame(ctx).merge(man[["flight_id", "gsd_thermal_cm", "thermal_speed_smoothed_p95_m_s", "pct_thermal_exceeds_3ms", "pct_exceeds_3ms_source"]], on="flight_id")  # XMP-primary after apply_xmp_speed_primary.py
    ctx.to_csv(out / "flight_environment_context.csv", index=False)
    print(len(hr), "hourly rows;", len(ctx), "flights ->", out)


if __name__ == "__main__":
    main()
