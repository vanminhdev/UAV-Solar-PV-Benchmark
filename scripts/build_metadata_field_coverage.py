#!/usr/bin/env python
"""Per camera system x metadata field: is the field recorded by the camera, and for how many images is it non-empty?

  python -X utf8 code/scripts/build_metadata_field_coverage.py
Inputs : dataset/metadata/inventory/image_inventory.csv (EXIF/XMP fields), thermal/thermal_parameters.csv (R-JPEG header),
         XMP cache written by xmp_speed_and_flags.py (%TEMP%/solar_uav_xmp_cache.csv; re-read from the JPGs if missing).
Output : dataset/metadata/inventory/metadata_field_coverage.csv (see README_field_coverage.md)
status : present                      field non-empty for >= 1 image of the system (n_nonempty / n_images gives the coverage)
         not_recorded_by_camera       field applicable to the system but empty for ALL its images
         not_applicable               field does not exist for this sensor type (e.g. thermal calibration on RGB) or the system has no images in the release
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import xmp_speed_and_flags as xs   # load_xmp(), paths

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "dataset/metadata/inventory"

SYSTEMS = [("H20T RGB", "ZH20T", "W"), ("H20T thermal", "ZH20T", "T"),
           ("Mavic2EA RGB", "MAVIC2-ENTERPRISE-ADVANCED", "W"), ("Mavic2EA thermal", "MAVIC2-ENTERPRISE-ADVANCED", "T")]
# (field, source, group, applies_to_sensor ('W','T','both'), description)
FIELDS = [
    ("timestamp", "inventory", "time", "both", "capture time (from file name)"),
    ("width", "inventory", "image", "both", "image width px"),
    ("focal_mm", "inventory", "optics", "both", "EXIF FocalLength (mm)"),
    ("focal_35mm", "inventory", "optics", "both", "EXIF FocalLengthIn35mmFormat (mm)"),
    ("GpsLatitude", "inventory", "position", "both", "EXIF GPS latitude"),
    ("GpsLongitude", "inventory", "position", "both", "EXIF GPS longitude"),
    ("AbsoluteAltitude", "inventory", "position", "both", "XMP AbsoluteAltitude (m, ellipsoidal)"),
    ("RelativeAltitude", "inventory", "position", "both", "XMP RelativeAltitude (m above take-off)"),
    ("GimbalRollDegree", "inventory", "attitude", "both", "XMP gimbal roll"),
    ("GimbalYawDegree", "inventory", "attitude", "both", "XMP gimbal yaw"),
    ("GimbalPitchDegree", "inventory", "attitude", "both", "XMP gimbal pitch"),
    ("FlightYawDegree", "inventory", "attitude", "both", "XMP aircraft yaw"),
    ("FlightPitchDegree", "inventory", "attitude", "both", "XMP aircraft pitch"),
    ("FlightRollDegree", "inventory", "attitude", "both", "XMP aircraft roll"),
    ("RtkFlag", "inventory", "RTK", "both", "XMP RtkFlag (50 = fixed)"),
    ("RtkStdLat", "inventory", "RTK", "both", "XMP RTK std latitude (m)"),
    ("RtkStdLon", "inventory", "RTK", "both", "XMP RTK std longitude (m)"),
    ("RtkStdHgt", "inventory", "RTK", "both", "XMP RTK std height (m)"),
    ("GpsStatus", "xmp", "RTK", "both", "XMP GpsStatus"),
    ("FlightXSpeed", "xmp", "speed", "both", "XMP flight-controller speed, north (m/s)"),
    ("FlightYSpeed", "xmp", "speed", "both", "XMP flight-controller speed, east (m/s)"),
    ("FlightZSpeed", "xmp", "speed", "both", "XMP flight-controller speed, down (m/s)"),
    ("LRFTargetDistance", "inventory", "LRF", "both", "XMP laser range finder distance (m)"),
    ("LRFTargetDistance>0", "inventory", "LRF", "both", "LRF distance with a non-zero value (0 = no valid return)"),
    ("emissivity", "thermal", "thermal_calibration", "T", "R-JPEG header emissivity"),
    ("reflected_temp_c", "thermal", "thermal_calibration", "T", "R-JPEG header reflected temperature"),
    ("distance_m", "thermal", "thermal_calibration", "T", "R-JPEG header object distance"),
    ("humidity_pct", "thermal", "thermal_calibration", "T", "R-JPEG header relative humidity"),
    ("ambient_temp_c", "thermal", "thermal_calibration", "T", "R-JPEG header ambient temperature"),
    ("raw_min/raw_max", "thermal", "thermal_calibration", "T", "raw 16-bit counts decodable (DJI Thermal SDK)"),
]


def main():
    inv = pd.read_csv(INV / "image_inventory.csv")
    x = xs.load_xmp()[["flight_id", "file", "FlightXSpeed", "FlightYSpeed", "FlightZSpeed", "GpsStatus"]]
    # GpsStatus is a string tag: count presence before numeric coercion
    x["GpsStatus"] = x["GpsStatus"].where(x["GpsStatus"].notna())
    tp = pd.read_csv(ROOT / "dataset/metadata/thermal/thermal_parameters.csv")
    tp = tp[tp.decode_ok.astype(str) == "True"].copy()
    tp["raw_min/raw_max"] = tp.raw_min.where(tp.raw_max.notna())
    inv = inv.merge(x, on=["flight_id", "file"], how="left", validate="1:1")
    inv = inv.merge(tp[["flight_id", "file", "emissivity", "reflected_temp_c", "distance_m", "humidity_pct",
                        "ambient_temp_c", "raw_min/raw_max"]], on=["flight_id", "file"], how="left", validate="1:1")
    inv["LRFTargetDistance>0"] = inv.LRFTargetDistance.where(inv.LRFTargetDistance > 0)
    rows = []
    for name, model, sens in SYSTEMS:
        g = inv[(inv.model == model) & (inv.sensor == sens)]
        for f, src, grp, app, desc in FIELDS:
            n = int(g[f].notna().sum())
            if app != "both" and app != sens:
                st = "not_applicable"
            else:
                st = "present" if n > 0 else "not_recorded_by_camera"
            rows.append(dict(camera_system=name, field=f, group=grp, source=src, description=desc, status=st,
                             n_images=len(g), n_nonempty=n, coverage_pct=round(100 * n / len(g), 1)))
    for f, src, grp, app, desc in FIELDS:     # Zenmuse P1: mapping products only, no raw images in the release
        rows.append(dict(camera_system="Zenmuse P1 mapping", field=f, group=grp, source=src, description=desc,
                         status="not_applicable", n_images=0, n_nonempty=0, coverage_pct=float("nan")))
    out = pd.DataFrame(rows)
    out.to_csv(INV / "metadata_field_coverage.csv", index=False)
    pv = out[out.camera_system != "Zenmuse P1 mapping"].pivot(index="field", columns="camera_system", values="n_nonempty")
    st = out.pivot(index="field", columns="camera_system", values="status")
    print(out.groupby("camera_system", sort=False).n_images.first().to_dict())
    print(pv.loc[[f[0] for f in FIELDS]].to_string())
    print(st.loc[[f[0] for f in FIELDS]].replace({"present": "P", "not_recorded_by_camera": "NRC", "not_applicable": "NA"}).to_string())


if __name__ == "__main__":
    main()
