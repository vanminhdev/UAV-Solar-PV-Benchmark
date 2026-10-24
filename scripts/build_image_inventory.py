"""Build an image-level inventory (EXIF + DJI XMP) for a folder of UAV JPGs.

Read-only on the source tree. Writes one CSV row per image and a per-flight
summary CSV. Usage:
    python build_image_inventory.py <raw_root> <out_dir> [--farm farm_01_thuan_nam_19]
"""
import argparse
import csv
import re
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

XMP_KEYS = [
    "GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "RelativeAltitude",
    "GimbalRollDegree", "GimbalYawDegree", "GimbalPitchDegree",
    "FlightYawDegree", "FlightPitchDegree", "FlightRollDegree",
    "RtkFlag", "RtkStdLat", "RtkStdLon", "RtkStdHgt", "LRFTargetDistance",
]
FIELDS = (
    ["flight", "file", "sensor", "timestamp", "seq", "width", "height", "bytes",
     "model", "focal_mm", "focal_35mm"]
    + XMP_KEYS
    + ["app3_segments"]
)


def parse_image(path: Path):
    head = path.read_bytes()[:400_000]
    row = {"flight": path.parent.name, "file": path.name, "bytes": path.stat().st_size}
    m = re.match(r"DJI_(?:(\d{14})_)?(\d{4})_([WTVZ])\.JPG", path.name, re.I)
    if m:
        row.update(seq=m.group(2), sensor=m.group(3).upper())
        if m.group(1):
            row["timestamp"] = m.group(1)
    for k in XMP_KEYS:
        mm = re.search(rb'drone-dji:' + k.encode() + rb'="([^"]*)"', head)
        if mm:
            row[k] = mm.group(1).decode()
    with Image.open(path) as im:
        row["width"], row["height"] = im.size
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        row["model"] = ex.get(0x0110, "")
        row["focal_mm"] = sub.get(0x920A, "")
        row["focal_35mm"] = sub.get(0xA405, "")
        if "timestamp" not in row and sub.get(0x9003):  # file names without a timestamp: use EXIF DateTimeOriginal
            row["timestamp"] = re.sub(r"\D", "", str(sub[0x9003]))[:14]
    row["app3_segments"] = head.count(b"\xff\xe3")
    return row


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(rows):
    out = []
    flights = sorted({r["flight"] for r in rows})
    for fl in flights:
        rs = [r for r in rows if r["flight"] == fl]
        for sensor in sorted({r.get("sensor", "?") for r in rs}):
            s = [r for r in rs if r.get("sensor", "?") == sensor]
            rel = [v for v in (f(r.get("RelativeAltitude")) for r in s) if v is not None]
            ab = [v for v in (f(r.get("AbsoluteAltitude")) for r in s) if v is not None]
            pit = [v for v in (f(r.get("GimbalPitchDegree")) for r in s) if v is not None]
            lat = [v for v in (f(r.get("GpsLatitude")) for r in s) if v is not None]
            lon = [v for v in (f(r.get("GpsLongitude")) for r in s) if v is not None]
            ts = sorted(r["timestamp"] for r in s if r.get("timestamp"))
            rtk = [r.get("RtkFlag") for r in s]
            out.append({
                "flight": fl, "sensor": sensor, "n_images": len(s),
                "size": f"{s[0]['width']}x{s[0]['height']}",
                "model": s[0]["model"], "focal_mm": s[0]["focal_mm"],
                "t_first": ts[0] if ts else "", "t_last": ts[-1] if ts else "",
                "rel_alt_mean": round(st.mean(rel), 2) if rel else "",
                "rel_alt_min": min(rel) if rel else "", "rel_alt_max": max(rel) if rel else "",
                "abs_alt_mean": round(st.mean(ab), 2) if ab else "",
                "pitch_median": round(st.median(pit), 2) if pit else "",
                "pitch_off_nadir_gt2deg": sum(abs(p + 90) > 2 for p in pit),
                "lat_mean": round(st.mean(lat), 6) if lat else "",
                "lon_mean": round(st.mean(lon), 6) if lon else "",
                "lat_min": min(lat) if lat else "", "lat_max": max(lat) if lat else "",
                "lon_min": min(lon) if lon else "", "lon_max": max(lon) if lon else "",
                "rtk_fix50_pct": round(100 * sum(x == "50" for x in rtk) / len(rtk), 1),
                "with_app3_pct": round(100 * sum(int(r["app3_segments"]) > 0 for r in s) / len(s), 1),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root")
    ap.add_argument("out_dir")
    ap.add_argument("--farm", default="")
    a = ap.parse_args()
    files = sorted(Path(a.raw_root).rglob("*.JPG"))
    print(f"{len(files)} images")
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(parse_image, files, chunksize=64))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "image_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    with open(out / "flight_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print("done", len(rows), "rows")


if __name__ == "__main__":
    main()
