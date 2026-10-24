"""Move downloaded raw images and V2 orthomosaics into dataset/ using the flight_id naming rule.

    python organise_dataset.py <cache_root> <dataset_root> <inv_dir> [--apply]

flight_id = <FARM>_<YYYYMMDD>_<alt>M_T<n>
  FARM  : BIM2 | TNDL | TT   (TN19 is already organised)
  date  : first EXIF timestamp of the flight folder
  alt   : round(median XMP RelativeAltitude) over all images of the folder (metres, 1 m)
Raw   -> dataset/farm_xx/raw_images/altitude_<alt>m_inspection/<flight_id>/
Ortho -> dataset/farm_xx/photogrammetry/<flight_id>/<flight_id>_orthomosaic_EPSG32649.tif  (SolarPanel*_V2.tif only)
DSM   -> same folder, <flight_id>_dsm_EPSG32649.tif (only float32 single-band *DEM_V2.tif)
Dry run unless --apply; every move is appended to dataset/provenance/organise_log.csv.
"""
import argparse
import csv
import re
import shutil
import statistics as st
from pathlib import Path

FARMS = {  # raw_src prefix -> (code, dataset dir, inventory subdir, ortho export dir)
    "1. ": ("BIM2", "farm_02_bim_2", "f1", "BIM2_Export"),
    "3. ": ("TNDL", "farm_03_thuan_nam_duc_long", "f2", "ThuanNamDucLong_Export"),
    "4. ": ("TT", "farm_04_thien_tan", "f3", "ThienTan_Export"),
}


def flight_table(inv_csv, code):
    rows = {}
    for r in csv.DictReader(open(inv_csv, encoding="utf-8")):
        rows.setdefault(r["flight"], []).append(r)
    out = {}
    for folder, rs in rows.items():
        alt = round(st.median(float(r["RelativeAltitude"]) for r in rs if r["RelativeAltitude"]))
        date = min(r["timestamp"] for r in rs if r["timestamp"])[:8]
        n = re.search(r"Thermal (\d+)", folder).group(1)
        out[folder] = dict(fid=f"{code}_{date}_{alt}M_T{n}", alt=alt, n=n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("dataset")
    ap.add_argument("inv")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cache, ds = Path(a.cache), Path(a.dataset)
    log = []
    for prefix, (code, fdir, sub, export) in FARMS.items():
        raw_farm = next(p for p in (cache / "raw_src").iterdir() if p.name.startswith(prefix)) / "2. Máy thuê"
        table = flight_table(Path(a.inv) / sub / "image_inventory.csv", code)
        for folder, t in table.items():
            src = raw_farm / folder
            dst = ds / fdir / "raw_images" / f"altitude_{t['alt']}m_inspection" / t["fid"]
            log.append((src, dst))
        # orthos: the first export sub-folder carries the same name as the raw flight folder
        for tif in sorted((cache / "d1_ortho_export" / export).rglob("*_V2.tif")):
            norm = {k.replace(" ", "").lower(): v for k, v in table.items()}  # "GSD3." vs "GSD 3."
            t = norm.get(tif.relative_to(cache / "d1_ortho_export" / export).parts[0].replace(" ", "").lower())
            if not t:
                print("NO FLIGHT FOR", tif)
                continue
            kind = "dsm" if tif.name.lower().startswith(("solarpaneldem", "dem")) else "orthomosaic"
            if "DEM" in tif.name.upper() and "DEM_V2" not in tif.name.upper():
                continue
            if kind == "orthomosaic" and "DEM" in tif.name.upper():
                continue
            dst = ds / fdir / "photogrammetry" / t["fid"] / f"{t['fid']}_{kind}_EPSG32649.tif"
            log.append((tif, dst))
    # TN19 orthos: raw flights are already organised; flight_id comes from the "<alt>M. Thermal <n>" folder
    for tif in sorted((cache / "d1_ortho_export" / "ThuanNam19_Export").rglob("*_V2.tif")):
        m = re.search(r"(\d+)M\. Thermal (\d+)", str(tif))
        fid = f"TN19_20260411_{m.group(1)}M_T{m.group(2)}"
        kind = "orthomosaic" if "DEM" not in tif.name.upper() else None
        if kind:
            log.append((tif, ds / "farm_01_thuan_nam_19" / "photogrammetry" / fid / f"{fid}_{kind}_EPSG32649.tif"))
    for s, d in log:
        print(("MOVE " if a.apply else "plan ") + str(s.relative_to(cache)) + "  ->  " + str(d.relative_to(ds)))
    if not a.apply:
        return
    prov = ds / "provenance"
    prov.mkdir(exist_ok=True)
    new = not (prov / "organise_log.csv").exists()
    with open(prov / "organise_log.csv", "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["source_relative_to_cache", "destination_relative_to_dataset"])
        for s, d in log:
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                continue
            shutil.move(str(s), str(d))
            w.writerow([str(s.relative_to(cache)), str(d.relative_to(ds))])


if __name__ == "__main__":
    main()
