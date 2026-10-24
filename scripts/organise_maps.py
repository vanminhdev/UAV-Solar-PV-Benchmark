"""Move the EPSG:32649 orthomosaics / DSMs of the Drive-2 'maps' export into dataset/ (mapping flights + per-flight DSMs).

    python organise_maps.py <d2_root> <dataset_root> [--apply]

Rules: keep only GeoTIFFs whose CRS is EPSG:32649 (rasterio check); the same product exported in EPSG:4326
(SonarPanel.tif, SonarPanelDEM3.tif), colour-coded 8-bit 'DEM' images and non-'Keep Key Points' duplicates are not copied.
The source typo "Sonar" is dropped: files are renamed <flight_id>_orthomosaic_EPSG32649.tif / <flight_id>_dsm_EPSG32649.tif.
Mapping flight ids use the median RelativeAltitude read from the raw images (87 m TN19/TNDL/TT, ~110/100 m BIM 2).
"""
import argparse
import csv
import shutil
from pathlib import Path

import rasterio

# (path fragment in the source tree) -> (farm dir, flight_id, suffix)
MAPPING = [
    ("1. Nhà máy BIM 2 - 10.4/Máy thuê/Mapping 2/1/", "farm_02_bim_2", "BIM2_20260411_100M_MAP2", "_part1"),
    ("1. Nhà máy BIM 2 - 10.4/Máy thuê/Mapping 2/2/", "farm_02_bim_2", "BIM2_20260411_100M_MAP2", "_part2"),
    ("1. Nhà máy BIM 2 - 10.4/Máy thuê/Mapping p1/", "farm_02_bim_2", "BIM2_20260410_100M_MAP3", ""),
    ("1. Nhà máy BIM 2 - 10.4/Máy thuê/Mapping/", "farm_02_bim_2", "BIM2_20260410_110M_MAP1", ""),
    ("Thuận Nam 19 -11/4/Máy thuê/Mapping 1/Bản Keep Key Points/", "farm_01_thuan_nam_19", "TN19_20260411_87M_MAP1", ""),
    ("Thuận Nam 19 -11/4/Máy thuê/Mapping 2/Bản Keep Key Points/", "farm_01_thuan_nam_19", "TN19_20260411_87M_MAP2", ""),
    ("3. Nhà máy Thuận Nam Đức Long - 11.4/Máy thuê/Bản Keep Key Points/", "farm_03_thuan_nam_duc_long", "TNDL_20260411_87M_MAP1", ""),
    ("4. Điện mặt trời Thiên Tân 1.2/Máy thuê/", "farm_04_thien_tan", "TT_20260412_87M_MAP1", ""),
]
DSM_PER_FLIGHT = {  # TN19 per-flight DSMs in the DEM folder
    "TN19_31M_Thermal2_RGB2.tif": "TN19_20260411_31M_T2",
    "TN19_31M_Thermal3_RGB3.tif": "TN19_20260411_31M_T3",
    "TN19_35M_Thermal3_RGB3.tif": "TN19_20260411_35M_T3",
    "TN19_35M_Thermal5_RGB5_DEM.tif": "TN19_20260411_35M_T5",
}
KIND = {"sonarpanelorthogeotiff.tif": "orthomosaic", "sonarpanelgeotiff.tif": "dsm"}


def epsg(p):
    with rasterio.open(p) as s:
        return s.crs.to_epsg() if s.crs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d2")
    ap.add_argument("dataset")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    d2, ds = Path(a.d2), Path(a.dataset)
    plan = []
    for tif in sorted(d2.rglob("*.tif")):
        rel = tif.relative_to(d2).as_posix()
        kind = KIND.get(tif.name.lower())
        if kind:
            hit = next((m for m in MAPPING if m[0] in rel + "/" and rel.startswith(m[0].split("Máy thuê")[0]) or m[0] in rel), None)
            # longest-prefix match so "Mapping/" does not swallow "Mapping 2/..."
            cands = [m for m in MAPPING if m[0] in rel]
            hit = max(cands, key=lambda m: len(m[0])) if cands else None
            if not hit:
                print("UNMATCHED", rel)
                continue
            farm, fid, suf = hit[1:]
            name = f"{fid}_{kind}{suf}_EPSG32649.tif"
        elif tif.name in DSM_PER_FLIGHT and "/DEM/" in rel:
            farm, fid = "farm_01_thuan_nam_19", DSM_PER_FLIGHT[tif.name]
            name = f"{fid}_dsm_EPSG32649.tif"
        else:
            continue
        if epsg(tif) != 32649:
            print("SKIP (not EPSG:32649)", rel)
            continue
        plan.append((tif, ds / farm / "photogrammetry" / fid / name))
    for s, d in plan:
        print(("MOVE " if a.apply else "plan ") + s.relative_to(d2).as_posix()[-70:] + "  ->  " + str(d.relative_to(ds)))
    if not a.apply:
        return
    log = ds / "provenance" / "organise_log.csv"
    with open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for s, d in plan:
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                print("EXISTS", d)
                continue
            shutil.move(str(s), str(d))
            w.writerow([f"d2_master_and_mapping/{s.relative_to(d2).as_posix()}", d.relative_to(ds).as_posix()])


if __name__ == "__main__":
    main()
