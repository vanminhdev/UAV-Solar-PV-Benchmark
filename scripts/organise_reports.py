"""Move Metashape camera files (cameras.xml) and processing reports (report.pdf) next to the orthomosaics.

    python organise_reports.py <cache_root> <dataset_root> <inv_dir> [--apply]

Targets: dataset/<farm>/photogrammetry/<flight_id>/<flight_id>_cameras.xml and <flight_id>_report.pdf.
Sources: d1_ortho_export/<Export>/<flight folder>/{cameras.xml,Cameras.xml,report.pdf,generate_report.pdf,generate_export.pdf}
         and the mapping flights of d2_master_and_mapping (SonarPanel.xml/.pdf in the 'Keep Key Points' or mapping folder).
Flight ids follow organise_dataset.py / organise_maps.py.
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

from organise_dataset import FARMS, flight_table
from organise_maps import MAPPING

XML = ("cameras.xml",)
PDF = ("report.pdf", "generate_report.pdf", "generate_export.pdf")


def kind(name):
    n = name.lower()
    if n in XML or (n.endswith(".xml") and n.startswith("sonarpanel")):
        return "cameras", ".xml"
    if n in PDF or (n.endswith(".pdf") and n.startswith("sonarpanel")):
        return "report", ".pdf"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("dataset")
    ap.add_argument("inv")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cache, ds = Path(a.cache), Path(a.dataset)
    plan = []
    # inspection flights: d1 exports
    for prefix, (code, fdir, sub, export) in FARMS.items():
        table = flight_table(Path(a.inv) / sub / "image_inventory.csv", code)
        norm = {k.replace(" ", "").lower(): v for k, v in table.items()}
        for f in sorted((cache / "d1_ortho_export" / export).rglob("*")):
            k, ext = kind(f.name)
            t = norm.get(f.relative_to(cache / "d1_ortho_export" / export).parts[0].replace(" ", "").lower()) if k else None
            if t:
                plan.append((f, ds / fdir / "photogrammetry" / t["fid"] / f"{t['fid']}_{k}{ext}"))
    for f in sorted((cache / "d1_ortho_export" / "ThuanNam19_Export").rglob("*")):
        k, ext = kind(f.name)
        m = re.search(r"(\d+)M\. Thermal (\d+)", str(f)) if k else None
        if m:
            fid = f"TN19_20260411_{m.group(1)}M_T{m.group(2)}"
            plan.append((f, ds / "farm_01_thuan_nam_19" / "photogrammetry" / fid / f"{fid}_{k}{ext}"))
    # mapping flights: d2 (only files that sit in the same folder as the kept orthos)
    d2 = cache / "d2_master_and_mapping"
    for f in sorted(d2.rglob("*")):
        k, ext = kind(f.name) if f.is_file() else (None, None)
        if not k:
            continue
        rel = f.relative_to(d2).as_posix()
        folder = "/".join(rel.split("/")[:-1]) + "/"
        cands = [m for m in MAPPING if folder.endswith(m[0])]  # the file must sit directly in the mapped folder
        if not cands:
            continue
        hit = max(cands, key=lambda m: len(m[0]))
        farm, fid, suf = hit[1:]
        plan.append((f, ds / farm / "photogrammetry" / fid / f"{fid}_{k}{suf}{ext}"))
    for s, d in plan:
        print(("MOVE " if a.apply else "plan ") + s.relative_to(cache).as_posix()[-70:] + "  ->  " + d.relative_to(ds).as_posix())
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
            w.writerow([s.relative_to(cache).as_posix(), d.relative_to(ds).as_posix()])


if __name__ == "__main__":
    main()
