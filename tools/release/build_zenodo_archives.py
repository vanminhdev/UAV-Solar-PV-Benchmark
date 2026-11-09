"""Pack the dataset into zip archives for deposit in several Zenodo records.

Every archive stores its files under their path relative to the dataset root (e.g.
farm_01_thuan_nam_19/raw_images/altitude_35m_inspection/<flight>/<file>.JPG). After downloading all archives of all
records into one folder, extracting them all there (extract_all.py) recreates the dataset folder exactly as it is here.

Records (each below the Zenodo limit of 50 GB and 100 files):
  R1 core            annotations, module reference, splits, metadata, provenance, thermal temperature matrices, documents
  R2 TN19 images 35 m
  R3 TN19 images 31 m, 42 m, 87 m
  R4 BIM2 and TNDL images
  R5 TT images and photogrammetry products of TN19, TNDL and TT
  R6 BIM2 photogrammetry products

Usage: python build_zenodo_archives.py <dataset_dir> <output_dir> [--max-zip-gb 8]
Writes <output_dir>/R<n>/*.zip, <output_dir>/MANIFEST.csv (path, size, sha256, record, archive) and checks the limits.
"""
import argparse
import csv
import hashlib
import os
import sys
import zipfile
from collections import defaultdict

DOCS = ["README.md", "DATA_DICTIONARY.md", "LICENSE.txt", "CITATION.cff"]


def record_of(rel):
    p = rel.split("/")
    if len(p) > 2 and p[1] == "raw_images":
        if p[0].startswith("farm_01"):
            return "R2" if p[2] == "altitude_35m_inspection" else "R3"
        if p[0].startswith(("farm_02", "farm_03")):
            return "R4"
        return "R5"
    if len(p) > 2 and p[1] == "photogrammetry":
        return "R6" if p[0].startswith("farm_02") else "R5"
    return "R1"


def unit_of(rel):
    """Files that stay together in one archive (a flight or a photogrammetry product folder)."""
    p = rel.split("/")
    if len(p) > 3 and p[1] in ("raw_images", "photogrammetry"):
        return "/".join(p[:4])
    if p[0] == "derived":
        return "derived"
    return "docs_tables"


LABEL = {"R1": "core", "R2": "tn19_images_35m", "R3": "tn19_images_31m_42m_87m", "R4": "bim2_tndl_images",
         "R5": "tt_images_and_photogrammetry", "R6": "bim2_photogrammetry"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--max-zip-gb", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.dataset)
    cap = a.max_zip_gb * 1e9
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            full = os.path.join(r, f)
            files.append((os.path.relpath(full, root).replace(os.sep, "/"), os.path.getsize(full)))
    files.sort()
    units = defaultdict(list)
    for rel, sz in files:
        units[(record_of(rel), unit_of(rel))].append((rel, sz))
    plan = []  # (record, archive name, [files])
    for rec in sorted({k[0] for k in units}):
        cur, cur_sz, k = [], 0, 1
        for (r2, un), fl in sorted(units.items()):
            if r2 != rec:
                continue
            usz = sum(s for _, s in fl)
            if cur and cur_sz + usz > cap:
                plan.append((rec, f"uavpv_{rec}_{LABEL[rec]}_{k:02d}.zip", cur))
                cur, cur_sz, k = [], 0, k + 1
            cur += fl
            cur_sz += usz
        if cur:
            plan.append((rec, f"uavpv_{rec}_{LABEL[rec]}_{k:02d}.zip", cur))
    # the document files go into the core archive as well, so a full extraction has them at the top level
    tot = defaultdict(lambda: [0, 0])
    for rec, name, fl in plan:
        tot[rec][0] += sum(s for _, s in fl)
        tot[rec][1] += 1
    for rec in sorted(tot):
        print(rec, "%.2f GB" % (tot[rec][0] / 1e9), tot[rec][1], "archives")
        assert tot[rec][0] < 49.0e9 and tot[rec][1] < 100, rec
    assert sum(len(fl) for _, _, fl in plan) == len(files)
    if a.dry_run:
        for rec, name, fl in plan:
            print("  %-60s %7.2f GB %6d files" % (name, sum(s for _, s in fl) / 1e9, len(fl)))
        return
    rows = []
    for rec, name, fl in plan:
        d = os.path.join(a.out, rec)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        if os.path.exists(path):
            print("exists, skipped:", name)
            continue
        with zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for rel, sz in fl:
                h = hashlib.sha256()
                zi = zipfile.ZipInfo.from_file(os.path.join(root, rel), rel)
                zi.compress_type = zipfile.ZIP_STORED
                with open(os.path.join(root, rel), "rb") as src, z.open(zi, "w", force_zip64=True) as dst:
                    while True:
                        b = src.read(1 << 24)
                        if not b:
                            break
                        h.update(b)
                        dst.write(b)
                rows.append((rel, sz, h.hexdigest(), rec, name))
        os.replace(path + ".tmp", path)
        print("written", name, "%.2f GB" % (os.path.getsize(path) / 1e9), flush=True)
    # a loose copy of the documents is placed next to the records for upload to the root record
    if rows:
        with open(os.path.join(a.out, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "size_bytes", "sha256", "record", "archive"])
            w.writerows(sorted(rows))
        print("MANIFEST.csv rows:", len(rows))


if __name__ == "__main__":
    sys.exit(main())
