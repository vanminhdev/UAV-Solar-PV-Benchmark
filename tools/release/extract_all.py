"""Recreate the dataset folder from the downloaded Zenodo archives.

Put the zip archives of ALL records of the dataset into one folder (sub-folders are fine), then run

    python extract_all.py <folder_with_zips> <target_dataset_dir> [--check MANIFEST.csv]

Every archive stores paths relative to the dataset root, so all of them are extracted into the same target folder and
together give the original layout (farm_01_thuan_nam_19/..., farm_02_bim_2/..., derived/, metadata/, ...).
You may also extract a single archive or a subset: the folder tree is the same, only the missing parts are absent.
With --check, every extracted file is compared with the SHA-256 in MANIFEST.csv.
"""
import argparse
import csv
import hashlib
import os
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips")
    ap.add_argument("target")
    ap.add_argument("--check", help="MANIFEST.csv for a SHA-256 check of the extracted files")
    a = ap.parse_args()
    archives = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.zips) for f in fs if f.lower().endswith(".zip"))
    if not archives:
        sys.exit("no .zip files found in " + a.zips)
    os.makedirs(a.target, exist_ok=True)
    for p in archives:
        print("extracting", os.path.basename(p), flush=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(a.target)
    print("extracted", len(archives), "archives into", a.target)
    if a.check:
        bad = missing = n = 0
        with open(a.check, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                full = os.path.join(a.target, row["path"])
                if not os.path.exists(full):
                    missing += 1
                    continue
                h = hashlib.sha256()
                with open(full, "rb") as g:
                    for b in iter(lambda: g.read(1 << 24), b""):
                        h.update(b)
                n += 1
                if h.hexdigest() != row["sha256"]:
                    bad += 1
                    print("CHECKSUM MISMATCH", row["path"])
        print("checked %d files, %d mismatches, %d listed files not present" % (n, bad, missing))


if __name__ == "__main__":
    main()
