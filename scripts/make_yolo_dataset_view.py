#!/usr/bin/env python
"""Build an ultralytics-style view (images/, labels/, split lists, data.yaml) of the frame-level thermal labels.

    python -X utf8 code/scripts/make_yolo_dataset_view.py --out <output_dir> \
        [--split-table dataset/metadata/splits/frame_label_split.csv] [--mode symlink|hardlink|copy] \
        [--splits train val test] [--no-negatives] [--box-table PATH|none]

The dataset itself keeps ONLY the original raw thermal R-JPEGs (no re-encoded copies). This script links (or copies) them
together with the matching YOLO-OBB label files into --out, so that

    <out>/images/<split>/<flight_id>__<thermal stem>.JPG      (symlink / hardlink / copy of the raw file)
    <out>/labels/<split>/<flight_id>__<thermal stem>.txt      (copy of labels_yolo_obb/<flight_id>/<stem>.txt)
    <out>/{train,val,test}.txt                                (absolute image paths)
    <out>/data.yaml                                           (task: obb, nc 4, names)

Frames with split == buffer are skipped. Frames without boxes (status reviewed_negative) get an empty label file, which
ultralytics treats as background. mode `symlink` may need administrator rights or Developer Mode on Windows; the script then
falls back to hardlink (same drive) and finally to copy.
ModuleOpenCircuit boxes flagged `ignore` in the per-box table (frame_label_split_moc_boxes.csv for the main split,
frame_label_split_xdomain_moc_boxes.csv for the xdomain split; picked automatically from the split-table name, `--box-table none`
switches it off) are DROPPED from the copied label files: they belong to another split's module group or to the gap between groups.
YOLO has no ignore regions, so a dropped box becomes plain background for the trainer; the modules are still visible in the image.
Use the main split (frame_label_split.csv) or the
cross-domain one (frame_label_split_xdomain_tn19_to_others.csv).
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
SUB = "annotations/model2_thermal_anomaly_obb"
FARM_DIR = {"TN19": "farm_01_thuan_nam_19", "BIM2": "farm_02_bim_2", "TNDL": "farm_03_thuan_nam_duc_long",
            "TT": "farm_04_thien_tan"}
NAMES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]


def link(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            mode = "hardlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(DS))
    ap.add_argument("--split-table", default=None)
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--no-negatives", action="store_true", help="skip reviewed_negative frames (no boxes)")
    ap.add_argument("--box-table", default=None, help="per-box ignore table (default: chosen from the split-table name; 'none' = keep all boxes)")
    a = ap.parse_args()
    ds = Path(a.dataset)
    tab = pd.read_csv(a.split_table or ds / "metadata/splits/frame_label_split.csv")
    tab = tab[tab.split.isin(a.splits)]
    drop = {}                                    # (flight_id, thermal_file) -> set of box_uid to drop
    bt = a.box_table
    if bt is None:
        nm = Path(a.split_table).name if a.split_table else "frame_label_split.csv"
        bt = {"frame_label_split.csv": "frame_label_split_moc_boxes.csv",
              "frame_label_split_xdomain_tn19_to_others.csv": "frame_label_split_xdomain_moc_boxes.csv"}.get(nm)
        bt = ds / "metadata/splits" / bt if bt else None
    if bt and str(bt).lower() != "none" and Path(bt).exists():
        ig = pd.read_csv(bt)
        ig = ig[ig.split == "ignore"]
        for r in ig.itertuples(index=False):
            drop.setdefault((r.flight_id, r.thermal_file), set()).add(r.box_uid)
    kept_uids = {}
    if drop:
        fb = pd.read_csv(ds / FARM_DIR["TN19"] / SUB / "thermal_frame_boxes.csv", usecols=["box_uid", "flight_id", "thermal_file", "kept_in_yolo"])
        fb = fb[fb.kept_in_yolo == 1]
        for (fl, tf), g in fb.groupby(["flight_id", "thermal_file"]):
            if (fl, tf) in drop:
                kept_uids[(fl, tf)] = list(g.box_uid)          # same order as the lines of the YOLO file
    out = Path(a.out)
    used, missing, lists = {}, 0, {s: [] for s in a.splits}
    n_dropped = 0
    index = {}
    for farm, d in FARM_DIR.items():
        for p in (ds / d / "raw_images").glob("*/*/"):
            index[(farm, p.name)] = p
    for r in tab.itertuples(index=False):
        farm_dir = FARM_DIR[r.farm]
        stem = Path(r.thermal_file).stem
        lab = ds / farm_dir / SUB / "labels_yolo_obb" / r.flight_id / (stem + ".txt")
        img = index.get((r.farm, r.flight_id), Path("/nonexistent")) / r.thermal_file
        if not img.exists() or not lab.exists():
            missing += 1
            continue
        if a.no_negatives and lab.stat().st_size == 0:
            continue
        name = f"{r.flight_id}__{stem}"
        (out / "images" / r.split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / r.split).mkdir(parents=True, exist_ok=True)
        dst = out / "images" / r.split / (name + ".JPG")
        m = link(img.resolve(), dst, a.mode)
        used[m] = used.get(m, 0) + 1
        dst_lab = out / "labels" / r.split / (name + ".txt")
        if (r.flight_id, r.thermal_file) in drop:
            lines = lab.read_text(encoding="utf-8").splitlines()
            uids = kept_uids[(r.flight_id, r.thermal_file)]
            assert len(lines) == len(uids), (r.flight_id, r.thermal_file)
            keep_l = [l for l, u in zip(lines, uids) if u not in drop[(r.flight_id, r.thermal_file)]]
            dst_lab.write_text("".join(l + chr(10) for l in keep_l), encoding="utf-8")
            n_dropped += len(lines) - len(keep_l)
        else:
            shutil.copyfile(lab, dst_lab)
        lists[r.split].append(str(dst.resolve()))
    for s, l in lists.items():
        (out / f"{s}.txt").write_text("\n".join(sorted(l)) + ("\n" if l else ""), encoding="utf-8")
    y = ["# generated by make_yolo_dataset_view.py (images = raw thermal R-JPEGs of the dataset)", "task: obb",
         f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test", "nc: 4", "names:"]
    y += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")
    print({s: len(l) for s, l in lists.items()}, "link modes:", used, "missing:", missing, "ignored MOC boxes dropped:", n_dropped)


if __name__ == "__main__":
    main()
