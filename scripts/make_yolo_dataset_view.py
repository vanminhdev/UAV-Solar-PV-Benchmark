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

Frames with split == buffer are skipped. Frames with split == crop (`crop_window = 1`: the ModuleOpenCircuit patch and, in the component split, frames that straddle a region boundary)
are NOT written as whole frames: for each crop window of `frame_label_split_crops.csv` (component split) or `frame_label_split_moc_crops.csv` (strip + buffer split; xdomain: `..._xdomain_...`)
the raw thermal JPEG is cropped with PIL (same rendition, re-encoded as JPEG quality 95; this is the only re-encoded file type) to
`images/<split>/<flight_id>__<stem>__<crop_id>.JPG` and the labels of `frame_label_split_moc_crop_boxes.csv` are written in crop-pixel
coordinates (normalised to the crop size) to the matching .txt. Names are unique, whole frames are never duplicated. Frames without boxes (status reviewed_negative) get an empty label file, which
ultralytics treats as background. mode `symlink` may need administrator rights or Developer Mode on Windows; the script then
falls back to hardlink (same drive) and finally to copy.
Split tables with the old relaxed rule (metadata/splits/_superseded_relaxed/) carry a per-box `ignore` table (`--box-table`, picked from the
file name; `none` switches it off): `ignore` boxes are dropped from the copied label files. The strict split has no `ignore` boxes.
Use the main split (frame_label_split.csv) or the cross-domain one (frame_label_split_xdomain_tn19_to_others.csv).
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
SUB = "annotations/thermal_anomaly_obb"
FARM_DIR = {"TN19": "farm_01_thuan_nam_19", "BIM2": "farm_02_bim_2", "TNDL": "farm_03_thuan_nam_duc_long",
            "TT": "farm_04_thien_tan"}
NAMES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]


def strip_app_segments(data: bytes) -> bytes:
    """Drop the APP1..APP15 / COM segments (radiometric payload, XMP) of a JPEG; the entropy-coded scan is copied byte for byte,
    so the decoded pixels are identical (checked on a sample) while the file gets much smaller. Used for the upload copy."""
    out, i = bytearray(b"\xff\xd8"), 2
    while i < len(data):
        m = data[i + 1]
        if data[i] != 0xFF:
            raise ValueError("bad JPEG marker")
        if m == 0xDA:
            out += data[i:]
            return bytes(out)
        n = 2 + int.from_bytes(data[i + 2:i + 4], "big")
        if not (0xE1 <= m <= 0xEF or m == 0xFE):
            out += data[i:i + n]
        i += n
    raise ValueError("no SOS marker")


def link(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "strip":
        dst.write_bytes(strip_app_segments(src.read_bytes()))
        return "strip"
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
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy", "strip"], default="symlink")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--no-negatives", action="store_true", help="skip reviewed_negative frames (no boxes)")
    ap.add_argument("--box-table", default=None, help="per-box ignore table (default: chosen from the split-table name; 'none' = keep all boxes)")
    a = ap.parse_args()
    ds = Path(a.dataset)
    tab = pd.read_csv(a.split_table or ds / "metadata/splits/frame_label_split.csv")
    drop = {}                                    # (flight_id, thermal_file) -> set of box_uid to drop
    bt = a.box_table
    stab = Path(a.split_table) if a.split_table else ds / "metadata/splits/frame_label_split.csv"
    xd = "xdomain" in stab.name
    pre = "frame_label_split_xdomain_" if xd else "frame_label_split_"
    crops_f, cboxes_f = stab.parent / (pre + "crops.csv"), stab.parent / (pre + "crop_boxes.csv")          # component split: MOC + boundary crops
    if not crops_f.exists():                                                                                  # strip + buffer strict split: MOC crops only
        crops_f, cboxes_f = stab.parent / (pre + "moc_crops.csv"), stab.parent / (pre + "moc_crop_boxes.csv")
    crops = pd.read_csv(crops_f) if "crop" in set(tab.split) and crops_f.exists() else None
    if crops is not None:
        crops = crops[crops.split.isin(a.splits)]
        cboxes = pd.read_csv(cboxes_f)
    if "crop" in set(tab.split) and crops is None:
        raise SystemExit(f"split table has crop frames but {crops_f} is missing")
    tab = tab[tab.split.isin(a.splits)]
    if bt is None and "crop_window" in tab.columns:
        bt = "none"
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
    n_crop = n_crop_lines = 0
    if crops is not None:
        from PIL import Image
        for r in crops.itertuples(index=False):
            img = index.get((r.farm, r.flight_id), Path("/nonexistent")) / r.thermal_file
            if not img.exists():
                missing += 1
                continue
            name = f"{r.flight_id}__{Path(r.thermal_file).stem}__{r.crop_id}"
            (out / "images" / r.split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / r.split).mkdir(parents=True, exist_ok=True)
            with Image.open(img) as im:
                W, H = im.size                                   # 640 x 512 for the H20T thermal frame
                sx, sy = W / 640.0, H / 512.0
                box = (int(round(r.x0 * sx)), int(round(r.y0 * sy)), int(round(r.x1 * sx)), int(round(r.y1 * sy)))
                cw, ch = box[2] - box[0], box[3] - box[1]
                kw = {"quality": 95}
                if "exif" in im.info:
                    kw["exif"] = im.info["exif"]
                im.crop(box).save(out / "images" / r.split / (name + ".JPG"), "JPEG", **kw)
            cb = cboxes[cboxes.crop_id == r.crop_id]
            lines = []
            for q in cb.itertuples(index=False):
                v = []
                for j in range(1, 5):
                    v += [min(max(getattr(q, f"x{j}") * sx / cw, 0.0), 1.0), min(max(getattr(q, f"y{j}") * sy / ch, 0.0), 1.0)]
                lines.append(f"{int(q.class_id)} " + " ".join(f"{x:.6f}" for x in v))
            (out / "labels" / r.split / (name + ".txt")).write_text("".join(l + chr(10) for l in lines), encoding="utf-8")
            lists[r.split].append(str((out / "images" / r.split / (name + ".JPG")).resolve()))
            n_crop += 1
            n_crop_lines += len(lines)
    for s, l in lists.items():
        (out / f"{s}.txt").write_text("\n".join(sorted(l)) + ("\n" if l else ""), encoding="utf-8")
    y = ["# generated by make_yolo_dataset_view.py (images = raw thermal R-JPEGs of the dataset)", "task: obb",
         f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test", "nc: 4", "names:"]
    y += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")
    print({s: len(l) for s, l in lists.items()}, "link modes:", used, "missing:", missing, "ignored MOC boxes dropped:", n_dropped,
          "crop images:", n_crop, "crop label lines:", n_crop_lines)


if __name__ == "__main__":
    main()
