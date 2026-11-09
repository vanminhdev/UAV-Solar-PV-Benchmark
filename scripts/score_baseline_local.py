"""Local scoring of the stored baseline weights (CPU): bootstrap over test frames and score against the annotator labels.

Needs ultralytics >= 8.3 (OBB) and the best.pt files written by colab_controls.py.

  bootstrap  per-frame statistics of a model on a list of frames (same validator settings as the benchmark: conf 0.001, NMS IoU 0.6,
             640 px), mAP@0.5:0.95 over the frames, and a percentile bootstrap over FRAMES (resample frames with replacement, B = 1000,
             seed 20261026; the same resample is used for all seeds, so the interval of the 3-seed mean is obtained from the mean of the
             three seed-wise mAPs in every resample).
  annot      mAP@0.5:0.95 of the same models against the labels of the other annotators (A2, A3, consensus) and A1 on the frames of the
             agreement sample that lie in the TEST split of protocol C (the model never saw them); frames without a box in the reference
             labels count as background (a false box on them is a false positive).

Usage examples (see COLAB_CONTROLS_README.md):
  python -X utf8 score_baseline_local.py bootstrap --tag C2 --weights <controls>/C2 --images <view>/images/test --out out/
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

SEED = 20261026
B = 1000


def per_image_stats(weights, img_list, lab_dir, work):
    """Return (list of per-image tuples (tp, conf, pred_cls, target_cls), mAP50-95 as computed by the validator)."""
    from ultralytics import YOLO
    shutil.rmtree(work, ignore_errors=True)
    for sub in ("images/t", "labels/t"):
        os.makedirs(os.path.join(work, sub))
    names = []
    for f in img_list:
        b = os.path.basename(f)
        shutil.copy(f, os.path.join(work, "images/t", b))
        stem = os.path.splitext(b)[0]
        src = os.path.join(lab_dir, stem + ".txt")
        dst = os.path.join(work, "labels/t", stem + ".txt")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            open(dst, "w").close()
        names.append(b)
    open(os.path.join(work, "t.txt"), "w").write("\n".join(os.path.join(work, "images/t", b) for b in names) + "\n")
    open(os.path.join(work, "t.yaml"), "w").write(
        "path: %s\ntrain: t.txt\nval: t.txt\ntest: t.txt\nnc: 4\nnames:\n  0: SingleHotSpot\n  1: MultiHotSpot\n  2: Bypass\n  3: ModuleOpenCircuit\n" % work)
    model = YOLO(weights)
    rows = []

    def cb(v):
        s = v.metrics.stats
        rows.append(tuple(np.array(s[k][-1]) for k in ("tp", "conf", "pred_cls", "target_cls")))
        assert len(s["tp"]) == len(rows), "stats are not one entry per image"

    model.add_callback("on_val_batch_end", cb)
    m = model.val(data=os.path.join(work, "t.yaml"), split="val", batch=1, imgsz=640, conf=0.001, iou=0.6, plots=False, verbose=False,
                  device="cpu", workers=0, project=os.path.join(work, "runs"), name="v", exist_ok=True)
    return rows, float(m.box.map)


def map5095(rows, idx=None):
    from ultralytics.utils.metrics import ap_per_class
    sel = range(len(rows)) if idx is None else idx
    tp = np.concatenate([rows[i][0].reshape(-1, 10) for i in sel], 0)
    conf = np.concatenate([rows[i][1].reshape(-1) for i in sel])
    pc = np.concatenate([rows[i][2].reshape(-1) for i in sel])
    tc = np.concatenate([rows[i][3].reshape(-1) for i in sel])
    if len(tc) == 0:
        return float("nan")
    r = ap_per_class(tp.astype(bool), conf, pc, tc)
    return float(r[5].mean())


def bootstrap(per_seed_rows):
    """per_seed_rows: list over seeds of per-image rows (same frames, same order). Returns dict with the seed-wise and mean mAP distributions."""
    n = len(per_seed_rows[0])
    rng = np.random.default_rng(SEED)
    out = np.zeros((B, len(per_seed_rows)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        for s, rows in enumerate(per_seed_rows):
            out[b, s] = map5095(rows, idx)
    return out


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bootstrap"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True, help="folder with s0/best.pt, s1/best.pt, s2/best.pt")
    ap.add_argument("--images", required=True, help="folder with the frames to score (e.g. the positive test frames)")
    ap.add_argument("--labels", help="label folder (default: sibling labels/ folder)")
    ap.add_argument("--list", help="optional text file with the image file names to use (one per line); default: all images in --images")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    labels = a.labels or os.path.join(os.path.dirname(os.path.dirname(a.images.rstrip("/\\"))), "labels", os.path.basename(a.images.rstrip("/\\")))
    imgs = sorted(glob.glob(os.path.join(a.images, "*")))
    if a.list:
        keep = {l.strip() for l in open(a.list) if l.strip()}
        imgs = [f for f in imgs if os.path.basename(f) in keep]
    os.makedirs(a.out, exist_ok=True)
    all_rows, res = [], {"tag": a.tag, "n_frames": len(imgs), "B": B, "seed": SEED, "per_seed": {}}
    for s in a.seeds:
        rows, m_val = per_image_stats(os.path.join(a.weights, "s%d" % s, "best.pt"), imgs, labels, a.work)
        mine = map5095(rows)
        res["per_seed"][s] = {"map_validator": m_val, "map_recomputed": mine}
        print(a.tag, "seed", s, "validator %.4f recomputed %.4f" % (m_val, mine), flush=True)
        all_rows.append(rows)
    dist = bootstrap(all_rows)
    np.save(os.path.join(a.out, "boot_%s.npy" % a.tag), dist)
    for s_i, s in enumerate(a.seeds):
        res["per_seed"][s]["boot_ci95"] = ci(dist[:, s_i])
        res["per_seed"][s]["boot_sd"] = float(np.nanstd(dist[:, s_i]))
    mean = np.nanmean(dist, 1)
    res["mean_of_seeds"] = {"point": float(np.mean([res["per_seed"][s]["map_recomputed"] for s in a.seeds])), "boot_ci95": ci(mean),
                            "boot_sd": float(np.nanstd(mean))}
    json.dump(res, open(os.path.join(a.out, "boot_%s.json" % a.tag), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
