"""Baseline oriented-box detector (YOLO11n-obb) used in Technical Validation.

Expects a YOLO-OBB dataset view built with make_yolo_dataset_view.py (images/<split>, labels/<split>) in which background
frames are listed, one image stem per line, in --neg-names. Trains one run per seed and writes metrics.json,
results.csv and args.yaml per seed to --out. Recipe: pretrained yolo11n-obb, 640 px, batch 16, at most 100 epochs,
patience 30, ultralytics defaults otherwise; the checkpoint with the best validation mAP is scored once on the test split.

Metrics: val_all, test_all (frames with boxes plus background frames), test_pos (frames with boxes only) and, on the
background test frames, false boxes per frame at confidence 0.25.
"""
import argparse
import glob
import json
import os
import shutil
import time

import torch
import ultralytics
from ultralytics import YOLO

NAMES = "nc: 4\nnames:\n  0: SingleHotSpot\n  1: MultiHotSpot\n  2: Bypass\n  3: ModuleOpenCircuit\n"


def write_list(path, files):
    with open(path, "w") as f:
        f.write("\n".join(files) + "\n")


def summarise(m, model):
    ap = {model.names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
    return {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map), "per_class_ap50_95": ap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset view with images/{train,val,test} and labels/{train,val,test}")
    ap.add_argument("--neg-names", default=None, help="text file with the image stems of the background frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    neg = set()
    if a.neg_names:
        neg = {l.strip() for l in open(a.neg_names) if l.strip()}
    lists = {sp: sorted(glob.glob(os.path.join(a.data, "images", sp, "*"))) for sp in ("train", "val", "test")}
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    test_neg = [f for f in lists["test"] if stem(f) in neg]
    test_pos = [f for f in lists["test"] if stem(f) not in neg]
    for name, files in (("train_all", lists["train"]), ("val_all", lists["val"]), ("test_all", lists["test"]),
                        ("test_pos", test_pos), ("test_neg", test_neg)):
        write_list(os.path.join(a.data, name + ".txt"), files)
    head = "path: %s\ntrain: train_all.txt\nval: val_all.txt\n" % a.data
    open(os.path.join(a.data, "all.yaml"), "w").write(head + "test: test_all.txt\n" + NAMES)
    open(os.path.join(a.data, "pos.yaml"), "w").write(head + "test: test_pos.txt\n" + NAMES)

    for seed in a.seeds:
        t0 = time.time()
        name = "s%d" % seed
        YOLO("yolo11n-obb.pt").train(data=os.path.join(a.data, "all.yaml"), epochs=100, imgsz=640, batch=16, seed=seed,
                                     deterministic=True, patience=30, project=a.work, name=name, workers=2,
                                     plots=False, exist_ok=True, device=a.device)
        best = YOLO(os.path.join(a.work, name, "weights", "best.pt"))
        out = {"seed": seed, "epochs_run": sum(1 for _ in open(os.path.join(a.work, name, "results.csv"))) - 1}
        for tag, yaml, sp in (("val_all", "all.yaml", "val"), ("test_all", "all.yaml", "test"), ("test_pos", "pos.yaml", "test")):
            m = best.val(data=os.path.join(a.data, yaml), split=sp, imgsz=640, conf=0.001, iou=0.6, plots=False,
                         name="%s_%s" % (name, tag), exist_ok=True, device=a.device)
            out[tag] = summarise(m, best)
        n_box = n_img = 0
        if test_neg:
            for r in best.predict(source=os.path.join(a.data, "test_neg.txt"), conf=0.25, imgsz=640, stream=True,
                                  verbose=False, device=a.device):
                k = len(r.obb) if r.obb is not None else 0
                n_box += k
                n_img += int(k > 0)
        n = max(len(test_neg), 1)
        out["test_neg"] = {"n_images": len(test_neg), "fp_boxes_conf025": n_box, "fp_per_image": n_box / n,
                           "frac_images_with_fp": n_img / n}
        out["minutes"] = (time.time() - t0) / 60
        out["env"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
        dst = os.path.join(a.out, name)
        os.makedirs(dst, exist_ok=True)
        json.dump(out, open(os.path.join(dst, "metrics.json"), "w"), indent=2)
        for fn in ("results.csv", "args.yaml"):
            shutil.copy(os.path.join(a.work, name, fn), os.path.join(dst, fn))


if __name__ == "__main__":
    main()
