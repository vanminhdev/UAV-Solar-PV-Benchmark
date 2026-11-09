"""Baseline controls for the data descriptor (Technical Validation), meant to run on a Colab GPU.

Jobs (each one is trained with the same recipe as train_baseline_obb.py: pretrained yolo11n-obb, 640 px, batch 16, <= 100 epochs,
patience 30, deterministic, best-validation checkpoint scored once on test; background frames are part of the lists):
  C2  protocol C (main footprint-disjoint split), re-run only to KEEP THE WEIGHTS (none were stored in the first round)
  X2  cross-domain protocol (train TN19, test BIM2/TNDL/TT), re-run to keep the weights, plus the test-time rotation control:
      the TRAINED-ON-TN19 model is scored on the test frames rotated by 90 deg clockwise (labels rotated with them), on the
      same frames rotated by 180 deg (null control: module orientation unchanged), and on the TN19 validation frames rotated
      by 90 deg (the model has seen only the unrotated orientation there), always per farm.
  RS  random image-level split control: the frames of protocol C (positives and background frames, all four classes) are pooled
      and re-assigned at random to train / val / test with exactly the sizes of protocol C (positives and background separately).
      The split seed is 1000 + seed, so every training seed has its own random split.
Outputs per job, in <results>/<TAG>/s<seed>/: metrics.json, best.pt, results.csv, args.yaml (+ rs_split.csv for RS).
Weights are small (about 6 MB); they are what the local scoring (bootstrap over test frames, score against the A2/A3/consensus
labels) needs. Nothing is uploaded from the user's machine: the dataset zips already sit on Drive.

Colab usage: see code/scripts/COLAB_CONTROLS_README.md. Local smoke test (no training):
  python -X utf8 colab_controls.py --dataset-dir <folder with the zips> --work <tmp> --results <tmp> --tags RS --prepare-only
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile

NAMES = "nc: 4\nnames:\n  0: SingleHotSpot\n  1: MultiHotSpot\n  2: Bypass\n  3: ModuleOpenCircuit\n"
ZIPS = {"C2": ["solar_uav_obb.zip", "solar_uav_obb_C2_neg.zip"], "RS": ["solar_uav_obb.zip", "solar_uav_obb_C2_neg.zip"],
        "X2": ["solar_uav_obb_X.zip", "solar_uav_obb_X2_neg.zip"]}
FARMS = ("TN19", "BIM2", "TNDL", "TT")


def stem(f):
    return os.path.splitext(os.path.basename(f))[0]


def farm(f):
    return os.path.basename(f).split("_")[0]


def write_list(path, files):
    with open(path, "w") as f:
        f.write("\n".join(files) + "\n")


def get_zip(dataset_dir, z, tmp):
    """Return a local path of the zip; very large zips on Drive may be stored as parts (z.partNNN + z.parts_done)."""
    q0 = os.path.join(tmp, z)
    if os.path.exists(q0) and os.path.getsize(q0) > 0:
        return q0
    p = os.path.join(dataset_dir, z)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    done = os.path.join(dataset_dir, "parts", z + ".parts_done")
    if not os.path.exists(done):
        raise FileNotFoundError(z)
    count, total = [int(x) for x in open(done).read().split()]
    parts = sorted(q for q in glob.glob(os.path.join(dataset_dir, "parts", z + ".part*")) if re.search(r"\.part\d{3}$", q))
    assert len(parts) == count and sum(os.path.getsize(q) for q in parts) == total, (z, len(parts), count)
    out = os.path.join(tmp, z)
    with open(out + ".tmp", "wb") as w:
        for q in parts:
            with open(q, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 24)
    os.rename(out + ".tmp", out)
    return out


def build_base(tag, dataset_dir, data):
    """Extract the zips of the protocol into data/{images,labels}/{train,val,test}; return the set of background stems."""
    ex = data + "_ex"
    shutil.rmtree(ex, ignore_errors=True)
    neg = set()
    for z in ZIPS[tag]:
        zipfile.ZipFile(get_zip(dataset_dir, z, os.path.dirname(data))).extractall(ex)
    for d in sorted(os.listdir(ex)):
        for sub in ("images", "labels"):
            for sp in ("train", "val", "test"):
                s = os.path.join(ex, d, sub, sp)
                if os.path.isdir(s):
                    os.makedirs(os.path.join(data, sub, sp), exist_ok=True)
                    for f in os.listdir(s):
                        shutil.move(os.path.join(s, f), os.path.join(data, sub, sp, f))
        nf = os.path.join(ex, d, "neg_names.txt")
        if os.path.exists(nf):
            neg |= {l.strip() for l in open(nf) if l.strip()}
    shutil.rmtree(ex, ignore_errors=True)
    return neg


def random_split(data, neg, seed):
    """Re-assign the frames of the C view at random to train/val/test with the same sizes (positives and background separately)."""
    pools = {sp: sorted(glob.glob(os.path.join(data, "images", sp, "*"))) for sp in ("train", "val", "test")}
    sizes = {(sp, isneg): sum((stem(f) in neg) == isneg for f in pools[sp]) for sp in pools for isneg in (False, True)}
    rng = random.Random(1000 + seed)
    rows = []
    new = {sp: [] for sp in pools}
    for isneg in (False, True):
        items = sorted(f for sp in pools for f in pools[sp] if (stem(f) in neg) == isneg)
        rng.shuffle(items)
        i = 0
        for sp in ("train", "val", "test"):
            new[sp] += items[i:i + sizes[(sp, isneg)]]
            i += sizes[(sp, isneg)]
        assert i == len(items)
    old_split = {f: sp for sp in pools for f in pools[sp]}
    # move files: first to a staging area, then into the new split
    stage = data + "_stage"
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(stage, sub), exist_ok=True)
    for f in old_split:
        lab = os.path.join(data, "labels", old_split[f], stem(f) + ".txt")
        shutil.move(f, os.path.join(stage, "images", os.path.basename(f)))
        if os.path.exists(lab):
            shutil.move(lab, os.path.join(stage, "labels", stem(f) + ".txt"))
    for sp in new:
        os.makedirs(os.path.join(data, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(data, "labels", sp), exist_ok=True)
        for f in new[sp]:
            b = os.path.basename(f)
            shutil.move(os.path.join(stage, "images", b), os.path.join(data, "images", sp, b))
            lab = os.path.join(stage, "labels", stem(f) + ".txt")
            if os.path.exists(lab):
                shutil.move(lab, os.path.join(data, "labels", sp, stem(f) + ".txt"))
            rows.append((b, old_split[f], sp, int(stem(f) in neg)))
    shutil.rmtree(stage, ignore_errors=True)
    return rows


def rotate_view(data, split, k, out):
    """Copy the split's images rotated by k*90 deg clockwise (k = 1 or 2) with the oriented-box labels rotated accordingly."""
    import cv2
    import numpy as np
    code = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180}[k]
    os.makedirs(os.path.join(out, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(data, "images", split, "*"))):
        im = cv2.imread(f)
        cv2.imwrite(os.path.join(out, "images", split, stem(f) + ".jpg"), cv2.rotate(im, code), [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for line in open(os.path.join(data, "labels", split, stem(f) + ".txt")):
            v = line.split()
            if len(v) == 9:
                xy = np.array(v[1:], float).reshape(4, 2)
                new = np.stack([1 - xy[:, 1], xy[:, 0]], 1) if k == 1 else 1 - xy
                lines.append(v[0] + " " + " ".join("%.6f" % t for t in new.reshape(-1)))
        open(os.path.join(out, "labels", split, stem(f) + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))


def write_yaml(dirpath, name, train, val, test):
    open(os.path.join(dirpath, name + ".yaml"), "w").write(
        "path: %s\ntrain: %s\nval: %s\ntest: %s\n" % (dirpath, train, val, test) + NAMES)


def ap_by_class(m, model):
    return {model.names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}


def summ(m, model):
    return {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "per_class_ap50_95": ap_by_class(m, model)}


def run_job(tag, seed, a):
    data = os.path.join(a.work, "%s_s%d" % (tag, seed))
    shutil.rmtree(data, ignore_errors=True)
    os.makedirs(data)
    neg = build_base(tag, a.dataset_dir, data)
    rows = random_split(data, neg, seed) if tag == "RS" else None
    lists = {sp: sorted(glob.glob(os.path.join(data, "images", sp, "*"))) for sp in ("train", "val", "test")}
    test_neg = [f for f in lists["test"] if stem(f) in neg]
    test_pos = [f for f in lists["test"] if stem(f) not in neg]
    for name, fs in (("train_all", lists["train"]), ("val_all", lists["val"]), ("test_all", lists["test"]),
                     ("test_pos", test_pos), ("test_neg", test_neg)):
        write_list(os.path.join(data, name + ".txt"), fs)
    write_yaml(data, "all", "train_all.txt", "val_all.txt", "test_all.txt")
    write_yaml(data, "pos", "train_all.txt", "val_all.txt", "test_pos.txt")
    for fm in FARMS:
        write_list(os.path.join(data, "test_pos_%s.txt" % fm), [f for f in test_pos if farm(f) == fm])
        write_yaml(data, "pos_%s" % fm, "train_all.txt", "val_all.txt", "test_pos_%s.txt" % fm)
    print(tag, seed, {sp: len(v) for sp, v in lists.items()}, "test positives", len(test_pos), "test background", len(test_neg), flush=True)
    res = os.path.join(a.results, tag, "s%d" % seed)
    os.makedirs(res, exist_ok=True)
    if rows is not None:
        with open(os.path.join(res, "rs_split.csv"), "w") as f:
            f.write("image,split_in_C,split_random,is_background\n")
            f.writelines("%s,%s,%s,%d\n" % r for r in rows)
    if a.prepare_only:
        return
    import torch
    import ultralytics
    from ultralytics import YOLO
    t0 = time.time()
    name = "%s_s%d" % (tag, seed)
    YOLO("yolo11n-obb.pt").train(data=os.path.join(data, "all.yaml"), epochs=100, imgsz=640, batch=16, seed=seed, deterministic=True,
                                 patience=30, project=os.path.join(a.work, "runs"), name=name, workers=2, plots=False, exist_ok=True)
    rd = os.path.join(a.work, "runs", name)
    best = YOLO(os.path.join(rd, "weights", "best.pt"))
    out = {"seed": seed, "tag": tag, "epochs_run": sum(1 for _ in open(os.path.join(rd, "results.csv"))) - 1}

    def val(key, yaml, split="test"):
        m = best.val(data=yaml, split=split, imgsz=640, conf=0.001, iou=0.6, plots=False, name="%s_%s" % (name, key), exist_ok=True)
        out[key] = summ(m, best)

    for key, yml, sp in (("val_all", "all.yaml", "val"), ("test_all", "all.yaml", "test"), ("test_pos", "pos.yaml", "test")):
        val(key, os.path.join(data, yml), sp)
    for fm in FARMS:
        if any(farm(f) == fm for f in test_pos):
            val("test_pos_" + fm, os.path.join(data, "pos_%s.yaml" % fm))
    n_box = n_img = 0
    for r in best.predict(source=os.path.join(data, "test_neg.txt"), conf=0.25, imgsz=640, stream=True, verbose=False):
        k = len(r.obb) if r.obb is not None else 0
        n_box += k
        n_img += int(k > 0)
    n = max(len(test_neg), 1)
    out["test_neg"] = {"n_images": len(test_neg), "fp_boxes_conf025": n_box, "fp_per_image": n_box / n, "frac_images_with_fp": n_img / n}
    if tag == "X2":  # test-time rotation control, positives only, per farm
        for k, lab in ((1, "rot90cw"), (2, "rot180")):
            rv = os.path.join(a.work, "%s_s%d_%s" % (tag, seed, lab))
            shutil.rmtree(rv, ignore_errors=True)
            rotate_view(data, "test", k, rv)
            rl = sorted(glob.glob(os.path.join(rv, "images", "test", "*")))
            pos_names = {stem(f) for f in test_pos}
            rp = [f for f in rl if stem(f) in pos_names]
            for fm in (None,) + FARMS:
                sel = [f for f in rp if fm is None or farm(f) == fm]
                if not sel:
                    continue
                write_list(os.path.join(rv, "t.txt"), sel)
                write_yaml(rv, "t", "t.txt", "t.txt", "t.txt")
                val("test_pos_%s_%s" % (lab, fm or "all"), os.path.join(rv, "t.yaml"))
            shutil.rmtree(rv, ignore_errors=True)
        rv = os.path.join(a.work, "%s_s%d_valrot" % (tag, seed))
        shutil.rmtree(rv, ignore_errors=True)
        rotate_view(data, "val", 1, rv)
        write_list(os.path.join(rv, "t.txt"), sorted(glob.glob(os.path.join(rv, "images", "val", "*"))))
        write_yaml(rv, "t", "t.txt", "t.txt", "t.txt")
        val("val_all_rot90cw", os.path.join(rv, "t.yaml"))
        shutil.rmtree(rv, ignore_errors=True)
    out["minutes"] = (time.time() - t0) / 60
    out["env"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    json.dump(out, open(os.path.join(res, "metrics.json"), "w"), indent=2)
    for fn in ("results.csv", "args.yaml"):
        shutil.copy(os.path.join(rd, fn), os.path.join(res, fn))
    shutil.copy(os.path.join(rd, "weights", "best.pt"), os.path.join(res, "best.pt"))
    shutil.rmtree(data, ignore_errors=True)
    print("DONE", tag, seed, "test_pos mAP50-95 %.3f" % out["test_pos"]["mAP50_95"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="folder holding the zips (and parts/ for zips stored in parts)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--tags", nargs="+", default=["X2", "C2", "RS"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--par", type=int, default=1, help="jobs at the same time (separate processes)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--one", help="internal: TAG:SEED")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.one:
        t, s = a.one.split(":")
        run_job(t, int(s), a)
        return
    jobs = [(t, s) for t in a.tags for s in a.seeds if not os.path.exists(os.path.join(a.results, t, "s%d" % s, "metrics.json"))]
    print("jobs:", jobs, flush=True)
    for z in sorted({z for t, _ in jobs for z in ZIPS[t]}):  # assemble zips stored in parts once, before the processes start
        get_zip(a.dataset_dir, z, a.work)
    procs = []
    while jobs or procs:
        while jobs and len(procs) < a.par:
            t, s = jobs.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--dataset-dir", a.dataset_dir, "--results", a.results, "--work", a.work,
                   "--one", "%s:%d" % (t, s)] + (["--prepare-only"] if a.prepare_only else [])
            procs.append(((t, s), subprocess.Popen(cmd)))
        for p in list(procs):
            if p[1].poll() is not None:
                print("finished", p[0], "rc", p[1].returncode, flush=True)
                procs.remove(p)
        time.sleep(10)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
