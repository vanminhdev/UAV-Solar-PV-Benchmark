"""Collect the per-seed baseline results behind Table `bench` of the data descriptor into two CSV files.

Reads the metrics.json written by train_baseline_obb.py (or by the equivalent Colab notebooks) for every protocol and seed:
  <root>/<protocol_dir>/s<seed>/metrics.json
and writes
  baseline_per_seed_results.csv   one row per protocol and seed (all metrics that enter the table)
  baseline_same_class_means.csv   mean +- sd over seeds of the unweighted mean AP over the SAME three classes
                                  (SingleHotSpot, MultiHotSpot, Bypass) for every protocol, plus the four-class mean
                                  where the test set holds all four classes (the cross-domain test set holds no
                                  ModuleOpenCircuit box).
The standard deviation uses divisor n-1 (sample standard deviation).

Usage: python -X utf8 collect_baseline_per_seed.py --root <dir with results_final, results_final_v2> --out <dir>
"""
import argparse
import csv
import json
import os
import statistics as st

CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]
THREE = CLASSES[:3]
# (protocol label, relative directory)
GROUPS = [
    ("C_main_split", "results_final_v2/C2"),
    ("A_strip_buffer10m", "results_final/A"),
    ("xdomain_released_colours", "results_final_v2/X2"),
    ("xdomain_fixed_T_range", "results_final/XTfix"),
    ("xdomain_per_frame_T_range", "results_final/XTframe"),
]


def load(root):
    rows = []
    for label, rel in GROUPS:
        for seed in (0, 1, 2):
            f = os.path.join(root, rel, "s%d" % seed, "metrics.json")
            if not os.path.exists(f):
                continue
            m = json.load(open(f))
            r = {"protocol": label, "seed": m["seed"], "epochs_run": m["epochs_run"], "minutes": round(m["minutes"], 1),
                 "gpu": m.get("env", {}).get("gpu", ""), "ultralytics": m.get("env", {}).get("ultralytics", "")}
            for part in ("val_all", "test_all", "test_pos"):
                for k, name in (("mAP50_95", "map50_95"), ("mAP50", "map50"), ("precision", "precision"), ("recall", "recall")):
                    r["%s_%s" % (part, name)] = round(m[part][k], 4)
            for c in CLASSES:
                v = m["test_pos"]["per_class_ap50_95"].get(c)
                r["test_pos_ap50_95_%s" % c] = "" if v is None else round(v, 4)
            ap = [m["test_pos"]["per_class_ap50_95"].get(c) for c in THREE]
            r["test_pos_mean_ap_3class"] = round(sum(ap) / 3, 4)
            r["bg_test_frames"] = m["test_neg"]["n_images"]
            r["bg_false_boxes_conf025"] = m["test_neg"]["fp_boxes_conf025"]
            r["bg_false_boxes_per_frame"] = round(m["test_neg"]["fp_per_image"], 4)
            r["bg_frac_frames_with_false_box"] = round(m["test_neg"]["frac_images_with_fp"], 4)
            rows.append(r)
    return rows


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.root)
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "baseline_per_seed_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    out = []
    for label, _ in GROUPS:
        rs = [r for r in rows if r["protocol"] == label]
        if not rs:
            continue
        four = [r for r in rs if all(r["test_pos_ap50_95_" + c] != "" for c in CLASSES)]
        m3, s3 = ms([r["test_pos_mean_ap_3class"] for r in rs])
        o = {"protocol": label, "n_seeds": len(rs), "test_pos_mean_ap_3class_mean": round(m3, 3), "test_pos_mean_ap_3class_sd": round(s3, 3)}
        if len(four) == len(rs):
            m4, s4 = ms([sum(r["test_pos_ap50_95_" + c] for c in CLASSES) / 4 for r in rs])
            o.update({"test_pos_mean_ap_4class_mean": round(m4, 3), "test_pos_mean_ap_4class_sd": round(s4, 3)})
        else:
            o.update({"test_pos_mean_ap_4class_mean": "", "test_pos_mean_ap_4class_sd": ""})
        mm, ss = ms([r["test_pos_map50_95"] for r in rs])
        o.update({"test_pos_map50_95_reported_mean": round(mm, 3), "test_pos_map50_95_reported_sd": round(ss, 3)})
        out.append(o)
    with open(os.path.join(a.out, "baseline_same_class_means.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)


if __name__ == "__main__":
    main()
