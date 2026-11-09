"""Collect the control runs (colab_controls.py) and the local bootstrap/annotator scoring (score_baseline_local.py) into three CSV files.

  baseline_controls_per_seed.csv   per run (X2 cross-domain with the test-time rotation control, C2 main split, RS random split) and seed:
                                   mAP@0.5:0.95 of every scored set
  baseline_bootstrap.csv           mAP@0.5:0.95 with 95 % percentile bootstrap intervals over test frames (1,000 resamples, seed 20261026)
  baseline_vs_annotators.csv       the main-split models scored against the labels of A1, A2, A3 and the consensus on the agreement frames that lie in the
                                   test split of protocol C

Usage: python -X utf8 collect_controls_results.py --root <results_controls> --out <dir>
"""
import argparse
import csv
import json
import os

KEYS = {
    "C2": ["val_all", "test_all", "test_pos", "test_pos_TN19", "test_pos_BIM2", "test_pos_TNDL", "test_pos_TT"],
    "RS": ["val_all", "test_all", "test_pos", "test_pos_TN19", "test_pos_BIM2", "test_pos_TNDL", "test_pos_TT"],
    "X2": ["val_all", "test_all", "test_pos", "test_pos_BIM2", "test_pos_TNDL", "test_pos_TT", "test_pos_rot90cw_all", "test_pos_rot90cw_BIM2",
           "test_pos_rot90cw_TNDL", "test_pos_rot90cw_TT", "test_pos_rot180_all", "test_pos_rot180_BIM2", "test_pos_rot180_TNDL",
           "test_pos_rot180_TT", "val_all_rot90cw"],
}
NAMES = {"C2": "C_main_split", "RS": "random_image_split_same_sizes_as_C", "X2": "xdomain_train_TN19"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for tag in ("C2", "RS", "X2"):
        for s in range(3):
            m = json.load(open(os.path.join(a.root, tag, "s%d" % s, "metrics.json")))
            r = {"protocol": NAMES[tag], "seed": s, "epochs_run": m["epochs_run"]}
            for k in KEYS[tag]:
                r["map50_95_" + k] = round(m[k]["mAP50_95"], 4) if k in m else ""
            rows.append(r)
    keys = ["protocol", "seed", "epochs_run"] + sorted({k for r in rows for k in r if k.startswith("map50_95_")}, key=lambda k: (k != "map50_95_val_all", k))
    with open(os.path.join(a.out, "baseline_controls_per_seed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    b = os.path.join(a.root, "bootstrap")
    out = []
    for tag, name in (("C2", "C_main_split"), ("X2", "xdomain_train_TN19")):
        d = json.load(open(os.path.join(b, "boot_%s.json" % tag)))
        for s, v in d["per_seed"].items():
            out.append({"protocol": name, "frames": d["n_frames"], "seed": s, "map50_95": round(v["map_recomputed"], 4),
                        "ci95_low": round(v["boot_ci95"][0], 4), "ci95_high": round(v["boot_ci95"][1], 4)})
        m = d["mean_of_seeds"]
        out.append({"protocol": name, "frames": d["n_frames"], "seed": "mean_of_3", "map50_95": round(m["point"], 4),
                    "ci95_low": round(m["boot_ci95"][0], 4), "ci95_high": round(m["boot_ci95"][1], 4)})
    for s in range(3):
        d = json.load(open(os.path.join(b, "boot_RS_s%d.json" % s)))
        v = d["per_seed"][str(s)]
        out.append({"protocol": NAMES["RS"], "frames": d["n_frames"], "seed": s, "map50_95": round(v["map_recomputed"], 4),
                    "ci95_low": round(v["boot_ci95"][0], 4), "ci95_high": round(v["boot_ci95"][1], 4)})
    with open(os.path.join(a.out, "baseline_bootstrap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    an = []
    for lab in ("A1", "A2", "A3", "consensus"):
        d = json.load(open(os.path.join(b, "boot_annot_%s.json" % lab)))
        m = d["mean_of_seeds"]
        an.append({"reference_labels": lab, "frames": d["n_frames"], "map50_95_mean_of_3_seeds": round(m["point"], 4),
                   "ci95_low": round(m["boot_ci95"][0], 4), "ci95_high": round(m["boot_ci95"][1], 4),
                   "seed0": round(d["per_seed"]["0"]["map_recomputed"], 4), "seed1": round(d["per_seed"]["1"]["map_recomputed"], 4),
                   "seed2": round(d["per_seed"]["2"]["map_recomputed"], 4)})
    with open(os.path.join(a.out, "baseline_vs_annotators.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(an[0].keys()))
        w.writeheader()
        w.writerows(an)


if __name__ == "__main__":
    main()
