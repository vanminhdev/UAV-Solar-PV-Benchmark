#!/usr/bin/env python
"""Consolidate the leakage check: Monte Carlo tables (mc_*.csv), margin cost of the variants and the headline table.

    python -X utf8 code/scripts/leakage_report.py
Output (dataset/metadata/splits/leakage_check/): leakage_mc_summary.csv (all rows), leakage_mc_headline.csv (split pair 'any', per farm),
leakage_margin_cost.csv (labels kept and expected leakage per variant: A, C 1.5 / 2.3 / 3.0 m).
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakage_common as lc  # noqa: E402

OUTD = lc.OUTD


def main():
    fs = sorted(glob.glob(str(OUTD / "mc_*.csv")))
    D = pd.concat([pd.read_csv(f).assign(file=Path(f).name) for f in fs], ignore_index=True)
    D.to_csv(OUTD / "leakage_mc_summary.csv", index=False)
    H = D[D.split_pair == "any"][["variant", "scenario", "bim2_sigma_m", "farm", "n_draws", "n_candidate_pairs", "pairs_intersecting_mean", "pairs_lo95", "pairs_hi95",
                                  "prob_any_pair", "overlap_share_of_smaller_mean", "overlap_share_p95", "overlap_share_max", "boxes_in_other_split_footprint_mean_pct",
                                  "boxes_lo95_pct", "boxes_hi95_pct", "boxes_total", "file"]]
    H.to_csv(OUTD / "leakage_mc_headline.csv", index=False)
    rows = []
    for key, name in (("C_margin0.0", "C margin 0 m"), ("C_margin1.5", "C margin 1.5 m"), ("C_margin2.3", "C margin 2.3 m (main)"), ("C_margin3.0", "C margin 3.0 m"),
                      ("A_strip_buffer", "A strip + 10 m buffer")):
        d = lc.VARIANTS[key][0]
        cn = pd.read_csv(d / "frame_label_split_counts.csv")
        cn = cn[(cn.farm == "ALL") & (cn.variant.isin(["main_component", "main_strict"]))].set_index("split")
        bl = pd.read_csv(d / "frame_label_split_buffer_loss.csv").set_index("farm")
        r = dict(variant=name, whole_frames=int(cn.loc[lc.SPL, "n_whole_frames"].sum()), crops=int(cn.loc[lc.SPL, "n_crops"].sum()), frames_in_buffer=int(cn.loc["buffer", "n_whole_frames"]),
                 boxes_kept=int(cn.loc[lc.SPL, "n_boxes"].sum()), boxes_lost_pct=float(bl.loc["ALL", "boxes_lost_pct"]),
                 train_boxes=int(cn.loc["train", "n_boxes"]), val_boxes=int(cn.loc["val", "n_boxes"]), test_boxes=int(cn.loc["test", "n_boxes"]),
                 moc_train=int(cn.loc["train", "n_ModuleOpenCircuit"]), moc_val=int(cn.loc["val", "n_ModuleOpenCircuit"]), moc_test=int(cn.loc["test", "n_ModuleOpenCircuit"]))
        for scen in ("S1", "S2", "S3"):
            f = OUTD / f"mc_{key}_{scen}.csv"
            if f.exists():
                m = pd.read_csv(f)
                m = m[(m.farm == "ALL") & (m.split_pair == "any")].iloc[0]
                r[f"{scen}_pairs_mean"] = m.pairs_intersecting_mean
                r[f"{scen}_pairs_lo95"] = m.pairs_lo95
                r[f"{scen}_pairs_hi95"] = m.pairs_hi95
                r[f"{scen}_boxes_in_overlap_pct"] = m.boxes_in_other_split_footprint_mean_pct
                r[f"{scen}_boxes_hi95_pct"] = m.boxes_hi95_pct
        rows.append(r)
    C = pd.DataFrame(rows)
    C.to_csv(OUTD / "leakage_margin_cost.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(C.T.to_string())


if __name__ == "__main__":
    main()
