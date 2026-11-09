#!/usr/bin/env python
"""One compact margin-sensitivity table for the paper (csv + markdown): C margin 0 / 1.5 / 2.3 (chosen) / 3.0 and the strip + buffer variant A.

    python -X utf8 code/scripts/leakage_report.py        # first: consolidates leakage_check/mc_*.csv and the variant counts
    python -X utf8 code/scripts/leakage_margin_table.py
Output: dataset/metadata/splits/frame_label_split_margin_table.csv and .md (copies in leakage_check/).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakage_common as lc  # noqa: E402

KEY = {"C margin 0 m": "C_margin0.0", "C margin 1.5 m": "C_margin1.5", "C margin 2.3 m (main)": "C_margin2.3", "C margin 3.0 m": "C_margin3.0",
       "A strip + 10 m buffer": "A_strip_buffer"}
MARGIN = {"C margin 0 m": "0", "C margin 1.5 m": "1.5", "C margin 2.3 m (main)": "2.3 (chosen)", "C margin 3.0 m": "3.0", "A strip + 10 m buffer": "1.5 + 10 m buffer"}


def image_check(key):
    f = lc.OUTD / f"images_{key}_summary.csv"
    if not f.exists():
        return "not run"
    s = pd.read_csv(f).set_index("group")
    c = s.loc["cross_split/ALL"]
    return (f"{int(c.thermal_window_overlap_gt2pct + c.rgb_window_overlap_gt2pct)} of {int(c.n_pairs):,} pairs < 20 m with window overlap "
            f"({int(c.thermal_verified)} thermal / {int(c.rgb_verified)} RGB matches outside the used windows)")


def fmt(m, lo, hi, nd=None):
    nd = (2 if m < 1 else 1) if nd is None else nd
    return f"{m:.{nd}f} ({lo:.0f}-{hi:.0f})"


def main():
    C = pd.read_csv(lc.OUTD / "leakage_margin_cost.csv")
    rows = []
    for _, r in C.iterrows():
        key = KEY[r.variant]
        tot = r.train_boxes + r.val_boxes + r.test_boxes
        rows.append({
            "variant": r.variant, "rtk_margin_m": MARGIN[r.variant],
            "whole_frames": int(r.whole_frames), "crop_windows": int(r.crops), "frames_in_buffer": int(r.frames_in_buffer),
            "boxes_kept": int(r.boxes_kept), "boxes_lost_pct": r.boxes_lost_pct,
            "test_boxes": int(r.test_boxes), "test_share_pct": round(100 * r.test_boxes / tot, 1), "val_share_pct": round(100 * r.val_boxes / tot, 1),
            "moc_train": int(r.moc_train), "moc_val": int(r.moc_val), "moc_test": int(r.moc_test),
            "S1_pairs_mean": r.S1_pairs_mean, "S1_pairs_lo95": r.S1_pairs_lo95, "S1_pairs_hi95": r.S1_pairs_hi95,
            "S2_pairs_mean": r.S2_pairs_mean, "S2_pairs_lo95": r.S2_pairs_lo95, "S2_pairs_hi95": r.S2_pairs_hi95,
            "S1_boxes_in_foreign_footprint_pct": r.S1_boxes_in_overlap_pct, "S1_boxes_hi95_pct": r.S1_boxes_hi95_pct,
            "S2_boxes_in_foreign_footprint_pct": r.S2_boxes_in_overlap_pct, "S2_boxes_hi95_pct": r.S2_boxes_hi95_pct,
            "image_check": image_check(key)})
    T = pd.DataFrame(rows)
    out = lc.SPLITS / "frame_label_split_margin_table.csv"
    T.to_csv(out, index=False)
    T.to_csv(lc.OUTD / "frame_label_split_margin_table.csv", index=False)
    md = ["| variant | RTK margin (m) | whole frames + crops | boxes kept (lost) | test boxes (share) | MOC train / val / test | expected intersecting pairs S1 (95 % int.) | S2 (95 % int.) | boxes in a foreign footprint S1 / S2 (95 % upper) | image check |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        nm = ("**" + r.variant + "**") if "main" in r.variant else r.variant
        md.append(f"| {nm} | {r.rtk_margin_m} | {r.whole_frames:,} + {r.crop_windows} | {r.boxes_kept:,} ({r.boxes_lost_pct:.1f} %) | {r.test_boxes:,} ({r.test_share_pct:.1f} %) | "
                  f"{r.moc_train} / {r.moc_val} / {r.moc_test} | {fmt(r.S1_pairs_mean, r.S1_pairs_lo95, r.S1_pairs_hi95)} | {fmt(r.S2_pairs_mean, r.S2_pairs_lo95, r.S2_pairs_hi95)} | "
                  f"{r.S1_boxes_in_foreign_footprint_pct:.4f} % / {r.S2_boxes_in_foreign_footprint_pct:.4f} % ({r.S1_boxes_hi95_pct:.3f} / {r.S2_boxes_hi95_pct:.3f} %) | {r.image_check} |")
    txt = "\n".join(md) + "\n"
    (lc.SPLITS / "frame_label_split_margin_table.md").write_text(txt, encoding="utf-8")
    (lc.OUTD / "frame_label_split_margin_table.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
