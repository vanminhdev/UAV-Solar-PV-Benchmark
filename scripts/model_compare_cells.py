#!/usr/bin/env python
"""Cell-level check of the orthomosaic transform models at TN19 (the only site with a reference grid).

    python -X utf8 code/scripts/model_compare_cells.py

For the four TN19 flights with ortho-level thermal boxes, each box centre (flight orthomosaic coordinates) is moved to the base
orthomosaic (= grid coordinates) with a translation-only, a similarity and an affine model fitted on the control points of that
flight (non-rejected points), and the reference cell that contains it is compared with the cell given by the box-to-cell
association (`metadata/association/thermal_box_to_cell.csv`, status `unique`). The association is derived from the boxes
themselves (best IoU after a per-flight shift), so it is a partly independent reference, not ground truth.

Output: metadata/ortho_registration/model_comparison_cells.csv  (per flight and model: boxes, share in the same cell,
share within one cell, median and 95th percentile distance from the transformed centre to the centre of the associated cell).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_footprints import read_gpkg_layer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
FARM = DS / "farm_01_thuan_nam_19"


def design(model, x, y):
    one, zero = np.ones_like(x), np.zeros_like(x)
    if model == "similarity":
        return np.r_[np.c_[x, -y, one, zero], np.c_[y, x, zero, one]]
    return np.r_[np.c_[x, y, one, zero, zero, zero], np.c_[zero, zero, zero, x, y, one]]  # affine


def transform(model, cp, x, y):
    x0, y0 = cp.E_src.mean(), cp.N_src.mean()
    if model == "translation":
        return x + (cp.E_ref - cp.E_src).mean(), y + (cp.N_ref - cp.N_src).mean()
    A = design(model, cp.E_src.values - x0, cp.N_src.values - y0)
    b = np.r_[cp.E_ref.values - x0, cp.N_ref.values - y0]
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    At = design(model, x - x0, y - y0)
    pr = At @ p
    return pr[: len(x)] + x0, pr[len(x):] + y0


def main():
    cells = read_gpkg_layer(FARM / "module_reference" / "TN19_modules_master.gpkg", "modules").set_index("module_id")
    polys = list(cells.geometry)
    ids = list(cells.index)
    tree = STRtree(polys)
    cent = {m: (g.centroid.x, g.centroid.y) for m, g in zip(ids, polys)}
    a = pd.read_csv(DS / "metadata" / "association" / "thermal_box_to_cell.csv")
    a = a[a.status == "unique"][["box_id", "flight_id", "module_id"]]
    bx = pd.read_csv(FARM / "annotations" / "thermal_anomaly_obb" / "thermal_boxes.csv")[["box_id", "centre_E", "centre_N"]]
    a = a.merge(bx, on="box_id")
    cp_all = pd.read_csv(DS / "metadata" / "ortho_registration" / "ortho_control_points.csv")
    cp_all = cp_all[cp_all.status != "rejected"]
    cw = 1.904  # cell length along the row, m (cell size 1.904 x 0.96 m)
    rows = []
    for fl, g in a.groupby("flight_id"):
        cp = cp_all[cp_all.flight_id == fl]
        if len(cp) < 4:
            continue
        for model in ("translation", "similarity", "affine"):
            x, y = transform(model, cp, g.centre_E.values, g.centre_N.values)
            hit = []
            for xi, yi in zip(x, y):
                idx = tree.query(Point(xi, yi), predicate="within")
                hit.append(ids[int(idx[0])] if len(idx) else None)
            d = np.array([np.hypot(xi - cent[m][0], yi - cent[m][1]) for xi, yi, m in zip(x, y, g.module_id)])
            same = np.array([h == m for h, m in zip(hit, g.module_id)])
            rows.append(dict(flight_id=fl, model=model, boxes=len(g), same_cell_pct=round(100 * same.mean(), 1),
                             within_one_cell_pct=round(100 * (d <= cw).mean(), 1), dist_median_m=round(float(np.median(d)), 3),
                             dist_p95_m=round(float(np.percentile(d, 95)), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(DS / "metadata" / "ortho_registration" / "model_comparison_cells.csv", index=False)
    print(out.to_string(index=False))
    print(out.groupby("model")[["same_cell_pct", "within_one_cell_pct", "dist_median_m", "dist_p95_m"]].mean().round(3))


if __name__ == "__main__":
    main()
