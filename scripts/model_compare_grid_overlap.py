#!/usr/bin/env python
"""Overlap of ortho-level thermal boxes with the reference grid after each transform model (all farms with a grid and boxes).

    python -X utf8 code/scripts/model_compare_grid_overlap.py

The boxes were drawn on the orthomosaic of a flight; the grid was drawn on the base (87 m) orthomosaic of the farm. For every
flight with boxes, the box corners are moved to the base orthomosaic with a translation-only, a similarity and an affine model
fitted on the control points of that flight (non-rejected points), then a small residual translation (search window +-0.5 m in E and N,
step 0.1 m, the same for every model) is chosen to maximise the mean best IoU between the boxes and the reference cells. The table
reports, per flight and model, the best-IoU median before and after the residual shift, the share of boxes whose best cell
has IoU >= 0.5 and at least twice the IoU of the second cell (status `unique` of the association), and the size of the residual shift.
No association table is needed, so the comparison does not depend on one transform model.

Output: metadata/ortho_registration/model_comparison_grid_overlap.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_footprints import read_gpkg_layer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
FARMS = {  # farm folder, grid file
    "TN19": ("farm_01_thuan_nam_19", "TN19_modules_master.gpkg"),
    "TNDL": ("farm_03_thuan_nam_duc_long", "TNDL_modules_master.gpkg"),
    "TT": ("farm_04_thien_tan", "TT_modules_master.gpkg"),
}
SHIFTS = np.round(np.arange(-0.5, 0.5001, 0.1), 2)


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
    pr = design(model, x - x0, y - y0) @ p
    return pr[: len(x)] + x0, pr[len(x):] + y0


def best_two(poly, tree, cells):
    ious = []
    for i in tree.query(poly):
        c = cells[int(i)]
        inter = poly.intersection(c).area
        if inter > 0:
            ious.append(inter / (poly.area + c.area - inter))
    ious.sort(reverse=True)
    return (ious[0] if ious else 0.0), (ious[1] if len(ious) > 1 else 0.0)


def score(polys, tree, cells):
    r = [best_two(p, tree, cells) for p in polys]
    return np.array([a for a, _ in r]), np.array([b for _, b in r])


def main():
    cp_all = pd.read_csv(DS / "metadata" / "ortho_registration" / "ortho_control_points.csv")
    cp_all = cp_all[cp_all.status != "rejected"]
    rows = []
    for farm, (folder, gp) in FARMS.items():
        cellsdf = read_gpkg_layer(DS / folder / "module_reference" / gp, "modules")
        cells = list(cellsdf.geometry)
        tree = STRtree(cells)
        bx = pd.read_csv(DS / folder / "annotations" / "thermal_anomaly_obb" / "thermal_boxes.csv")
        for fl, g in bx.groupby("flight_id"):
            cp = cp_all[cp_all.flight_id == fl]
            if len(cp) < 4:
                continue
            E = g[["E1", "E2", "E3", "E4"]].values
            N = g[["N1", "N2", "N3", "N4"]].values
            for model in ("translation", "similarity", "affine"):
                tx, ty = transform(model, cp, E.ravel(), N.ravel())
                tx, ty = tx.reshape(E.shape), ty.reshape(N.shape)
                polys = [Polygon(zip(tx[i], ty[i])) for i in range(len(g))]
                polys = [p if p.is_valid else p.buffer(0) for p in polys]
                raw1, _ = score(polys, tree, cells)
                best = None
                for dx in SHIFTS:
                    for dy in SHIFTS:
                        sh = [translate(p, dx, dy) for p in polys]
                        a, b = score(sh, tree, cells)
                        m = a.mean()
                        if best is None or m > best[0]:
                            best = (m, dx, dy, a, b)
                _, dx, dy, a, b = best
                uniq = (a >= 0.5) & (b <= a / 2)
                rows.append(dict(farm=farm, flight_id=fl, model=model, boxes=len(g),
                                 iou_median_before_shift=round(float(np.median(raw1)), 3),
                                 iou_median_after_shift=round(float(np.median(a)), 3),
                                 unique_pct=round(100 * uniq.mean(), 1),
                                 shift_E_m=dx, shift_N_m=dy,
                                 shift_mag_m=round(float(np.hypot(dx, dy)), 2)))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(DS / "metadata" / "ortho_registration" / "model_comparison_grid_overlap.csv", index=False)
    print(out.groupby(["farm", "model"])[["iou_median_before_shift", "iou_median_after_shift", "unique_pct", "shift_mag_m"]].mean().round(3))


if __name__ == "__main__":
    main()
