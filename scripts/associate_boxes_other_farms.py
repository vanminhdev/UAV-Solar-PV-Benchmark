#!/usr/bin/env python
"""Box-to-cell association for the farms with a reference grid other than TN19 (TNDL, TT).

    python -X utf8 code/scripts/associate_boxes_other_farms.py

Simpler than the TN19 procedure (`associate_boxes_to_grid.py`, which uses the grid-audit windows of the 87 m orthomosaic): the box corners of a
flight are moved to the base orthomosaic with the primary (similarity) transform fitted on the control points of the flight, a residual
translation (+-0.5 m in E and N, step 0.05 m, one value per flight) is chosen to maximise the mean best IoU with the reference cells, and every
box gets its best and second-best cell. Status as for TN19: `unique` if the best IoU is >= 0.5 and the second best is at most half of it,
`unassigned` if the best IoU is below 0.3, otherwise `ambiguous`.

Output: metadata/association/thermal_box_to_cell_TNDL_TT.csv
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
from model_compare_grid_overlap import DS, transform  # noqa: E402

FARMS = {"TNDL": ("farm_03_thuan_nam_duc_long", "TNDL_modules_master.gpkg"), "TT": ("farm_04_thien_tan", "TT_modules_master.gpkg")}
SHIFTS = np.round(np.arange(-0.5, 0.5001, 0.05), 2)


def top2(poly, tree, cells, ids):
    r = []
    for i in tree.query(poly):
        c = cells[int(i)]
        inter = poly.intersection(c).area
        if inter > 0:
            r.append((inter / (poly.area + c.area - inter), ids[int(i)], inter))
    r.sort(reverse=True)
    return r


def main():
    cp_all = pd.read_csv(DS / "metadata" / "ortho_registration" / "ortho_control_points.csv")
    cp_all = cp_all[cp_all.status != "rejected"]
    out = []
    for farm, (folder, gp) in FARMS.items():
        cdf = read_gpkg_layer(DS / folder / "module_reference" / gp, "modules")
        cells, ids = list(cdf.geometry), list(cdf.module_id)
        tree = STRtree(cells)
        bx = pd.read_csv(DS / folder / "annotations" / "thermal_anomaly_obb" / "thermal_boxes.csv")
        for fl, g in bx.groupby("flight_id"):
            cp = cp_all[cp_all.flight_id == fl]
            E = g[["E1", "E2", "E3", "E4"]].values
            N = g[["N1", "N2", "N3", "N4"]].values
            tx, ty = transform("similarity", cp, E.ravel(), N.ravel())
            tx, ty = tx.reshape(E.shape), ty.reshape(N.shape)
            polys = [Polygon(zip(tx[i], ty[i])) for i in range(len(g))]
            polys = [p if p.is_valid else p.buffer(0) for p in polys]
            best = None
            for dx in SHIFTS:
                for dy in SHIFTS:
                    m = np.mean([(top2(translate(p, dx, dy), tree, cells, ids) or [(0,)])[0][0] for p in polys])
                    if best is None or m > best[0]:
                        best = (m, dx, dy)
            _, dx, dy = best
            for (_, b), p in zip(g.iterrows(), polys):
                p2 = translate(p, dx, dy)
                t = top2(p2, tree, cells, ids)
                b1 = t[0] if t else (0.0, None, 0.0)
                b2 = t[1] if len(t) > 1 else (0.0, None, 0.0)
                status = "unassigned" if b1[0] < 0.3 else ("unique" if (b1[0] >= 0.5 and b2[0] <= b1[0] / 2) else "ambiguous")
                cc = cells[ids.index(b1[1])].centroid if b1[1] else None
                out.append(dict(farm=farm, box_id=b.box_id, module_code=b.module_code, module_id=b1[1], iou=round(b1[0], 4),
                                centroid_offset_m=round(float(p2.centroid.distance(cc)), 3) if cc is not None else None,
                                status=status, second_best_module_id=b2[1], second_best_iou=round(b2[0], 4), flight_id=fl,
                                class_final=b.class_final, in_eval_zone=b.in_eval_zone, reg_dE=dx, reg_dN=dy))
            print(farm, fl, len(g), "shift", dx, dy, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(DS / "metadata" / "association" / "thermal_box_to_cell_TNDL_TT.csv", index=False)
    print(df.groupby(["farm", "status"]).size())
    print(df.groupby("farm")[["iou", "centroid_offset_m"]].describe().round(3).T)


if __name__ == "__main__":
    main()
