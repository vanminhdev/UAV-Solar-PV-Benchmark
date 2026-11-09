#!/usr/bin/env python
"""Cell-level inspection layer of TN19: which reference cells carry a thermal anomaly, in which flight and class.

    python -X utf8 code/scripts/build_inspection_cell_layer.py

Inputs (read only): farm_01_thuan_nam_19/module_reference/TN19_modules_master.gpkg (layer `modules`),
  metadata/association/thermal_box_to_cell.csv, farm_01_thuan_nam_19/annotations/thermal_anomaly_obb/thermal_boxes.csv.
Outputs (new files):
  farm_01_thuan_nam_19/annotations/thermal_anomaly_obb/TN19_inspection_cells.gpkg   (EPSG:32649)
      layer `defect_cells`  one polygon per reference cell with at least one uniquely assigned box
      layer `defect_boxes`  the ortho-level boxes (map coordinates) with the cell they were assigned to
  metadata/association/cell_repeat_flights.csv  cells hit by boxes of two or more flights (class per flight)
  metadata/association/cell_repeat_summary.json agreement of repeated cells

Only boxes with status `unique` (best IoU >= 0.5 and second best <= half of it) are assigned to a cell; `ambiguous` and
`unassigned` boxes stay in `defect_boxes` with an empty module_id.  The flights are consecutive passes on the same day, so repeated
cells measure the consistency of drawing and registration, not a change of the defect over time.
"""
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import wkb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_footprints import read_gpkg_layer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
FARM = DS / "farm_01_thuan_nam_19"
OUT_GPKG = FARM / "annotations" / "thermal_anomaly_obb" / "TN19_inspection_cells.gpkg"
SRID = 32649


def gpkg_blob(geom):
    """GeoPackage geometry blob: 'GP', version 0, flags (little endian, no envelope), srs id, then WKB."""
    return b"GP" + struct.pack("<BBi", 0, 1, SRID) + wkb.dumps(geom, byte_order=1)


def write_layer(con, name, df, cols):
    """df has a `geometry` column; cols = [(name, sqltype)]."""
    defs = ", ".join(f'"{c}" {t}' for c, t in cols)
    con.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB, {defs})')
    ph = ",".join("?" * (len(cols) + 1))
    rows = []
    for r in df.itertuples(index=False):
        rows.append([gpkg_blob(r.geometry)] + [None if (isinstance(getattr(r, c), float) and np.isnan(getattr(r, c))) else getattr(r, c) for c, _ in cols])
    con.executemany(f'INSERT INTO "{name}" (geom,{",".join(c for c, _ in cols)}) VALUES ({ph})', rows)
    b = np.array([g.bounds for g in df.geometry])
    con.execute("INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?,?,?,?)",
                (name, "features", name, f"TN19 {name}", "2026-11-08T00:00:00Z", b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()) + (SRID,))
    con.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)", (name, "geom", "POLYGON", SRID, 0, 0))


def main():
    cells = read_gpkg_layer(FARM / "module_reference" / "TN19_modules_master.gpkg", "modules").set_index("module_id")
    a = pd.read_csv(DS / "metadata" / "association" / "thermal_box_to_cell.csv")
    bx = pd.read_csv(FARM / "annotations" / "thermal_anomaly_obb" / "thermal_boxes.csv")
    a = a.merge(bx[["box_id", "n_frames", "E1", "N1", "E2", "N2", "E3", "N3", "E4", "N4"]], on="box_id", how="left")
    a["module_id_assigned"] = np.where(a.status == "unique", a.module_id, None)

    # ---- layer defect_boxes
    a["geometry"] = [Polygon([(r.E1, r.N1), (r.E2, r.N2), (r.E3, r.N3), (r.E4, r.N4)]) for r in a.itertuples()]
    boxes = a[["box_id", "flight_id", "class_final", "status", "module_id_assigned", "iou", "centroid_offset_m", "in_eval_zone", "n_frames", "geometry"]].copy()
    boxes.columns = ["box_id", "flight_id", "class_final", "status", "module_id", "iou", "centroid_offset_m", "in_eval_zone", "n_frames", "geometry"]
    boxes["in_eval_zone"] = boxes.in_eval_zone.astype(int)

    # ---- layer defect_cells
    u = a[a.status == "unique"].copy()
    rows = []
    for mid, g in u.groupby("module_id"):
        c = cells.loc[mid]
        fl = sorted(g.flight_id.unique())
        cl = sorted(g.class_final.unique())
        rows.append(dict(module_id=mid, block=c["layer"], string_id=c["String_ID"], row_id=int(c["Row_ID"]), id_pin=int(c["ID_Pin"]),
                         n_boxes=len(g), n_flights=len(fl), flights=";".join(fl), classes=";".join(cl), n_classes=len(cl),
                         iou_max=float(g.iou.max()), geometry=c["geometry"]))
    dc = pd.DataFrame(rows).sort_values("module_id").reset_index(drop=True)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    con = sqlite3.connect(OUT_GPKG)
    con.execute("PRAGMA application_id = 1196444487")  # 'GPKG'
    con.execute("PRAGMA user_version = 10200")
    con.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY, organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT)")
    for r in [("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None), ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None)]:
        con.execute("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", r)
    from pyproj import CRS
    crs = CRS.from_epsg(SRID)
    con.execute("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", (crs.name, SRID, "EPSG", SRID, crs.to_wkt(), None))
    con.execute("CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL, min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER)")
    con.execute("CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL, PRIMARY KEY (table_name, column_name))")
    write_layer(con, "defect_cells", dc, [("module_id", "TEXT"), ("block", "TEXT"), ("string_id", "TEXT"), ("row_id", "INTEGER"), ("id_pin", "INTEGER"),
                                          ("n_boxes", "INTEGER"), ("n_flights", "INTEGER"), ("flights", "TEXT"), ("classes", "TEXT"), ("n_classes", "INTEGER"), ("iou_max", "REAL")])
    write_layer(con, "defect_boxes", boxes, [("box_id", "TEXT"), ("flight_id", "TEXT"), ("class_final", "TEXT"), ("status", "TEXT"), ("module_id", "TEXT"),
                                             ("iou", "REAL"), ("centroid_offset_m", "REAL"), ("in_eval_zone", "INTEGER"), ("n_frames", "REAL")])
    con.commit()
    con.close()

    # ---- repeated cells
    rep = u.groupby("module_id").filter(lambda g: g.flight_id.nunique() >= 2)
    rows = []
    for mid, g in rep.groupby("module_id"):
        per = {f: sorted(x.class_final.unique()) for f, x in g.groupby("flight_id")}
        rows.append(dict(module_id=mid, n_flights=len(per), flights=";".join(sorted(per)),
                         classes_per_flight=" | ".join(f"{f}: {','.join(c)}" for f, c in sorted(per.items())),
                         same_class_set=int(len({tuple(c) for c in per.values()}) == 1)))
    rp = pd.DataFrame(rows)
    rp.to_csv(DS / "metadata" / "association" / "cell_repeat_flights.csv", index=False)
    # pairs of flights
    pairs = {}
    for mid, g in rep.groupby("module_id"):
        fls = sorted(g.flight_id.unique())
        for i in range(len(fls)):
            for j in range(i + 1, len(fls)):
                ci = set(g[g.flight_id == fls[i]].class_final)
                cj = set(g[g.flight_id == fls[j]].class_final)
                d = pairs.setdefault(f"{fls[i]} + {fls[j]}", dict(n_cells=0, same_class_set=0))
                d["n_cells"] += 1
                d["same_class_set"] += int(ci == cj)
    summ = dict(n_cells_defect=len(dc), n_cells_repeated=len(rp), n_cells_repeated_same_class=int(rp.same_class_set.sum()) if len(rp) else 0,
                n_boxes=len(boxes), n_boxes_assigned=int((boxes.module_id.notna()).sum()), by_flight_pair=pairs)
    json.dump(summ, open(DS / "metadata" / "association" / "cell_repeat_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
