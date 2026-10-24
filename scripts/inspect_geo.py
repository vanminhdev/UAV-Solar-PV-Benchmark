"""Audit GeoTIFF headers and GeoPackage contents (read-only).

    python inspect_geo.py rasters <root_dir> <out.csv>     # every *.tif under root
    python inspect_geo.py gpkg <file.gpkg>                  # layers, module stats, ids
"""
import csv
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np


def rasters(root, out_csv):
    import rasterio

    rows = []
    for p in sorted(Path(root).rglob("*.tif")):
        r = {"path": str(p), "mb": round(p.stat().st_size / 1e6, 1)}
        try:
            with rasterio.open(p) as s:
                t = s.transform
                r.update(
                    width=s.width, height=s.height, bands=s.count, dtype=s.dtypes[0],
                    epsg=s.crs.to_epsg() if s.crs else "", crs_units="deg" if s.crs and s.crs.is_geographic else "m",
                    px_x=t.a, px_y=-t.e, west=s.bounds.left, south=s.bounds.bottom,
                    east=s.bounds.right, north=s.bounds.top, nodata=s.nodata,
                )
        except Exception as e:  # noqa: BLE001
            r["error"] = str(e)[:100]
        rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["path", "mb"].index(k) if k in ("path", "mb") else 9)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rasters ->", out_csv)


def _poly_sides(blob):
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    w = blob[8 + env:]
    bo = "<" if w[0] == 1 else ">"
    n = struct.unpack(bo + "I", w[9:13])[0]
    pts = np.frombuffer(w[13:13 + 16 * n], dtype=bo + "f8").reshape(-1, 2)
    e = np.hypot(*(pts[1:] - pts[:-1]).T)[:2]
    x, y = pts[:-1, 0], pts[:-1, 1]
    area = 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    return sorted(e), area


def gpkg(path):
    c = sqlite3.connect(path)
    print("contents:", c.execute("select table_name,data_type,srs_id,min_x,min_y,max_x,max_y from gpkg_contents").fetchall())
    n, d = c.execute("select count(*), count(distinct module_id) from modules").fetchone()
    print(f"modules: {n}  unique module_id: {d}")
    print("blocks:", c.execute("select layer, count(*) from modules group by 1").fetchall())
    sides, areas = zip(*(_poly_sides(g) for (g,) in c.execute("select geom from modules")))
    s = np.array(sides)
    print(f"short side {s[:,0].mean():.3f}±{s[:,0].std():.3f} m, long side {s[:,1].mean():.3f}±{s[:,1].std():.3f} m, "
          f"area {np.mean(areas):.3f} m2")
    if "path" in [r[1] for r in c.execute("pragma table_info(modules)")]:
        print("WARNING: 'path' column holds local absolute paths - strip before release:",
              c.execute("select path from modules limit 1").fetchone()[0])


if __name__ == "__main__":
    if sys.argv[1] == "rasters":
        rasters(sys.argv[2], sys.argv[3])
    else:
        gpkg(sys.argv[2])
