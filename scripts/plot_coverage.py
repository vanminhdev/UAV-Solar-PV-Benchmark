"""Plot flight coverage per altitude class from image_inventory.csv (EXIF GPS + nominal sensor footprint).

    python plot_coverage.py <image_inventory.csv> <out.png> [--title TXT]

Layout: left = overlay (largest/highest flights at the back, labels outside with leader lines);
right = one small panel per altitude class so overlaps can be read without text collisions.
Footprint = convex hull of image centres buffered by half the ground footprint of that sensor
(thermal 7.68x6.14 mm, f 13.5 mm; RGB pitch 1.55 um, f 4.5 mm -> replace with ortho bounds when available).
"""
import argparse
import re

import matplotlib
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import MultiPoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (sensor width m, height m, focal m) per camera model and stream; Mavic 2 Enterprise Advanced values are nominal
SENSOR = {
    "ZH20T": {"T": (7.68e-3, 6.144e-3, 13.5e-3), "W": (4056 * 1.55e-6, 3040 * 1.55e-6, 4.5e-3)},
    "MAVIC2-ENTERPRISE-ADVANCED": {"T": (7.68e-3, 6.144e-3, 9.0e-3), "W": (6.4e-3, 4.8e-3, 4.5e-3)},
}


def alt_class(fl, med):
    return int(round(med))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inv")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--farm", default="", help="filter on the 'farm' column of a multi-farm inventory")
    a = ap.parse_args()
    d = pd.read_csv(a.inv)
    if a.farm:
        d = d[d.farm == a.farm]
    tr = Transformer.from_crs(4326, 32649, always_xy=True)
    d["E"], d["N"] = tr.transform(d.GpsLongitude.values, d.GpsLatitude.values)
    d["rel"] = d.RelativeAltitude.astype(float)
    polys = []
    for (fl, sensor), g in d.groupby(["flight", "sensor"]):
        if sensor == "T" or (sensor == "W" and (g.rel.median() > 60)):  # RGB only drawn for high mapping flights
            sw, sh, f = SENSOR[g.model.iloc[0]][sensor]
            alt = g.rel.median()
            half = 0.5 * np.hypot(sw, sh) * alt / f / 1.0 * 0.5
            poly = MultiPoint(g[["E", "N"]].values).convex_hull.buffer(half)
            polys.append(dict(flight=fl, sensor=sensor, alt=alt, cls=alt_class(fl, alt), poly=poly, n=len(g)))
    classes = sorted({p["cls"] for p in polys}, reverse=True)
    cmap = dict(zip(classes, ["#7f7f7f", "#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]))
    x0, y0, x1, y1 = (min(p["poly"].bounds[i] for p in polys) for i in range(4))
    x1 = max(p["poly"].bounds[2] for p in polys); y1 = max(p["poly"].bounds[3] for p in polys)
    pad = 30
    n = len(classes)
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, max(2, (n + 1) // 2 + 1), width_ratios=[2.6] + [1] * ((n + 1) // 2) if n > 1 else [1, 1])
    axm = fig.add_subplot(gs[:, 0])

    def draw(ax, ps, label=False):
        for p in sorted(ps, key=lambda p: -p["cls"]):
            xs, ys = p["poly"].exterior.xy
            ax.fill(xs, ys, color=cmap[p["cls"]], alpha=0.18 if p["cls"] > 60 else 0.28, lw=0)
            ax.plot(xs, ys, color=cmap[p["cls"]], lw=1.1)
        ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad); ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    draw(axm, polys)
    axm.set_title(f"{a.title}  overlay, outermost = highest altitude" if a.title else "overlay", fontsize=10)
    from matplotlib.patches import Patch
    axm.legend(handles=[Patch(color=cmap[c], alpha=0.6, label=f"{c} m") for c in classes], fontsize=7, loc="lower left")
    # outside labels with leader lines, per altitude class, sorted by northing
    right = x1 + pad
    lab = sorted(polys, key=lambda p: -p["poly"].centroid.y)
    ys = np.linspace(y1 + pad * 0.5, y0 - pad * 0.5, len(lab))
    axm.set_xlim(x0 - pad, x1 + pad + 0.42 * (x1 - x0))
    for yy, p in zip(ys, lab):
        c = p["poly"].representative_point()
        txt = f"{p['flight'].replace('. Thermal ', ' T').replace('Maping ', 'MAP')} ({p['alt']:.0f} m, {p['n']} img)"
        axm.annotate(txt, xy=(c.x, c.y), xytext=(right + 8, yy), fontsize=7, color=cmap[p["cls"]],
                     va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=cmap[p["cls"]], alpha=0.8))
    axm.set_xlabel("Easting (m, EPSG:32649)", fontsize=8); axm.set_ylabel("Northing (m)", fontsize=8)
    for i, c in enumerate(classes):
        ax = fig.add_subplot(gs[i % 2, 1 + i // 2])
        draw(ax, [p for p in polys if p["cls"] == c])
        ax.set_title(f"{c} m  ({sum(p['n'] for p in polys if p['cls']==c)} images)", fontsize=9, color=cmap[c])
    fig.tight_layout(); fig.savefig(a.out, dpi=120)


if __name__ == "__main__":
    main()
