"""Plot where the thermal box labels sit relative to the thermal image coverage of each farm.

    python plot_label_coverage.py <labels_dir> <dataset_root> <out_dir>

Per farm: overview (all thermal flights as grey footprints, labelled flights in colour, evaluation zones in red)
and one panel per labelled flight (flight footprint, evaluation zone, boxes coloured by class).
"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import build_footprints as bf  # noqa: E402

FLIGHTS = {
    "31M_T3": ("TN19", "TN19_20260411_31M_T3"), "35M_T3": ("TN19", "TN19_20260411_35M_T3"),
    "35M_T4": ("TN19", "TN19_20260411_35M_T4"), "35M_T5": ("TN19", "TN19_20260411_35M_T5"),
    "TNDL_GSD3_T2": ("TNDL", "TNDL_20260411_35M_T2"), "TT_GSD3_T1": ("TT", "TT_20260412_39M_T1"),
}
COL = {"SingleHotSpot": "#2ca02c", "MultiHotSpot": "#d62728", "Bypass": "#1f77b4", "ModuleOpenCircuit": "#9467bd"}
TR = Transformer.from_crs(4326, 32649, always_xy=True)


def draw_poly(ax, geom, **kw):
    """Filled polygon if 'fc' is given, outline otherwise."""
    for g in getattr(geom, "geoms", [geom]):
        x, y = g.exterior.xy
        if "fc" in kw:
            ax.fill(x, y, **kw)
        else:
            ax.plot(x, y, **kw)


def main():
    src, ds, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    fp = bf.read_gpkg_layer(bf.GPKG, "image_footprints")
    fp = fp[fp.sensor == "T"]
    boxes = pd.concat([pd.read_csv(p) for p in ds.glob("farm_0*/annotations/thermal_anomaly_obb/thermal_boxes.csv")])
    zones = {}
    for k, (farm, fid) in FLIGHTS.items():
        j = json.loads((src / k / "nhan_tay.json").read_text(encoding="utf-8"))
        zones[fid] = Polygon([TR.transform(lo, la) for lo, la in j["vung_danh_gia"]["lonlat"]])
    for farm in ("TN19", "TNDL", "TT"):
        lab = [fid for _, (f, fid) in FLIGHTS.items() if f == farm]
        flights = sorted(fp[fp.farm == farm].flight_id.unique())
        unions = {f: unary_union(list(fp[fp.flight_id == f].geometry)) for f in flights}
        n = len(lab)
        ncol = 3 if n > 2 else n + 1
        nrow = (n + 1 + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.4 * nrow), squeeze=False)
        axs = axs.ravel()
        ax = axs[0]
        for f, u in unions.items():
            draw_poly(ax, u, fc="#bbbbbb", alpha=0.25, lw=0) if f not in lab else None
            draw_poly(ax, u, color="#bbbbbb", lw=0.6)
        palette = plt.cm.tab10.colors
        for i, f in enumerate(lab):
            draw_poly(ax, unions[f], fc=palette[i], alpha=0.30, lw=0)
            draw_poly(ax, zones[f], color="red", lw=1.3)
        ax.set_title(f"{farm}: all thermal flights (grey) | labelled flights (colour) | evaluation zones (red)", fontsize=8)
        for ax in axs[1:]:
            ax.set_visible(False)
        for i, f in enumerate(lab):
            ax = axs[i + 1]
            ax.set_visible(True)
            draw_poly(ax, unions[f], fc="#9ecae1", alpha=0.35, lw=0)
            draw_poly(ax, unions[f], color="#3182bd", lw=0.8)
            draw_poly(ax, zones[f], color="red", lw=1.5)
            b = boxes[boxes.flight_id == f]
            for st, c in COL.items():
                s = b[b.class_final == st]
                if len(s):
                    ax.scatter(s.centre_E, s.centre_N, s=7, c=c, label=f"{st} ({len(s)})", linewidths=0)
            inz = int(b.in_eval_zone.sum())
            za = zones[f].intersection(unions[f]).area / unions[f].area * 100
            ax.set_title(f"{f}: zone covers {za:.0f}% of flight, {inz}/{len(b)} boxes inside zone", fontsize=8)
            ax.legend(fontsize=7, loc="lower left")
        for ax in axs:
            if ax.get_visible():
                ax.set_aspect("equal")
                ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(out / f"label_coverage_{farm}.png", dpi=110)
        plt.close(fig)
        print("wrote", out / f"label_coverage_{farm}.png")


if __name__ == "__main__":
    main()
