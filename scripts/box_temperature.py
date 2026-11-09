"""Per-box apparent-temperature statistics for thermal_frame_boxes.csv (all farms).
Output (in .local/annotation_review/): box_temperature.csv, box_temperature_summary.csv, fig_box_delta_by_class.png
The OBB is rasterised on the 640x512 grid (PIL polygon); clipped coordinates are used when the box leaves the frame.
Ring = dilation of the mask by max(6 px, 0.25*short side) minus the box (ring_clean additionally excludes all other boxes of the frame).
Run: python -X utf8 code/scripts/box_temperature.py"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from review_common import *


def poly_mask(poly):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in poly], fill=1, outline=1)
    return np.asarray(m, dtype=bool)


def short_long(poly):
    p = np.array(poly)
    a = np.linalg.norm(p[1] - p[0]); b = np.linalg.norm(p[2] - p[1])
    return min(a, b), max(a, b)


def process_frame(args):
    npz, rows = args
    t = load_temp(npz)
    fmed = float(np.median(t))
    masks = []
    for r in rows:
        masks.append(poly_mask(box_polygon(r, clipped=True)))
    allm = np.any(masks, axis=0)
    out = []
    for r, m in zip(rows, masks):
        poly = box_polygon(r, clipped=True)
        sh, lo = short_long(poly)
        rec = dict(box_uid=r["box_uid"], n_pixels=int(m.sum()), box_short_px=sh, box_long_px=lo, frame_median=fmed)
        v = t[m]
        if len(v):
            rec.update(t_min=v.min(), t_mean=v.mean(), t_median=np.median(v), t_p95=np.percentile(v, 95), t_max=v.max())
        else:
            rec.update(t_min=np.nan, t_mean=np.nan, t_median=np.nan, t_p95=np.nan, t_max=np.nan)
        rad = max(6.0, 0.25 * sh)
        ys, xs = np.nonzero(m)
        if len(ys):
            pad = int(rad) + 2
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
            sub = m[y0:y1, x0:x1]
            d = ndi.distance_transform_edt(~sub)
            ringc = (d <= rad) & ~sub
            ring = np.zeros_like(m); ring[y0:y1, x0:x1] = ringc
            rv = t[ring]
            rc = t[ring & ~allm]
            if len(rv):
                rec.update(ring_n=int(len(rv)), ring_median=np.median(rv), ring_p25=np.percentile(rv, 25), ring_p75=np.percentile(rv, 75))
            if len(rc) >= 10:
                rec["ring_clean_median"] = np.median(rc)
        out.append(rec)
    return out


def main():
    b = load_boxes()
    fr = load_frames().set_index(["flight_id", "file"])
    jobs = []
    for (fid, f), g in b.groupby(["flight_id", "thermal_file"]):
        jobs.append((fr.loc[(fid, f)].npz_path, [r for _, r in g.iterrows()]))
    with Pool(6) as p:
        res = [x for chunk in p.imap_unordered(process_frame, jobs, chunksize=20) for x in chunk]
    d = pd.DataFrame(res)
    for c in ["ring_median", "ring_p25", "ring_p75", "ring_clean_median", "ring_n"]:
        if c not in d: d[c] = np.nan
    m = b[["box_uid", "flight_id", "thermal_file", "farm_dir", "class_name", "clipped_flag", "frac_area_inside_frame", "kept_in_yolo"]].merge(d, on="box_uid", how="left")
    m["clipped"] = m.clipped_flag.astype(int)
    m["delta_max"] = m.t_max - m.frame_median
    m["delta_p95"] = m.t_p95 - m.frame_median
    m["delta_ring"] = m.t_p95 - m.ring_median
    m["delta_ring_clean"] = m.t_p95 - m.ring_clean_median
    m["flag_small_mask"] = (m.n_pixels < 20).astype(int)
    m = m.set_index("box_uid").loc[b.box_uid].reset_index()      # original box order
    m.round(3).to_csv(OUT / "box_temperature.csv", index=False)
    print("boxes", len(m), "small masks", m.flag_small_mask.sum(), "clipped", m.clipped.sum(), "ring NaN", m.ring_median.isna().sum())
    summary(m)


def summary(m):
    rows = []
    for by in ("class_name", "farm_dir"):
        for key, g in m.groupby(by):
            for lab, sub in ([(by, g)] if by == "class_name" else []):
                pass
    recs = []
    def add(level, key_c, key_f, g):
        r = dict(level=level, class_name=key_c, farm=key_f, n=len(g), n_pixels_median=g.n_pixels.median())
        for c in ("delta_ring", "delta_max"):
            x = g[c].dropna()
            r[c + "_p10"], r[c + "_median"], r[c + "_p90"] = x.quantile(.1), x.median(), x.quantile(.9)
        recs.append(r)
    add("all", "ALL", "ALL", m)
    for c, g in m.groupby("class_name"): add("class", c, "ALL", g)
    for f, g in m.groupby("farm_dir"): add("farm", "ALL", f, g)
    for (c, f), g in m.groupby(["class_name", "farm_dir"]): add("class_farm", c, f, g)
    s = pd.DataFrame(recs).round(2)
    s.to_csv(OUT / "box_temperature_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(s.to_string(index=False))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for i, c in enumerate(("delta_ring", "delta_max")):
        data = [m.loc[m.class_name == k, c].dropna() for k in CLASSES]
        ax = axs[i, 0]
        ax.boxplot(data, labels=[SHORT[k] for k in CLASSES], showfliers=False)
        ax.set_title(f"{c} by class (degC), n=" + "/".join(str(len(x)) for x in data)); ax.axhline(0, color="grey", lw=.5)
        farms = sorted(m.farm_dir.unique())
        for j, fm in enumerate(farms):
            pass
        ax = axs[i, 1]
        pos, dat, lab = [], [], []
        for fi, fm in enumerate(farms):
            for ki, k in enumerate(CLASSES):
                x = m.loc[(m.farm_dir == fm) & (m.class_name == k), c].dropna()
                if len(x) >= 5:
                    pos.append(fi * 5 + ki); dat.append(x); lab.append(f"{fm[5:7]}\n{SHORT[k]}")
        ax.boxplot(dat, positions=pos, labels=lab, showfliers=False)
        ax.tick_params(axis="x", labelsize=6); ax.set_title(f"{c} by farm x class (n>=5)"); ax.axhline(0, color="grey", lw=.5)
    for k in CLASSES:
        axs[0, 2].hist(m.loc[m.class_name == k, "n_pixels"].clip(upper=3000), bins=60, histtype="step", label=SHORT[k], log=True)
    axs[0, 2].legend(); axs[0, 2].set_title("mask size (px, clipped at 3000)")
    data = [m.loc[m.class_name == k, "t_median"] - m.loc[m.class_name == k, "ring_median"] for k in CLASSES]
    axs[1, 2].boxplot([x.dropna() for x in data], labels=[SHORT[k] for k in CLASSES], showfliers=False)
    axs[1, 2].set_title("box median - ring median (degC)"); axs[1, 2].axhline(0, color="grey", lw=.5)
    plt.tight_layout(); plt.savefig(OUT / "fig_box_delta_by_class.png", dpi=90)


if __name__ == "__main__":
    main()
