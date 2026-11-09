#!/usr/bin/env python
"""Image check of the cross-split pairs of a split, independent of the footprints: do two frames of different splits really share scene content?

    python -X utf8 code/scripts/leakage_images.py pairs   --variant C_margin1.5 --max-dist 20 [--sample N]
    python -X utf8 code/scripts/leakage_images.py match   --variant C_margin1.5 [--workers 7]
    python -X utf8 code/scripts/leakage_images.py report  --variant C_margin1.5

For every cross-split pair of units (whole frames / crop windows) whose nominal footprints are closer than --max-dist metres, the RGB frames paired with the two
thermal frames (metadata/pairing) are matched with SIFT (Lowe ratio 0.85, RANSAC homography, USAC-MAGSAC); the thermal images (rendered R-JPEG, CLAHE) are matched
the same way as a second view. A match is *verified* with >= 25 RANSAC inliers (RGB) / >= 12 (thermal), an inlier ratio >= 0.25 and a plausible homography. The
homography maps the thermal window of frame A (the whole 640 x 512 frame or the crop window; in the RGB image it is the centre rectangle given by the
RGB/thermal scale of the optics) into frame B; the overlap with the thermal window of B gives the shared share of the smaller window. Controls: far-apart
pairs (false positives) and overlapping neighbours of one flight (recall).
Output: dataset/metadata/splits/leakage_check/images_<variant>_pairs.csv, images_<variant>_results.csv, images_<variant>_summary.csv
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakage_common as lc  # noqa: E402

cv2.setNumThreads(1)
CACHE = lc.ROOT / ".local/tmp/leak_feat"
FARM_DIR = {"TN19": "farm_01_thuan_nam_19", "BIM2": "farm_02_bim_2", "TNDL": "farm_03_thuan_nam_duc_long", "TT": "farm_04_thien_tan"}
RGB_W = 2000                      # RGB images are matched at this width
MIN_INL = {"rgb": 25, "thermal": 12}
MIN_RATIO = 0.25
RATIO = 0.85


# ----------------------------------------------------------------------------- files
def raw_index():
    idx = {}
    for farm, d in FARM_DIR.items():
        for p in (lc.DS / d / "raw_images").glob("*/*/"):
            idx[(farm, p.name)] = p
    return idx


def pairing():
    p = pd.read_csv(lc.DS / "metadata/pairing/rgb_thermal_pairing.csv", usecols=["flight_id", "rgb_file", "thermal_file", "pairing_status"])
    return {(r.flight_id, r.thermal_file): r.rgb_file for r in p.itertuples() if isinstance(r.rgb_file, str)}


def rgb_scale_table():
    """RGB px per thermal px for every flight: from the footprints (ground width of the RGB / thermal footprint) and the image widths."""
    fp = lc.bf.read_gpkg_layer(lc.bf.GPKG, "image_footprints")
    inv = pd.read_csv(lc.bf.INVENTORY, usecols=["flight_id", "sensor", "width", "height"]).drop_duplicates(["flight_id", "sensor"])
    w = inv.set_index(["flight_id", "sensor"])[["width", "height"]]
    g = fp.groupby(["flight_id", "sensor"]).ground_w_m.median()
    out = {}
    for fl in fp.flight_id.unique():
        try:
            tw, rw = g[(fl, "T")], g[(fl, "W")]
            rpx = w.loc[(fl, "W")].width
            out[fl] = dict(scale=rpx * tw / (640.0 * rw), rgb_w=int(rpx), rgb_h=int(w.loc[(fl, "W")].height))
        except KeyError:
            pass
    return out


# ----------------------------------------------------------------------------- pairs
def select_pairs(variant, max_dist, sample, seed=1):
    U, B = lc.load_units(variant)
    polys = shapely.polygons(np.concatenate([np.stack(U.corners.values), np.stack(U.corners.values)[:, :1]], 1))
    rows = []
    for farm in lc.mf.FARMS:
        idx = np.where(U.farm.values == farm)[0]
        if len(idx) == 0:
            continue
        tree = STRtree(polys[idx])
        ii, jj = tree.query(polys[idx], predicate="dwithin", distance=max_dist)
        for i, j in zip(ii, jj):
            gi, gj = idx[i], idx[j]
            if gi < gj and U.split.values[gi] != U.split.values[gj] and U.frame_idx.values[gi] != U.frame_idx.values[gj]:
                rows.append((gi, gj, float(shapely.distance(polys[gi], polys[gj]))))
    P = pd.DataFrame(rows, columns=["i", "j", "nominal_distance_m"])
    P["farm"] = U.farm.values[P.i]
    P["split_a"], P["split_b"] = U.split.values[P.i], U.split.values[P.j]
    # one test per (frame pair, rect pair): keep the closest unit pair of each frame pair
    P["fa"] = [f"{U.flight_id.values[i]}|{U.thermal_file.values[i]}" for i in P.i]
    P["fb"] = [f"{U.flight_id.values[j]}|{U.thermal_file.values[j]}" for j in P.j]
    P = P.sort_values("nominal_distance_m").drop_duplicates(["fa", "fb"]).reset_index(drop=True)
    P["kind_a"], P["kind_b"] = U.kind.values[P.i], U.kind.values[P.j]
    P["rect_a"] = [U.rect.values[i] for i in P.i]
    P["rect_b"] = [U.rect.values[j] for j in P.j]
    P["uid_a"], P["uid_b"] = U.uid.values[P.i], U.uid.values[P.j]
    if sample and len(P) > sample:
        near = P[P.nominal_distance_m < 8]
        rest = P[P.nominal_distance_m >= 8]
        k = max(sample - len(near), 0)
        P = pd.concat([near, rest.sample(min(k, len(rest)), random_state=seed)]).sort_values("nominal_distance_m").reset_index(drop=True)
    P["role"] = "cross_split"
    # controls
    rng = np.random.default_rng(seed)
    ctrl = []
    fl = U[U.kind == "frame"]
    fpoly_all = polys[fl.index.values]
    fl = fl.reset_index(drop=True)
    cen = np.array([[p_.centroid.x, p_.centroid.y] for p_ in fpoly_all])
    n_ctrl = 150
    for farm in lc.mf.FARMS:
        m = np.where(fl.farm.values == farm)[0]
        if len(m) < 20:
            continue
        for _ in range(n_ctrl // 4):
            a = rng.choice(m)
            far = m[np.hypot(*(cen[m] - cen[a]).T) > 120]
            far = far[fl.split.values[far] != fl.split.values[a]]
            if len(far):
                b = rng.choice(far)
                ctrl.append(dict(farm=farm, fa=fl.uid.values[a], fb=fl.uid.values[b], role="control_far", nominal_distance_m=float(np.hypot(*(cen[a] - cen[b]))),
                                 rect_a=(0, 0, 640, 512), rect_b=(0, 0, 640, 512), split_a=fl.split.values[a], split_b=fl.split.values[b], kind_a="frame", kind_b="frame"))
        # recall controls: same-flight neighbours with a nominal footprint overlap in three bands
        f_polys = fpoly_all[m]
        tree = STRtree(f_polys)
        bands = [(0.02, 0.10, "control_overlap_2_10"), (0.10, 0.30, "control_overlap_10_30"), (0.30, 0.70, "control_overlap_30_70")]
        got = {b[2]: 0 for b in bands}
        for k in rng.permutation(len(m)):
            if all(v >= 30 for v in got.values()):
                break
            ii = tree.query(f_polys[k], predicate="intersects")
            for q in ii:
                if q == k or fl.flight_id.values[m[q]] != fl.flight_id.values[m[k]]:
                    continue
                inter = f_polys[k].intersection(f_polys[q]).area / f_polys[k].area
                for lo, hi, name in bands:
                    if lo < inter <= hi and got[name] < 30:
                        ctrl.append(dict(farm=farm, fa=fl.uid.values[m[k]], fb=fl.uid.values[m[q]], role=name, nominal_distance_m=0.0,
                                         rect_a=(0, 0, 640, 512), rect_b=(0, 0, 640, 512), split_a=fl.split.values[m[k]], split_b=fl.split.values[m[q]],
                                         kind_a="frame", kind_b="frame", nominal_overlap=float(inter)))
                        got[name] += 1
                        break
    C = pd.DataFrame(ctrl)
    P = pd.concat([P[["farm", "fa", "fb", "role", "nominal_distance_m", "rect_a", "rect_b", "split_a", "split_b", "kind_a", "kind_b", "uid_a", "uid_b"]], C], ignore_index=True)
    P["nominal_overlap"] = P.get("nominal_overlap", 0.0)
    return P


# ----------------------------------------------------------------------------- features
def feat_path(kind, flight, file):
    return CACHE / f"{kind}__{flight}__{Path(file).stem}.npz"


def extract(args):
    kind, farm, flight, file, path = args
    out = feat_path(kind, flight, file)
    if out.exists():
        return str(out)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if kind == "rgb":
        h, w = img.shape
        f = RGB_W / w
        img = cv2.resize(img, (RGB_W, int(round(h * f))), interpolation=cv2.INTER_AREA)
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)          # thermal frames are 640 x 512: work at 2x
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(kp) < 5:
        np.savez_compressed(out, pts=np.zeros((0, 2), np.float32), des=np.zeros((0, 128), np.uint8), shape=np.array(img.shape))
        return str(out)
    pts = np.array([k.pt for k in kp], np.float32)
    np.savez_compressed(out, pts=pts, des=np.clip(des, 0, 255).astype(np.uint8), shape=np.array(img.shape))
    return str(out)


def load_feat(kind, flight, file):
    d = np.load(feat_path(kind, flight, file))
    return d["pts"], d["des"].astype(np.float32), tuple(d["shape"])


# ----------------------------------------------------------------------------- matching
def rect_poly_rgb(rect, shape, scale, img_scale):
    """thermal window (x0, y0, x1, y1) -> polygon in the resized RGB image: centre rectangle of the RGB, scale = RGB px per thermal px (full resolution)."""
    h, w = shape
    cx, cy = w / 2.0, h / 2.0
    s = scale * img_scale
    x0, y0, x1, y1 = rect
    return np.array([[cx + (x0 - 320) * s, cy + (y0 - 256) * s], [cx + (x1 - 320) * s, cy + (y0 - 256) * s],
                     [cx + (x1 - 320) * s, cy + (y1 - 256) * s], [cx + (x0 - 320) * s, cy + (y1 - 256) * s]])


def match_one(fa, fb):
    """SIFT ratio-test matches + RANSAC homography A -> B. -> dict(n_matches, inliers, H) or None."""
    pa, da, sa = fa
    pb, db, sb = fb
    if len(pa) < 8 or len(pb) < 8:
        return dict(n_matches=0, inliers=0, H=None)
    bf_ = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))          # kd-tree matcher (brute force: ~4 s per RGB pair)
    m = bf_.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < RATIO * x[1].distance]
    if len(good) < 8:
        return dict(n_matches=len(good), inliers=0, H=None)
    src = np.float32([pa[g.queryIdx] for g in good])
    dst = np.float32([pb[g.trainIdx] for g in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    if H is None:
        return dict(n_matches=len(good), inliers=0, H=None)
    return dict(n_matches=len(good), inliers=int(mask.sum()), H=H)


def plausible(H, shape):
    if H is None:
        return False
    A = H[:2, :2]
    det = np.linalg.det(A)
    if not (0.25 < abs(det) < 4.0):
        return False
    if abs(H[2, 0]) > 2e-3 or abs(H[2, 1]) > 2e-3:
        return False
    sv = np.linalg.svd(A, compute_uv=False)
    return sv[0] / sv[1] < 1.8


def overlap_share(H, poly_a, poly_b):
    """share of the smaller thermal window shared by A (warped into B) and B."""
    pa = cv2.perspectiveTransform(poly_a[None].astype(np.float64), H)[0]
    A = shapely.Polygon(pa)
    B = shapely.Polygon(poly_b)
    if not A.is_valid or not B.is_valid or A.area <= 0:
        return np.nan, np.nan
    inter = A.intersection(B).area
    return float(inter / min(A.area, B.area)), float(inter)


def job(args):
    row, pair_files, scales = args
    (farm, fa, fb, rect_a, rect_b) = row
    fla, tha = fa.split("|")
    flb, thb = fb.split("|")
    res = dict(fa=fa, fb=fb)
    try:
        # ---- RGB view
        ra, rb = pair_files.get((fla, tha)), pair_files.get((flb, thb))
        sc_a, sc_b = scales.get(fla), scales.get(flb)
        if ra and rb and sc_a and sc_b:
            Fa, Fb = load_feat("rgb", fla, ra), load_feat("rgb", flb, rb)
            r = match_one(Fa, Fb)
            res.update(rgb_matches=r["n_matches"], rgb_inliers=r["inliers"])
            ok = r["inliers"] >= MIN_INL["rgb"] and r["inliers"] >= MIN_RATIO * max(r["n_matches"], 1) and plausible(r["H"], Fa[2])
            res["rgb_verified"] = bool(ok)
            if ok:
                pa = rect_poly_rgb(rect_a, Fa[2], sc_a["scale"], Fa[2][1] / sc_a["rgb_w"])
                pb = rect_poly_rgb(rect_b, Fb[2], sc_b["scale"], Fb[2][1] / sc_b["rgb_w"])
                sh, ar = overlap_share(r["H"], pa, pb)
                res["rgb_overlap_share"], res["rgb_overlap_px2"] = sh, ar
                # RGB-RGB overlap (whole images) as context
                full_a = np.array([[0, 0], [Fa[2][1], 0], [Fa[2][1], Fa[2][0]], [0, Fa[2][0]]], float)
                full_b = np.array([[0, 0], [Fb[2][1], 0], [Fb[2][1], Fb[2][0]], [0, Fb[2][0]]], float)
                res["rgb_full_image_overlap_share"], _ = overlap_share(r["H"], full_a, full_b)
        # ---- thermal view
        Ta, Tb = load_feat("thermal", fla, tha), load_feat("thermal", flb, thb)
        r = match_one(Ta, Tb)
        res.update(th_matches=r["n_matches"], th_inliers=r["inliers"])
        ok = r["inliers"] >= MIN_INL["thermal"] and r["inliers"] >= MIN_RATIO * max(r["n_matches"], 1) and plausible(r["H"], Ta[2])
        res["th_verified"] = bool(ok)
        if ok:
            def tpoly(rect):
                x0, y0, x1, y1 = rect
                return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) * 2.0
            sh, ar = overlap_share(r["H"], tpoly(rect_a), tpoly(rect_b))
            res["th_overlap_share"], res["th_overlap_px2"] = sh, ar
    except Exception as e:  # pragma: no cover
        res["error"] = repr(e)[:120]
    return res


def cmd_pairs(a):
    P = select_pairs(a.variant, a.max_dist, a.sample)
    lc.OUTD.mkdir(parents=True, exist_ok=True)
    P.to_csv(lc.OUTD / f"images_{a.variant}_pairs.csv", index=False)
    print(P.role.value_counts().to_dict(), "cross-split pairs:", int((P.role == "cross_split").sum()), "frames:", len(set(P.fa) | set(P.fb)))
    print(P[P.role == "cross_split"].groupby("farm").size().to_dict(), P[P.role == "cross_split"].nominal_distance_m.describe().round(1).to_dict())


def parse_rect(s):
    return tuple(int(float(x)) for x in str(s).strip("()").split(","))


def cmd_match(a):
    P = pd.read_csv(lc.OUTD / f"images_{a.variant}_pairs.csv")
    CACHE.mkdir(parents=True, exist_ok=True)
    idx, pf, scales = raw_index(), pairing(), rgb_scale_table()
    frames_ = sorted(set(P.fa) | set(P.fb))
    jobs = []
    for f in frames_:
        fl, th = f.split("|")
        farm = fl.split("_")[0]
        d = idx.get((farm, fl))
        if d is None:
            continue
        jobs.append(("thermal", farm, fl, th, d / th))
        rg = pf.get((fl, th))
        if rg:
            jobs.append(("rgb", farm, fl, rg, d / rg))
    t0 = time.time()
    with Pool(a.workers) as pool:
        n = 0
        for _ in pool.imap_unordered(extract, jobs, chunksize=4):
            n += 1
            if n % 300 == 0:
                print(f"  features {n}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    # rgb feature file name uses the rgb file name; map in job via pairing
    tasks = [((r.farm, r.fa, r.fb, parse_rect(r.rect_a), parse_rect(r.rect_b)), pf, scales) for r in P.itertuples()]
    import pickle
    ck = lc.ROOT / f".local/tmp/leak_ckpt_{a.variant}.pkl"           # checkpoint: the run can be resumed after an interruption
    out = pickle.load(open(ck, "rb")) if ck.exists() else []
    print(f"  resuming at pair {len(out)}", flush=True)
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap(job, tasks[len(out):], chunksize=8), start=len(out)):
            out.append(res)
            if k % 500 == 0:
                print(f"  pairs {k}/{len(tasks)} ({time.time() - t0:.0f} s)", flush=True)
                pickle.dump(out, open(ck, "wb"))
    R = pd.concat([P.reset_index(drop=True), pd.DataFrame(out).drop(columns=["fa", "fb"])], axis=1)
    R.to_csv(lc.OUTD / f"images_{a.variant}_results.csv", index=False)
    print("done", time.time() - t0)


def cmd_report(a):
    R = pd.read_csv(lc.OUTD / f"images_{a.variant}_results.csv")
    R["pair_kind"] = R.kind_a + "-" + R.kind_b
    R["dist_bin"] = pd.cut(R.nominal_distance_m, [-1, 8, 14, 20.1], labels=["<8 m", "8-14 m", "14-20 m"]).astype(str)
    rows = []

    def add(name, g, extra=None):
        vr, vt = g.rgb_verified.fillna(False).astype(bool), g.th_verified.fillna(False).astype(bool)
        ov_r = g.rgb_overlap_share[vr & (g.rgb_overlap_share > 0.02)]
        ov_t = g.th_overlap_share[vt & (g.th_overlap_share > 0.02)]
        rows.append(dict(variant=a.variant, group=name, n_pairs=len(g), rgb_verified=int(vr.sum()), thermal_verified=int(vt.sum()), either_verified=int((vr | vt).sum()),
                         both_verified=int((vr & vt).sum()), rgb_window_overlap_gt2pct=int(len(ov_r)), thermal_window_overlap_gt2pct=int(len(ov_t)),
                         rgb_window_overlap_max=float(ov_r.max()) if len(ov_r) else 0.0, thermal_window_overlap_max=float(ov_t.max()) if len(ov_t) else 0.0,
                         median_rgb_inliers=float(g.rgb_inliers.median()), median_thermal_inliers=float(g.th_inliers.median()),
                         recall_thermal=round(float(vt.mean()), 3), recall_rgb=round(float(vr.mean()), 3)))

    for role in sorted(R.role.unique()):
        g = R[R.role == role]
        add(role + "/ALL", g)
        if role == "cross_split":
            for f in lc.mf.FARMS:
                if (g.farm == f).any():
                    add(f"{role}/farm={f}", g[g.farm == f])
            for k in sorted(g.pair_kind.unique()):
                add(f"{role}/{k}", g[g.pair_kind == k])
            for b in ["<8 m", "8-14 m", "14-20 m"]:
                if (g.dist_bin == b).any():
                    add(f"{role}/nominal distance {b}", g[g.dist_bin == b])
        else:
            for f in lc.mf.FARMS:
                if (g.farm == f).any():
                    add(f"{role}/farm={f}", g[g.farm == f])
    S = pd.DataFrame(rows)
    S.to_csv(lc.OUTD / f"images_{a.variant}_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.to_string())


def match_with_inliers(fa, fb):
    pa, da, sa = fa
    pb, db, sb = fb
    bf_ = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))          # kd-tree matcher (brute force: ~4 s per RGB pair)
    m = bf_.knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < RATIO * x[1].distance]
    if len(good) < 8:
        return None, None, good
    src = np.float32([pa[g.queryIdx] for g in good])
    dst = np.float32([pb[g.trainIdx] for g in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    return H, (mask.ravel() > 0 if mask is not None else None), (src, dst)


def cmd_examples(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = pd.read_csv(lc.OUTD / f"images_{a.variant}_results.csv")
    idx, pf, scales = raw_index(), pairing(), rgb_scale_table()
    X = R[R.role == "cross_split"]
    ex = []
    v = X[(X.th_verified == True) & (X.th_overlap_share.fillna(0) > 0.02)]
    if len(v):
        ex.append(("VERIFIED OVERLAP of the used windows", v.sort_values("th_overlap_share", ascending=False).iloc[0]))
    v1 = X[(X.th_verified == True) & (X.th_overlap_share.fillna(0) <= 0.02) & (X.kind_a == "crop") & (X.kind_b == "crop")].sort_values("th_inliers", ascending=False)
    if len(v1):
        ex.append(("matches found, but the used windows (crops) do not overlap", v1.iloc[0]))
    v2 = X[(X.rgb_verified == True) & (X.rgb_overlap_share.fillna(0) <= 0.02)].sort_values("rgb_inliers", ascending=False)
    if len(v2):
        ex.append(("wide RGB views overlap, thermal windows do not", v2.iloc[len(v2) // 3]))
    nn = X[(X.kind_a == "frame") & (X.kind_b == "frame") & (X.nominal_distance_m < 5)].assign(t=lambda d: d.rgb_inliers.fillna(0) + d.th_inliers.fillna(0)).sort_values("t")
    if len(nn):
        ex.append(("no shared scene found (closest whole-frame pair of different splits)", nn.iloc[0]))
    c = R[(R.role == "control_overlap_30_70") & (R.th_verified == True)].sort_values("th_inliers", ascending=False)
    if len(c):
        ex.append(("positive control: neighbouring frames of one flight (nominal overlap 30-70 %)", c.iloc[len(c) // 2]))
    fig, axs = plt.subplots(len(ex), 4, figsize=(16, 3.3 * len(ex)))
    axs = np.atleast_2d(axs)

    def thermal_img(path, shape):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
        return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    for r_, (title, row) in zip(axs, ex):
        (fla, tha), (flb, thb) = row.fa.split("|"), row.fb.split("|")
        farm = fla.split("_")[0]
        ra, rb = pf[(fla, tha)], pf[(flb, thb)]
        rect_a, rect_b = parse_rect(row.rect_a), parse_rect(row.rect_b)
        # thermal view
        Ta, Tb = load_feat("thermal", fla, tha), load_feat("thermal", flb, thb)
        Ht, it, mt = match_with_inliers(Ta, Tb)
        for ax_, img, rect in ((r_[0], thermal_img(idx[(farm, fla)] / tha, Ta[2]), rect_a), (r_[1], thermal_img(idx[(farm, flb)] / thb, Tb[2]), rect_b)):
            ax_.imshow(img, cmap="gray")
            x0, y0, x1, y1 = [v_ * 2 for v_ in rect]
            ax_.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.8))
        if Ht is not None and it is not None and it.sum() >= MIN_INL["thermal"]:
            src, dst = mt
            for (x1, y1), (x2, y2) in list(zip(src[it], dst[it]))[:60]:
                r_[0].plot(x1, y1, "c.", ms=2); r_[1].plot(x2, y2, "c.", ms=2)
            wa = cv2.perspectiveTransform((np.array([[rect_a[0], rect_a[1]], [rect_a[2], rect_a[1]], [rect_a[2], rect_a[3]], [rect_a[0], rect_a[3]]], float) * 2.0)[None], Ht)[0]
            r_[1].add_patch(plt.Polygon(wa, fill=False, ec="yellow", lw=1.5, ls="--"))
        # RGB view
        Fa, Fb = load_feat("rgb", fla, ra), load_feat("rgb", flb, rb)
        Hr, ir, mr = match_with_inliers(Fa, Fb)
        for ax_, fl, rg, F_, rect in ((r_[2], fla, ra, Fa, rect_a), (r_[3], flb, rb, Fb, rect_b)):
            img = cv2.imread(str(idx[(farm, fl)] / rg), cv2.IMREAD_GRAYSCALE)
            ax_.imshow(cv2.resize(img, (F_[2][1], F_[2][0]), interpolation=cv2.INTER_AREA), cmap="gray")
            ax_.add_patch(plt.Polygon(rect_poly_rgb(rect, F_[2], scales[fl]["scale"], F_[2][1] / scales[fl]["rgb_w"]), fill=False, ec="red", lw=1.8))
        if Hr is not None and ir is not None and ir.sum() >= MIN_INL["rgb"]:
            src, dst = mr
            for (x1, y1), (x2, y2) in list(zip(src[ir], dst[ir]))[:60]:
                r_[2].plot(x1, y1, "c.", ms=2); r_[3].plot(x2, y2, "c.", ms=2)
            wa = cv2.perspectiveTransform(rect_poly_rgb(rect_a, Fa[2], scales[fla]["scale"], Fa[2][1] / scales[fla]["rgb_w"])[None].astype(np.float64), Hr)[0]
            r_[3].add_patch(plt.Polygon(wa, fill=False, ec="yellow", lw=1.5, ls="--"))
        for ax_ in r_:
            ax_.set_xticks([]); ax_.set_yticks([])
        nl = chr(10)
        ov_t = row.th_overlap_share if row.th_overlap_share == row.th_overlap_share else 0.0
        ov_r = row.rgb_overlap_share if row.rgb_overlap_share == row.rgb_overlap_share else 0.0
        r_[0].set_title(title + nl + f"A {row.split_a} ({row.kind_a}) {tha[-20:-6]}, thermal; nominal distance {row.nominal_distance_m:.1f} m", fontsize=7.5, loc="left")
        r_[1].set_title(f"B {row.split_b} ({row.kind_b}) {thb[-20:-6]}; thermal inliers {int(row.th_inliers)}, window overlap {ov_t:.2f}", fontsize=7.5)
        r_[2].set_title("paired RGB of A", fontsize=7.5)
        r_[3].set_title(f"paired RGB of B; inliers {int(row.rgb_inliers)}, window overlap {ov_r:.2f}", fontsize=7.5)
    fig.suptitle("Image check: red = used thermal window (frame or crop), cyan = RANSAC inliers, yellow dashed = window of A mapped into B by the homography", fontsize=10)
    fig.tight_layout()
    fig.savefig(lc.OUTD / f"images_{a.variant}_examples.png", dpi=95)
    print("examples:", [t for t, _ in ex])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "match", "report", "examples"])
    ap.add_argument("--variant", default="C_margin1.5")
    ap.add_argument("--max-dist", type=float, default=20.0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    {"pairs": cmd_pairs, "match": cmd_match, "report": cmd_report, "examples": cmd_examples}[a.cmd](a)


if __name__ == "__main__":
    main()
