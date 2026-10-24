#!/usr/bin/env python
"""Consistency / transfer-error evaluation of the RGB <-> thermal registration of DJI H20T shots.

NO manually measured check points exist, so nothing here is absolute accuracy.  Every error is a
*hold-out transfer error*: automatic patch matches of a pair are split at random into a fit half and a
hold-out half; a model is estimated without the hold-out half and the residual on the hold-out half is
reported (thermal pixels and metres on the ground).  See dataset/metadata/registration/README.md.

Frames
  T   thermal image, 640x512 (pixel indices).
  A   RGB image converted to grey and down-scaled by the focal/pixel-pitch ratio so that it has the
      thermal pixel scale (pinhole prior).  A' = A - (OX, OY) is the "prior aligned" frame: the identity
      transform T -> A' is the geometric prior (same optical axis, same height, only the focal ratio).
  M (3x3) maps thermal pixel -> A' pixel.  The published matrices map thermal pixel -> full-resolution
  RGB pixel (and the inverse) and are built by to_full_rgb().

Methods
  M1   per-image homography (RANSAC) from automatic matches: dense grid of thermal patches matched to the
       RGB by NCC of gradient-magnitude maps inside a window centred on the nominal position
       (+ the GPS-predicted along-track shift between the two exposures, see below).
  M2   one fixed similarity per altitude class = robust median of the per-image similarity fits of the
       OTHER flights of the class (leave-flight-out; 4 contiguous folds when the class has a single flight).
  M2m  M2 + a motion term: translation = c + k * g, g = displacement of the aircraft between the GPS
       fixes of the W and T file, expressed in thermal pixels (robust Theil-Sen/IRLS fit).  This is the
       fixed-rig hypothesis corrected for the time offset between the two exposures (extension, not asked
       for in the task, added because the plain M2 translation is not stable).
  M3   geometric prior only (pinhole scale, zero offset), no matches and no GPS: coarse global NMI search of
       the translation (+-48 x +-112 px) followed by a Nelder-Mead NMI refinement of (tx, ty, scale).
Metrics are described in the README.

Usage (from the project root):
  python -X utf8 code/scripts/register_rgb_thermal.py                       # H20T + flagged BIM2 secondary run
  python -X utf8 code/scripts/register_rgb_thermal.py --cameras H20T --n-per-flight 40 --seed 20260420
"""
import argparse
import glob
import math
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import spearmanr, theilslopes

TW, TH = 640, 512
INL_LOOSE_PX = 6.0    # looser hold-out inlier threshold (sensitivity)
INL_PX = 3.0          # hold-out inlier threshold (thermal px)
CORE_PX = 10.0        # capture radius for the "core" hold-out error (thermal px)
SUCCESS_FRAC = 0.20   # min share of hold-out matches within INL_PX
SUCCESS_N = 8         # min number of hold-out matches within INL_PX
MIN_PUTATIVE = 16     # below this a pair has no usable reference match set
SIM_MIN_INL = 12
SIM_MIN_RATIO = 0.30
NSPLIT = 5
PATCH_H, SEARCH_R, PATCH_STEP, SCORE_MIN = 40, 64, 36, 0.40
PAD = 160

CAMERAS = {
    "H20T": dict(label="DJI Zenmuse H20T", farms=["TN19", "TNDL", "TT"], rgb_w=4056, rgb_h=3040,
                 rgb_pitch_um=1.55, rgb_f_mm=4.5, th_pitch_um=12.0, th_f_mm=13.5, secondary=False,
                 flights=["TN19_20260411_31M_T2", "TN19_20260411_31M_T3", "TN19_20260411_35M_T2",
                          "TN19_20260411_35M_T4", "TN19_20260411_35M_T5", "TN19_20260411_42M_T1",
                          "TN19_20260411_42M_T4", "TNDL_20260411_30M_T1", "TNDL_20260411_35M_T2",
                          "TT_20260412_39M_T1", "TT_20260412_39M_T2", "TT_20260412_39M_T6"]),
    # Mavic 2 Enterprise Advanced: NOMINAL values (thermal f=9 mm, RGB 0.8 um pixel, f=4.5 mm). Not verified.
    "M2EA": dict(label="DJI Mavic 2 Enterprise Advanced (nominal specs, flagged secondary)", farms=["BIM2"],
                 rgb_w=8000, rgb_h=6000, rgb_pitch_um=0.8, rgb_f_mm=4.5, th_pitch_um=12.0, th_f_mm=9.0,
                 secondary=True,
                 flights=["BIM2_20260410_20M_T1", "BIM2_20260411_30M_T1", "BIM2_20260411_30M_T2"]),
}
for _c in CAMERAS.values():
    _c["S0"] = (_c["th_pitch_um"] / _c["th_f_mm"]) / (_c["rgb_pitch_um"] / _c["rgb_f_mm"])  # RGB px per thermal px
    _c["dw"] = int(round(_c["rgb_w"] / _c["S0"]))
    _c["dh"] = int(round(_c["rgb_h"] / _c["S0"]))
    _c["ox"] = (_c["dw"] - TW) // 2
    _c["oy"] = (_c["dh"] - TH) // 2
    _c["sx"] = _c["rgb_w"] / _c["dw"]
    _c["sy"] = _c["rgb_h"] / _c["dh"]

XC = np.array([(TW - 1) / 2.0, (TH - 1) / 2.0])


# ----------------------------------------------------------------------------- geometry helpers
def sim_matrix(s, th, tcx, tcy):
    """Similarity thermal -> A' with scale s, rotation th (rad) about the thermal centre, centre displaced by tc."""
    c, sn = s * math.cos(th), s * math.sin(th)
    R = np.array([[c, -sn], [sn, c]])
    t = XC + np.array([tcx, tcy]) - R @ XC
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1.0]])


def sim_params(M):
    s = math.hypot(M[0, 0], M[1, 0])
    th = math.atan2(M[1, 0], M[0, 0])
    tc = M[:2, :2] @ XC + M[:2, 2] - XC
    return s, th, tc[0], tc[1]


def apply_H(M, P):
    ph = np.c_[P, np.ones(len(P))] @ M.T
    return ph[:, :2] / ph[:, 2:3]


def to_full_rgb(M_A, cam):
    """thermal px -> full-resolution RGB px (pixel-index convention, centre of first pixel = 0)."""
    Tr = np.array([[1, 0, cam["ox"]], [0, 1, cam["oy"]], [0, 0, 1.0]])
    S = np.array([[cam["sx"], 0, 0.5 * cam["sx"] - 0.5], [0, cam["sy"], 0.5 * cam["sy"] - 0.5], [0, 0, 1.0]])
    return S @ Tr @ M_A


# ----------------------------------------------------------------------------- image helpers
def gm(g, sigma):
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (np.percentile(m, 98) + 1e-6)


def norm01(g):
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def load_pair(rgb_path, th_path, cam):
    a = cv2.imread(rgb_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    a = cv2.resize(a, (cam["dw"], cam["dh"]), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = cv2.imread(th_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    assert t.shape == (TH, TW), t.shape
    return a, t


def warp_rgb(A, M_A, cam):
    Tr = np.array([[1, 0, cam["ox"]], [0, 1, cam["oy"]], [0, 0, 1.0]])
    return cv2.warpPerspective(A, Tr @ M_A, (TW, TH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def nmi_q(x, y, bins):
    """Normalised mutual information (Hx+Hy)/Hxy of two integer images in [0,bins)."""
    h = np.bincount((x.astype(np.int32) * bins + y.astype(np.int32)).ravel(), minlength=bins * bins).astype(np.float64)
    h = h.reshape(bins, bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    hxy = -(p[p > 0] * np.log(p[p > 0])).sum()
    return (hx + hy) / hxy


def quant(g01, bins):
    return np.minimum((g01 * bins).astype(np.int32), bins - 1)


def metrics_for(A, T, M_A, cam, cache):
    """NMI (grey, 32 bins) and edge NCC (gradient magnitude) between T and warped RGB over the inner region."""
    W_ = warp_rgb(A, M_A, cam)
    b = 16
    ws = norm01(cv2.GaussianBlur(W_, (0, 0), 1.0))[b:-b, b:-b]
    nm = nmi_q(quant(cache["t_q"], 32), quant(ws, 32), 32)
    ge = gm(W_, 1.5)[b:-b, b:-b]
    gt = cache["t_edge"]
    ncc = float(np.corrcoef(ge.ravel(), gt.ravel())[0, 1])
    return nm, ncc


def metric_cache(T):
    b = 16
    return dict(t_q=norm01(cv2.GaussianBlur(T, (0, 0), 1.0))[b:-b, b:-b], t_edge=gm(T, 1.5)[b:-b, b:-b])


# ----------------------------------------------------------------------------- M1: patch matches
def patch_matches(A, T, g, cam):
    Ga = cv2.copyMakeBorder(gm(A, 2.0), PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    Gt = gm(T, 2.0)
    H, R = PATCH_H, SEARCH_R
    P, Q, S = [], [], []
    for cy in range(H + 4, TH - H - 3, PATCH_STEP):
        for cx in range(H + 4, TW - H - 3, PATCH_STEP):
            t = Gt[cy - H:cy + H + 1, cx - H:cx + H + 1]
            if t.std() < 0.12:
                continue
            x0 = int(round(cam["ox"] + cx + g[0])) - H - R + PAD
            y0 = int(round(cam["oy"] + cy + g[1])) - H - R + PAD
            ref = Ga[y0:y0 + 2 * (H + R) + 1, x0:x0 + 2 * (H + R) + 1]
            res = cv2.matchTemplate(ref, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < SCORE_MIN:
                continue
            lx, ly = loc
            dx = dy = 0.0
            if 0 < lx < res.shape[1] - 1:
                a, b_, c = res[ly, lx - 1], res[ly, lx], res[ly, lx + 1]
                den = a - 2 * b_ + c
                dx = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            if 0 < ly < res.shape[0] - 1:
                a, b_, c = res[ly - 1, lx], res[ly, lx], res[ly + 1, lx]
                den = a - 2 * b_ + c
                dy = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            qx = x0 + lx + dx + H - PAD - cam["ox"]
            qy = y0 + ly + dy + H - PAD - cam["oy"]
            P.append((cx, cy))
            Q.append((qx, qy))
            S.append(mx)
    return np.float32(P).reshape(-1, 2), np.float32(Q).reshape(-1, 2), np.float32(S)


# ----------------------------------------------------------------------------- M3: prior + NMI
def run_m3(A, T, cam):
    Ab = norm01(cv2.GaussianBlur(A, (0, 0), 2.0))
    Tb = norm01(cv2.GaussianBlur(T, (0, 0), 2.0))
    ox, oy = cam["ox"], cam["oy"]
    tq = quant(Tb[::4, ::4], 16)
    aq = quant(Ab, 16)
    DX, DY = 48, 112
    best = []
    for dy in range(-DY, DY + 1, 4):
        for dx in range(-DX, DX + 1, 4):
            y0, x0 = oy + dy, ox + dx
            if y0 < 0 or x0 < 0 or y0 + TH > Ab.shape[0] or x0 + TW > Ab.shape[1]:
                continue
            best.append((nmi_q(tq, aq[y0:y0 + TH:4, x0:x0 + TW:4], 16), dx, dy))
    best.sort(reverse=True)
    v0, dx0, dy0 = best[0]
    far = [b for b in best if abs(b[1] - dx0) > 16 or abs(b[2] - dy0) > 16]
    distinct = v0 - far[0][0] if far else float("nan")
    tqf = quant(norm01(cv2.GaussianBlur(T, (0, 0), 1.0)), 24)
    Af = norm01(cv2.GaussianBlur(A, (0, 0), 1.0))
    sl = (slice(24, TH - 24), slice(24, TW - 24))

    def obj(p):
        dx, dy, ds = p
        pen = 0.0
        if abs(dx - dx0) > 10 or abs(dy - dy0) > 10 or abs(ds) > 0.04:
            pen = 0.05
        M = sim_matrix(1.0 + ds, 0.0, dx, dy)
        Wf = warp_rgb(Af, M, cam)
        return -nmi_q(tqf[sl], quant(Wf, 24)[sl], 24) + pen

    r = minimize(obj, [dx0, dy0, 0.0], method="Nelder-Mead",
                 options=dict(initial_simplex=np.array([[dx0, dy0, 0], [dx0 + 3, dy0, 0], [dx0, dy0 + 3, 0], [dx0, dy0, 0.015]]),
                              xatol=0.1, fatol=1e-5, maxfev=140))
    dx, dy, ds = r.x
    return dict(m3_s=1.0 + ds, m3_tcx=float(dx), m3_tcy=float(dy), m3_coarse_nmi=float(v0), m3_nmi=float(-r.fun),
                m3_distinct=float(distinct), m3_dx0=int(dx0), m3_dy0=int(dy0))


# ----------------------------------------------------------------------------- stage A worker
def stage_a(task):
    cv2.setNumThreads(1)
    cam = CAMERAS[task["camera"]]
    A, T = load_pair(task["rgb_path"], task["th_path"], cam)
    P, Q, S = patch_matches(A, T, (task["gx"], task["gy"]), cam)
    out = dict(idx=task["idx"], P=P, Q=Q, S=S)
    out.update(sim_n_in=0, sim_ratio=0.0, sim_s=np.nan, sim_th=np.nan, sim_tcx=np.nan, sim_tcy=np.nan)
    if len(P) >= 4:
        Ms, inl = cv2.estimateAffinePartial2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=INL_PX,
                                              maxIters=3000, confidence=0.995)
        if Ms is not None:
            M3x3 = np.vstack([Ms, [0, 0, 1]])
            s, th, tcx, tcy = sim_params(M3x3)
            out.update(sim_n_in=int(inl.sum()), sim_ratio=float(inl.sum()) / len(P), sim_s=s,
                       sim_th=th, sim_tcx=tcx, sim_tcy=tcy)
        H, inl = cv2.findHomography(P, Q, cv2.RANSAC, INL_PX, maxIters=3000, confidence=0.995)
        out["H_all"] = H
        out["H_inl"] = int(inl.sum()) if inl is not None else 0
    else:
        out["H_all"] = None
        out["H_inl"] = 0
    out.update(run_m3(A, T, cam))
    return out


# ----------------------------------------------------------------------------- hold-out evaluation
def ho_stats(err):
    n = len(err)
    inl = int((err <= INL_PX).sum())
    core = err[err < CORE_PX]
    inl6 = int((err <= INL_LOOSE_PX).sum())
    return inl / n, inl, (float(np.median(core)) if len(core) else np.nan), float(np.median(err)), inl6 / n, inl6


def evaluate_pair(P, Q, fixed, rng):
    """fixed: dict name -> 3x3 matrix.  Returns dict of per-method hold-out statistics (median over splits)."""
    n = len(P)
    acc = {m: [] for m in ["M1"] + list(fixed)}
    fit_ratio = []
    for _ in range(NSPLIT):
        idx = rng.permutation(n)
        fit, hold = idx[: n // 2], idx[n // 2:]
        H = inl = None
        if len(fit) >= 8:
            H, inl = cv2.findHomography(P[fit], Q[fit], cv2.RANSAC, INL_PX, maxIters=3000, confidence=0.995)
        if H is not None and inl.sum() >= SUCCESS_N and inl.sum() / len(fit) >= SUCCESS_FRAC:
            err = np.linalg.norm(apply_H(H, P[hold]) - Q[hold], axis=1)
            acc["M1"].append(ho_stats(err))
            fit_ratio.append(inl.sum() / len(fit))
        else:
            acc["M1"].append((0.0, 0, np.nan, np.nan, 0.0, 0))
            fit_ratio.append(0.0 if H is None else inl.sum() / len(fit))
        for m, M in fixed.items():
            err = np.linalg.norm(apply_H(M, P[hold]) - Q[hold], axis=1)
            acc[m].append(ho_stats(err))
    out = {}
    for m, lst in acc.items():
        a = np.array(lst, dtype=float)
        frac, ninl, core, med = np.median(a[:, 0]), np.median(a[:, 1]), np.nanmedian(a[:, 2]) if np.isfinite(a[:, 2]).any() else np.nan, \
            np.nanmedian(a[:, 3]) if np.isfinite(a[:, 3]).any() else np.nan
        f6, n6 = np.median(a[:, 4]), np.median(a[:, 5])
        out[m] = dict(frac=frac, n_inl=ninl, core=core, med=med, frac6=f6,
                      success=bool(frac >= SUCCESS_FRAC and ninl >= SUCCESS_N),
                      success6=bool(f6 >= SUCCESS_FRAC and n6 >= SUCCESS_N))
    out["M1"]["fit_ratio"] = float(np.median(fit_ratio))
    return out


# ----------------------------------------------------------------------------- fixed transform estimation
def fit_fixed(df):
    """df columns: sim_s, sim_th, sim_tcx, sim_tcy, gx, gy.  Returns dict of parameters (A' frame)."""
    s, th = float(np.median(df.sim_s)), float(np.median(df.sim_th))
    cx0, cy0 = float(np.median(df.sim_tcx)), float(np.median(df.sim_tcy))
    out = dict(s=s, th=th, tcx=cx0, tcy=cy0, n=len(df), k=0.0, cx=cx0, cy=cy0)
    if len(df) >= 5 and np.ptp(df.gy.values) > 5:
        k, c, _, _ = theilslopes(df.sim_tcy.values, df.gy.values)
        # IRLS (Huber) refinement on both axes with a shared slope
        y = np.r_[df.sim_tcx.values, df.sim_tcy.values]
        X = np.zeros((2 * len(df), 3))
        X[: len(df), 0] = 1
        X[len(df):, 1] = 1
        X[: len(df), 2] = df.gx.values
        X[len(df):, 2] = df.gy.values
        beta = np.array([np.median(df.sim_tcx.values - k * df.gx.values), c, k])
        for _ in range(30):
            r = y - X @ beta
            sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
            w = np.minimum(1.0, 1.345 * sc / (np.abs(r) + 1e-9))
            beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        out.update(cx=float(beta[0]), cy=float(beta[1]), k=float(beta[2]),
                   resid_mad=float(sc))
    return out


def matrices(par, g, motion):
    if motion:
        return sim_matrix(par["s"], par["th"], par["cx"] + par["k"] * g[0], par["cy"] + par["k"] * g[1])
    return sim_matrix(par["s"], par["th"], par["tcx"], par["tcy"])


# ----------------------------------------------------------------------------- stage C worker (image metrics)
def stage_c(task):
    cv2.setNumThreads(1)
    cam = CAMERAS[task["camera"]]
    A, T = load_pair(task["rgb_path"], task["th_path"], cam)
    cache = metric_cache(T)
    out = {"idx": task["idx"]}
    for name, M in task["mats"].items():
        if M is None:
            out[f"nmi_{name}"], out[f"ncc_{name}"] = np.nan, np.nan
        else:
            out[f"nmi_{name}"], out[f"ncc_{name}"] = metrics_for(A, T, M, cam, cache)
    return out


# ----------------------------------------------------------------------------- sampling
def build_samples(root, cam_key, n_per_flight, seed):
    cam = CAMERAS[cam_key]
    ds = os.path.join(root, "dataset")
    inv = pd.read_csv(os.path.join(ds, "metadata", "inventory", "image_inventory.csv")).set_index(["flight_id", "file"])
    pr = pd.read_csv(os.path.join(ds, "metadata", "pairing", "rgb_thermal_pairing.csv"))
    spd = pd.read_csv(os.path.join(ds, "metadata", "inventory", "thermal_speed.csv")).set_index(["flight_id", "file"])
    rng = np.random.default_rng(seed)
    rows = []
    for fl in cam["flights"]:
        d = glob.glob(os.path.join(ds, "farm_*", "raw_images", "*", fl))
        if len(d) != 1:
            print(f"  WARNING flight {fl}: dir not found", file=sys.stderr)
            continue
        d = d[0]
        p = pr[(pr.flight_id == fl) & (pr.pairing_status == "exact_pair")]
        cand = []
        for r in p.itertuples():
            w, t = inv.loc[(fl, r.rgb_file)], inv.loc[(fl, r.thermal_file)]
            tilt = max(math.hypot(w.GimbalPitchDegree + 90, w.GimbalRollDegree),
                       math.hypot(t.GimbalPitchDegree + 90, t.GimbalRollDegree))
            if not np.isfinite(tilt) or tilt >= 2.0:
                continue
            cand.append((r, w, t, tilt))
        pick = rng.choice(len(cand), size=min(n_per_flight, len(cand)), replace=False)
        for j in sorted(pick):
            r, w, t, tilt = cand[j]
            lat0 = math.radians(w.GpsLatitude)
            dE = (t.GpsLongitude - w.GpsLongitude) * 111320.0 * math.cos(lat0)
            dN = (t.GpsLatitude - w.GpsLatitude) * 110574.0
            psi = math.radians(w.GimbalYawDegree)
            fwd = np.array([math.sin(psi), math.cos(psi)])
            rgt = np.array([math.cos(psi), -math.sin(psi)])
            H = 0.5 * (w.RelativeAltitude + t.RelativeAltitude)
            gsd_t = cam["th_pitch_um"] * 1e-6 / (cam["th_f_mm"] * 1e-3) * H
            dm = np.array([dE, dN])
            sp = spd.loc[(fl, r.thermal_file)] if (fl, r.thermal_file) in spd.index else None
            speed = np.nan
            if sp is not None:
                speed = sp.speed_smoothed_m_s if np.isfinite(sp.speed_smoothed_m_s) else sp.speed_m_s
            m = re.search(r"_(\d+)M_", fl)
            rows.append(dict(
                camera=cam_key, farm=r.farm, flight_id=fl, pair_id=r.pair_id, rgb_file=r.rgb_file,
                thermal_file=r.thermal_file, rgb_path=os.path.join(d, r.rgb_file), th_path=os.path.join(d, r.thermal_file),
                alt_class=int(m.group(1)), rel_alt_m=H, gsd_thermal_m=gsd_t, delta_t_s=r.delta_t_s,
                delta_pos_m=float(np.hypot(dE, dN)), gx=float(dm @ rgt / gsd_t), gy=float(-(dm @ fwd) / gsd_t),
                speed_m_s=speed, tilt_deg=tilt, pitch_dev_deg=abs(w.GimbalPitchDegree + 90),
                roll_abs_deg=abs(w.GimbalRollDegree), gimbal_yaw_deg=w.GimbalYawDegree))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def small_components_removed(e, min_px=150):
    n, lab, st, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), connectivity=8)
    keep = np.isin(lab, np.where(st[:, cv2.CC_STAT_AREA] >= min_px)[0]) & (lab > 0)
    return keep


def overlay_figure(row, mats, cam, path, titles):
    A, T = load_pair(row["rgb_path"], row["th_path"], cam)
    e = cv2.Canny(cv2.GaussianBlur(T.astype(np.uint8), (0, 0), 1.5), 25, 70)
    e = small_components_removed(e)
    panels = []
    for name, M in mats:
        W_ = warp_rgb(A, M, cam)
        W_ = cv2.resize(cv2.cvtColor(W_.astype(np.uint8), cv2.COLOR_GRAY2BGR), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        ee = cv2.resize(e.astype(np.uint8), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST) > 0
        W_[ee] = (0, 0, 255)
        cv2.putText(W_, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(W_)
    img = np.hstack(panels)
    cv2.putText(img, titles, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


# ----------------------------------------------------------------------------- summary helpers
def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def summarise(df, cam_key, methods):
    rows = []

    def block(section, group, sub):
        for m in methods:
            ref = sub[sub.ref_ok]
            r = dict(section=section, camera=cam_key, group=group, method=m, n_pairs=len(sub), n_ref=len(ref))
            if len(ref):
                suc = ref[ref[f"success_{m}"]]
                r.update(success_rate=float(ref[f"success_{m}"].mean()), success_rate_6px=float(ref[f"success6_{m}"].mean()), n_success=len(suc),
                         ho_core_err_px_median=q(suc[f"ho_core_px_{m}"], 50), ho_core_err_px_p25=q(suc[f"ho_core_px_{m}"], 25),
                         ho_core_err_px_p75=q(suc[f"ho_core_px_{m}"], 75),
                         ho_core_err_m_median=q(suc[f"ho_core_px_{m}"] * suc.gsd_thermal_m, 50),
                         ho_core_err_m_p75=q(suc[f"ho_core_px_{m}"] * suc.gsd_thermal_m, 75),
                         ho_all_med_err_px_median=q(ref[f"ho_med_px_{m}"], 50),
                         ho_inlier_frac_median=q(ref[f"ho_inl_frac_{m}"], 50))
            both = sub[np.isfinite(sub[f"nmi_{m}"])]
            if len(both):
                r.update(nmi_before_median=q(both.nmi_prior, 50), nmi_after_median=q(both[f"nmi_{m}"], 50),
                         nmi_gain_median=q(both[f"nmi_{m}"] - both.nmi_prior, 50),
                         ncc_before_median=q(both.ncc_prior, 50), ncc_after_median=q(both[f"ncc_{m}"], 50),
                         ncc_gain_median=q(both[f"ncc_{m}"] - both.ncc_prior, 50))
            rows.append(r)

    block("by_altitude_method", "all", df)
    for a, s in df.groupby("alt_class"):
        block("by_altitude_method", f"{a}m", s)
    for (f, a), s in df.groupby(["farm", "alt_class"]):
        block("by_farm_altitude_method", f"{f}_{a}m", s)
    for f, s in df.groupby("flight_id"):
        block("by_flight_method", f, s)
    return rows


def stability_rows(df, cam_key):
    rows = []
    ok = df[df.sim_ok]
    m1 = [("scale", "sim_s"), ("rotation_deg", "sim_th_deg"), ("centre_shift_x_px", "sim_tcx"),
          ("centre_shift_y_px", "sim_tcy"), ("centre_shift_y_minus_k_gy_px", "sim_tcy_res")]
    m3 = [("scale", "m3_s"), ("centre_shift_x_px", "m3_tcx"), ("centre_shift_y_px", "m3_tcy")]
    for meth, d, specs in [("M1_sim", ok, m1), ("M3", df, m3)]:
        groups = [("all", d)] + [(f"{a}m", s) for a, s in d.groupby("alt_class")] + [(f, s) for f, s in d.groupby("flight_id")]
        for gname, sub in groups:
            for par, col in specs:
                v = sub[col].values.astype(float)
                rows.append(dict(section="stability", camera=cam_key, group=gname, method=meth, param=par,
                                 n_pairs=int(np.isfinite(v).sum()), value_median=q(v, 50), value_p25=q(v, 25),
                                 value_p75=q(v, 75), value_iqr=q(v, 75) - q(v, 25)))
    return rows


def dependence_rows(df, cam_key, methods):
    rows = []
    ref = df[df.ref_ok]
    xs = {"altitude_m": "rel_alt_m", "speed_m_s": "speed_m_s", "tilt_deg": "tilt_deg", "pitch_dev_deg": "pitch_dev_deg",
          "abs_gps_shift_y_px": "abs_gy"}
    for m in methods:
        for xn, xc in xs.items():
            x = ref[xc].values.astype(float)
            for yn, yc in [("ho_all_med_err_px", f"ho_med_px_{m}"), ("ho_inlier_frac", f"ho_inl_frac_{m}"),
                           ("success", f"success_{m}")]:
                y = ref[yc].values.astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 10 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
                    rho, p = spearmanr(x[ok], y[ok])
                    rows.append(dict(section="dependence_spearman", camera=cam_key, group=f"{xn}~{yn}", method=m,
                                     n_pairs=int(ok.sum()), rho=float(rho), p_value=float(p)))
        # binned success / error by speed and tilt (terciles)
        for xn, xc in [("speed_m_s", "speed_m_s"), ("tilt_deg", "tilt_deg")]:
            x = ref[xc]
            if x.notna().sum() < 30:
                continue
            try:
                bins = pd.qcut(x, 3, duplicates="drop")
            except ValueError:
                continue
            for b, s in ref.groupby(bins, observed=True):
                suc = s[s[f"success_{m}"]]
                rows.append(dict(section="dependence_bin", camera=cam_key, group=f"{xn} {b}", method=m, n_pairs=len(s),
                                 n_ref=len(s), success_rate=float(s[f"success_{m}"].mean()),
                                 ho_core_err_px_median=q(suc[f"ho_core_px_{m}"], 50)))
    return rows


# ----------------------------------------------------------------------------- per-camera run
def run_camera(root, cam_key, args, pool, outdir, figdir):
    cam = CAMERAS[cam_key]
    t0 = time.time()
    df = build_samples(root, cam_key, args.n_per_flight if not cam["secondary"] else args.n_per_flight_secondary, args.seed)
    if df.empty:
        return None
    df["idx"] = np.arange(len(df))
    df["abs_gy"] = df.gy.abs()
    print(f"[{cam_key}] {len(df)} pairs sampled from {df.flight_id.nunique()} flights", flush=True)

    # ---- stage A
    tasks = [dict(idx=int(r.idx), camera=cam_key, rgb_path=r.rgb_path, th_path=r.th_path, gx=r.gx, gy=r.gy) for r in df.itertuples()]
    resA = sorted(pool.imap_unordered(stage_a, tasks, chunksize=2), key=lambda d: d["idx"])
    print(f"[{cam_key}] stage A (matching + M3) done in {time.time() - t0:.0f}s", flush=True)
    for k in ["sim_n_in", "sim_ratio", "sim_s", "sim_th", "sim_tcx", "sim_tcy", "m3_s", "m3_tcx", "m3_tcy", "m3_coarse_nmi",
              "m3_nmi", "m3_distinct", "m3_dx0", "m3_dy0", "H_inl"]:
        df[k] = [r[k] for r in resA]
    df["n_putative"] = [len(r["P"]) for r in resA]
    df["match_score_median"] = [float(np.median(r["S"])) if len(r["S"]) else np.nan for r in resA]
    df["sim_th_deg"] = np.degrees(df.sim_th)
    df["sim_ok"] = (df.sim_n_in >= SIM_MIN_INL) & (df.sim_ratio >= SIM_MIN_RATIO)
    df["ref_ok"] = df.n_putative >= MIN_PUTATIVE

    # ---- fixed transforms (grouped CV)
    df["group"] = df.flight_id
    for a, s in df.groupby("alt_class"):
        if s.flight_id.nunique() < 2:  # single flight: 4 contiguous folds
            order = s.index.values
            df.loc[order, "group"] = [f"{a}m_fold{int(i * 4 / len(order))}" for i in range(len(order))]
    pars_cv = {}
    for i in df.index:
        sub = df[(df.alt_class == df.at[i, "alt_class"]) & (df.group != df.at[i, "group"]) & df.sim_ok]
        key = (df.at[i, "alt_class"], df.at[i, "group"])
        if key not in pars_cv:
            pars_cv[key] = fit_fixed(sub) if len(sub) >= 5 else None
    par_all = {a: fit_fixed(s[s.sim_ok]) for a, s in df.groupby("alt_class")}
    par_all["all"] = fit_fixed(df[df.sim_ok])
    df["sim_tcy_res"] = np.nan
    pg = par_all["all"]
    df["sim_tcy_res"] = df.sim_tcy - pg["k"] * df.gy

    # ---- stage B: hold-out evaluation
    methods = ["M1", "M2", "M2m", "M3"]
    recs, mats_C = [], []
    for r, res in zip(df.itertuples(), resA):
        P, Q = res["P"], res["Q"]
        par = pars_cv[(r.alt_class, r.group)]
        g = (r.gx, r.gy)
        fixed = {"M3": sim_matrix(r.m3_s, 0.0, r.m3_tcx, r.m3_tcy)}
        if par is not None:
            fixed["M2"] = matrices(par, g, False)
            fixed["M2m"] = matrices(par, g, True)
        rec = dict(idx=r.idx)
        if r.ref_ok:
            ev = evaluate_pair(P, Q, fixed, np.random.default_rng(10_000 + r.idx))
        else:
            ev = {}
        for m in methods:
            e = ev.get(m)
            rec[f"success_{m}"] = bool(e["success"]) if e else False
            rec[f"success6_{m}"] = bool(e["success6"]) if e else False
            rec[f"ho_core_px_{m}"] = e["core"] if e else np.nan
            rec[f"ho_med_px_{m}"] = e["med"] if e else np.nan
            rec[f"ho_inl_frac_{m}"] = e["frac"] if e else np.nan
            rec[f"ho_n_inl_{m}"] = e["n_inl"] if e else np.nan
        rec["m1_fit_inlier_ratio"] = ev["M1"]["fit_ratio"] if ev else np.nan
        recs.append(rec)
        # matrices for the image metrics (full-fit transforms)
        H = res["H_all"]
        m1 = H if (ev and ev["M1"]["success"] and H is not None) else None
        mats_C.append({"prior": np.eye(3), "M1": m1 if m1 is not None else np.eye(3),
                       "M2": fixed.get("M2"), "M2m": fixed.get("M2m"), "M3": fixed["M3"]})
    df = df.merge(pd.DataFrame(recs), on="idx")

    # ---- stage C: NMI / edge NCC
    tasks = [dict(idx=int(r.idx), camera=cam_key, rgb_path=r.rgb_path, th_path=r.th_path, mats=mats_C[int(r.idx)]) for r in df.itertuples()]
    resC = sorted(pool.imap_unordered(stage_c, tasks, chunksize=2), key=lambda d: d["idx"])
    for k in resC[0]:
        if k != "idx":
            df[k] = [r[k] for r in resC]
    # M1 failures fall back to the prior in the NMI/NCC columns (gain 0), by construction of mats_C
    print(f"[{cam_key}] stage C done in {time.time() - t0:.0f}s", flush=True)

    # M1 vs M3 agreement of the centre shift
    df["m1_m3_centre_diff_px"] = np.hypot(df.sim_tcx - df.m3_tcx, df.sim_tcy - df.m3_tcy)

    # ---- fixed_transforms rows
    ft = []
    for scope, par in list(par_all.items()):
        n_flights = df[df.sim_ok & ((df.alt_class == scope) if scope != "all" else True)].flight_id.nunique()
        for name, motion in [("M2_static", False), ("M2m_motion_compensated", True)]:
            M_A = matrices(par, (0.0, 0.0), motion)
            Mf = to_full_rgb(M_A, cam)
            Mi = np.linalg.inv(Mf)
            row = dict(camera=cam_key, scope=("all_altitudes" if scope == "all" else f"{scope}m"), method=name,
                       n_pairs_used=par["n"], n_flights_used=n_flights, scale_rgbds_per_thermal=par["s"],
                       rotation_deg=math.degrees(par["th"]),
                       centre_shift_x_px=(par["cx"] if motion else par["tcx"]), centre_shift_y_px=(par["cy"] if motion else par["tcy"]),
                       k_motion=(par["k"] if motion else 0.0), prior_scale_rgb_per_thermal=cam["S0"],
                       nominal=cam["secondary"])
            for i in range(3):
                for j in range(3):
                    row[f"T2R_{i}{j}"] = Mf[i, j]
            for i in range(3):
                for j in range(3):
                    row[f"R2T_{i}{j}"] = Mi[i, j]
            ft.append(row)
    ft = pd.DataFrame(ft)

    # ---- tables
    summ = summarise(df, cam_key, methods) + stability_rows(df, cam_key) + dependence_rows(df, cam_key, methods)

    # ---- figures
    figs = []
    if not cam["secondary"]:
        want = [("TN19", 31), ("TN19", 35), ("TN19", 42), ("TNDL", 30), ("TT", 39)]
        rec_m = "M2m"
        for fm, a in want:
            s = df[(df.farm == fm) & (df.alt_class == a) & df.ref_ok & df.success_M1 & df.success_M2m & (df.gy.abs() > 15)]
            if s.empty:
                s = df[(df.farm == fm) & (df.alt_class == a)]
            if s.empty:
                continue
            r = s.iloc[(s[f"ho_core_px_{rec_m}"].fillna(99) - s[f"ho_core_px_{rec_m}"].median()).abs().argsort().iloc[0] if s[f"ho_core_px_{rec_m}"].notna().any() else 0]
            figs.append((f"overlay_{fm}_{a}m.png", r, f"{fm} {a} m {r.pair_id}"))
        f = df[(~df.success_M1) & df.ref_ok]
        if f.empty:
            f = df[~df.success_M1]
        if f.empty:
            f = df.sort_values("ho_inl_frac_M1").head(1)
        r = f.iloc[0]
        figs.append((f"overlay_failure_{r.farm}_{r.alt_class}m.png", r, f"M1 FAILURE example {r.pair_id}"))
        for fn, r, title in figs:
            i = int(r.idx)
            mats = [("prior (no offset)", np.eye(3)), ("M1 per-image" if r.success_M1 else "M1 (failed, prior shown)", mats_C[i]["M1"]),
                    ("M2m fixed rig + motion", mats_C[i]["M2m"] if mats_C[i]["M2m"] is not None else np.eye(3))]
            overlay_figure(r.to_dict(), mats, cam, os.path.join(figdir, fn),
                           title + f" | dt={r.delta_t_s:+.0f}s gy={r.gy:+.0f}px")
    df = df.drop(columns=["rgb_path", "th_path", "idx", "group"])
    return df, ft, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")), help="project root (contains dataset/)")
    ap.add_argument("--cameras", default="H20T,M2EA")
    ap.add_argument("--n-per-flight", type=int, default=40)
    ap.add_argument("--n-per-flight-secondary", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    outdir = os.path.join(a.root, "dataset", "metadata", "registration")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    cv2.setNumThreads(1)
    dfs, fts, summ = [], [], []
    with Pool(a.workers) as pool:
        for ck in a.cameras.split(","):
            res = run_camera(a.root, ck, a, pool, outdir, figdir)
            if res is None:
                continue
            d, f, s = res
            dfs.append(d)
            fts.append(f)
            summ += s
    pp = pd.concat(dfs, ignore_index=True)
    pp["secondary_flag"] = pp.camera.map(lambda c: CAMERAS[c]["secondary"])
    pp = pp.round(4)
    pp.to_csv(os.path.join(outdir, "registration_per_pair.csv"), index=False)
    pd.concat(fts, ignore_index=True).to_csv(os.path.join(outdir, "fixed_transforms.csv"), index=False)
    pd.DataFrame(summ).round(4).to_csv(os.path.join(outdir, "registration_summary.csv"), index=False)
    print("written to", outdir)


if __name__ == "__main__":
    main()
