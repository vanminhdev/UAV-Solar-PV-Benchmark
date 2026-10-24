"""Control-point search + robust transform fit, one (flight, base) pair. Used by ortho_control_points.py `match`."""
import numpy as np, cv2
from orthoreg_match import *

def fit_sim(src, dst, ransac_thr=None):
    """Similarity dst = s R (src-c) + c + t about centroid c. Returns dict + inlier mask."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    c = src.mean(0)
    if ransac_thr is not None and len(src) >= 4:
        M, inl = cv2.estimateAffinePartial2D((src - c).astype(np.float32), (dst - c).astype(np.float32),
                                             method=cv2.RANSAC, ransacReprojThreshold=ransac_thr,
                                             maxIters=5000, confidence=0.999)
        if M is None: return None, np.zeros(len(src), bool)
        inl = inl.ravel().astype(bool)
    else:
        inl = np.ones(len(src), bool)
    return ls_fit(src[inl], dst[inl], 'similarity', c), inl

def ls_fit(src, dst, model, c=None):
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    c = src.mean(0) if c is None else np.asarray(c, float)
    s = src - c; d = dst - c
    if model == 'similarity':
        # unknowns a,b,tx,ty: x' = a x - b y + tx ; y' = b x + a y + ty
        A = np.zeros((2 * len(s), 4)); y = np.zeros(2 * len(s))
        A[0::2] = np.c_[s[:, 0], -s[:, 1], np.ones(len(s)), np.zeros(len(s))]
        A[1::2] = np.c_[s[:, 1], s[:, 0], np.zeros(len(s)), np.ones(len(s))]
        y[0::2] = d[:, 0]; y[1::2] = d[:, 1]
        a, b, tx, ty = np.linalg.lstsq(A, y, rcond=None)[0]
        m = np.array([[a, -b], [b, a]])
    else:
        A = np.c_[s, np.ones(len(s))]
        px = np.linalg.lstsq(A, d[:, 0], rcond=None)[0]; py = np.linalg.lstsq(A, d[:, 1], rcond=None)[0]
        m = np.array([[px[0], px[1]], [py[0], py[1]]]); tx, ty = px[2], py[2]
    return dict(model=model, c=c, m=m, t=np.array([tx, ty]))

def apply_fit(F, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    return (pts - F['c']) @ F['m'].T + F['c'] + F['t']

def loo(src, dst, model):
    n = len(src); res = np.zeros(n)
    for i in range(n):
        k = np.arange(n) != i
        F = ls_fit(src[k], dst[k], model)
        res[i] = np.linalg.norm(apply_fit(F, src[i])[0] - dst[i])
    return res

def coarse(D, spacing=40.0, search=15.0, half=18.0, vmin=0.9):
    sb = D['sb']; out = []
    for E in np.arange(sb[0] + half + 2, sb[2] - half - 2, spacing):
        for N in np.arange(sb[1] + half + 2, sb[3] - half - 2, spacing):
            m = match_window(D, E, N, half=half, search=search, vmin=vmin)
            if m and m['zncc'] >= 0.3: out.append((E, N, m))
    return out

def search_pair(D, spacing_fine=20.0, half=18.0, coarse_search=15.0, log=print, vmin=0.9, cspacing=40.0):
    cc = coarse(D, spacing=cspacing, search=coarse_search, half=half, vmin=vmin)
    if len(cc) < 3: return None
    log(f'  coarse candidates {len(cc)}')
    src = np.array([[e, n] for e, n, m in cc]); dst = src + np.array([[m['dE'], m['dN']] for e, n, m in cc])
    F, inl = fit_sim(src, dst, 1.0)
    if F is None or inl.sum() < 3: return None
    log(f'  coarse: {len(cc)} windows, {inl.sum()} inliers')
    sb = D['sb']; cand = []
    for it, (sr, thr) in enumerate([(3.0, 0.8), (2.5, 0.6)]):
        cand = []
        for E in np.arange(sb[0] + half + 2, sb[2] - half - 2, spacing_fine):
            for N in np.arange(sb[1] + half + 2, sb[3] - half - 2, spacing_fine):
                pr = apply_fit(F, [E, N])[0] - [E, N]
                m = match_window(D, E, N, half=half, search=sr, prior=(pr[0], pr[1]), vmin=vmin)
                if m is None or m['zncc'] < 0.35: continue
                m['E'] = E; m['N'] = N; m['pred_dE'] = pr[0]; m['pred_dN'] = pr[1]; cand.append(m)
        if len(cand) < 4: return None
        src = np.array([[m['E'], m['N']] for m in cand]); dst = src + np.array([[m['dE'], m['dN']] for m in cand])
        F2, inl = fit_sim(src, dst, thr)
        if F2 is None: return None
        F = F2
        for m, i in zip(cand, inl): m['inlier'] = bool(i)
        log(f'  fine pass {it}: {len(cand)} windows, {inl.sum()} inliers')
    return dict(F=F, cand=cand)

def select_spread(cand, k=12, min_sep=0.0):
    """Greedy farthest-point selection among inlier candidates weighted by quality."""
    c = [m for m in cand if m.get('inlier') and m['ratio'] <= 0.92 and not m['edge']]
    if not c: return []
    q = np.array([m['zncc'] * (1 - 0.5 * m['ratio']) for m in c])
    xy = np.array([[m['E'], m['N']] for m in c])
    sel = [int(np.argmax(q))]
    while len(sel) < min(k, len(c)):
        d = np.min(np.linalg.norm(xy[:, None] - xy[sel][None], axis=2), axis=1)
        score = d * (0.5 + q)          # distance with a quality bonus
        score[sel] = -1
        j = int(np.argmax(score))
        if score[j] <= 0: break
        sel.append(j)
    return [c[i] for i in sel]


def search_pair_robust(D, log=print):
    for kw in [dict(), dict(half=12.0, vmin=0.8, cspacing=25.0, spacing_fine=14.0), dict(half=9.0, vmin=0.7, cspacing=15.0, spacing_fine=10.0, coarse_search=18.0)]:
        R = search_pair(D, log=log, **kw)
        if R is not None:
            R['params'] = kw; return R
        log(f'  retry after failure with {kw}')
    return None
