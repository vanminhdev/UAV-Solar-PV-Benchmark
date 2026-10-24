"""Automatic window matching thermal-flight ortho -> base mapping ortho (same RGB modality, structure/gradient NCC)."""
import numpy as np, cv2, rasterio
from orthoreg_common import *

RES = 0.1

def feat(rgb, valid):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = valid.astype(np.float32)
    mean = (g * m).sum() / max(m.sum(), 1)
    g = np.where(valid, g, mean)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    f = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    f = f - cv2.GaussianBlur(f, (0, 0), 20)
    f[~cv2.erode(valid.astype(np.uint8), np.ones((5, 5))).astype(bool)] = 0
    return f.astype(np.float32)

def load_pair(src_path, base_path, pad=40, res=RES):
    with rasterio.open(src_path) as r: b = r.bounds
    sb = (b.left, b.bottom, b.right, b.top)
    bb = (b.left - pad, b.bottom - pad, b.right + pad, b.top + pad)
    s_rgb, s_v = read_rgb(src_path, sb, res)
    b_rgb, b_v = read_rgb(base_path, bb, res)
    return dict(s_rgb=s_rgb, s_v=s_v, b_rgb=b_rgb, b_v=b_v, sb=sb, bb=bb, res=res,
                s_f=feat(s_rgb, s_v), b_f=feat(b_rgb, b_v))

def px_to_en(bounds, res, col, row):
    return bounds[0] + col * res, bounds[3] - row * res

def en_to_px(bounds, res, E, N):
    return (E - bounds[0]) / res, (bounds[3] - N) / res

def match_window(P, E, N, half=18.0, search=6.0, prior=(0.0, 0.0), vmin=0.9):
    """Template of the source ortho centred at (E,N), half-size `half` m. Search in base around (E,N)+prior (+-search m).
    Returns dict(dE, dN, zncc, second, ratio) with base_point = src_point + (dE,dN), or None."""
    res = P['res']
    c, r = en_to_px(P['sb'], res, E, N)
    h = int(half / res)
    c0, r0 = int(round(c)) - h, int(round(r)) - h
    if c0 < 0 or r0 < 0 or c0 + 2 * h > P['s_f'].shape[1] or r0 + 2 * h > P['s_f'].shape[0]: return None
    vfrac = P['s_v'][r0:r0 + 2 * h, c0:c0 + 2 * h].mean()
    if vfrac < vmin: return None
    t = P['s_f'][r0:r0 + 2 * h, c0:c0 + 2 * h]
    if t.std() < 1e-3: return None
    bc, br = en_to_px(P['bb'], res, E + prior[0], N + prior[1])
    s = int(search / res)
    x0, y0 = int(round(bc)) - h - s, int(round(br)) - h - s
    x1, y1 = x0 + 2 * h + 2 * s + 1, y0 + 2 * h + 2 * s + 1
    if x0 < 0 or y0 < 0 or x1 > P['b_f'].shape[1] or y1 > P['b_f'].shape[0]: return None
    if P['b_v'][y0:y1, x0:x1].mean() < vmin: return None
    R = cv2.matchTemplate(P['b_f'][y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED)
    iy, ix = np.unravel_index(np.argmax(R), R.shape)
    z = float(R[iy, ix])
    # sub-pixel
    def sub(a, b, c): d = a - 2 * b + c; return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    dx = sub(R[iy, ix - 1], R[iy, ix], R[iy, ix + 1]) if 0 < ix < R.shape[1] - 1 else 0.0
    dy = sub(R[iy - 1, ix], R[iy, ix], R[iy + 1, ix]) if 0 < iy < R.shape[0] - 1 else 0.0
    R2 = R.copy(); k = int(0.6 / res)
    R2[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1] = -1
    z2 = float(R2.max()); j2, i2 = np.unravel_index(np.argmax(R2), R2.shape)
    edge = (ix in (0, R.shape[1] - 1)) or (iy in (0, R.shape[0] - 1))
    # displacement of template position in base relative to unshifted position
    ox = (ix + dx) - s; oy = (iy + dy) - s          # pixels, relative to prior position
    dE = prior[0] + ox * res; dN = prior[1] - oy * res
    return dict(dE=dE, dN=dN, zncc=z, second=z2, ratio=z2 / z if z > 0 else 9, edge=edge, vfrac=float(vfrac),
                sec_dE=prior[0] + (i2 - s) * res, sec_dN=prior[1] - (j2 - s) * res)
