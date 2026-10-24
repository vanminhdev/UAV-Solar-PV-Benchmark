"""Panel-structure descriptors and paired crops for visual verification of control points."""
import numpy as np, cv2
from orthoreg_match import *

def panel_mask(rgb, valid):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    gray = (r + g + b) / 3
    return valid & (b >= r - 4) & (b >= g - 8) & (gray < 150)

def _compass(dx, dy):
    a = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360   # image y down -> north is -dy
    return ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'][int(((a + 22.5) % 360) // 45)]

def describe(D, E, N, half=15.0):
    """Auto description of the structure inside the window (heuristic colour rules); confirmed/overridden by the visual review log."""
    res = D['res']; c, r = en_to_px(D['sb'], res, E, N); h = int(half / res)
    c, r = int(round(c)), int(round(r))
    rgb = D['s_rgb'][r - h:r + h, c - h:c + h]; v = D['s_v'][r - h:r + h, c - h:c + h]
    if rgb.size == 0: return 'unknown', 0.0
    pm = panel_mask(rgb, v); f = float(pm.mean()); H, W = pm.shape
    R, G, B = [rgb[..., i].astype(int) for i in range(3)]
    gray = (R + G + B) / 3
    parts = []
    nod = (~v).mean()
    if nod > 0.02:
        ys, xs = np.nonzero(~v); parts.append('ortho/plant edge to the ' + _compass(xs.mean() - W / 2, ys.mean() - H / 2))
    track = v & (gray > 185) & (np.abs(R - B) < 28) & (np.abs(G - B) < 28)
    track = cv2.morphologyEx(track.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool)
    if track.mean() > 0.02:
        ys, xs = np.nonzero(track)
        parts.append('pale road/track or roof (%s part of window)' % _compass(xs.mean() - W / 2, ys.mean() - H / 2))
    green = v & (G > R + 8) & (G > B + 8)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    # row ends: transitions along E-W inside table bands
    pmc = cv2.morphologyEx(pm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    pmc = cv2.morphologyEx(pmc, cv2.MORPH_OPEN, np.ones((5, 25), np.uint8))
    prow = cv2.GaussianBlur(pmc.astype(np.float32), (0, 0), 3)
    rowfrac = (prow > 0.5).mean(1)
    bands = rowfrac > 0.35
    ends = 0; side = []
    y = 0
    while y < H:
        if bands[y]:
            y2 = y
            while y2 < H and bands[y2]: y2 += 1
            if y2 - y > int(1.5 / res):
                prof = prow[y + (y2 - y) // 3:y2 - (y2 - y) // 3].mean(0)
                b = prof > 0.5
                tr = np.nonzero(b[1:] != b[:-1])[0]
                tr = [t for t in tr if 0.1 * W < t < 0.9 * W]
                ends += len(tr); side += [t - W / 2 for t in tr]
            y = y2
        else: y += 1
    n_bands = int(((bands[1:] != bands[:-1]).sum() + 1) // 2)
    if ends:
        parts.append('%d table-row end/gap%s in %d table rows%s' % (ends, 's' if ends > 1 else '', n_bands, ' (mostly %s)' % ('west' if np.mean(side) < -0.1 * W else 'east' if np.mean(side) > 0.1 * W else 'near the centre') if ends > 2 else ''))
    elif f > 0.3: parts.append('table-block interior, %d table rows, no row end in window (periodic-lattice risk)' % n_bands)
    if green.mean() > 0.05: parts.append('vegetation patches/ground strips')
    if f < 0.08 and not parts: parts.append('open ground, no panels')
    return '; '.join(parts) if parts else 'table rows', f

def tile(rgb, c, r, half_px, size, mark=True):
    h = half_px
    pad = np.zeros((2 * h + 2, 2 * h + 2, 3), np.uint8)
    y0, x0 = r - h, c - h
    sy0, sx0 = max(y0, 0), max(x0, 0); sy1, sx1 = min(y0 + 2 * h, rgb.shape[0]), min(x0 + 2 * h, rgb.shape[1])
    t = np.zeros((2 * h, 2 * h, 3), np.uint8)
    if sy1 > sy0 and sx1 > sx0: t[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    t = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA)
    if mark:
        m = size // 2; g = 6; L = 22
        for (a, b, c2, d) in [(m - L, m, m - g, m), (m + g, m, m + L, m), (m, m - L, m, m - g), (m, m + g, m, m + L)]:
            cv2.line(t, (a, b), (c2, d), (255, 0, 0), 2)
    return t

def montage(D, pts, title, out, crop_m=36.0, size=230, ncol=3):
    """pts: list of dict(E,N,dE,dN,point_id,zncc). Left: source crop, right: base crop at matched location."""
    res = D['res']; h = int(crop_m / 2 / res)
    tiles = []
    for p in pts:
        c, r = en_to_px(D['sb'], res, p['E'], p['N']); bc, br = en_to_px(D['bb'], res, p['E'] + p['dE'], p['N'] + p['dN'])
        a = tile(D['s_rgb'], int(round(c)), int(round(r)), h, size); b = tile(D['b_rgb'], int(round(bc)), int(round(br)), h, size)
        pair = np.hstack([a, np.full((size, 3, 3), 255, np.uint8), b])
        cv2.putText(pair, p['point_id'].split('_')[-1] + ' z%.2f' % p['zncc'], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(pair)
    while len(tiles) % ncol: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack([t if i == 0 else np.hstack([np.full((size, 8, 3), 40, np.uint8), t]) for i, t in enumerate(tiles[k:k + ncol])]) for k in range(0, len(tiles), ncol)]
    img = np.vstack([np.vstack([r, np.full((6, r.shape[1], 3), 40, np.uint8)]) for r in rows])
    cv2.putText(img, title, (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out, img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80] if out.endswith('.jpg') else [])
