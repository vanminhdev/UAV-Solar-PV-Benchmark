"""Audit how the QGIS reference grid (TN19_modules_master.gpkg, layer `modules`) relates to the real PV modules.

Read-only on all inputs.  Writes to dataset/metadata/grid_audit/ (outputs) and .local/tmp/ (scratch).
All image statistics are HEURISTIC (colour thresholds, edge scores), not ground truth.

    python -X utf8 code/scripts/audit_grid.py meta      # gpkg metadata, id checks, which 87 m ortho the grid was drawn on
    python -X utf8 code/scripts/audit_grid.py sample    # sample_600_cells.csv (100 per block, fixed seed)
    python -X utf8 code/scripts/audit_grid.py measure   # module pitch / cell grid from seam profiles in 31/35/42/87 m orthos
    python -X utf8 code/scripts/audit_grid.py drift     # grid-vs-ortho shift on a lattice of windows (87 m and low-altitude orthos)
    python -X utf8 code/scripts/audit_grid.py register  # registration of 31/35/42 m orthos to the 87 m ortho
    python -X utf8 code/scripts/audit_grid.py sitewide  # per-cell panel-like fraction, uncovered panel blobs (87 m ortho)
    python -X utf8 code/scripts/audit_grid.py crops     # overlay PNGs + per-crop panel fraction / residual offsets
    python -X utf8 code/scripts/audit_grid.py figs      # close-up + edge examples, copies the chosen PNGs into the output folder
    python -X utf8 code/scripts/audit_grid.py all

Requires: numpy, scipy, rasterio, matplotlib, pillow (no geopandas).
"""
import csv
import glob
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FARM = ROOT / "dataset" / "farm_01_thuan_nam_19"
GPKG = FARM / "module_reference" / "TN19_modules_master.gpkg"
OUT = ROOT / "dataset" / "metadata" / "grid_audit"
TMP = ROOT / ".local" / "tmp"
SEED = 20260420
PW, PH = 1.904, 0.96  # nominal grid cell (long x short side), metres


# --------------------------------------------------------------------------- data access
def ortho(tag):
    """tag like '87M_MAP1' or '42M_T1'."""
    return glob.glob(str(FARM / "photogrammetry" / f"*_{tag}" / "*orthomosaic*.tif"))[0]


def load_modules(path=GPKG):
    """Return dict of numpy arrays: fid, block, string, module_id, num (int counter), P (n,4,2 corner coords)."""
    c = sqlite3.connect(path)
    rows = c.execute("select fid,geom,layer,String_ID,module_id from modules order by fid").fetchall()
    fid, blk, st, mid, P = [], [], [], [], []
    for f, g, layer, sid, m in rows:
        flags = g[3]
        env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
        w = g[8 + env:]
        n = struct.unpack("<I", w[9:13])[0]
        pts = np.frombuffer(w[13:13 + 16 * n], dtype="<f8").reshape(-1, 2)
        fid.append(f); blk.append(layer); st.append(sid); mid.append(m); P.append(pts[:4])
    mid = np.array(mid)
    num = np.array([int(m.split("_")[-1]) for m in mid])
    return dict(fid=np.array(fid), block=np.array(blk), string=np.array(st), module_id=mid, num=num, P=np.array(P))


def read_rgb(path, x0, y0, x1, y1, res):
    """Read RGB(+alpha) window (bounds in map units) resampled to `res` m/pixel; returns (h,w,4) float, transform origin."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    with rasterio.open(path) as s:
        w = max(int(round((x1 - x0) / res)), 2)
        h = max(int(round((y1 - y0) / res)), 2)
        win = from_bounds(x0, y0, x1, y1, s.transform)
        a = s.read([1, 2, 3, 4], window=win, out_shape=(4, h, w), boundless=True, fill_value=0,
                   resampling=Resampling.average).astype(np.float32)
    return np.moveaxis(a, 0, 2)  # pixel (i,j) centre at x0+(j+.5)*res, y1-(i+.5)*res


def panel_mask(a, sigma_px=0):
    """HEURISTIC panel-like pixels: not warm/tan ground (R>B), not green vegetation, not near-white seams/very dark shadow."""
    from scipy import ndimage as ndi
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    mx = a[..., :3].max(-1)
    m = (b - r > -8) & (g - b < 10) & (mx < 215) & (mx > 25) & (al > 200)
    m = m.astype(np.float32)
    return ndi.gaussian_filter(m, sigma_px) if sigma_px else m


def fill_polys(P, x0, y1, res, shape, shift=(0.0, 0.0)):
    from PIL import Image, ImageDraw
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    for p in P:
        xy = [((x + shift[0] - x0) / res, (y1 - (y + shift[1])) / res) for x, y in p]
        d.polygon(xy, fill=255)
    return np.asarray(im) > 0


# --------------------------------------------------------------------------- 1. metadata
def meta():
    import rasterio
    OUT.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(GPKG)
    rep = {}
    rep["contents"] = [dict(zip(["table", "type", "srs", "minx", "miny", "maxx", "maxy"], r)) for r in
                       c.execute("select table_name,data_type,srs_id,min_x,min_y,max_x,max_y from gpkg_contents")]
    rep["srs"] = c.execute("select srs_id,organization,organization_coordsys_id,srs_name from gpkg_spatial_ref_sys where srs_id=32649").fetchall()
    rep["geometry_columns"] = c.execute("select * from gpkg_geometry_columns").fetchall()
    rep["modules_columns"] = [r[1] for r in c.execute("pragma table_info(modules)")]
    M = load_modules()
    n = M["num"]
    rep["n_modules"] = int(len(n)); rep["n_unique_module_id"] = int(len(set(M["module_id"])))
    rep["counter_min_max"] = [int(n.min()), int(n.max())]
    rep["counter_gaps"] = int(n.max() - len(set(n)))
    rep["counter_unique"] = int(len(set(n)))
    # prefix consistency
    rep["id_prefix_mismatch"] = int(sum(1 for m, b in zip(M["module_id"], M["block"]) if not m.startswith(b.lower() + "_")))
    rep["id_string_mismatch"] = int(sum(1 for m, s in zip(M["module_id"], M["string"]) if m.split("_")[1] != s))
    side = np.hypot(*(M["P"][:, 1] - M["P"][:, 0]).T), np.hypot(*(M["P"][:, 2] - M["P"][:, 1]).T)
    rep["side_a_mean_std"] = [float(side[0].mean()), float(side[0].std())]
    rep["side_b_mean_std"] = [float(side[1].mean()), float(side[1].std())]
    ang = np.degrees(np.arctan2(*(M["P"][:, 1] - M["P"][:, 0]).T[::-1]))
    rep["rotation_deg_percentiles_0_5_50_95_100"] = np.percentile(ang, [0, 5, 50, 95, 100]).round(3).tolist()
    # overlaps (shapely if available) - neighbour overlap check via centre distances
    cen = M["P"].mean(1)
    from scipy.spatial import cKDTree
    t = cKDTree(cen)
    pairs = t.query_pairs(0.9)  # centres closer than 0.9 m cannot be two non-overlapping 1.9x0.96 cells
    rep["centre_pairs_closer_than_0.9m"] = len(pairs)
    # per block
    per = {}
    for b in sorted(set(M["block"])):
        m = M["block"] == b
        nb = n[m]
        # string / row structure: strings -> counts
        strs = M["string"][m]
        per[b] = dict(n=int(m.sum()), counter_range=[int(nb.min()), int(nb.max())],
                      contiguous=bool(nb.max() - nb.min() + 1 == m.sum()), n_strings=len(set(strs)),
                      centre_E_range=[float(cen[m, 0].min()), float(cen[m, 0].max())],
                      centre_N_range=[float(cen[m, 1].min()), float(cen[m, 1].max())],
                      rot_deg_median=float(np.median(ang[m])))
    rep["per_block"] = per
    # ordering along strings: consecutive counters -> same row (W->E), next row?
    order = np.argsort(n)
    d = cen[order]
    step = np.diff(d, axis=0)
    ok_we = (step[:, 0] > 0) & (np.abs(step[:, 1]) < 0.5)
    rep["consecutive_ids_step_east_same_row_frac"] = float(ok_we.mean())
    rep["consecutive_ids_step_median_m"] = np.median(step[ok_we], axis=0).round(3).tolist()
    # row-jump size distribution (new row): dy negative (north->south) ?
    jump = ~ok_we
    rep["row_change_count"] = int(jump.sum())
    rep["row_change_dy_median"] = float(np.median(step[jump, 1]))
    rep["row_change_dy_neg_frac(N->S)"] = float((step[jump, 1] < 0).mean())
    # path column leak
    rep["path_column_distinct"] = [r[0] for r in c.execute("select distinct path from modules")]
    # which ortho
    tiles = c.execute("select * from gpkg_tile_matrix_set").fetchall()
    rep["tile_matrix_set"] = tiles
    ext = tiles[0][2:6]
    cmp_ = []
    for tag in ["87M_MAP1", "87M_MAP2"]:
        with rasterio.open(ortho(tag)) as s:
            b = s.bounds
            cmp_.append(dict(tag=tag, left=b.left, bottom=b.bottom, right=b.right, top=b.top, res=s.res[0],
                             dx_left=b.left - ext[0], dtop=b.top - ext[3]))
    rep["raster_layer_vs_orthos"] = cmp_
    # decode a tile from the raster layer and correlate with MAP1
    import io
    from PIL import Image
    mm = c.execute("select zoom_level,tile_column,tile_row from orthophoto_87m where zoom_level=7").fetchall()
    zs = c.execute("select * from gpkg_tile_matrix where zoom_level=7").fetchone()
    rep["tile_matrix_z7"] = zs
    mw, mh, tw, th, res_t = zs[2], zs[3], zs[4], zs[5], zs[6]
    # choose a tile near the centre of the modules extent
    best = None
    ox, oy = ext[0], ext[3]
    for z, col, row in mm:
        cx = ox + (col + .5) * tw * res_t
        cy = oy - (row + .5) * th * res_t
        if 265500 < cx < 265700 and 1263650 < cy < 1263850:
            best = (z, col, row); break
    if best:
        blob = c.execute("select tile_data from orthophoto_87m where zoom_level=? and tile_column=? and tile_row=?", best).fetchone()[0]
        im = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB")).astype(float)
        x0 = ox + best[1] * tw * res_t; y1 = oy - best[2] * th * res_t
        out = {}
        for tag in ["87M_MAP1", "87M_MAP2"]:
            a = read_rgb(ortho(tag), x0, y1 - th * res_t, x0 + tw * res_t, y1, res_t)[..., :3]
            if a.shape[:2] != im.shape[:2]:
                out[tag] = "shape mismatch"; continue
            out[tag] = dict(mean_abs_diff=float(np.abs(a - im).mean()),
                            corr=float(np.corrcoef(a.ravel(), im.ravel())[0, 1]))
        rep["tile_vs_ortho_pixel_check"] = dict(tile=list(best), result=out)
    (OUT / "meta_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))


# --------------------------------------------------------------------------- 2. sample
def sample():
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_modules()
    cen = M["P"].mean(1)
    rng = np.random.default_rng(SEED)
    rows = []
    for b in sorted(set(M["block"])):
        idx = np.where(M["block"] == b)[0]
        pick = np.sort(rng.choice(idx, 100, replace=False))
        for i in pick:
            rows.append(dict(module_id=M["module_id"][i], block=b, string_id=M["string"][i],
                             centre_E_32649=round(float(cen[i, 0]), 3), centre_N_32649=round(float(cen[i, 1]), 3),
                             check_status="", check_notes=""))
    with open(OUT / "sample_600_cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(len(rows), "rows; seed", SEED)


# --------------------------------------------------------------------------- 3. pitch measurement
class Src:
    """Pre-read window around a centre so that oriented strips can be sampled repeatedly without re-reading the raster."""
    def __init__(self, path, c, res, rad=24.0):
        self.res = res
        self.x0, self.y1 = c[0] - rad, c[1] + rad
        self.img = read_rgb(path, c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad, res)


def strip(src, c, theta, L, H, res=None):
    """Sample an oriented strip (length L along theta deg, height H across) -> float (ny,nx,4)."""
    from scipy import ndimage as ndi
    res = res or src.res
    t = np.radians(theta)
    u = np.array([np.cos(t), np.sin(t)]); v = np.array([-np.sin(t), np.cos(t)])
    nx, ny = int(L / res), int(H / res)
    a = (np.arange(nx) - nx / 2 + .5) * res; b = (np.arange(ny) - ny / 2 + .5) * res
    A, B = np.meshgrid(a, b)
    X = c[0] + A * u[0] + B * v[0]; Y = c[1] + A * u[1] + B * v[1]
    col = (X - src.x0) / src.res - .5; row = (src.y1 - Y) / src.res - .5
    return np.stack([ndi.map_coordinates(src.img[..., k], [row, col], order=1) for k in range(4)], -1)


def seam_positions(prof, res, lo=1.5, hi=2.3, prom_frac=0.5):
    """Positions (m) of the strongest periodic bright peaks (module seams) in a 1-D profile."""
    from scipy.signal import find_peaks
    p = prof - np.median(prof)
    rngv = np.percentile(p, 99.5) - np.percentile(p, 5)
    pk, pr = find_peaks(p, distance=int(0.5 / res), prominence=prom_frac * rngv)
    # sub-pixel centroid
    pos = []
    for k in pk:
        s = slice(max(k - 3, 0), k + 4)
        w = np.clip(p[s] - p[s].min(), 0, None)
        pos.append((np.arange(s.start, s.stop) * w).sum() / w.sum() * res if w.sum() > 0 else k * res)
    return np.array(pos), pk


def table_rows(a, res, close_m=0.0):
    """Longest run of panel-like rows (the table, 3.3-4.6 m high) in a strip -> (i0,i1) or None."""
    br = (a[..., 2] - a[..., 0]).mean(1)
    alpha = a[..., 3].mean(1)
    panel = (br > -8) & (alpha > 200) & (a[..., :3].mean((1, 2)) < 200)
    if close_m:  # bridge the bright module-seam rows (used for the 1 cm close-up; measure() was run with close_m=0)
        from scipy import ndimage as ndi
        panel = ndi.binary_closing(panel, iterations=int(close_m / res))
    best = (0, 0, 0); i = 0
    while i < len(panel):
        if panel[i]:
            j = i
            while j < len(panel) and panel[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    n, i0, i1 = best
    return (i0, i1) if 3.3 < n * res < 4.6 else None


def measure():
    """Module pitch (seam spacing) along (x) and across (y) a table, per ortho.  HEURISTIC peak detection."""
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_modules()
    cen = M["P"].mean(1)
    rng = np.random.default_rng(SEED + 1)
    import rasterio
    rows = []
    tags = ["31M_T1", "31M_T2", "31M_T3", "35M_T1", "35M_T2", "35M_T3", "35M_T4", "35M_T5", "35M_T6",
            "42M_T1", "42M_T2", "42M_T3", "42M_T4", "87M_MAP1", "87M_MAP2"]
    ths = np.degrees(np.arctan2(*(M["P"][:, 1] - M["P"][:, 0]).T[::-1]))
    for tag in tags:
        path = ortho(tag)
        with rasterio.open(path) as s:
            b = s.bounds; nat = s.res[0]
        res = 0.03 if tag.startswith("87M") else 0.015
        ok = np.where((cen[:, 0] > b.left + 25) & (cen[:, 0] < b.right - 25) & (cen[:, 1] > b.bottom + 12) & (cen[:, 1] < b.top - 12))[0]
        if len(ok) == 0:
            continue
        tries = rng.choice(ok, min(80, len(ok)), replace=False)
        got = 0
        for i in tries:
            if got >= 8:
                break
            th = float(ths[i])
            src = Src(path, cen[i], res)
            a = strip(src, cen[i], th, 36.0, 14.0)
            loc = table_rows(a, res)
            if loc is None:
                continue
            m = int(0.25 / res)
            r0, r1 = loc[0] + m, loc[1] - m
            best = (-1, th)
            for dth in np.arange(-0.8, 0.81, 0.1):
                aa = strip(src, cen[i], th + dth, 36.0, 14.0)
                pr = aa[r0:r1, :, :3].mean((0, 2))
                sc = np.var(np.diff(pr))
                if sc > best[0]:
                    best = (sc, th + dth)
            th2 = best[1]
            a = strip(src, cen[i], th2, 36.0, 14.0)
            px = a[r0:r1, :, :3].mean((0, 2))
            pos, _ = seam_positions(px, res)
            d = np.diff(pos)
            dx = d[(d > 1.7) & (d < 2.2)]
            py = a[max(loc[0] - int(0.3 / res), 0):loc[1] + int(0.3 / res), int(3 / res):int(-3 / res), :3].mean((1, 2))
            posy, _ = seam_positions(py, res, prom_frac=0.45) if len(py) > 10 else (np.array([]), None)
            dy = np.diff(posy); dy = dy[(dy > 0.8) & (dy < 1.15)]
            fw = []
            for p in pos:
                k = int(round(p / res))
                seg = px[max(k - int(0.3 / res), 0):k + int(0.3 / res)]
                base = np.median(px); half = base + 0.5 * (px[min(k, len(px) - 1)] - base)
                fw.append((seg > half).sum() * res)
            cells = None
            from scipy.signal import find_peaks
            cnt = []
            for p0, p1 in zip(pos[:-1], pos[1:]):
                if 1.7 < p1 - p0 < 2.2:
                    s0, s1 = int((p0 + 0.12) / res), int((p1 - 0.12) / res)
                    seg = px[s0:s1] - np.median(px[s0:s1])
                    pk, _ = find_peaks(seg, distance=int(0.08 / res), prominence=0.25 * (np.percentile(seg, 99) - np.percentile(seg, 1)))
                    cnt.append(len(pk) + 1)
            cells = float(np.median(cnt)) if cnt else None
            # cumulative position check: fit seam index vs position over the strip (robust slope)
            slope = None
            if len(dx) >= 6 and len(dx) == len(d):
                k = np.arange(len(pos)); slope = float(np.polyfit(k, pos, 1)[0])
            rows.append(dict(ortho=tag, gsd_cm=round(nat * 100, 2), E=round(float(cen[i, 0]), 1), N=round(float(cen[i, 1]), 1),
                             block=M["block"][i], theta_deg=round(th2, 2), n_seam_x=len(pos), n_dx=len(dx),
                             pitch_x_median=float(np.median(dx)) if len(dx) else None,
                             pitch_x_fit_slope=slope,
                             pitch_x_std=float(dx.std()) if len(dx) else None,
                             n_dy=len(dy), pitch_y_median=float(np.median(dy)) if len(dy) else None,
                             seam_width_x_m=float(np.median(fw)) if fw else None, cell_cols_per_module=cells))
            got += 1
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    with open(OUT / "pitch_measurements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for fam in ["31M", "35M", "42M", "87M"]:
        rr = [r for r in rows if r["ortho"].startswith(fam)]
        px = np.array([r["pitch_x_median"] for r in rr if r["pitch_x_median"]])
        py = np.array([r["pitch_y_median"] for r in rr if r["pitch_y_median"]])
        summ[fam] = dict(n_strips=len(rr), pitch_x_mean=float(px.mean()) if len(px) else None, pitch_x_sd=float(px.std(ddof=1)) if len(px) > 1 else None,
                         n_y=len(py), pitch_y_mean=float(py.mean()) if len(py) else None, pitch_y_sd=float(py.std(ddof=1)) if len(py) > 1 else None)
    (OUT / "pitch_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


# --------------------------------------------------------------------------- 4. grid vs ortho shift
def hp_gray(a, res, sigma_m=0.6):
    from scipy import ndimage as ndi
    g = a[..., :3].mean(-1)
    return g - ndi.gaussian_filter(g, sigma_m / res)


def edge_points(p, step):
    pts = []
    for k in range(4):
        a, b = p[k], p[(k + 1) % 4]
        n = max(int(np.hypot(*(b - a)) / step), 2)
        t = (np.arange(n) + .5) / n
        pts.append(a[None] + t[:, None] * (b - a)[None])
    return np.vstack(pts)


def seam_score(hp, x0, y1, res, pts, dx, dy):
    from scipy import ndimage as ndi
    col = (pts[:, 0] + dx - x0) / res - .5
    row = (y1 - (pts[:, 1] + dy)) / res - .5
    return ndi.map_coordinates(hp, [row, col], order=1, mode="nearest").mean()


def window_surface(path, P, x0, y0, x1, y1, res, R=4.0):  # R: search radius (m) for the polygon shift
    """Normalised cross-correlation surface (panel mask vs rasterised polygons) for polygon shifts within +-R m.
    Returns (surface[(dN,dE)], image array a, X0, Y1) or None when the window is mostly outside the ortho."""
    from scipy import ndimage as ndi
    a = read_rgb(path, x0 - R, y0 - R, x1 + R, y1 + R, res)
    X0, Y1 = x0 - R, y1 + R
    if (a[..., 3] > 200).mean() < 0.6:
        return None
    m = ndi.gaussian_filter(panel_mask(a), 0.12 / res)
    F = ndi.gaussian_filter(fill_polys(P, X0, Y1, res, m.shape).astype(np.float32), 0.12 / res)
    mz = m - m.mean(); fz = F - F.mean()
    cc = np.fft.irfft2(np.fft.rfft2(mz) * np.conj(np.fft.rfft2(fz)), s=m.shape) / (np.sqrt((mz ** 2).sum() * (fz ** 2).sum()) + 1e-9)
    n = int(R / res)
    idx = np.r_[-n:n + 1]
    # cc[di,dj]: polygons moved by (+dj*res E, -di*res N)
    S = cc[np.ix_(idx % m.shape[0], idx % m.shape[1])]  # rows: di=-n..n ; cols: dj=-n..n
    S = S[::-1, :]  # rows now dN = -n..n  (dN = -di)
    return S, a, X0, Y1


def surf_peak(S, res, near=None, rad=None):
    n = (S.shape[0] - 1) // 2
    dN = (np.arange(-n, n + 1)) * res
    T = S.copy()
    if near is not None:
        mm = (np.abs(dN[:, None] - near[1]) <= rad) & (np.abs(dN[None, :] - near[0]) <= rad)
        T = np.where(mm, T, -np.inf)
    k = np.unravel_index(np.argmax(T), T.shape)
    return dN[k[1]], dN[k[0]], float(S[k])  # dE, dN, value


def fine_seam(a, X0, Y1, res, P, c0, rad=0.5):
    hp = hp_gray(a, res)
    pts = np.vstack([edge_points(p, 0.05) for p in P])
    best = (-1e9, c0[0], c0[1])
    for dN in np.arange(c0[1] - rad, c0[1] + rad + 1e-6, 0.03):
        for dE in np.arange(c0[0] - rad, c0[0] + rad + 1e-6, 0.03):
            sc = seam_score(hp, X0, Y1, res, pts, dE, dN)
            if sc > best[0]:
                best = (sc, dE, dN)
    return best[1], best[2], best[0]


def window_shift(path, P, x0, y0, x1, y1, res, near=(0.0, 0.0), rad=0.6):
    """Refine a polygon shift for a window by seam brightness along polygon edges, searching +-rad m around `near`
    (the registration estimate).  HEURISTIC; rad must stay below half a module (0.95 m) to avoid column aliasing."""
    a = read_rgb(path, x0 - 1.5 - abs(near[0]), y0 - 1.5 - abs(near[1]), x1 + 1.5 + abs(near[0]), y1 + 1.5 + abs(near[1]), res)
    X0, Y1 = x0 - 1.5 - abs(near[0]), y1 + 1.5 + abs(near[1])
    dE, dN, sc = fine_seam(a, X0, Y1, res, P, near, rad)
    return dict(dE=dE, dN=dN, score=sc)


def drift():
    """Grid-vs-ortho registration on a lattice of 30x30 m windows.
    Level 1: per-ortho rigid shift from the summed correlation surfaces of all windows (resolves the 1.9 m column aliasing
    because table ends differ between windows).  Level 2: per (ortho, block) shift.  Level 3: per-window residual.  HEURISTIC."""
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_modules(); cen = M["P"].mean(1); P = M["P"]
    import rasterio
    rows = []; summ = {}
    tags = ["87M_MAP1"]  # grid-based estimate is only reliable on the 87 m ortho; low-altitude orthos: see register()
    if len(sys.argv) > 2:  # re-run only some orthos and merge into the existing outputs
        keep = [t for t in tags if t not in sys.argv[2:]]
        tags = [t for t in tags if t in sys.argv[2:]]
        if (OUT / "drift_summary.json").exists():
            old = json.loads((OUT / "drift_summary.json").read_text())
            summ = {k: v for k, v in old.items() if k in keep}
            rows = [r for r in csv.DictReader(open(OUT / "drift_windows.csv", encoding="utf-8")) if r["ortho"] in keep]
            for r in rows:
                for k in ("E", "N", "dE", "dN"):
                    r[k] = float(r[k])
    res = 0.03
    for tag in tags:
        path = ortho(tag)
        with rasterio.open(path) as s:
            b = s.bounds
        wins = []
        E0 = max(b.left, 265360); N0 = max(b.bottom, 1263590)
        for e in np.arange(E0 + 20, min(b.right, 265920) - 10, 40):
            for n in np.arange(N0 + 15, min(b.top, 1263905) - 10, 40):
                x0, x1, y0, y1 = e - 15, e + 15, n - 15, n + 15
                sel = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1)
                if sel.sum() < 120:
                    continue
                r = window_surface(path, P[sel], x0, y0, x1, y1, res, R=4.0 if tag.startswith("87M") else 9.0)
                if r is None:
                    continue
                wins.append(dict(e=e, n=n, sel=sel, blk=Counter_most(M["block"][sel]), S=r[0], a=r[1], X0=r[2], Y1=r[3]))
        if not wins:
            continue
        Sall = np.mean([w["S"] for w in wins], 0)
        c1 = surf_peak(Sall, res)
        for w in wins:
            w["c1"] = c1
        blocks = sorted(set(w["blk"] for w in wins))
        bres = {}
        for blk in blocks:
            ww = [w for w in wins if w["blk"] == blk]
            Sb = np.mean([w["S"] for w in ww], 0)
            bres[blk] = surf_peak(Sb, res, near=c1[:2], rad=1.0)
        for w in wins:
            cb = bres[w["blk"]]
            dE, dN, sc = fine_seam(w["a"], w["X0"], w["Y1"], res, P[w["sel"]], (cb[0], cb[1]), 0.5)
            rows.append(dict(ortho=tag, E=w["e"], N=w["n"], block=w["blk"], n_cells=int(w["sel"].sum()),
                             ortho_dE=round(c1[0], 3), ortho_dN=round(c1[1], 3), block_dE=round(cb[0], 3), block_dN=round(cb[1], 3),
                             dE=round(dE, 3), dN=round(dN, 3), seam_score=round(float(sc), 2)))
        rr = [r for r in rows if r["ortho"] == tag]
        dE = np.array([r["dE"] for r in rr]); dN = np.array([r["dN"] for r in rr])
        summ[tag] = dict(n_windows=len(rr), ortho_level_shift_E_N=[round(c1[0], 3), round(c1[1], 3)], corr_peak=round(c1[2], 3),
                         per_block_shift={k: [round(v[0], 2), round(v[1], 2)] for k, v in bres.items()},
                         window_dE_median=float(np.median(dE)), window_dN_median=float(np.median(dN)),
                         window_dE_p5_p95=np.percentile(dE, [5, 95]).round(2).tolist(), window_dN_p5_p95=np.percentile(dN, [5, 95]).round(2).tolist(),
                         window_dev_from_median_p95=float(np.percentile(np.hypot(dE - np.median(dE), dN - np.median(dN)), 95)))
        print(tag, summ[tag], flush=True)
    with open(OUT / "drift_windows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "drift_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")


LOW = ["42M_T1", "42M_T2", "42M_T3", "42M_T4", "35M_T1", "35M_T2", "35M_T3", "35M_T4", "35M_T5", "35M_T6", "31M_T1", "31M_T2", "31M_T3"]


def register():
    """Image-to-image registration of each low-altitude ortho to the 87 m ortho (the one the grid was drawn on):
    zero-mean normalised cross-correlation of high-passed grey images in 36 m windows, search +-12 m.  Result = displacement (E, N)
    of the low-altitude ortho content relative to the 87 m ortho = shift to apply to the grid to fit that ortho.  HEURISTIC."""
    from scipy import ndimage as ndi
    from scipy.signal import fftconvolve
    import rasterio
    OUT.mkdir(parents=True, exist_ok=True)
    res, W, R, step = 0.06, 36, 12, 30
    pr = ortho("87M_MAP1")
    rows = []; summ = {}
    for tag in LOW:
        pa = ortho(tag)
        b = rasterio.open(pa).bounds
        for e in np.arange(b.left + W / 2 + R, b.right - W / 2 - R, step):
            for n in np.arange(b.bottom + W / 2 + R, b.top - W / 2 - R, step):
                A = read_rgb(pa, e - W / 2 - R, n - W / 2 - R, e + W / 2 + R, n + W / 2 + R, res)
                Rr = read_rgb(pr, e - W / 2, n - W / 2, e + W / 2, n + W / 2, res)
                if (A[..., 3] > 200).mean() < 0.9 or (Rr[..., 3] > 200).mean() < 0.9 or panel_mask(Rr).mean() < 0.2:
                    continue
                ga = A[..., :3].mean(-1); gr = Rr[..., :3].mean(-1)
                ga = ndi.gaussian_filter(ga - ndi.gaussian_filter(ga, 1.5 / res), 0.1 / res)
                gr = ndi.gaussian_filter(gr - ndi.gaussian_filter(gr, 1.5 / res), 0.1 / res)
                gz = gr - gr.mean(); ones = np.ones_like(gr)
                s2 = fftconvolve(ga ** 2, ones, mode="valid"); s1 = fftconvolve(ga, ones, mode="valid")
                num = fftconvolve(ga, gz[::-1, ::-1], mode="valid")
                z = num / np.sqrt(np.maximum(s2 - s1 ** 2 / gr.size, 1e-6) * (gz ** 2).sum())
                k = np.unravel_index(np.argmax(z), z.shape)
                z2 = z.copy(); q = int(0.5 / res)
                z2[max(k[0] - q, 0):k[0] + q + 1, max(k[1] - q, 0):k[1] + q + 1] = -1
                c = int(R / res)
                rows.append(dict(ortho=tag, E=e, N=n, dE=round((k[1] - c) * res, 2), dN=round(-(k[0] - c) * res, 2),
                                 zncc=round(float(z[k]), 3), second_peak=round(float(z2.max()), 3)))
        rr = [r for r in rows if r["ortho"] == tag and r["zncc"] > 0.5 and r["zncc"] - r["second_peak"] > 0.05]
        if rr:
            dE = np.array([r["dE"] for r in rr]); dN = np.array([r["dN"] for r in rr])
            summ[tag] = dict(n_windows_ok=len(rr), n_windows_all=sum(1 for r in rows if r["ortho"] == tag),
                             dE_median=float(np.median(dE)), dN_median=float(np.median(dN)),
                             dE_min_max=[float(dE.min()), float(dE.max())], dN_min_max=[float(dN.min()), float(dN.max())],
                             radial_median=float(np.median(np.hypot(dE, dN))))
        print(tag, summ.get(tag), flush=True)
    with open(OUT / "registration_windows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "registration_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")


def local_shift(tag, cx, cy):
    """Nearest reliable registration window shift for a low-altitude ortho."""
    rr = [r for r in csv.DictReader(open(OUT / "registration_windows.csv", encoding="utf-8")) if r["ortho"] == tag
          and float(r["zncc"]) > 0.5 and float(r["zncc"]) - float(r["second_peak"]) > 0.05]
    if not rr:
        return (0.0, 0.0)
    r = min(rr, key=lambda r: np.hypot(float(r["E"]) - cx, float(r["N"]) - cy))
    return (float(r["dE"]), float(r["dN"]))


def sitewide():
    """Whole-site HEURISTIC checks on the 87 m ortho at 0.1 m: panel-like fraction of every cell, and panel blobs not covered by any cell."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    OUT.mkdir(parents=True, exist_ok=True)
    M = load_modules(); P = M["P"]
    res = 0.1
    x0, y0, x1, y1 = 265363.458, 1263596.4554, 265913.062, 1263901.9076
    a = read_rgb(ortho("87M_MAP1"), x0, y0, x1, y1, res)
    mask = panel_mask(a, 0.1 / res * 1.0) > 0.5
    valid = a[..., 3] > 200
    h, w = mask.shape
    im = Image.new("I", (w, h), 0); d = ImageDraw.Draw(im)
    for i, p in enumerate(P):
        d.polygon([((x - x0) / res, (y1 - y) / res) for x, y in p], fill=i + 1)
    lab = np.asarray(im)
    cnt = np.bincount(lab.ravel(), minlength=len(P) + 1)[1:]
    pan = np.bincount(lab.ravel(), weights=(mask & valid).ravel().astype(float), minlength=len(P) + 1)[1:]
    val = np.bincount(lab.ravel(), weights=valid.ravel().astype(float), minlength=len(P) + 1)[1:]
    pf = pan / np.maximum(cnt, 1)
    cen = P.mean(1)
    with open(OUT / "cell_panel_fraction.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["module_id", "block", "panel_like_fraction_heuristic", "valid_ortho_fraction"])
        for i in range(len(P)):
            wr.writerow([M["module_id"][i], M["block"][i], round(pf[i], 3), round(val[i] / max(cnt[i], 1), 3)])
    rep = dict(n_cells=len(P), pf_percentiles_1_5_25_50_75=np.percentile(pf, [1, 5, 25, 50, 75]).round(3).tolist(),
               n_pf_lt_0p3=int((pf < 0.3).sum()), n_pf_lt_0p5=int((pf < 0.5).sum()), n_pf_lt_0p7=int((pf < 0.7).sum()),
               n_partly_outside_ortho=int((val / np.maximum(cnt, 1) < 0.95).sum()))
    rep["pf_lt_0p5_by_block"] = {b: int(((pf < 0.5) & (M["block"] == b)).sum()) for b in sorted(set(M["block"]))}
    low = np.where(pf < 0.5)[0]
    rep["lowest_cells"] = [dict(module_id=M["module_id"][i], pf=round(float(pf[i]), 2), E=round(float(cen[i, 0]), 1), N=round(float(cen[i, 1]), 1)) for i in low[np.argsort(pf[low])][:15]]
    # uncovered panel blobs
    union = lab > 0
    near = ndi.binary_dilation(union, iterations=int(0.5 / res))
    lb, n = ndi.label(mask & valid & ~near)
    sizes = ndi.sum(np.ones_like(lb), lb, range(1, n + 1))
    coms = ndi.center_of_mass(np.ones_like(lb), lb, range(1, n + 1))
    blobs = [(float(sz * res * res), x0 + (c[1] + .5) * res, y1 - (c[0] + .5) * res) for sz, c in zip(sizes, coms) if sz * res * res >= 4.0]
    blobs.sort(reverse=True)
    with open(OUT / "uncovered_panel_blobs.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["area_m2", "E", "N"])
        for b in blobs:
            wr.writerow([round(b[0], 1), round(b[1], 1), round(b[2], 1)])
    rep["uncovered_blobs_ge4m2"] = len(blobs)
    rep["uncovered_blobs_total_area_m2"] = round(sum(b[0] for b in blobs), 1)
    rep["panel_like_area_m2_total"] = round(float((mask & valid).sum() * res * res), 1)
    rep["cells_area_m2_total"] = round(len(P) * PW * PH, 1)
    rep["top_uncovered"] = [[round(v, 1) for v in b] for b in blobs[:12]]
    (OUT / "sitewide_summary.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


def figs():
    """Extra figures: ortho-edge examples (87 m) and a close-up of one 31 m table with seam profiles; then copy the chosen PNGs to OUT."""
    import shutil
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    M = load_modules()
    crop_stats_and_png("87M_MAP1", "87m_south_edge_warp", 265600.1, 1263606.4, 40, 26, M, 0.03, png=TMP / "87m_south_edge_warp.png", residuals=False)
    crop_stats_and_png("87M_MAP1", "87m_west_edge_warp", 265398.1, 1263630.8, 40, 26, M, 0.03, png=TMP / "87m_west_edge_warp.png", residuals=False)
    # close-up on 31M_T3 (use the crop centre / shift recorded by crops())
    row = [r for r in csv.DictReader(open(OUT / "crop_stats.csv", encoding="utf-8")) if r["name"] == "31M_T3_Block2_shifted"][0]
    cx, cy = float(row["cx"]), float(row["cy"]); sh = (float(row["est_shift_E"]), float(row["est_shift_N"]))
    path = ortho("31M_T3")
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(2, 2, (1, 3))
    w, h = 9.0, 6.0
    x0, y0 = cx - 9, cy - 3
    sub = read_rgb(path, x0, y0, x0 + w, y0 + h, 0.01)
    ax.imshow(np.clip(sub[..., :3], 0, 255).astype(np.uint8), extent=(x0, x0 + w, y0, y0 + h))
    cen = M["P"].mean(1) + np.array(sh)
    sel = (cen[:, 0] > x0 - 2) & (cen[:, 0] < x0 + w + 2) & (cen[:, 1] > y0 - 2) & (cen[:, 1] < y0 + h + 2)
    ax.add_collection(PolyCollection(M["P"][sel] + np.array(sh)[None, None], facecolors="none", edgecolors="yellow", linewidths=1.2))
    ax.set_xlim(x0, x0 + w); ax.set_ylim(y0, y0 + h); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"31 m ortho (1.05 cm GSD), grid shifted by ({sh[0]:+.2f},{sh[1]:+.2f}) m: one yellow cell = one 12x6-cell module", fontsize=9)
    # seam profile along a table (strip centred on the grid cell nearest to the crop centre, in ortho coordinates)
    ii = int(np.argmin(np.hypot(*(cen - [cx, cy]).T)))
    c0 = cen[ii]
    th = float(np.degrees(np.arctan2(*(M["P"][ii][1] - M["P"][ii][0])[::-1])))
    src = Src(path, c0, 0.01, rad=24)
    st = strip(src, c0, th, 24.0, 14.0)
    loc = table_rows(st, 0.01, close_m=0.15)
    if loc:
        m = int(0.25 / 0.01)
        px = st[loc[0] + m:loc[1] - m, :, :3].mean((0, 2))
        pos, _ = seam_positions(px, 0.01)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(np.arange(len(px)) * 0.01, px, lw=0.7)
        for q in pos:
            ax2.axvline(q, color="r", lw=0.4)
        d = np.diff(pos)
        ax2.set_title(f"brightness along a table (mean over its 4 rows): seam spacing median {np.median(d[(d>1.7)&(d<2.2)]):.3f} m; 12 cell gaps between seams", fontsize=9)
        ax2.set_xlabel("m along table")
        py = st[:, int(2 / 0.01):int(-2 / 0.01), :3].mean((1, 2))[max(loc[0] - 30, 0):loc[1] + 30]
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(np.arange(len(py)) * 0.01, py, lw=0.7)
        pos2, _ = seam_positions(py, 0.01, prom_frac=0.45)
        for q in pos2:
            ax3.axvline(q, color="r", lw=0.4)
        d2 = np.diff(pos2)
        ax3.set_title(f"brightness across the table: seam spacing {np.round(d2, 3).tolist()} m; 6 cell rows per module", fontsize=9)
        ax3.set_xlabel("m across table")
    plt.tight_layout(); fig.savefig(TMP / "detail_31m_module_profile.png", dpi=55, pil_kwargs={"optimize": True}); plt.close(fig)
    picks = {"87m_Block1_corner": "87m_01_block1_corner", "87m_Block2_interior": "87m_02_block2_interior", "87m_Block2_corner": "87m_03_block2_corner",
             "87m_Block3_interior": "87m_04_block3_interior", "87m_Block3_corner": "87m_05_block3_corner", "87m_Block4_corner": "87m_06_block4_corner",
             "87m_Block5_interior": "87m_07_block5_interior", "87m_Block6_corner": "87m_08_block6_corner",
             "87m_south_edge_warp": "87m_09_south_edge_warp", "87m_west_edge_warp": "87m_10_west_edge_warp",
             "31M_T3_Block2_shifted": "low_11_31m_T3_shifted", "31M_T2_Block2_shifted": "low_12_31m_T2_shifted",
             "35M_T6_Block6_raw": "low_13_35m_T6_raw_unshifted", "35M_T6_Block6_shifted": "low_14_35m_T6_shifted",
             "42M_T1_Block1_shifted": "low_15_42m_T1_rigid_shift_fails", "detail_31m_module_profile": "detail_16_31m_module_profile"}
    for k, v in picks.items():
        shutil.copy(TMP / f"{k}.png", OUT / f"{v}.png")
    # drift map of the 87 m windows (colour = |shift| of the best-fitting polygon position, in cm)
    rows = [r for r in csv.DictReader(open(OUT / "drift_windows.csv", encoding="utf-8"))]
    cen_all = M["P"].mean(1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.scatter(cen_all[::15, 0], cen_all[::15, 1], s=0.3, c="0.8")
    E_ = np.array([float(r["E"]) for r in rows]); N_ = np.array([float(r["N"]) for r in rows])
    mag = np.hypot([float(r["dE"]) for r in rows], [float(r["dN"]) for r in rows]) * 100
    sc = ax.scatter(E_, N_, c=mag, s=120, cmap="viridis", vmin=0, vmax=max(15, mag.max()))
    plt.colorbar(sc, label="residual shift of grid vs 87 m ortho, cm")
    ax.set_aspect("equal"); ax.set_title("Grid-vs-87 m-ortho registration residual per 30x30 m window (HEURISTIC seam-score fit)", fontsize=9)
    plt.tight_layout(); fig.savefig(OUT / "drift_17_map_87m.png", dpi=60, pil_kwargs={"optimize": True}); plt.close(fig)
    print("figs done")


def Counter_most(arr):
    from collections import Counter
    return Counter(arr).most_common(1)[0][0]


# --------------------------------------------------------------------------- 5. crops + quantification
def module_residuals(a, res, x0, y1, P, hp, max_off=0.3):
    """Per-module HEURISTIC offset: shift (within +-max_off) maximising seam brightness along that cell's edges."""
    out = []
    for p in P:
        pts = edge_points(p, res)
        best = (-1e9, 0, 0)
        for dN in np.arange(-max_off, max_off + 1e-6, res):
            for dE in np.arange(-max_off, max_off + 1e-6, res):
                s = seam_score(hp, x0, y1, res, pts, dE, dN)
                if s > best[0]:
                    best = (s, dE, dN)
        out.append(best[1:])
    return np.array(out)


def crop_stats_and_png(tag, name, cx, cy, w, h, M, res, png=None, shift=(0.0, 0.0), residuals=True):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from scipy import ndimage as ndi
    path = ortho(tag)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    a = read_rgb(path, x0, y0, x1, y1, res)
    P = M["P"]
    cen = P.mean(1) + np.array(shift)[None]
    sel = np.where((cen[:, 0] > x0 - 3) & (cen[:, 0] < x1 + 3) & (cen[:, 1] > y0 - 3) & (cen[:, 1] < y1 + 3))[0]
    Ps = P[sel]
    pm = panel_mask(a, 0.06 / res)
    mask = pm > 0.5
    F_one = [fill_polys(Ps[i:i + 1], x0, y1, res, mask.shape, shift) for i in range(len(Ps))]
    fr = np.array([(mask & f).sum() / max(f.sum(), 1) for f in F_one], dtype=float)
    valid = np.array([f.sum() > 0 and ((a[..., 3] > 200) & f).sum() / f.sum() > 0.9 for f in F_one], dtype=bool)
    inside = np.array([f.sum() > 0.9 * (PW * PH) / res ** 2 for f in F_one], dtype=bool)  # fully inside crop
    use = valid & inside
    union = np.any(F_one, 0) if len(F_one) else np.zeros(mask.shape, bool)
    union_d = ndi.binary_dilation(union, iterations=max(int(0.3 / res), 1))
    valid_px = a[..., 3] > 200
    panel_px = mask & valid_px
    # panel pixels not near any polygon, ignore tiny blobs
    lab, n = ndi.label(panel_px & ~union_d)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else np.array([])
    big = int((sizes * res * res > 2.0).sum()) if n else 0
    unc_area = float(sizes[sizes * res * res > 2.0].sum() * res * res) if big else 0.0
    # polygons mostly on ground
    ground = int((use & (fr < 0.3)).sum())
    st = dict(name=name, ortho=tag, cx=cx, cy=cy, w=w, h=h, res=res, n_cells=int(use.sum()),
              panel_frac_mean=float(fr[use].mean()) if use.any() else None,
              panel_frac_median=float(np.median(fr[use])) if use.any() else None,
              cells_panel_frac_lt_0p3=ground, cells_panel_frac_lt_0p6=int((use & (fr < 0.6)).sum()),
              panel_px_area_m2=float(panel_px.sum() * res * res),
              panel_not_covered_blobs_gt2m2=big, panel_not_covered_area_m2=round(unc_area, 1),
              panel_not_covered_frac=round(unc_area / max(panel_px.sum() * res * res, 1e-9), 4))
    hp = hp_gray(a, res)
    if residuals and use.any():
        good = np.where(use & (fr > 0.75))[0]
        r = module_residuals(a, res, x0, y1, Ps[good] + np.array(shift)[None, None], hp)
        if len(r):
            st.update(n_resid=int(len(r)), resid_dE_median=float(np.median(r[:, 0])), resid_dN_median=float(np.median(r[:, 1])),
                      resid_abs_median=float(np.median(np.hypot(*r.T))), resid_abs_p90=float(np.percentile(np.hypot(*r.T), 90)))
    if png:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20 * h / w / 2 + 0.6))
        for k in range(2):
            x = ax[k]
            x.imshow(np.clip(a[..., :3], 0, 255).astype(np.uint8), extent=(x0, x1, y0, y1))
            x.set_xticks([]); x.set_yticks([])
            if k == 1:
                x.add_collection(PolyCollection(Ps + np.array(shift)[None, None], facecolors="none", edgecolors="yellow", linewidths=0.5))
                x.set_xlim(x0, x1); x.set_ylim(y0, y1)
        ax[0].set_title(f"{tag}  centre E{cx:.0f} N{cy:.0f}  {w:.0f}x{h:.0f} m   (left: ortho only)", fontsize=9)
        ax[1].set_title(f"grid overlay (shift {shift[0]:+.2f},{shift[1]:+.2f} m); panel-like fraction in cells {st['panel_frac_mean']:.2f}" if st["panel_frac_mean"] is not None else "grid overlay", fontsize=9)
        plt.tight_layout(pad=0.3)
        fig.savefig(png, dpi=55, pil_kwargs={"optimize": True})
        plt.close(fig)
    return st


def crops():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    M = load_modules()
    cen = M["P"].mean(1)
    stats = []
    # 12 crops on the 87 m ortho: interior + edge per block
    dirs = {"Block1": (-1, 1), "Block2": (1, 1), "Block3": (-1, -1), "Block4": (1, -1), "Block5": (1, -1), "Block6": (1, 1)}
    only_low = len(sys.argv) > 2 and sys.argv[2] == "low"
    if only_low:
        stats = [dict(r) for r in csv.DictReader(open(OUT / "crop_stats.csv", encoding="utf-8")) if r["name"].startswith("87m")]
    for b in ([] if only_low else sorted(set(M["block"]))):
        idx = np.where(M["block"] == b)[0]
        c = cen[idx]
        ctr = c.mean(0)
        i_in = idx[np.argmin(np.hypot(*(c - ctr).T))]
        sx, sy = dirs[b]
        key = sx * c[:, 0] + sy * c[:, 1]
        i_ed = idx[np.argmax(key)]
        for kind, i, off in (("interior", i_in, (0, 0)), ("corner", i_ed, (-sx * 20, -sy * 10))):
            cx, cy = cen[i, 0] + off[0], cen[i, 1] + off[1]
            name = f"87m_{b}_{kind}"
            st = crop_stats_and_png("87M_MAP1", name, cx, cy, 60, 40, M, 0.03, png=TMP / f"{name}.png", residuals=True)
            st["block"] = b; st["kind"] = kind
            stats.append(st); print(st)
    # 6 low-altitude crops (grid shifted by the locally estimated offset)
    low = [("42M_T1", "Block1", None), ("42M_T3", "Block3", None), ("35M_T5", "Block4", None), ("35M_T6", "Block6", None),
           ("31M_T2", "Block2", None), ("31M_T3", "Block2", None)]
    import rasterio
    rng = np.random.default_rng(SEED + 7)
    for tag, blk, _ in low:
        with rasterio.open(ortho(tag)) as s:
            bb = s.bounds
        idx = np.where((M["block"] == blk) & (cen[:, 0] > bb.left + 20) & (cen[:, 0] < bb.right - 20) & (cen[:, 1] > bb.bottom + 14) & (cen[:, 1] < bb.top - 14))[0]
        if not len(idx):
            idx = np.where((cen[:, 0] > bb.left + 20) & (cen[:, 0] < bb.right - 20) & (cen[:, 1] > bb.bottom + 14) & (cen[:, 1] < bb.top - 14))[0]
        i = rng.choice(idx)
        cx, cy = cen[i]
        w, h = 36.0, 22.0
        near = local_shift(tag, cx, cy)
        sel = (cen[:, 0] + near[0] > cx - 18) & (cen[:, 0] + near[0] < cx + 18) & (cen[:, 1] + near[1] > cy - 11) & (cen[:, 1] + near[1] < cy + 11)
        wx, wy = cx + near[0], cy + near[1]  # where the content of this grid area appears in the ortho
        r = window_shift(ortho(tag), M["P"][sel], wx - 18, wy - 11, wx + 18, wy + 11, 0.03, near=near, rad=0.6)
        shift = (r["dE"], r["dN"])
        cx, cy = cx + shift[0], cy + shift[1]  # crop centre in ortho coordinates
        for nm, sh in (("raw", (0.0, 0.0)), ("shifted", shift)):
            name = f"{tag}_{blk}_{nm}"
            st = crop_stats_and_png(tag, name, cx, cy, w, h, M, 0.02, png=TMP / f"{name}.png", shift=sh, residuals=(nm == "shifted"))
            st["block"] = blk; st["kind"] = nm; st["est_shift_E"] = shift[0]; st["est_shift_N"] = shift[1]
            stats.append(st); print(st)
    keys = sorted({k for s in stats for k in s}, key=lambda k: (k not in ("name", "ortho", "block", "kind"), k))
    with open(OUT / "crop_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(stats)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("meta", meta), ("sample", sample), ("measure", measure), ("drift", drift), ("register", register), ("sitewide", sitewide), ("crops", crops), ("figs", figs)]:
        if cmd in (name, "all"):
            fn()
