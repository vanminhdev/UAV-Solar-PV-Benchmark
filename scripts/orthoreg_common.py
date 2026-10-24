"""Shared helpers for ortho-to-ortho control points (see ortho_control_points.py)."""
import glob, os, re
import numpy as np, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DS = os.path.join(ROOT, 'dataset')

def ortho_paths():
    d = {}
    for f in sorted(glob.glob(os.path.join(DS, 'farm_*', 'photogrammetry', '*', '*orthomosaic*.tif'))):
        b = os.path.basename(f)
        m = re.match(r'(.+?)_orthomosaic(?:_(part\d))?_EPSG32649\.tif', b)
        fid = m.group(1) + ('_' + m.group(2) if m.group(2) else '')
        d[fid] = f
    return d

def read_rgb(path, bounds, res):
    """Read bounds=(x0,y0,x1,y1) at res m/px. Returns uint8 (h,w,3) + alpha mask (h,w) bool."""
    x0, y0, x1, y1 = bounds
    w = int(round((x1 - x0) / res)); h = int(round((y1 - y0) / res))
    with rasterio.open(path) as r:
        win = from_bounds(x0, y0, x1, y1, r.transform)
        a = r.read(window=win, out_shape=(r.count, h, w), resampling=Resampling.average, boundless=True, fill_value=0)
    rgb = np.moveaxis(a[:3], 0, -1)
    valid = (a[3] > 0) if a.shape[0] > 3 else (rgb.sum(-1) > 0)
    return rgb, valid
