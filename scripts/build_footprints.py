#!/usr/bin/env python
"""Ground footprints of every UAV image, overlap report and split-leakage audit.

Sub-commands (run with ``python -X utf8 code/scripts/build_footprints.py <cmd>``):

  build                        compute footprints -> dataset/metadata/footprints/
  audit SPLIT.csv [--buffer-m] count cross-split image pairs with intersecting footprints
  selftest                     unit test of the audit with a synthetic split

Only numpy / pandas / shapely / pyproj / rasterio / matplotlib are needed. The GeoPackage is
written with the stdlib ``sqlite3`` (GeoPackage 1.2 binary geometry), so geopandas/GDAL-vector
are NOT required; the file opens in QGIS / GDAL / geopandas as usual.

Geometry model
--------------
* Work CRS EPSG:32649 (WGS 84 / UTM 49N). Image position = (GpsLongitude, GpsLatitude) of the
  XMP metadata (the camera/antenna position; for RTK flights ~cm, for BIM2 without RTK ~1-3 m).
* Yaw convention (verified, see ``verify_yaw``): GimbalYawDegree is the heading of the image
  "up" axis, in degrees CLOCKWISE from true north (DJI convention, 0 = north, 90 = east).
  Heading unit vector (E, N) = (sin psi, cos psi); image "right" = (cos psi, -sin psi).
  The image height (sensor short side) therefore lies ALONG the heading, the width ACROSS it.
  On 24/24 flights the bearing of the step to the next image equals the gimbal yaw within
  10 deg for 94-99 % of the steps (never reversed), so the yaw is the flight direction and
  the rectangles line up along the flight line.
* Footprint size (nadir): ground_w = alt * sensor_w / f ; ground_h = alt * sensor_h / f with
  alt = RelativeAltitude (height above take-off ~ AGL on flat sites, flat-terrain assumption;
  terrain relief / take-off point elevation differences are NOT modelled).
* Tilt: tilt = angle between optical axis and nadir from (GimbalPitch, GimbalRoll).
  tilt <= 2 deg: nadir rectangle (ground shift = alt*tan(tilt) < 1.1 m at 31 m).
  2 < tilt <= 35 deg: the four corner rays are intersected with the flat ground plane
  (footprint_mode = 'tilted'; roll sign convention is ambiguous but |roll| <= 4.3 deg).
  tilt > 35 deg or any corner ray above the horizon (gimbal still moving after take-off):
  nadir rectangle is used as fallback (footprint_mode = 'fallback', flagged).
* Altitude outliers (|alt - flight median| > 3 m, e.g. take-off/landing) are kept as measured
  but flagged (alt_outlier).
* Sensor models (pixel pitch/focal length): H20T wide RGB 1.55 um / 4.5 mm, H20T thermal
  12 um / 13.5 mm; Mavic 2 Enterprise Advanced thermal 12 um / 9 mm (nominal, no focal in
  EXIF), RGB 6.4 mm sensor width / 4.5 mm (nominal). Sensor height = width * px_h / px_w, so
  non-standard crops (one TT 1920x1080 image) get a full-width 16:9 FOV (assumption).
* BIM2 has no RTK (RtkFlag missing) -> flagged ``rtk_fix = False``; use a safety buffer of
  at least 3 m when auditing BIM2 splits.
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely import wkb as shp_wkb
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore", message="invalid value encountered in", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "dataset/metadata/inventory/image_inventory.csv"
OUTDIR = ROOT / "dataset/metadata/footprints"
GPKG = OUTDIR / "image_footprints.gpkg"
REPORT = OUTDIR / "overlap_report.csv"
EPSG = 32649

# (model, sensor) -> (pixel pitch um | None, sensor width mm | None, nominal focal mm)
SENSORS = {
    ("ZH20T", "W"): (1.55, None, 4.5),
    ("ZH20T", "T"): (12.0, None, 13.5),
    ("MAVIC2-ENTERPRISE-ADVANCED", "T"): (12.0, None, 9.0),
    ("MAVIC2-ENTERPRISE-ADVANCED", "W"): (None, 6.4, 4.5),
}
TILT_NADIR_MAX = 2.0
TILT_PROJ_MAX = 35.0
ALT_OUTLIER_M = 3.0
ALT_GROUP_TOL_M = 2.0  # flights whose median altitude differs by more are "different altitude"


# --------------------------------------------------------------------------- geometry
def _sensor_mm(model, sensor, w_px, h_px):
    pitch, sw, f = SENSORS[(model, sensor)]
    w = sw if sw is not None else w_px * pitch / 1000.0
    return w, w * h_px / w_px, f


def _footprint(x, y, alt, yaw, pitch, roll, sw, sh, f):
    """Return (polygon, mode, tilt_deg). x, y in metres (UTM)."""
    psi = np.radians(yaw)
    h = np.array([np.sin(psi), np.cos(psi), 0.0])
    right = np.array([np.cos(psi), -np.sin(psi), 0.0])
    p = np.radians(pitch)
    axis = np.cos(p) * h + np.sin(p) * np.array([0, 0, 1.0])  # optical axis (pitch=-90 -> down)
    up = -np.sin(p) * h + np.cos(p) * np.array([0, 0, 1.0])  # image "up"
    tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
    rr = np.radians(roll)
    if abs(roll) > 1e-9:
        # gimbal roll = rotation about the heading axis h (tilts the optical axis sideways; sign ambiguous)
        k = h
        def rot(v):
            return v * np.cos(rr) + np.cross(k, v) * np.sin(rr) + k * np.dot(k, v) * (1 - np.cos(rr))
        axis0 = np.cos(p) * h + np.sin(p) * np.array([0, 0, 1.0])
        up0 = -np.sin(p) * h + np.cos(p) * np.array([0, 0, 1.0])
        axis, up, right = rot(axis0), rot(up0), rot(np.array([np.cos(psi), -np.sin(psi), 0.0]))
        tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
    gw, gh = alt * sw / f, alt * sh / f
    nadir = lambda: Polygon(  # noqa: E731
        [(x + sx * gw / 2 * right[0] + sy * gh / 2 * h[0], y + sx * gw / 2 * right[1] + sy * gh / 2 * h[1])
         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    if tilt <= TILT_NADIR_MAX:
        return nadir(), "nadir", tilt
    if tilt <= TILT_PROJ_MAX:
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            d = f * axis + sx * sw / 2 * right + sy * sh / 2 * up
            if d[2] >= -0.2 * np.linalg.norm(d):  # ray (nearly) horizontal / above horizon
                pts = None
                break
            t = alt / -d[2]
            pts.append((x + t * d[0], y + t * d[1]))
        if pts is not None:
            poly = Polygon(pts)
            if poly.is_valid and poly.area < 20 * gw * gh:
                return poly, "tilted", tilt
    return nadir(), "fallback", tilt


def compute_footprints(inv: pd.DataFrame) -> pd.DataFrame:
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    d = inv.copy().sort_values(["flight_id", "sensor", "seq"]).reset_index(drop=True)
    d["x"], d["y"] = tr.transform(d.GpsLongitude.values, d.GpsLatitude.values)
    med = d.groupby(["flight_id", "sensor"]).RelativeAltitude.transform("median")
    bad = ~(d.RelativeAltitude > 1.0)
    d["altitude_m"] = d.RelativeAltitude.where(~bad, med)
    d["alt_outlier"] = bad | ((d.altitude_m - med).abs() > ALT_OUTLIER_M)
    d["rtk_fix"] = d.RtkFlag.eq(50)
    geoms, modes, tilts, gws, ghs = [], [], [], [], []
    for r in d.itertuples(index=False):
        sw, sh, f = _sensor_mm(r.model, r.sensor, r.width, r.height)
        if r.model == "ZH20T" and not np.isnan(r.focal_mm):
            f = r.focal_mm
        poly, mode, tilt = _footprint(r.x, r.y, r.altitude_m, r.GimbalYawDegree, r.GimbalPitchDegree,
                                      r.GimbalRollDegree, sw, sh, f)
        geoms.append(poly); modes.append(mode); tilts.append(tilt)
        gws.append(r.altitude_m * sw / f); ghs.append(r.altitude_m * sh / f)
    d["geometry"] = geoms
    d["footprint_mode"] = modes
    d["tilt_deg"] = tilts
    d["ground_w_m"] = gws  # across-track (image width)
    d["ground_h_m"] = ghs  # along-track (image height)
    d["area_m2"] = [g.area for g in geoms]
    d["yaw_deg"] = d.GimbalYawDegree
    return d


# --------------------------------------------------------------------------- GeoPackage I/O
def _gpkg_blob(geom, srid=EPSG):
    minx, miny, maxx, maxy = geom.bounds
    return (b"GP" + bytes([0, 0b00000011]) + struct.pack("<i", srid)
            + struct.pack("<4d", minx, maxx, miny, maxy) + shp_wkb.dumps(geom, byte_order=1))


def _parse_blob(b):
    flags = b[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return shp_wkb.loads(bytes(b[8 + env:]))


def write_gpkg(path: Path, layers: dict):
    """layers: name -> (DataFrame with 'geometry' col, gpkg geometry type)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("PRAGMA application_id=1196444487")  # 'GPKG'
    con.execute("PRAGMA user_version=10200")
    con.executescript("""
    CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
      organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL,
      description TEXT);
    CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
      identifier TEXT UNIQUE, description TEXT DEFAULT '',
      last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
    CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
      geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
      PRIMARY KEY (table_name, column_name));
    """)
    srs = [("Undefined cartesian SRS", -1, "NONE", -1, "undefined"),
           ("Undefined geographic SRS", 0, "NONE", 0, "undefined"),
           ("WGS 84 / UTM zone 49N", EPSG, "EPSG", EPSG, CRS.from_epsg(EPSG).to_wkt())]
    con.executemany("INSERT INTO gpkg_spatial_ref_sys(srs_name,srs_id,organization,organization_coordsys_id,"
                    "definition) VALUES (?,?,?,?,?)", srs)
    for name, (df, gtype) in layers.items():
        cols = [c for c in df.columns if c != "geometry"]
        decl = []
        for c in cols:
            s = df[c]
            t = ("INTEGER" if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s)
                 else "REAL" if pd.api.types.is_float_dtype(s) else "TEXT")
            decl.append(f'"{c}" {t}')
        con.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB, '
                    + ", ".join(decl) + ")")
        rows = []
        for rec, g in zip(df[cols].itertuples(index=False, name=None), df.geometry):
            vals = [None if (isinstance(v, float) and np.isnan(v)) else
                    (int(v) if isinstance(v, (bool, np.bool_)) else v.item() if hasattr(v, "item") else v)
                    for v in rec]
            rows.append([_gpkg_blob(g)] + vals)
        con.executemany(f'INSERT INTO "{name}" (geom,' + ",".join(f'"{c}"' for c in cols) + ") VALUES ("
                        + ",".join("?" * (len(cols) + 1)) + ")", rows)
        minx, miny, maxx, maxy = unary_union(list(df.geometry)).bounds if len(df) else (0, 0, 0, 0)
        con.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,min_x,min_y,max_x,max_y,srs_id) "
                    "VALUES (?,?,?,?,?,?,?,?)", (name, "features", name, minx, miny, maxx, maxy, EPSG))
        con.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)", (name, "geom", gtype, EPSG))
    con.commit(); con.close()


def read_gpkg_layer(path: Path, layer: str) -> pd.DataFrame:
    con = sqlite3.connect(path)
    df = pd.read_sql_query(f'SELECT * FROM "{layer}"', con)
    con.close()
    df["geometry"] = [_parse_blob(b) for b in df.pop("geom")]
    return df


# --------------------------------------------------------------------------- overlaps
def _flight_overlaps(g: pd.DataFrame) -> dict:
    """Forward / side overlap of one (flight, sensor) from neighbouring footprints (sorted by seq)."""
    x, y = g.x.values, g.y.values
    yaw = g.yaw_deg.values
    n = len(g)
    psi = np.radians(yaw)
    hx, hy = np.sin(psi), np.cos(psi)
    rx, ry = np.cos(psi), -np.sin(psi)
    gw, gh = g.ground_w_m.values, g.ground_h_m.values
    dyaw = (yaw[1:] - yaw[:-1] + 180) % 360 - 180
    step = np.hypot(np.diff(x), np.diff(y))
    brk = (np.abs(dyaw) > 45) | (step > 5 * np.median(step))
    pass_id = np.concatenate([[0], np.cumsum(brk)])
    # forward: consecutive images of the same pass, along-track distance vs footprint height
    a = (np.diff(x) * hx[:-1] + np.diff(y) * hy[:-1])
    same = ~brk
    fwd = 1 - np.abs(a[same]) / ((gh[:-1][same] + gh[1:][same]) / 2)
    fwd = np.clip(fwd, 0, 1)
    # side: nearest parallel neighbour in another pass whose along-track offset is < half height
    tree = cKDTree(np.c_[x, y])
    side = np.full(n, np.nan)
    R = 1.5 * np.hypot(gw.max(), gh.max())
    for i, nb in enumerate(tree.query_ball_point(np.c_[x, y], R)):
        nb = np.array([j for j in nb if pass_id[j] != pass_id[i]], int)
        if nb.size == 0:
            continue
        dx, dy = x[nb] - x[i], y[nb] - y[i]
        al = dx * hx[i] + dy * hy[i]
        cr = dx * rx[i] + dy * ry[i]
        par = np.abs(((yaw[nb] - yaw[i]) + 90) % 180 - 90) < 20
        ok = par & (np.abs(al) <= gh[i] / 2) & (np.abs(cr) > 0.05 * gw[i]) & (np.abs(cr) < gw[i])
        if ok.any():
            side[i] = 1 - np.abs(cr[ok]).min() / gw[i]
    pct = lambda v, q: float(np.nanpercentile(v, q)) if np.isfinite(v).any() else np.nan  # noqa: E731
    return dict(n_passes=int(pass_id.max() + 1), fwd_overlap_median=pct(fwd, 50), fwd_overlap_p10=pct(fwd, 10),
                side_overlap_median=pct(side, 50), side_overlap_p10=pct(side, 10),
                side_overlap_n_images=int(np.isfinite(side).sum()))


def verify_yaw(d: pd.DataFrame) -> pd.DataFrame:
    """Share of consecutive-image steps whose bearing equals gimbal yaw (+-10 deg) / its reverse / perpendicular."""
    out = []
    for (fid, s), g in d.groupby(["flight_id", "sensor"]):
        dx, dy = np.diff(g.x.values), np.diff(g.y.values)
        st = np.hypot(dx, dy)
        ok = st > 0.5 * np.median(st)
        diff = ((np.degrees(np.arctan2(dx, dy)) - g.yaw_deg.values[:-1]) + 180) % 360 - 180
        out.append(dict(flight_id=fid, sensor=s, steps=int(ok.sum()),
                        fwd=float((np.abs(diff) < 10)[ok].mean()),
                        reversed=float((np.abs(np.abs(diff) - 180) < 10)[ok].mean()),
                        perpendicular=float((np.abs(np.abs(diff) - 90) < 10)[ok].mean())))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- ortho sanity check
def ortho_bounds() -> dict:
    import rasterio
    from rasterio.warp import transform_bounds
    res = {}
    for p in (ROOT / "dataset").glob("farm_*/photogrammetry/*/*orthomosaic*.tif"):
        with rasterio.open(p) as ds:  # header only
            b = ds.bounds
            if ds.crs and ds.crs.to_epsg() != EPSG:
                b = transform_bounds(ds.crs, f"EPSG:{EPSG}", *b)
        res.setdefault(p.parent.name, []).append(box(*b))
    return {k: unary_union(v) for k, v in res.items()}


# --------------------------------------------------------------------------- figures
def plot_farm(farm, unions: pd.DataFrame, share, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    u = unions[(unions.farm == farm) & (unions.sensor == "T")].copy()
    u["grp"] = u.altitude_median_m.round().astype(int)
    w = unions[(unions.farm == farm) & (unions.sensor == "W")].copy()
    w["grp"] = w.altitude_median_m.round().astype(int)
    w = w[~w.grp.isin(u.grp)]  # RGB-only altitudes (e.g. 87 m MAP flights): dashed outline
    grps = sorted(u.grp.unique(), reverse=True)  # draw highest first -> it ends up outermost/at the bottom
    cmap = plt.get_cmap("viridis")
    col = {gv: cmap(0.9 * i / max(1, len(grps) - 1)) for i, gv in enumerate(grps)}
    fig, ax = plt.subplots(figsize=(9, 7.5))
    allg = list(u.geometry) + list(w.geometry)
    x0 = min(g.bounds[0] for g in allg); y0 = min(g.bounds[1] for g in allg)
    from shapely import affinity
    for z, gv in enumerate(grps):
        for geom in u[u.grp == gv].geometry:
            geom = affinity.translate(geom, -x0, -y0)
            for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                xs, ys = poly.exterior.xy
                ax.fill(xs, ys, color=col[gv], alpha=0.30, lw=0, zorder=z)
                ax.plot(xs, ys, color=col[gv], lw=1.4, zorder=z + 0.5)
    wgrps = sorted(w.grp.unique(), reverse=True)
    for gv in wgrps:
        for geom in w[w.grp == gv].geometry:
            geom = affinity.translate(geom, -x0, -y0)
            for poly in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color="0.35", lw=1.3, ls="--", zorder=-1)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("Easting - %.0f m (EPSG:32649)" % x0); ax.set_ylabel("Northing - %.0f m" % y0)
    handles = [Patch(facecolor=col[gv], edgecolor=col[gv], alpha=0.6,
                     label=f"{gv:.0f} m AGL ({(u.grp == gv).sum()} flight(s))") for gv in grps]
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color="0.35", ls="--", label=f"{gv:.0f} m AGL, RGB only ({(w.grp == gv).sum()} flight(s))") for gv in wgrps]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              title="Flight footprint union\n(filled = thermal; outermost =\nhighest altitude)")
    ax.set_title(f"{farm}: area flown at >1 altitude = {share:.0%} of the union", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)


def _farm_summary(farm, f, sensor):
    """Union area, and share of it flown at >1 altitude (altitude group = median altitude rounded to 1 m)."""
    groups = {}
    for r in f.itertuples():
        groups.setdefault(int(round(r.altitude_median_m)), []).append(r.geometry)
    gu = {k: unary_union(v) for k, v in groups.items()}
    ks = sorted(gu)
    multi = unary_union([gu[a].intersection(gu[b]) for i, a in enumerate(ks) for b in ks[i + 1:]])         if len(ks) > 1 else Polygon()
    tot = unary_union(list(gu.values()))
    return dict(record_type="farm_summary_" + ("thermal" if sensor == "T" else "rgb_info"), farm=farm,
                sensor=sensor, n_images=int(f.n_images.sum()), n_altitude_groups=len(ks),
                union_area_m2=round(tot.area, 1), multi_altitude_area_m2=round(multi.area, 1),
                share_multi_altitude=multi.area / tot.area, altitude_list_m=" ".join(map(str, ks)))


# --------------------------------------------------------------------------- build
def build():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVENTORY)
    d = compute_footprints(inv)
    yv = verify_yaw(d)
    print("yaw verification (share of steps along gimbal yaw, min over flights): fwd=%.2f  reversed max=%.2f"
          % (yv.fwd.min(), yv.reversed.max()))

    # ortho sanity check
    ob = ortho_bounds()
    cx = []
    for r in d.itertuples():
        b = ob.get(r.flight_id)
        cx.append((np.nan, np.nan) if b is None else
                  (float(b.contains(box(r.x, r.y, r.x, r.y).centroid)),
                   1 - r.geometry.intersection(b).area / r.geometry.area))
    d["centre_in_ortho"] = [c[0] for c in cx]
    d["frac_outside_ortho"] = [c[1] for c in cx]

    # layers
    cols = ["farm", "flight_id", "file", "sensor", "altitude_m", "area_m2", "ground_w_m", "ground_h_m",
            "yaw_deg", "tilt_deg", "footprint_mode", "alt_outlier", "rtk_fix", "centre_in_ortho",
            "frac_outside_ortho", "geometry"]
    layer1 = d[cols].copy()
    layer1["altitude_m"] = layer1.altitude_m.round(3)
    uni = []
    for (fid, s), g in d.groupby(["flight_id", "sensor"]):
        geom = unary_union(list(g.geometry))
        med = float(g.altitude_m.median())
        uni.append(dict(farm=g.farm.iloc[0], flight_id=fid, sensor=s, n_images=len(g),
                        altitude_median_m=round(med, 2), altitude_group_m=round(med),
                        area_m2=geom.area, geometry=geom))
    uni = pd.DataFrame(uni)
    write_gpkg(GPKG, {"image_footprints": (layer1, "POLYGON"), "flight_union": (uni, "GEOMETRY")})

    # ---------------- overlap report
    rows = []
    for (fid, s), g in d.groupby(["flight_id", "sensor"]):
        r = dict(record_type="flight_sensor", farm=g.farm.iloc[0], flight_id=fid, sensor=s, n_images=len(g),
                 altitude_median_m=round(g.altitude_m.median(), 2),
                 footprint_w_m_median=round(g.ground_w_m.median(), 2), footprint_h_m_median=round(g.ground_h_m.median(), 2),
                 n_alt_outlier=int(g.alt_outlier.sum()), n_tilt_gt2=int((g.tilt_deg > TILT_NADIR_MAX).sum()),
                 n_fallback=int((g.footprint_mode == "fallback").sum()), rtk_fix_pct=round(100 * g.rtk_fix.mean(), 1))
        r.update(_flight_overlaps(g))
        if g.centre_in_ortho.notna().any():
            r["n_centre_outside_ortho"] = int((g.centre_in_ortho == 0).sum())
            r["n_gt50pct_outside_ortho"] = int((g.frac_outside_ortho > 0.5).sum())
        rows.append(r)

    tu = uni[uni.sensor == "T"]
    shares = {}
    for farm, f in tu.groupby("farm"):
        f = f.reset_index(drop=True)
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                a, b = f.loc[i], f.loc[j]
                if abs(a.altitude_median_m - b.altitude_median_m) <= ALT_GROUP_TOL_M:
                    continue
                inter = a.geometry.intersection(b.geometry).area
                un = a.geometry.union(b.geometry).area
                rows.append(dict(record_type="flight_pair_thermal", farm=farm, flight_id=a.flight_id,
                                 flight_id_b=b.flight_id, sensor="T", altitude_median_m=a.altitude_median_m,
                                 altitude_b_median_m=b.altitude_median_m, inter_area_m2=round(inter, 1),
                                 iou=inter / un, frac_of_a=inter / a.geometry.area, frac_of_b=inter / b.geometry.area))
        rows.append(_farm_summary(farm, f, "T"))
        shares[farm] = rows[-1]["share_multi_altitude"]
    for farm, f in uni[uni.sensor == "W"].groupby("farm"):  # informational: RGB incl. the 87 m MAP flights
        rows.append(_farm_summary(farm, f.reset_index(drop=True), "W"))
    pd.DataFrame(rows).to_csv(REPORT, index=False, float_format="%.4f")

    for farm in shares:
        plot_farm(farm, uni, shares[farm], OUTDIR / f"overlap_{farm}.png")

    # console summary
    print(d.groupby("farm").size().rename("images (all sensors)").to_string())
    s = d.groupby(["farm", "sensor", d.altitude_m.round().astype(int)]).agg(
        n=("area_m2", "size"), w=("ground_w_m", "median"), h=("ground_h_m", "median"), area=("area_m2", "median"))
    print(s.round(1).to_string())
    print(d.footprint_mode.value_counts().to_string(), "\nalt_outliers:", int(d.alt_outlier.sum()))
    print("ortho: centres outside", int((d.centre_in_ortho == 0).sum()), "of", int(d.centre_in_ortho.notna().sum()))
    print("wrote", GPKG, REPORT)


# --------------------------------------------------------------------------- audit
def audit_split(split_csv, buffer_m: float = 0.0, gpkg: Path = GPKG, verbose: bool = True) -> dict:
    """Count cross-split image pairs whose footprints intersect (or lie within ``buffer_m`` metres).

    split_csv : csv with columns ``file,split`` (split in train/val/test); an optional ``flight_id``
                column disambiguates files that exist in several flights (BIM2 reuses DJI_000x names;
                without flight_id such files raise ValueError).
    Pairs are counted over ALL sensors and flights (a thermal and an RGB frame of the same place in
    different splits is leakage as well). Returns a dict (also printed when verbose).
    """
    sp = pd.read_csv(split_csv) if not isinstance(split_csv, pd.DataFrame) else split_csv.copy()
    if not {"file", "split"} <= set(sp.columns):
        raise ValueError("split csv needs columns file,split")
    bad = set(sp.split) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"unknown split values {bad}")
    fp = read_gpkg_layer(gpkg, "image_footprints")
    keys = ["flight_id", "file"] if "flight_id" in sp.columns else ["file"]
    if keys == ["file"]:
        dup = fp.groupby("file").size()
        amb = sp[sp.file.isin(dup[dup > 1].index)]
        if len(amb):
            raise ValueError(f"{len(amb)} split rows have file names that exist in several flights "
                             f"(e.g. {amb.file.iloc[0]}); add a flight_id column")
    m = sp[keys + ["split"]].drop_duplicates().merge(fp[keys + ["geometry"]], on=keys, how="left")
    n_missing = int(m.geometry.isna().sum())
    m = m.dropna(subset=["geometry"]).reset_index(drop=True)
    if m.duplicated(keys).any():
        raise ValueError("an image is assigned to more than one split")
    geoms = m.geometry.values
    tree = STRtree(geoms)
    if buffer_m > 0:
        i, j = tree.query(geoms, predicate="dwithin", distance=buffer_m)
    else:
        i, j = tree.query(geoms, predicate="intersects")
    k = i < j
    i, j = i[k], j[k]
    s = m.split.values
    cross = s[i] != s[j]
    ci, cj = i[cross], j[cross]
    pair_names = {}
    for a, b in zip(s[ci], s[cj]):
        key = "-".join(sorted((a, b), key=["train", "val", "test"].index))
        pair_names[key] = pair_names.get(key, 0) + 1
    touched = np.unique(np.r_[ci, cj])
    res = dict(n_images_audited=len(m), n_missing_footprint=n_missing, buffer_m=buffer_m,
               n_pairs_same_split=int((~cross).sum()), n_cross_split_pairs=int(cross.sum()),
               pairs_by_split_pair=pair_names,
               n_images_with_cross_split_neighbour={sp_: int((s[touched] == sp_).sum())
                                                    for sp_ in ("train", "val", "test")},
               share_val_test_images_leaking=float(np.isin(np.flatnonzero(s != "train"), touched).mean())
               if (s != "train").any() else 0.0)
    if verbose:
        print(f"[audit] {len(m)} images, buffer {buffer_m} m -> {res['n_cross_split_pairs']} cross-split "
              f"intersecting pairs {pair_names}; images touching another split: "
              f"{res['n_images_with_cross_split_neighbour']}; missing footprints: {n_missing}")
    return res


def selftest():
    """Synthetic split: ten 10x10 m squares every 8 m along a line; train = 0-4, test = 5-9."""
    n = 10
    df = pd.DataFrame(dict(farm="S", flight_id="S_1", file=[f"i{k}.JPG" for k in range(n)], sensor="T",
                           geometry=[box(8 * k, 0, 8 * k + 10, 10) for k in range(n)]))
    df["split"] = ["train"] * 5 + ["test"] * 5
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "t.gpkg"
        write_gpkg(g, {"image_footprints": (df.drop(columns="split"), "POLYGON")})
        back = read_gpkg_layer(g, "image_footprints")
        assert all(a.equals(b) for a, b in zip(back.geometry, df.geometry)), "gpkg roundtrip"
        sp = df[["file", "split"]]
        r0 = audit_split(sp, 0.0, g, verbose=False)
        assert r0["n_cross_split_pairs"] == 1, r0           # only (4,5) overlap
        r1 = audit_split(sp, 6.5, g, verbose=False)          # i, i+2 are 6 m apart -> 3 pairs
        assert r1["n_cross_split_pairs"] == 3, r1
        sp2 = sp.copy(); sp2["split"] = ["train", "test"] * 5  # interleaved -> every neighbour leaks
        assert audit_split(sp2, 0.0, g, verbose=False)["n_cross_split_pairs"] == 9
        sp3 = sp.copy(); sp3["split"] = "train"
        assert audit_split(sp3, 0.0, g, verbose=False)["n_cross_split_pairs"] == 0
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build")
    a = sub.add_parser("audit"); a.add_argument("split_csv"); a.add_argument("--buffer-m", type=float, default=0.0)
    a.add_argument("--gpkg", default=str(GPKG))
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "audit":
        audit_split(args.split_csv, args.buffer_m, Path(args.gpkg))
    elif args.cmd == "selftest":
        selftest()
    else:
        build()
