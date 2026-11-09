#!/usr/bin/env python
"""Leakage-aware split protocols for the UAV PV dataset (deterministic, seed fixed).

    python -X utf8 code/scripts/make_splits.py build            # zones, all protocols, farm txt lists
    python -X utf8 code/scripts/make_splits.py audit [--no-phash]   # audit every protocol -> audit_summary.csv
    python -X utf8 code/scripts/make_splits.py figures          # one figure per protocol
    python -X utf8 code/scripts/make_splits.py all              # build + audit + figures
    python -X utf8 code/scripts/make_splits.py label_aware      # SUPERSEDED (ortho-box based, v1; outputs archived outside the release):
                                                                # use make_frame_label_split.py for the labelled frames

Unit of splitting = RGB/thermal PAIR (pair_id). A pair is never split. Footprint of a pair = union
of its thermal and RGB ground footprints (image_footprints.gpkg, flat-terrain model).
Mapping flights (``*_MAP*``, RGB only, 87-110 m) are not part of any split (photogrammetry only).

Protocols (names = csv file names in dataset/metadata/splits/):
  A  main_spatial_buf{05,10,15,20}   TN19, 24 sheared strip tiles ("zones"), tiles -> train/val/test
  B1 xalt_shared_<src>to<tgt>        TN19 blocks A-C; train src altitude, TEST tgt altitude on the SAME
                                     modules (train zones). Deliberately shares physical modules.
  B2 xalt_strict_<src>to<tgt>        same, but test = tgt altitude in the disjoint TEST zones (no sharing)
  C  xfarm_tn19_to_{tndl,tt,bim2}    train/val TN19, test = whole other farm
  D  tndl_xalt_{shared,strict}_<src>to<tgt>   TNDL 30 m vs 35 m, strips along the long axis
Rule (A..D): a pair is kept in split s iff its footprint buffered by b metres meets only zones of split s,
otherwise ``buffer_excluded``. Hence the footprint of any kept pair of another split is >= b m away.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.affinity import affine_transform
from shapely.geometry import Point, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_footprints as bf  # noqa: E402

ROOT = bf.ROOT
DS = ROOT / "dataset"
OUT = DS / "metadata/splits"
FARM_DIR = {"TN19": DS / "farm_01_thuan_nam_19", "BIM2": DS / "farm_02_bim_2",
            "TNDL": DS / "farm_03_thuan_nam_duc_long", "TT": DS / "farm_04_thien_tan"}
MASTER = FARM_DIR["TN19"] / "module_reference/TN19_modules_master.gpkg"
SEED = 20260421
TARGET = np.array([0.60, 0.15, 0.25])
SPLITS = ["train", "val", "test"]
BUFFERS = [5, 10, 15, 20]
CHOSEN_BUFFER = 10
ALTS = [31, 35, 42]

# TN19 zoning: sheared frame u = x - K*(y - V0), v = y. Strings are E-W rows slanted ~22 deg from the
# grid axes (left edges of Block2/Block4 strings: dx/dy = 0.405 / 0.399); cut lines in u run along the
# module-free alleys between plant blocks (Block1|Block2 at u~265396, Block2|Block4 at ~265577,
# Block4|Block6 at ~265757) and through the block interiors; v cuts separate the flown blocks C|B|A.
K_SHEAR, V0 = 0.40, 1263600.0
U_CUTS = [265396.0, 265488.0, 265577.0, 265667.0, 265757.0]
V_CUTS = [1263636.0, 1263739.0, 1263816.0]
V_NAMES = ["S", "C", "B", "A"]  # south of C | C | B | A (north)


# ----------------------------------------------------------------------------- data
def alt_of(flight_id: str) -> int:
    return int(re.search(r"_(\d+)M_", flight_id).group(1))


def load_units(farms=("TN19", "TNDL", "TT", "BIM2")):
    """Return (units, imgs). units: one row per pair, imgs: one row per image (thermal / rgb)."""
    inv = pd.read_csv(bf.INVENTORY)
    fp = bf.read_gpkg_layer(bf.GPKG, "image_footprints")
    pr = pd.read_csv(ROOT / "dataset/metadata/pairing/rgb_thermal_pairing.csv")
    pr = pr[pr.farm.isin(farms) & ~pr.flight_id.str.contains("_MAP")]
    rows = []
    for r in pr.itertuples(index=False):
        for col, sensor in (("thermal_file", "T"), ("rgb_file", "W")):
            f = getattr(r, col)
            if isinstance(f, str):
                rows.append((r.farm, r.flight_id, f, sensor, r.pair_id))
    imgs = pd.DataFrame(rows, columns=["farm", "flight_id", "file", "sensor", "pair_id"])
    imgs = imgs.merge(fp[["flight_id", "file", "geometry"]], on=["flight_id", "file"], how="left")
    imgs = imgs.merge(inv[["flight_id", "file", "block", "timestamp"]], on=["flight_id", "file"], how="left")
    assert imgs.geometry.notna().all(), "image without footprint"
    imgs["altitude_m"] = imgs.flight_id.map(alt_of)
    g = imgs.groupby("pair_id")
    units = g.agg(farm=("farm", "first"), flight_id=("flight_id", "first"), altitude_m=("altitude_m", "first"),
                  block=("block", "first"), timestamp=("timestamp", "min"),
                  geometry=("geometry", lambda s: unary_union(list(s)))).reset_index()
    units["cx"] = [g_.centroid.x for g_ in units.geometry]
    units["cy"] = [g_.centroid.y for g_ in units.geometry]
    units = units.sort_values(["farm", "flight_id", "pair_id"]).reset_index(drop=True)
    imgs = imgs.sort_values(["farm", "flight_id", "pair_id", "sensor"]).reset_index(drop=True)
    return units, imgs


def load_cells():
    import sqlite3
    c = sqlite3.connect(MASTER)
    m = pd.read_sql("select layer, module_id, Tam_X x, Tam_Y y, geom from modules", c)
    m["geometry"] = [bf._parse_blob(b) for b in m.pop("geom")]
    return m


# ----------------------------------------------------------------------------- zoning
class Strips:
    """Three contiguous strips (zones) bounded by two parallel cut lines f(x,y) = const,
    f = a*x + b*y + c. ``order`` gives the split owning each strip (west->east / low->high f).
    Every strip is further divided into equal sub-zones (~``sub_w`` m wide) only for labelling
    (zone_id); splits are decided at strip level, so same-split neighbours never cost images."""

    def __init__(self, prefix, frame, cuts, order, extent, sub_w=60.0):
        self.prefix, self.frame, self.cuts, self.order = prefix, frame, tuple(cuts), tuple(order)
        a, b, _ = frame
        self.norm = float(np.hypot(a, b))
        self.edges = [extent[0], *cuts, extent[1]]          # f-units are metres (|(a,b)| = 1 normalised)
        self.sub = []                                       # (zone_id, f0, f1, split)
        k = 1
        for i, s in enumerate(order):
            w = self.edges[i + 1] - self.edges[i]
            n = max(1, int(round(w / sub_w)))
            for j in range(n):
                f0 = self.edges[i] + w * j / n
                f1 = self.edges[i] + w * (j + 1) / n
                self.sub.append((f"{prefix}-Z{k:02d}", f0, f1, s))
                k += 1

    def f(self, x, y):
        a, b, c = self.frame
        return (a * x + b * y + c) / self.norm

    def extent_of(self, geoms, buf):
        bg = shapely.buffer(np.asarray(geoms, dtype=object), buf) if buf > 0 else np.asarray(geoms, dtype=object)
        lo = np.empty(len(bg)); hi = np.empty(len(bg))
        for i, g in enumerate(bg):
            xy = np.asarray(g.exterior.coords) if g.geom_type == "Polygon" else np.concatenate(
                [np.asarray(p.exterior.coords) for p in g.geoms])
            fv = self.f(xy[:, 0], xy[:, 1])
            lo[i], hi[i] = fv.min(), fv.max()
        return lo, hi

    def seg_of(self, f):
        return np.searchsorted(np.asarray(self.cuts), f, side="right")

    def assign(self, units, buf, ext=None):
        lo, hi = ext if ext is not None else self.extent_of(units.geometry.values, buf)
        sl, sh = self.seg_of(lo), self.seg_of(hi)
        split = np.where(sl == sh, np.array(self.order, dtype=object)[sl], "buffer_excluded")
        fc = self.f(units.cx.values, units.cy.values)
        eds = np.array([e for _, e, _, _ in self.sub][1:])
        zid = [self.sub[i][0] for i in np.searchsorted(eds, fc, side="right")]
        return pd.DataFrame({"pair_id": units.pair_id.values, "zone_id": zid, "split": split})

    def polygons(self, ref, half=250.0):
        """zone_id -> (polygon, split) for plotting / documentation (strip polygons in EPSG:32649)."""
        a, b, c = self.frame
        n = np.array([a, b]) / self.norm
        d = np.array([-n[1], n[0]])
        ref = np.asarray(ref, float)
        t0 = float(ref @ d)
        out = {}
        for zid, f0, f1, s in self.sub:
            p0, p1 = n * f0 - n * c / self.norm, n * f1 - n * c / self.norm
            p0, p1 = p0 + t0 * d, p1 + t0 * d
            out[zid] = (shapely.Polygon([p0 - half * d, p0 + half * d, p1 + half * d, p1 - half * d]), s)
        return out


PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def search_cuts(frame, lo, hi, score, step=2.5, min_gap=0.0):
    """Exhaustive search of two cut positions and the split order of the 3 strips.
    ``score(seg_lo, seg_hi, order) -> (cost, info)``. Deterministic (first minimum wins)."""
    grid = np.arange(np.floor(lo.min()), np.ceil(hi.max()) + step, step)
    best = (1e18, None)
    cut_arr = grid
    for i in range(len(grid) - 1):
        for j in range(i + 1, len(grid)):
            c = np.array([grid[i], grid[j]])
            if c[1] - c[0] < min_gap:
                continue
            sl = np.searchsorted(c, lo, side="right"); sh = np.searchsorted(c, hi, side="right")
            for order in PERMS:
                cost, info = score(sl, sh, order)
                if cost < best[0] - 1e-12:
                    best = (cost, (c, order, info))
    return best


def score_main(alt, w_alt=3.0):
    """cost = 3*L1(kept share - 60/15/25) + 4*loss share + w_alt*sum_s share_s*L1(altitude mix_s - overall mix).
    The altitude term keeps every split (especially test) representative of the 31/35/42 m mix."""
    levels = sorted(set(alt.tolist()))
    overall = np.array([(alt == a).mean() for a in levels])

    def f(sl, sh, order):
        kept = sl == sh
        sp = np.array(order)[sl]
        cnt = np.bincount(sp[kept], minlength=3).astype(float)
        if (cnt == 0).any():
            return 1e9, None
        c = 3 * np.abs(cnt / cnt.sum() - TARGET).sum() + 4 * (1 - kept.mean())
        for s_ in range(3):
            m = kept & (sp == s_)
            d = np.array([(alt[m] == a).mean() for a in levels])
            c += w_alt * (cnt[s_] / cnt.sum()) * np.abs(d - overall).sum()
        return c, cnt.astype(int).tolist()
    return f


def score_xalt(alt, src, tgt, min_n=20):
    def f(sl, sh, order):
        kept = sl == sh
        sp = np.array(order)[sl]
        ks, kt = kept & (alt == src), kept & (alt == tgt)
        n3 = np.array([(ks & (sp == 0)).sum(), (ks & (sp == 1)).sum(), (kt & (sp == 2)).sum()], float)
        if n3.min() < min_n:
            return 1e9, None
        return np.abs(n3 / n3.sum() - TARGET).sum(), n3.astype(int).tolist()
    return f


def tn19_frame():
    return (1.0, -K_SHEAR, K_SHEAR * V0)


def tn19_main(units):
    """Optimise the 3-strip zoning of TN19 on all inspection pairs (buffer = CHOSEN_BUFFER)."""
    u = units[units.farm == "TN19"].reset_index(drop=True)
    fr = tn19_frame()
    probe = Strips("TN19", fr, (0, 1), (0, 1, 2), (0, 1))
    lo, hi = probe.extent_of(u.geometry.values, CHOSEN_BUFFER)
    cost, (cuts, order, info) = search_cuts(fr, lo, hi, score_main(u.altitude_m.values), step=2.5)
    fc = probe.f(u.cx.values, u.cy.values)
    st = Strips("TN19", fr, cuts, [SPLITS[i] for i in order], (fc.min() - 5, fc.max() + 5))
    return u, st, cost, info


# ----------------------------------------------------------------------------- csv builders
COLS = ["farm", "flight_id", "file", "sensor", "pair_id", "split", "zone_id", "altitude_m"]


def to_rows(imgs, pair_split):
    """pair_split: DataFrame(pair_id, split, zone_id) -> image-level rows."""
    r = imgs.merge(pair_split, on="pair_id", how="inner")
    return r[COLS + ["geometry"]].sort_values(["flight_id", "file"]).reset_index(drop=True)


def write_csv(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    rows[COLS].to_csv(OUT / f"{name}.csv", index=False)


def xalt_zoning(u_sub, frame, src, tgt, b, prefix, extent_pad=5):
    """Direction-specific 3-strip zoning optimised only on pairs of the two altitudes involved."""
    sub = u_sub[u_sub.altitude_m.isin([src, tgt])].reset_index(drop=True)
    probe = Strips(prefix, frame, (0, 1), (0, 1, 2), (0, 1))
    lo, hi = probe.extent_of(sub.geometry.values, b)
    cost, (cuts, order, info) = search_cuts(frame, lo, hi, score_xalt(sub.altitude_m.values, src, tgt), step=2.5)
    fc = probe.f(sub.cx.values, sub.cy.values)
    return Strips(prefix, frame, cuts, [SPLITS[i] for i in order], (fc.min() - extent_pad, fc.max() + extent_pad),
                  sub_w=1e9), cost, info


def xalt_rows(u_sub, strips, src, tgt, kind, b):
    a = strips.assign(u_sub, b)
    a = a.merge(u_sub[["pair_id", "altitude_m"]], on="pair_id")
    s = a[a.altitude_m == src]
    t = a[a.altitude_m == tgt]
    tz = "train" if kind == "shared" else "test"
    tt = t[t.split == tz].copy()
    tt["split"] = "test"
    ps = pd.concat([s[s.split.isin(["train", "val", "buffer_excluded"])], tt, t[t.split == "buffer_excluded"]],
                   ignore_index=True)
    return ps[["pair_id", "zone_id", "split"]]


def build_all():
    units, imgs = load_units()
    cells = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    u19, st, cost, info = tn19_main(units)
    print(f"[zones] cuts (f, m) {np.round(st.cuts, 1)}, order {st.order}, cost {cost:.3f}, kept per split {info}")
    protocols, strips = {}, {"main": st}
    ass = {}
    for b in BUFFERS:
        a = st.assign(u19, b)
        ass[b] = a
        rows = to_rows(imgs[imgs.farm == "TN19"], a)
        write_csv(rows, f"main_spatial_buf{b:02d}")
        protocols[f"main_spatial_buf{b:02d}"] = rows
        n = a.split.value_counts()
        print(f"[A] buffer {b:2d} m: " + ", ".join(f"{s}={n.get(s, 0)}" for s in SPLITS + ["buffer_excluded"]) + " pairs")
    rows = protocols[f"main_spatial_buf{CHOSEN_BUFFER:02d}"]
    sp_dir = FARM_DIR["TN19"] / "splits"
    for s in SPLITS + ["buffer_excluded"]:
        sub = rows[rows.split == s]
        paths = sorted(f"raw_images/altitude_{alt_of(f)}m_inspection/{f}/{fn}"
                       for f, fn in zip(sub.flight_id, sub.file))
        (sp_dir / f"{s}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    # ---- B: cross altitude, blocks A-C, direction-specific zoning
    acb = u19[u19.block.isin(list("ABC"))].reset_index(drop=True)
    for src in ALTS:
        for tgt in ALTS:
            if src == tgt:
                continue
            sz, c2, inf = xalt_zoning(acb, tn19_frame(), src, tgt, CHOSEN_BUFFER, f"X{src}to{tgt}")
            strips[f"xalt_{src}to{tgt}"] = sz
            print(f"[B] {src}->{tgt}: cuts {np.round(sz.cuts, 1)} order {sz.order} (train,val,test pairs) {inf}")
            for kind in ("shared", "strict"):
                ps = xalt_rows(acb, sz, src, tgt, kind, CHOSEN_BUFFER)
                name = f"xalt_{kind}_{src}to{tgt}"
                protocols[name] = to_rows(imgs[imgs.farm == "TN19"], ps)
                write_csv(protocols[name], name)
    # ---- C: cross farm
    main = protocols[f"main_spatial_buf{CHOSEN_BUFFER:02d}"].copy()
    main["split"] = main.split.replace({"test": "train"})     # test is another farm: test-zone strip joins train
    for farm in ("TNDL", "TT", "BIM2"):
        o = imgs[imgs.farm == farm].copy()
        o["split"], o["zone_id"] = "test", f"{farm}-all"
        r = pd.concat([main, o[COLS + ["geometry"]]], ignore_index=True)
        name = f"xfarm_tn19_to_{farm.lower()}"
        write_csv(r, name)
        protocols[name] = r
    # ---- D: TNDL 30 vs 35 m
    uT = units[units.farm == "TNDL"].reset_index(drop=True)
    c = np.c_[uT.cx, uT.cy]
    c0 = c.mean(0)
    w, v = np.linalg.eigh(np.cov((c - c0).T))
    ax = v[:, 1] if v[0, 1] > 0 else -v[:, 1]
    frame = (ax[0], ax[1], -float(c0 @ ax))
    for src, tgt in ((30, 35), (35, 30)):
        sz, c2, inf = xalt_zoning(uT, frame, src, tgt, CHOSEN_BUFFER, f"TNDL{src}to{tgt}")
        strips[f"tndl_{src}to{tgt}"] = sz
        print(f"[D] {src}->{tgt}: cuts {np.round(sz.cuts, 1)} order {sz.order} {inf}")
        for kind in ("shared", "strict"):
            ps = xalt_rows(uT, sz, src, tgt, kind, CHOSEN_BUFFER)
            name = f"tndl_xalt_{kind}_{src}to{tgt}"
            protocols[name] = to_rows(imgs[imgs.farm == "TNDL"], ps)
            write_csv(protocols[name], name)
    # zone tables
    zt = []
    for key, z in strips.items():
        for zid, f0, f1, s in z.sub:
            zt.append(dict(zoning=key, zone_id=zid, split=s, coord_from_m=round(f0 * z.norm, 1), coord_to_m=round(f1 * z.norm, 1)))
    pd.DataFrame(zt).to_csv(OUT / "zones.csv", index=False)
    return units, imgs, cells, strips, protocols


# ----------------------------------------------------------------------------- audit
def phash(path, size=32):
    from PIL import Image
    from scipy.fft import dctn
    im = Image.open(path)
    im.draft("L", (size * 8, size * 8))
    a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), float)
    d = dctn(a, norm="ortho")[:8, :8].ravel()
    return (d > np.median(d[1:])).astype(np.uint8)


def img_path(farm, flight_id, f):
    return FARM_DIR[farm] / f"raw_images/altitude_{alt_of(flight_id)}m_inspection/{flight_id}/{f}"


def near_dup_check(rows, units, frac=0.05, seed=SEED):
    """pHash on a 5 % sample of cross-split nearest-neighbour pairs (val/test unit -> nearest train unit),
    compared with within-train nearest neighbours and random cross-split pairs. Thermal + RGB."""
    from scipy.spatial import cKDTree
    pu = rows.drop_duplicates("pair_id").merge(units[["pair_id", "cx", "cy"]], on="pair_id")
    pu = pu[pu.split.isin(SPLITS)].reset_index(drop=True)
    tr = pu[pu.split == "train"].reset_index(drop=True)
    te = pu[pu.split != "train"].reset_index(drop=True)
    tree = cKDTree(np.c_[tr.cx, tr.cy])
    d, j = tree.query(np.c_[te.cx, te.cy])
    rng = np.random.default_rng(seed)
    n = max(10, int(round(frac * len(te))))
    pick = rng.choice(len(te), n, replace=False)
    cross = [(te.pair_id[i], tr.pair_id[j[i]], d[i]) for i in pick]
    d2, j2 = tree.query(np.c_[tr.cx, tr.cy], k=2)       # within-train NN (k=1 is itself)
    pk = rng.choice(len(tr), n, replace=False)
    within = [(tr.pair_id[i], tr.pair_id[j2[i, 1]], d2[i, 1]) for i in pk]
    rnd = [(te.pair_id[i], tr.pair_id[rng.integers(len(tr))], np.nan) for i in pick]
    files = rows.set_index(["pair_id", "sensor"])

    def ham(p1, p2, sensor):
        r1, r2 = files.loc[(p1, sensor)], files.loc[(p2, sensor)]
        h1 = phash(img_path(r1.farm, r1.flight_id, r1.file))
        h2 = phash(img_path(r2.farm, r2.flight_id, r2.file))
        return int((h1 != h2).sum())

    res = {}
    for name, lst in (("cross_split_NN", cross), ("within_train_NN", within), ("random_cross", rnd)):
        for sensor in ("T", "W"):
            hs = []
            for p1, p2, _ in lst:
                try:
                    hs.append(ham(p1, p2, sensor))
                except KeyError:
                    pass
            res[(name, sensor)] = hs
    out = []
    for (name, sensor), hs in res.items():
        hs = np.array(hs)
        out.append(dict(group=name, sensor=sensor, n=len(hs), ham_median=float(np.median(hs)),
                        ham_p10=float(np.percentile(hs, 10)), share_le6=float((hs <= 6).mean()), share_le10=float((hs <= 10).mean())))
    dd = np.array([x[2] for x in cross])
    return pd.DataFrame(out), dict(n_sampled=n, nn_dist_median_m=float(np.median(dd)), nn_dist_min_m=float(dd.min()))


def cell_sharing(rows, cells, tree):
    """reference cells intersecting kept footprints: shared between splits (image-level footprints)."""
    k = rows[rows.split.isin(SPLITS)]
    geoms = k.geometry.values
    i, j = tree.query(geoms, predicate="intersects")        # (geom idx, cell idx)
    sp = k.split.values[i]
    cellsets = {s: set(j[sp == s]) for s in SPLITS}
    shared = {f"{a}&{b}": len(cellsets[a] & cellsets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    return shared, {s: len(cellsets[s]) for s in SPLITS}, cellsets


def balance(rows, units):
    """per-split pair count, altitude mix, time-of-day, footprint area."""
    k = rows[rows.split.isin(SPLITS)]
    pu = k.drop_duplicates("pair_id").merge(units[["pair_id", "timestamp"]], on="pair_id", how="left")
    out = {}
    for s in SPLITS:
        x = pu[pu.split == s]
        if x.empty:
            continue
        ts = pd.to_datetime(x.timestamp.astype("Int64").astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        hh = ts.dt.hour + ts.dt.minute / 60
        out[s] = dict(pairs=len(x), images=int((k.split == s).sum()),
                      altitudes=x.altitude_m.value_counts().sort_index().to_dict(),
                      hour_median=float(hh.median()), hour_min=float(hh.min()), hour_max=float(hh.max()),
                      union_area_m2=float(unary_union(list(k[k.split == s].geometry)).area))
    return out


STRICT = {"main_spatial": True, "xalt_strict": True, "xfarm": True, "tndl_xalt_strict": True,
          "xalt_shared": False, "tndl_xalt_shared": False}


def family(name):
    for f in ("main_spatial", "xalt_strict", "xalt_shared", "xfarm", "tndl_xalt_strict", "tndl_xalt_shared"):
        if name.startswith(f):
            return f


def audit_all(protocols, units, cells, do_phash=True):
    tree = STRtree(cells.geometry.values)
    recs, extra = [], {}
    for name, rows in protocols.items():
        if name.startswith("_"):
            continue
        fam = family(name)
        used = rows[rows.split.isin(SPLITS)]
        buf = int(name[-2:]) if fam == "main_spatial" else CHOSEN_BUFFER
        aud = bf.audit_split(used[["farm", "flight_id", "file", "split"]], buf, verbose=False)
        r = dict(protocol=name, family=fam, buffer_m=buf, strict=STRICT[fam],
                 **{f"pairs_{s}": int(used[used.split == s].pair_id.nunique()) for s in SPLITS},
                 pairs_buffer_excluded=int(rows[rows.split == "buffer_excluded"].pair_id.nunique()),
                 images_train=int((used.split == "train").sum()), images_val=int((used.split == "val").sum()),
                 images_test=int((used.split == "test").sum()),
                 cross_split_footprint_pairs=aud["n_cross_split_pairs"],
                 by_split_pair=";".join(f"{k}:{v}" for k, v in aud["pairs_by_split_pair"].items()))
        if (rows.farm == "TN19").all() or name.startswith("xfarm"):
            t19 = used[used.farm == "TN19"]
            sh, nc, _ = cell_sharing(t19, cells, tree)
            r.update({f"cells_shared_{k.replace('&', '_')}": v for k, v in sh.items()})
            r.update({f"cells_{s}": v for s, v in nc.items()})
        r["verdict"] = ("PASS" if (r["cross_split_footprint_pairs"] == 0 and
                                   all(v == 0 for k, v in r.items() if k.startswith("cells_shared"))) else
                        ("EXPECTED (shares modules by design)" if not r["strict"] else "FAIL"))
        recs.append(r)
        extra[name] = balance(rows, units)
        if name == f"main_spatial_buf{CHOSEN_BUFFER:02d}":
            _, _, cs = cell_sharing(used, cells, tree)
            blk = [dict(split=s_, block=b_, cells=int((cells.layer.values[list(cs[s_])] == b_).sum()))
                   for s_ in SPLITS for b_ in sorted(cells.layer.unique())]
            pd.DataFrame(blk).pivot(index="split", columns="block", values="cells").to_csv(
                OUT / "audit_cells_by_block.csv")
    # negative control: random 60/15/25 split of PAIRS (what a naive split would do)
    rng = np.random.default_rng(SEED)
    base = protocols[f"main_spatial_buf{CHOSEN_BUFFER:02d}"]
    ids = base.pair_id.unique()
    lab = rng.choice(SPLITS, size=len(ids), p=TARGET)
    rnd = base.drop(columns="split").merge(pd.DataFrame({"pair_id": ids, "split": lab}), on="pair_id")
    aud = bf.audit_split(rnd[["farm", "flight_id", "file", "split"]], 0.0, verbose=False)
    sh, nc, _ = cell_sharing(rnd, cells, tree)
    recs.append(dict(protocol="NEGATIVE_CONTROL_random_pairs", family="control", buffer_m=0, strict=True,
                     **{f"pairs_{s_}": int((lab == s_).sum()) for s_ in SPLITS}, pairs_buffer_excluded=0,
                     cross_split_footprint_pairs=aud["n_cross_split_pairs"],
                     by_split_pair=";".join(f"{k}:{v}" for k, v in aud["pairs_by_split_pair"].items()),
                     **{f"cells_shared_{k.replace('&', '_')}": v for k, v in sh.items()},
                     verdict="FAIL (expected: control)"))
    # buffer sensitivity of the main split
    sens = []
    u19 = units[units.farm == "TN19"].set_index("pair_id")
    for b in BUFFERS:
        r_ = protocols[f"main_spatial_buf{b:02d}"].drop_duplicates("pair_id")
        r_ = r_.assign(alt=r_.pair_id.map(u19.altitude_m))
        d = dict(buffer_m=b, pairs_total=len(r_), excluded=int((r_.split == "buffer_excluded").sum()))
        d["excluded_pct"] = round(100 * d["excluded"] / d["pairs_total"], 1)
        for a_ in ALTS:
            x = r_[r_.alt == a_]
            d[f"excluded_pct_{a_}m"] = round(100 * (x.split == "buffer_excluded").mean(), 1)
        for s_ in SPLITS:
            d[f"kept_{s_}"] = int((r_.split == s_).sum())
        k_ = max(1, sum(d[f"kept_{s_}"] for s_ in SPLITS))
        for s_ in SPLITS:
            d[f"share_{s_}_pct"] = round(100 * d[f"kept_{s_}"] / k_, 1)
        sens.append(d)
    pd.DataFrame(sens).to_csv(OUT / "audit_buffer_sensitivity.csv", index=False)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "audit_summary.csv", index=False)
    nd = None
    if do_phash:
        name = f"main_spatial_buf{CHOSEN_BUFFER:02d}"
        nd, info = near_dup_check(protocols[name], units)
        nd.to_csv(OUT / "audit_near_duplicates.csv", index=False)
        print("[near-dup]", info)
        print(nd.to_string(index=False))
    bal = []
    for name in (f"main_spatial_buf{CHOSEN_BUFFER:02d}", "xalt_strict_31to42", "xalt_shared_31to42",
                 "tndl_xalt_strict_30to35", "xfarm_tn19_to_tndl"):
        if name in extra:
            for s, d in extra[name].items():
                bal.append(dict(protocol=name, split=s, **{k: (str(v) if isinstance(v, dict) else v) for k, v in d.items()}))
    pd.DataFrame(bal).to_csv(OUT / "audit_balance.csv", index=False)
    return df, nd, pd.DataFrame(bal)


# ----------------------------------------------------------------------------- figures
COL = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c", "buffer_excluded": "#d62728"}


def _zones(ax, strips, labels=True, clip=None, ref=(265600.0, 1263740.0)):
    from matplotlib.patches import Polygon as MPoly
    for zid, (poly, s) in strips.polygons(ref).items():
        g = poly.intersection(clip) if clip is not None else poly
        if g.is_empty:
            continue
        xs, ys = g.exterior.xy
        ax.add_patch(MPoly(np.c_[xs, ys], closed=True, fc=COL[s], ec="k", lw=0.6, alpha=0.18, zorder=0))
        if labels:
            c = g.representative_point()
            ax.text(c.x, g.bounds[3] - 4, zid.split("-")[-1], fontsize=7, ha="center", va="top", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def _draw(ax, rows, lw=0.3, only_sensor="T", cells=None, every_excl=1, dots=False):
    """thermal footprints coloured by split; excluded ones thinned (every_excl) to keep the plot readable."""
    from matplotlib.collections import PolyCollection
    if cells is not None:
        ax.scatter(cells.x, cells.y, s=0.15, c="0.75", zorder=1, linewidths=0)
    r = rows[rows.sensor == only_sensor]
    for s in ["buffer_excluded"] + SPLITS:
        g = r[r.split == s]
        if s == "buffer_excluded":
            g = g.iloc[::every_excl]
        if len(g):
            ax.add_collection(PolyCollection([np.asarray(p.exterior.coords) for p in g.geometry], facecolors="none",
                                             edgecolors=COL[s], linewidths=lw, alpha=0.5 if s == "buffer_excluded" else 0.8,
                                             zorder=2))
            if dots:
                c = np.array([[p.centroid.x, p.centroid.y] for p in g.geometry])
                ax.scatter(c[:, 0], c[:, 1], s=2, color=COL[s], zorder=3, linewidths=0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")


def _legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=COL[s], lw=2, label=s) for s in SPLITS + ["buffer_excluded"]]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8, frameon=False)


def make_figures(protocols, strips, cells, units):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = OUT / "figures"
    FIG.mkdir(exist_ok=True)
    win = box(265330, 1263570, 265880, 1263890)

    def lim(ax, xl=(265330, 265880), yl=(1263590, 1263880)):
        ax.set_xlim(*xl); ax.set_ylim(*yl)

    st = strips["main"]
    name = f"main_spatial_buf{CHOSEN_BUFFER:02d}"
    fig, ax = plt.subplots(figsize=(12, 6.6))
    _zones(ax, st, True, box(265330, 1263590, 265880, 1263880))
    _draw(ax, protocols[name], cells=cells, lw=0.15, every_excl=12)
    lim(ax)
    ax.set_title(f"A  main spatial split, TN19, buffer {CHOSEN_BUFFER} m: zones (shaded, sub-zone ids) and thermal "
                 "footprints of all altitudes (grey dots = reference cells)", fontsize=9)
    _legend(fig); fig.subplots_adjust(bottom=0.1, top=0.93)
    fig.savefig(FIG / f"A_{name}.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for ax, b in zip(axs.ravel(), BUFFERS):
        r = protocols[f"main_spatial_buf{b:02d}"]
        _zones(ax, st, False, win)
        _draw(ax, r, lw=0.15, every_excl=12)
        lim(ax)
        lost = int(r[r.sensor == "T"].split.eq("buffer_excluded").sum())
        ax.set_title(f"buffer {b} m: {lost} of {int((r.sensor == 'T').sum())} thermal frames excluded", fontsize=9)
    _legend(fig); fig.subplots_adjust(bottom=0.08, hspace=0.2, wspace=0.12)
    fig.savefig(FIG / "A_main_spatial_buffer_sensitivity.png", dpi=130); plt.close(fig)

    for kind, tag in (("shared", "B1"), ("strict", "B2")):
        fig, axs = plt.subplots(2, 2, figsize=(15, 9))
        for row, (src, tgt) in zip(axs, ((31, 42), (42, 31))):
            r = protocols[f"xalt_{kind}_{src}to{tgt}"]
            sz = strips[f"xalt_{src}to{tgt}"]
            for ax, alt, role in zip(row, (src, tgt), ("source: train/val", "target: test")):
                _zones(ax, sz, False, win)
                _draw(ax, r[r.altitude_m == alt], lw=0.25, every_excl=6)
                lim(ax, (265390, 265700), (1263625, 1263880))
                ax.set_title(f"{src} -> {tgt} m, {alt} m pairs ({role})", fontsize=9)
        fig.suptitle(f"{tag}  cross-altitude {kind} (blocks A-C); shading = zone owner. "
                     + ("test = target altitude on the SAME modules as train (shared by design)" if kind == "shared"
                        else "test = target altitude in the disjoint test strip"), fontsize=10)
        _legend(fig); fig.subplots_adjust(bottom=0.08, top=0.92, wspace=0.12, hspace=0.18)
        fig.savefig(FIG / f"{tag}_xalt_{kind}.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, farm in zip(axs.ravel(), ["TN19", "TNDL", "TT", "BIM2"]):
        r = protocols[f"xfarm_tn19_to_{farm.lower()}"] if farm != "TN19" else protocols["xfarm_tn19_to_tndl"]
        r = r[r.farm == farm]
        if farm == "TN19":
            _zones(ax, st, False, win); _draw(ax, r, lw=0.15, every_excl=12); lim(ax)
            ax.set_title("TN19: train (train + former test strip) / val / excluded", fontsize=9)
        else:
            _draw(ax, r, lw=0.3)
            xy = np.concatenate([np.asarray(g.exterior.coords) for g in r.geometry])
            ax.set_xlim(xy[:, 0].min() - 10, xy[:, 0].max() + 10); ax.set_ylim(xy[:, 1].min() - 10, xy[:, 1].max() + 10)
            extra = " (Mavic 2 Ent. Adv., no RTK)" if farm == "BIM2" else ""
            ax.set_title(f"{farm}: test, all pairs{extra}", fontsize=9)
    _legend(fig); fig.subplots_adjust(bottom=0.07, hspace=0.2)
    fig.savefig(FIG / "C_xfarm.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    k = 0
    for kind in ("shared", "strict"):
        for src, tgt in ((30, 35), (35, 30)):
            ax = axs[k]; k += 1
            r = protocols[f"tndl_xalt_{kind}_{src}to{tgt}"]
            _zones(ax, strips[f"tndl_{src}to{tgt}"], False, box(265700, 1262500, 266300, 1262800), ref=(265980.0, 1262640.0))
            _draw(ax, r, every_excl=3)
            ax.set_xlim(265770, 266190); ax.set_ylim(1262575, 1262710)
            ax.set_title(f"D {kind}: {src} -> {tgt} m (source and target footprints overlaid)", fontsize=9)
    _legend(fig); fig.subplots_adjust(bottom=0.06, hspace=0.45)
    fig.savefig(FIG / "D_tndl_xalt.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- label-aware protocols
# Thermal fault boxes exist only for 6 flights and labels were assigned only inside a reviewer-fixed evaluation
# zone per flight: an image whose footprint holds no in-zone box is a NEGATIVE only where it is covered by the zone.
# Zones for the label-aware splits are searched on box counts per class (objective in ``la_search``), identical
# for all labelled TN19 flights (31 m and 35 m flights of the same blocks share the zones, so a physical module
# is never split across altitudes).  Everything is deterministic (exhaustive search, no random draws).
LA_CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]    # 4th class: empty today
LA_TN19 = [f"TN19_20260411_{x}" for x in ("31M_T3", "35M_T3", "35M_T4", "35M_T5")]
LA_TARGET = {"tndl": "TNDL_20260411_35M_T2", "tt": "TT_20260412_39M_T1"}
LA_LABEL_SRC = ROOT / ".local/cache/d4_labels"      # original label JSON (holds the evaluation zones); local only
LA_LABEL_DIR = {"31M_T3": LA_TN19[0], "35M_T3": LA_TN19[1], "35M_T4": LA_TN19[2], "35M_T5": LA_TN19[3],
                "TNDL_GSD3_T2": LA_TARGET["tndl"], "TT_GSD3_T1": LA_TARGET["tt"]}
LA_ZONES_CSV = OUT / "label_aware_eval_zones.csv"
LA_IOU_SAME_MODULE = 0.3     # polygon IoU above which two boxes (any flights) are the same physical module
LA_NEG_COVER = 0.95          # image is a negative only if >= 95 % of its thermal footprint is inside the eval zone
LA_W_SHORT, LA_W_CLASS, LA_W_ALT = 8.0, 0.7, 2.0
LA_MIN_ABS = {"train": 0, "val": 10, "test": 15}           # soft minimum number of boxes per class and split ...
LA_MIN_FRAC = {"train": 0.40, "val": 0.08, "test": 0.12}   # ... or this share of the class total, whichever is larger


def la_classes(boxes):
    return LA_CLASSES + sorted(set(boxes.class_final) - set(LA_CLASSES))


def la_boxes():
    """All thermal boxes of the labelled flights + physical-module clusters (IoU-linked, within a farm)."""
    from shapely.geometry import Polygon
    fs = [pd.read_csv(d / "annotations/thermal_anomaly_obb/thermal_boxes.csv")
          for d in FARM_DIR.values() if (d / "annotations/thermal_anomaly_obb/thermal_boxes.csv").exists()]
    b = pd.concat(fs, ignore_index=True)
    b = b[b.flight_id.isin(LA_TN19 + list(LA_TARGET.values()))].copy()
    b["farm"] = b.flight_id.str.split("_").str[0]
    b["in_eval_zone"] = b.in_eval_zone.astype(str).str.lower().eq("true")
    b["_n"] = b.box_id.str.extract(r"_(\d+)$")[0].astype(float).fillna(1e9)     # self-drawn boxes have random ids
    b = b.sort_values(["flight_id", "_n", "box_id"]).drop(columns="_n").reset_index(drop=True)
    polys = [Polygon([(r.E1, r.N1), (r.E2, r.N2), (r.E3, r.N3), (r.E4, r.N4)]) for r in b.itertuples()]
    b["geometry"] = polys
    par = list(range(len(b)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    tree = STRtree(polys)
    ii, jj = tree.query(np.array(polys, dtype=object), predicate="intersects")
    for i, j in sorted(zip(ii.tolist(), jj.tolist())):
        if i < j and b.farm[i] == b.farm[j]:
            if polys[i].intersection(polys[j]).area / polys[i].union(polys[j]).area >= LA_IOU_SAME_MODULE:
                par[find(j)] = find(i)
    root = [find(i) for i in range(len(b))]
    num, ids = {}, []
    for r_, f_ in zip(root, b.farm):
        if r_ not in num:
            num[r_] = 1 + sum(1 for k in num if b.farm[k] == f_)
        ids.append(f"{f_}-PM{num[r_]:05d}")
    b["phys_module_id"] = ids
    return b


def la_eval_zones():
    """flight_id -> evaluation-zone polygon (EPSG:32649).  Read from the label JSON when present (and cached in
    label_aware_eval_zones.csv, which is what a release user has); otherwise from the csv."""
    import json
    from pyproj import Transformer
    from shapely import wkt
    from shapely.geometry import Polygon
    if LA_LABEL_SRC.exists():
        tr = Transformer.from_crs(4326, 32649, always_xy=True)
        rows = []
        for k, fid in LA_LABEL_DIR.items():
            j = json.loads((LA_LABEL_SRC / k / "nhan_tay.json").read_text(encoding="utf-8"))
            poly = Polygon([tr.transform(lo, la) for lo, la in j["vung_danh_gia"]["lonlat"]])
            rows.append(dict(flight_id=fid, area_m2=round(poly.area, 1),
                             wkt=wkt.dumps(poly, rounding_precision=3)))
        df = pd.DataFrame(rows).sort_values("flight_id")
        OUT.mkdir(parents=True, exist_ok=True)
        txt = df.to_csv(index=False, lineterminator="\n")
        if not LA_ZONES_CSV.exists() or LA_ZONES_CSV.read_bytes().decode("utf-8") != txt:
            LA_ZONES_CSV.write_bytes(txt.encode("utf-8"))
    df = pd.read_csv(LA_ZONES_CSV)
    return {r.flight_id: wkt.loads(r.wkt) for r in df.itertuples()}


class LA:
    """Units/images of the labelled flights, boxes, eval zones, box -> candidate pair links, pair label table."""

    def __init__(self, units, imgs):
        self.boxes = la_boxes()
        self.classes = la_classes(self.boxes)
        self.zones = la_eval_zones()
        fl = set(self.boxes.flight_id)
        self.units = units[units.flight_id.isin(fl)].reset_index(drop=True)
        self.imgs = imgs[imgs.flight_id.isin(fl)].reset_index(drop=True)
        t = self.imgs[self.imgs.sensor == "T"].drop_duplicates("pair_id").set_index("pair_id").geometry
        tf = self.units.pair_id.map(t)
        miss = tf.isna().values
        self.tfp = np.where(miss, self.units.geometry.values, tf.values)   # thermal footprint (own footprint if no thermal)
        pts = np.array([Point(x, y) for x, y in zip(self.boxes.centre_E, self.boxes.centre_N)], dtype=object)
        self.pts = pts
        self.link = {}              # box index -> unit indices of the same flight whose thermal footprint contains its centre
        cnt = np.zeros((len(self.units), len(self.classes)), int)
        out_zone = np.zeros(len(self.units), int)
        cover = np.zeros(len(self.units))
        cidx = {c: i for i, c in enumerate(self.classes)}
        for fid, ub in self.units.groupby("flight_id"):
            ui_all = ub.index.values
            bi = np.flatnonzero((self.boxes.flight_id == fid).values)
            tree = STRtree(pts[bi])
            ui, tj = tree.query(self.tfp[ui_all], predicate="contains")
            for a, c in sorted(zip(ui.tolist(), tj.tolist())):
                u_, b_ = ui_all[a], bi[c]
                self.link.setdefault(b_, []).append(u_)
                if self.boxes.in_eval_zone[b_]:
                    cnt[u_, cidx[self.boxes.class_final[b_]]] += 1
                else:
                    out_zone[u_] += 1
            z = self.zones[fid]
            for u_ in ui_all:
                g = self.tfp[u_]
                cover[u_] = g.intersection(z).area / g.area
        lab = pd.DataFrame(cnt, columns=[f"n_{c}" for c in self.classes])
        lab.insert(0, "pair_id", self.units.pair_id.values)
        lab["n_boxes"] = cnt.sum(1)
        lab["has_boxes"] = lab.n_boxes > 0
        lab["n_boxes_outside_zone"] = out_zone
        lab["in_eval_zone_cover"] = np.round(cover, 3)
        lab["label_status"] = np.where(lab.n_boxes > 0, "positive", np.where(
            cover >= LA_NEG_COVER, "negative", np.where(cover > 0, "partial_unknown", "unknown")))
        self.pair_labels = lab
        self.boxes["ref_cell"] = ""
        try:                         # optional: approximate reference cell containing the box centre (TN19 only)
            cells = load_cells()
            ct = STRtree(cells.geometry.values)
            ci, bj = ct.query(pts, predicate="within")        # (point idx, cell idx)
            ref = np.array([""] * len(self.boxes), dtype=object)
            for p_, c_ in sorted(zip(ci.tolist(), bj.tolist()), reverse=True):
                ref[p_] = f"{cells.layer.values[c_]}:{cells.module_id.values[c_]}"
            ref[(self.boxes.farm != "TN19").values] = ""
            self.boxes["ref_cell"] = ref
        except Exception as e:       # never a dependency
            print("[la] reference cells not available:", e)


def la_minreq(tot_c):
    """(nc, 3) soft minimum box counts per class and split (0 for classes with < 3 boxes, e.g. the empty 4th class)."""
    m = np.array([[max(LA_MIN_ABS[s], np.floor(LA_MIN_FRAC[s] * t)) for s in SPLITS] for t in tot_c])
    return np.where(tot_c[:, None] >= 3, np.minimum(m, np.floor(tot_c[:, None] * 0.5)), 0)


def la_cost(cs, calt, loss, act, minreq):
    """cost = 3 L1(total box shares - 60/15/25) + W_CLASS mean_class L1(class shares - target)
    + W_SHORT mean_class soft-minimum shortfall + W_ALT mean_altitude L1(box shares of the altitude) + 3 image loss share.
    cs (..., 3 splits, nc classes) usable boxes, calt (..., 3 splits, 2) usable boxes of the 31 m / 35 m flights
    (each altitude must be spread 60/15/25 too: the cross-altitude protocols train on 31 m and test on 35 m)."""
    tot = cs.sum(-1)
    cost = 3 * np.abs(tot / np.maximum(tot.sum(-1, keepdims=True), 1) - TARGET).sum(-1)
    sh = cs / np.maximum(cs.sum(-2, keepdims=True), 1)
    cost = cost + LA_W_CLASS * np.abs(sh - TARGET[:, None]).sum(-2)[..., act].mean(-1)
    short = np.maximum(0, minreq.T - cs) / np.maximum(minreq.T, 1)
    cost = cost + LA_W_SHORT * short[..., act].sum((-1, -2)) / max(act.sum(), 1)
    cost = cost + LA_W_ALT * np.abs(calt / np.maximum(calt.sum(-2, keepdims=True), 1) - TARGET[:, None]).sum(-2).mean(-1)
    return cost + 3 * loss + 1e3 * (tot.min(-1) <= 0)


class LAZones:
    """Guillotine zoning of the TN19 plant in the sheared frame u = x - K (y - V0), v = y:
    ``ucuts`` split the plant into strips; at most one strip (``vstrip``) is cut once more at ``vcut`` (v) into a
    southern and a northern cell.  Every cell is owned by one split (``order``, one entry per cell).  Cells are
    numbered LA-Z01.. in the order strip 0 (south, north), strip 1, ...  Two images of the same cell never cost
    anything; an image meeting two cells (after buffering) is ``buffer_excluded``."""

    def __init__(self, ucuts, vstrip, vcut, order):
        self.ucuts = np.asarray(ucuts, float)
        self.vstrip, self.vcut = vstrip, vcut
        self.nstrip = len(self.ucuts) + 1
        self.ncell = self.nstrip + (1 if vstrip is not None else 0)
        self.order = list(order)
        assert len(self.order) == self.ncell
        names, k = {}, 1
        for i in range(self.nstrip):
            for part in ((0, 1) if i == vstrip else (0,)):
                names[(i, part)] = f"LA-Z{k:02d}"
                k += 1
        self.names = names
        self.cell_names = [None] * self.ncell
        for (i, part), nm in names.items():
            self.cell_names[self._cell(i, part)] = nm

    def _cell(self, seg, part):
        return seg if part == 0 else self.nstrip

    @staticmethod
    def uv(x, y):
        return np.asarray(x) - K_SHEAR * (np.asarray(y) - V0), np.asarray(y)

    def cell_of(self, x, y):
        u, v = self.uv(x, y)
        seg = np.searchsorted(self.ucuts, u, side="right")
        up = (seg == self.vstrip) & (v >= self.vcut) if self.vstrip is not None else np.zeros(len(seg), bool)
        return np.where(up, self.nstrip, seg)

    def extents(self, geoms, buf):
        bg = shapely.buffer(np.asarray(geoms, dtype=object), buf) if buf > 0 else np.asarray(geoms, dtype=object)
        out = np.empty((len(bg), 4))
        for i, g in enumerate(bg):
            xy = np.asarray(g.exterior.coords) if g.geom_type == "Polygon" else np.concatenate(
                [np.asarray(p.exterior.coords) for p in g.geoms])
            u, v = self.uv(xy[:, 0], xy[:, 1])
            out[i] = u.min(), u.max(), v.min(), v.max()
        return out

    def assign(self, units, buf, ext=None):
        e = ext if ext is not None else self.extents(units.geometry.values, buf)
        sl = np.searchsorted(self.ucuts, e[:, 0], "right"); sh = np.searchsorted(self.ucuts, e[:, 1], "right")
        ok = sl == sh
        if self.vstrip is not None:
            ok &= ~((sl == self.vstrip) & (e[:, 2] < self.vcut) & (e[:, 3] >= self.vcut))
        cell = self.cell_of(units.cx.values, units.cy.values)
        order = np.array(self.order, dtype=object)
        # a kept image lies in one cell; its centre cell is that cell
        split = np.where(ok, order[cell], "buffer_excluded")
        return pd.DataFrame({"pair_id": units.pair_id.values, "zone_id": np.array(self.cell_names)[cell], "split": split})

    def zone_of(self, x, y):
        return np.array(self.cell_names)[self.cell_of(x, y)]

    def polygons(self, ref=None, v0=1263560.0, v1=1263900.0):
        def quad(a, b, c, d):
            return shapely.Polygon([(a + K_SHEAR * (c - V0), c), (b + K_SHEAR * (c - V0), c),
                                    (b + K_SHEAR * (d - V0), d), (a + K_SHEAR * (d - V0), d)])
        edges = [265300.0, *self.ucuts, 265900.0]
        out = {}
        for i in range(self.nstrip):
            a, b = edges[i], edges[i + 1]
            if i == self.vstrip:
                out[self.names[(i, 0)]] = (quad(a, b, v0, self.vcut), self.order[self._cell(i, 0)])
                out[self.names[(i, 1)]] = (quad(a, b, self.vcut, v1), self.order[self._cell(i, 1)])
            else:
                out[self.names[(i, 0)]] = (quad(a, b, v0, v1), self.order[self._cell(i, 0)])
        return out

    def table(self):
        rows = []
        for zid, (poly, s) in self.polygons().items():
            rows.append(dict(zone_id=zid, split=s, cell_corners_EN=";".join(f"{x:.1f},{y:.1f}" for x, y in list(poly.exterior.coords)[:4])))
        return pd.DataFrame(rows)

    def describe(self):
        d = f"u cuts {np.round(self.ucuts, 1).tolist()}"
        if self.vstrip is not None:
            d += f", strip {self.vstrip} also cut at v(northing) = {self.vcut:.1f}"
        return d + f", owners {self.order}"


def la_search(L, b, ks=(2, 3), step_u=10.0, step_v=10.0, refine=2.5, verbose=True):
    """Deterministic search of guillotine zonings (see ``LAZones``) for the labelled TN19 flights.
    A box is *usable* iff at least one pair that sees it is kept (buffered footprint inside one cell).
    cost = 3 L1(total box shares - 60/15/25) + 1.5 mean_class L1(class shares - target)
           + LA_W_SHORT soft-minimum shortfall (val >= 10 %, test >= 18 %, train >= 40 % of each class)
           + W_ALT (31 m and 35 m box shares) + 3 image loss share + 0.1 (number of cuts).   Classes with 0 boxes are ignored.
    Coarse exhaustive search on step_u/step_v grids, then coordinate refinement at ``refine`` metres."""
    import itertools
    uidx = np.flatnonzero(L.units.farm.eq("TN19").values)
    u19 = L.units.iloc[uidx]
    Z0 = LAZones([0.0], None, None, ["train", "val"])
    pe = Z0.extents(u19.geometry.values, b)                       # (n, 4) buffered extents
    pos = {u: i for i, u in enumerate(uidx)}
    bx = L.boxes[L.boxes.farm.eq("TN19") & L.boxes.in_eval_zone].index.values
    cls = np.array([L.classes.index(c) for c in L.boxes.class_final[bx]])
    is35 = (~L.boxes.flight_id[bx].str.contains("_31M_").values).astype(int)
    bu, bv = LAZones.uv(L.boxes.centre_E[bx].values, L.boxes.centre_N[bx].values)
    rb, rpe = [], []
    for k, b_ in enumerate(bx):
        for u_ in L.link.get(b_, []):
            rb.append(k); rpe.append(pe[pos[u_]])
    rb, rpe = np.array(rb), np.array(rpe)
    nb, nc = len(bx), len(L.classes)
    tot_c = np.bincount(cls, minlength=nc).astype(float)
    act = tot_c > 0
    minreq = la_minreq(tot_c)
    assigns = {}

    def assignments(n):
        if n not in assigns:
            A = np.array([a for a in itertools.product(range(3), repeat=n) if len(set(a)) == 3])
            S = np.zeros((len(A), n, 3)); S[np.arange(len(A))[:, None], np.arange(n)[None, :], A] = 1
            assigns[n] = (A, S)
        return assigns[n]

    def ev(uc, vs, vc):
        uc = np.asarray(uc)
        nst = len(uc) + 1
        ncell = nst + (vs is not None)
        sl = np.searchsorted(uc, rpe[:, 0], "right"); sh = np.searchsorted(uc, rpe[:, 1], "right")
        ok_r = sl == sh
        if vs is not None:
            ok_r &= ~((sl == vs) & (rpe[:, 2] < vc) & (rpe[:, 3] >= vc))
        ok = np.bincount(rb[ok_r], minlength=nb) > 0
        seg = np.searchsorted(uc, bu, "right")
        cell = np.where((seg == vs) & (bv >= vc), nst, seg) if vs is not None else seg
        cnt = np.bincount((cell * nc + cls)[ok], minlength=ncell * nc).reshape(ncell, nc).astype(float)
        calt = np.bincount((cell * 2 + is35)[ok], minlength=ncell * 2).reshape(ncell, 2).astype(float)
        sp_ = np.searchsorted(uc, pe[:, 0], "right"); ep_ = np.searchsorted(uc, pe[:, 1], "right")
        okp = sp_ == ep_
        if vs is not None:
            okp &= ~((sp_ == vs) & (pe[:, 2] < vc) & (pe[:, 3] >= vc))
        loss = 1 - okp.mean()
        A, S = assignments(ncell)
        cs = np.einsum("aks,kc->asc", S, cnt)                          # (nA, 3 splits, nc)
        cost = la_cost(cs, np.einsum("aks,kt->ast", S, calt), loss, act, minreq) + 0.1 * (len(uc) + (vs is not None))
        a = int(np.argmin(cost))
        return float(cost[a]), [SPLITS[i] for i in A[a]], cs[a].astype(int)

    ug = np.arange(np.floor(bu.min()) + 1, np.ceil(bu.max()), step_u)
    vg = np.arange(np.floor(bv.min()) + 1, np.ceil(bv.max()), step_v)
    best = (1e18, None)
    cand = []
    for K in ks:
        for uc in itertools.combinations(ug, K):
            cand.append((uc, None, None))
            if K == 2:
                for vs in range(K + 1):
                    for vc in vg:
                        cand.append((uc, vs, vc))
    for uc, vs, vc in cand:
        c, order, cs = ev(uc, vs, vc)
        if c < best[0] - 1e-12:
            best = (c, (list(uc), vs, vc, order, cs))
    if verbose:
        print(f"[la_search] coarse ({len(cand)} configs): cost {best[0]:.3f}")
    c, (uc, vs, vc, order, cs) = best
    improved = True
    while improved:                                                    # coordinate refinement (deterministic)
        improved = False
        for k in range(len(uc) + (vs is not None)):
            for d in (-refine, refine, -2 * refine, 2 * refine):
                uc2, vc2 = list(uc), vc
                if k < len(uc):
                    uc2[k] += d
                    if sorted(uc2) != uc2:
                        continue
                else:
                    vc2 = vc + d
                c2, o2, cs2 = ev(uc2, vs, vc2)
                if c2 < c - 1e-9:
                    c, uc, vc, order, cs, improved = c2, uc2, vc2, o2, cs2, True
    if verbose:
        print(f"[la_search] refined: cost {c:.3f} u cuts {np.round(uc, 1).tolist()} vstrip {vs} vcut {vc} order {order}")
    return c, LAZones(uc, vs, vc, order), cs


LA_CELL_M = 8.0              # edge of the lattice cells (sheared frame) from which the label-aware zones are built
# annealing: best of 4 seeds (21-24, final cost 3.56-3.61) at 300k iterations; seed 21 is the one kept
LA_SA = dict(iters=300000, t0=0.06, t1=0.002, frag=0.002, seed=21)


def _to_uv(g):
    return affine_transform(g, [1, -K_SHEAR, 0, 1, K_SHEAR * V0, 0])          # u = x - K (y - V0), v = y


def _from_uv(g):
    return affine_transform(g, [1, K_SHEAR, 0, 1, -K_SHEAR * V0, 0])


class Lattice:
    """Regular lattice in the sheared (u, v) frame; zones are unions of its cells."""

    def __init__(self, geoms, h=LA_CELL_M, pad=25.0):
        g = [_to_uv(x) for x in shapely.buffer(np.asarray(geoms, dtype=object), pad)]
        b = np.array([x.bounds for x in g])
        self.h = h
        self.u0, self.v0 = np.floor(b[:, 0].min() / h) * h, np.floor(b[:, 1].min() / h) * h
        self.nu = int(np.ceil((b[:, 2].max() - self.u0) / h)); self.nv = int(np.ceil((b[:, 3].max() - self.v0) / h))
        self.polys = np.array([box(self.u0 + i * h, self.v0 + j * h, self.u0 + (i + 1) * h, self.v0 + (j + 1) * h)
                               for i in range(self.nu) for j in range(self.nv)], dtype=object)
        self.tree = STRtree(self.polys)

    def cell_of(self, x, y):
        u, v = LAZones.uv(x, y)
        i = np.clip(np.floor((u - self.u0) / self.h).astype(int), 0, self.nu - 1)
        j = np.clip(np.floor((v - self.v0) / self.h).astype(int), 0, self.nv - 1)
        return i * self.nv + j

    def touched(self, geoms, buf):
        """list (per geometry) of lattice cells meeting the geometry buffered by buf metres."""
        g = np.array([_to_uv(x) for x in (shapely.buffer(np.asarray(geoms, dtype=object), buf) if buf > 0
                                          else np.asarray(geoms, dtype=object))], dtype=object)
        gi, ci = self.tree.query(g, predicate="intersects")
        out = [[] for _ in range(len(g))]
        for a, c in zip(gi.tolist(), ci.tolist()):
            out[a].append(c)
        return [np.array(sorted(o)) for o in out]


class LAMosaic:
    """Zoning = split owner per lattice cell.  An image is kept in split s iff its footprint buffered by b m touches
    only cells owned by s (neighbouring cells of one split cost nothing).  Zones (zone_id) are the 4-connected
    components of same-split cells, numbered west to east."""

    def __init__(self, lat, code, active):
        self.lat, self.code, self.active = lat, np.asarray(code, np.int8), np.asarray(active, bool)
        nv = lat.nv
        comp = -np.ones(len(code), int)
        k = 0
        for c in sorted(np.flatnonzero(self.active)):
            if comp[c] >= 0:
                continue
            st, comp[c] = [c], k
            while st:
                a = st.pop()
                i, j = divmod(a, nv)
                for ii, jj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= ii < lat.nu and 0 <= jj < nv:
                        n = ii * nv + jj
                        if self.active[n] and comp[n] < 0 and code[n] == code[c]:
                            comp[n] = k
                            st.append(n)
            k += 1
        self.comp = comp
        self.names = {i: f"LA-Z{i + 1:02d}" for i in range(k)}

    def zone_of(self, x, y):
        c = self.lat.cell_of(x, y)
        return np.array([self.names.get(self.comp[i], "") for i in c])

    def split_of_cells(self, cells):
        return self.code[cells]

    def assign(self, units, buf):
        t = self.lat.touched(units.geometry.values, buf)
        keep = np.array([len(c) > 0 and (self.code[c] == self.code[c[0]]).all() and self.code[c[0]] >= 0 for c in t])
        cc = self.lat.cell_of(units.cx.values, units.cy.values)
        own = np.array(SPLITS, dtype=object)[np.maximum(self.code[cc], 0)]
        return pd.DataFrame({"pair_id": units.pair_id.values, "zone_id": self.zone_of(units.cx.values, units.cy.values),
                             "split": np.where(keep, own, "buffer_excluded")})

    def polygons(self, ref=None):
        out = {}
        lat = self.lat
        for k, nm in self.names.items():
            cells = np.flatnonzero(self.comp == k)
            g = _from_uv(unary_union(list(lat.polys[cells])))
            out[nm] = (g, SPLITS[int(self.code[cells[0]])])
        return out

    def table(self):
        rows = []
        for nm, (g, s) in self.polygons().items():
            k = int(nm.split("Z")[-1]) - 1
            rows.append(dict(zone_id=nm, split=s, n_cells=int((self.comp == k).sum()), area_m2=round(g.area, 0),
                             centroid_E=round(g.centroid.x, 1), centroid_N=round(g.centroid.y, 1),
                             bounds_EN=";".join(f"{v:.0f}" for v in g.bounds)))
        return pd.DataFrame(rows)

    def describe(self):
        t = self.table()
        return f"mosaic of {int(self.active.sum())} cells of {self.lat.h:g} m in {len(t)} zones " + \
            ", ".join(f"{s}: {int((t.split == s).sum())} zones/{int(t[t.split == s].area_m2.sum())} m2" for s in SPLITS)


def la_anneal(L, b, init_zoning, verbose=True, **kw):
    """Simulated annealing (seeded, deterministic) over the lattice-cell owners, started from a guillotine zoning.
    Same cost as ``la_search`` + ``frag`` * (number of cell edges between different owners)."""
    prm = {**LA_SA, **kw}
    uidx = np.flatnonzero(L.units.farm.eq("TN19").values)
    u19 = L.units.iloc[uidx].reset_index(drop=True)
    lat = Lattice(u19.geometry.values)
    t10 = lat.touched(u19.geometry.values, b)
    t20 = lat.touched(u19.geometry.values, max(BUFFERS))
    bx = L.boxes[L.boxes.farm.eq("TN19") & L.boxes.in_eval_zone].index.values
    bxy = (L.boxes.centre_E[bx].values, L.boxes.centre_N[bx].values)
    bcell = lat.cell_of(*bxy)
    active = np.zeros(lat.nu * lat.nv, bool)
    for c in t20:
        active[c] = True
    active[bcell] = True
    active[lat.cell_of(u19.cx.values, u19.cy.values)] = True
    W = max(len(c) for c in t10)
    idx = np.array([np.r_[c, np.full(W - len(c), c[0])] for c in t10])
    pos = {u: i for i, u in enumerate(uidx)}
    rb, rp = [], []
    for k, b_ in enumerate(bx):
        for u_ in L.link.get(b_, []):
            rb.append(k); rp.append(pos[u_])
    rb, rp = np.array(rb), np.array(rp)
    nb, nc = len(bx), len(L.classes)
    cls = np.array([L.classes.index(c) for c in L.boxes.class_final[bx]])
    is35 = (~L.boxes.flight_id[bx].str.contains("_31M_").values).astype(int)
    tot_c = np.bincount(cls, minlength=nc).astype(float)
    act = tot_c > 0
    minreq = la_minreq(tot_c)
    # neighbours (4-connected, active only)
    nv = lat.nv
    ac = np.flatnonzero(active)
    nbr = np.full((len(active), 4), -1)
    for k, (di, dj) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        i, j = np.divmod(ac, nv)
        ii, jj = i + di, j + dj
        ok = (ii >= 0) & (ii < lat.nu) & (jj >= 0) & (jj < nv)
        n = np.where(ok, ii * nv + jj, -1)
        n = np.where((n >= 0) & active[np.maximum(n, 0)], n, -1)
        nbr[ac, k] = n
    ea = np.repeat(ac, 4); eb = nbr[ac].ravel()
    m = eb > ea
    ea, eb = ea[m], eb[m]

    def cost_of(code):
        sp = code[idx]
        kept = sp.max(1) == sp.min(1)
        ok = np.bincount(rb[kept[rp]], minlength=nb) > 0
        bs = code[bcell]
        cs = np.bincount((bs * nc + cls)[ok], minlength=3 * nc).reshape(3, nc).astype(float)
        calt = np.bincount((bs * 2 + is35)[ok], minlength=6).reshape(3, 2).astype(float)
        return float(la_cost(cs, calt, 1 - kept.mean(), act, minreq) + prm["frag"] * (code[ea] != code[eb]).sum()), cs

    # init from the guillotine zoning (cell centres)
    ctr = np.array([[lat.u0 + (c // nv + .5) * lat.h, lat.v0 + (c % nv + .5) * lat.h] for c in range(len(active))])
    xc = ctr[:, 0] + K_SHEAR * (ctr[:, 1] - V0)
    cell = init_zoning.cell_of(xc, ctr[:, 1])
    code = np.array([SPLITS.index(init_zoning.order[c]) for c in cell], np.int8)
    code[~active] = -1
    rng = np.random.default_rng(prm["seed"])
    cur, _ = cost_of(code)
    best_c, best_code = cur, code.copy()
    n = prm["iters"]
    T = prm["t0"] * (prm["t1"] / prm["t0"]) ** (np.arange(n) / n)
    cand = ac
    win = {}

    def window(c, r):
        if (c, r) not in win:
            i, j = divmod(c, nv)
            w = [ii * nv + jj for ii in range(max(0, i - r), min(lat.nu, i + r + 1))
                 for jj in range(max(0, j - r), min(nv, j + r + 1)) if active[ii * nv + jj]]
            win[(c, r)] = np.array(w)
        return win[(c, r)]
    for it in range(n):
        c = cand[rng.integers(len(cand))]
        r = int(rng.choice([0, 0, 1, 2, 3]))
        if r == 0:
            nb_ = nbr[c][nbr[c] >= 0]
            if (code[nb_] == code[c]).all() and rng.random() < 0.9:
                continue                                            # interior cell: rarely worth flipping
        w = window(c, r)
        old = code[w].copy()
        code[w] = (code[c] + 1 + rng.integers(2)) % 3
        new, _ = cost_of(code)
        if new <= cur or rng.random() < np.exp((cur - new) / T[it]):
            cur = new
            if cur < best_c - 1e-12:
                best_c, best_code = cur, code.copy()
        else:
            code[w] = old
        if verbose and it % 15000 == 0:
            print(f"[la_anneal] it {it} T {T[it]:.4f} cost {cur:.3f} best {best_c:.3f}")
    c_, cs = cost_of(best_code)
    if verbose:
        print(f"[la_anneal] final cost {c_:.3f} (start {cost_of(np.where(active, np.array([SPLITS.index(init_zoning.order[k]) for k in cell], np.int8), -1))[0]:.3f})")
    return c_, LAMosaic(lat, best_code, active), cs.astype(int)


# ------------------------------------------------------------------ protocol assembly
LA_COLS = ["farm", "flight_id", "file", "sensor", "pair_id", "split", "zone_id", "altitude_m"]


def la_image_rows(L, pair_split):
    """pair_split: DataFrame(pair_id, split, zone_id) -> image-level rows with label columns.
    Label columns are thermal-derived and apply to both files of the pair (RGB panel boxes will come later)."""
    r = L.imgs.merge(pair_split, on="pair_id", how="inner").merge(L.pair_labels, on="pair_id", how="left")
    cols = LA_COLS + ["has_boxes", "n_boxes"] + [f"n_{c}" for c in L.classes] + \
        ["n_boxes_outside_zone", "in_eval_zone_cover", "label_status"]
    return r[cols + ["geometry"]].sort_values(["flight_id", "file"]).reset_index(drop=True)


def la_box_rows(L, rows, zone_of_box=None):
    """Box-level table.  A box takes the split of the kept pairs that see it (all candidate pairs are of its own
    flight so they cannot disagree); no kept pair -> buffer_excluded (flight is in the protocol) or unused."""
    first = rows.drop_duplicates("pair_id").set_index("pair_id")
    pair_split, pair_zone = first.split, first.zone_id
    flights = set(rows.flight_id)
    pid = L.units.pair_id.values
    out = []
    for i, r in L.boxes.iterrows():
        cand = [pid[u] for u in L.link.get(i, [])]
        sp = [pair_split[p] for p in cand if p in pair_split.index and pair_split[p] in SPLITS]
        assert len(set(sp)) <= 1, f"box {r.box_id} seen in several splits {set(sp)}"
        if sp:
            s, reason = sp[0], ""
        elif r.flight_id in flights:
            s = "buffer_excluded"
            reason = "no kept image sees the box" if cand else "box centre outside every thermal footprint"
        else:
            s, reason = "unused", "flight not in protocol"
        zc = [pair_zone[p] for p in cand if p in pair_zone.index]
        out.append((r.box_id, s, reason, len(sp), zc[0] if zc else ""))
    o = pd.DataFrame(out, columns=["box_id", "split", "reason", "n_pairs_kept", "zone_id"])
    b = L.boxes.drop(columns=["geometry"]).merge(o, on="box_id")
    if zone_of_box is not None:
        b["zone_id"] = b.box_id.map(zone_of_box).fillna(b.zone_id)
    used = b[b.split.isin(SPLITS)]                     # module_code is NOT a module identity (README): flag collisions
    cs = used.groupby(["farm", "module_code"]).split.nunique()
    coll = set(cs[cs > 1].index)
    b["code_collision"] = [(f, m) in coll for f, m in zip(b.farm, b.module_code)]
    keep = ["box_id", "split", "reason", "farm", "flight_id", "class_final", "in_eval_zone", "module_code",
            "phys_module_id", "ref_cell", "zone_id", "n_pairs_kept", "code_collision", "centre_E", "centre_N",
            "self_drawn"]
    return b[keep]


def la_zone_of_boxes(L, st):
    b = L.boxes[L.boxes.farm == "TN19"]
    return dict(zip(b.box_id, st.zone_of(b.centre_E.values, b.centre_N.values)))


def la_counts(boxes_df, classes, protocol):
    """long table protocol x split x class: boxes inside the eval zone, all boxes, physical modules (in zone)."""
    rec = []
    for s in SPLITS:
        x = boxes_df[boxes_df.split == s]
        for c in classes + ["ALL"]:
            xc = x if c == "ALL" else x[x.class_final == c]
            z = xc[xc.in_eval_zone]
            rec.append(dict(protocol=protocol, split=s, class_final=c, boxes_in_eval_zone=len(z),
                            boxes_all=len(xc), phys_modules_in_eval_zone=z.phys_module_id.nunique()))
    return rec


def la_audit(L, name, rows, boxes_df, buf, strict, cells_tree, cells):
    """footprint audit (build_footprints.audit_split) + box-level leakage audit."""
    aud = bf.audit_split(rows[rows.split.isin(SPLITS)][["farm", "flight_id", "file", "split"]], buf, verbose=False)
    used = boxes_df[boxes_df.split.isin(SPLITS)].reset_index(drop=True)
    r = dict(protocol=name, buffer_m=buf, strict=strict)
    for s in SPLITS:
        r[f"pairs_{s}"] = int(rows[rows.split == s].pair_id.nunique())
    r["pairs_buffer_excluded"] = int(rows[rows.split == "buffer_excluded"].pair_id.nunique())
    for s in SPLITS:
        r[f"images_{s}"] = int((rows.split == s).sum())
    r["cross_split_footprint_pairs"] = aud["n_cross_split_pairs"]
    for key in ("module_code", "phys_module_id", "ref_cell"):
        d = used[used[key].astype(str) != ""]
        d = d.assign(**{key: d.farm + "/" + d[key].astype(str)}) if key == "module_code" else d
        sets = {s: set(d[d.split == s][key]) for s in SPLITS}
        tot = 0
        for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
            n = len(sets[a_] & sets[b_])
            r[f"shared_{key}_{a_}_{b_}"] = n
            tot += n
        r[f"shared_{key}_total"] = tot
    pts = np.array([Point(x, y) for x, y in zip(used.centre_E, used.centre_N)], dtype=object)
    pr = rows[rows.split.isin(SPLITS)].drop_duplicates("pair_id")
    ug = L.units.set_index("pair_id").geometry.reindex(pr.pair_id).values      # union footprints of kept pairs
    bi, gj = STRtree(ug).query(pts, predicate="within")
    cross = used.split.values[bi] != pr.split.values[gj]
    r["boxes_visible_in_other_split_images"] = int(len(np.unique(bi[cross])))
    from scipy.spatial import cKDTree
    xy = np.c_[used.centre_E, used.centre_N]
    ds = [cKDTree(xy[(used.split == a_).values]).query(xy[(used.split == b_).values])[0].min()
          for a_ in SPLITS for b_ in SPLITS
          if a_ < b_ and (used.split == a_).any() and (used.split == b_).any()
          and used[used.split.isin([a_, b_])].farm.nunique() == 1]
    r["min_dist_between_split_boxes_m"] = round(float(min(ds)), 2) if ds else np.nan
    r["code_collision_boxes"] = int(used.code_collision.sum())
    dd = []                                           # distance between same-module_code boxes of different splits
    for _, g in used[used.code_collision].groupby(["farm", "module_code"]):
        for i_, a_ in g.iterrows():
            for j_, b_ in g.iterrows():
                if a_.split < b_.split:
                    dd.append(float(np.hypot(a_.centre_E - b_.centre_E, a_.centre_N - b_.centre_N)))
    r["code_collision_min_dist_m"] = round(min(dd), 1) if dd else np.nan
    k = rows[rows.split.isin(SPLITS) & (rows.farm == "TN19")]
    if len(k):
        sh, _, _ = cell_sharing(k, cells, cells_tree)
        r.update({f"cells_shared_{kk.replace('&', '_')}": v for kk, v in sh.items()})
    ok = (r["cross_split_footprint_pairs"] == 0 and r["shared_phys_module_id_total"] == 0
          and r["boxes_visible_in_other_split_images"] == 0)
    r["verdict"] = "PASS" if ok else ("EXPECTED (shares modules by design)" if not strict else "FAIL")
    return r


def la_xalt_rows(L, st, kind, b):
    """31 m -> 35 m on the label-aware zones.  strict: test = 35 m pairs in test zones.  shared: test = 35 m pairs
    of the TRAIN zones whose thermal footprint lies >= 50 % over kept 31 m train footprints (same modules)."""
    u = L.units[L.units.farm == "TN19"]
    a = st.assign(u, b).merge(u[["pair_id", "altitude_m"]], on="pair_id")
    s31, t35 = a[a.altitude_m == 31], a[a.altitude_m == 35]
    tt = t35[t35.split == ("test" if kind == "strict" else "train")].copy()
    n_unp = 0
    if kind == "shared":
        tr31 = unary_union(list(u.set_index("pair_id").loc[s31[s31.split == "train"].pair_id].geometry))
        tfp = pd.Series(L.tfp, index=L.units.pair_id.values)
        ov = np.array([tfp[p].intersection(tr31).area / tfp[p].area for p in tt.pair_id])
        n_unp = int((ov < 0.5).sum())
        tt = tt[ov >= 0.5].copy()
    tt["split"] = "test"
    ps = pd.concat([s31[s31.split.isin(["train", "val", "buffer_excluded"])], tt,
                    t35[t35.split == "buffer_excluded"]], ignore_index=True)
    return ps[["pair_id", "zone_id", "split"]], n_unp


def run_label_aware(figures=True):
    units, imgs = load_units(("TN19", "TNDL", "TT"))
    cells = load_cells()
    ctree = STRtree(cells.geometry.values)
    L = LA(units, imgs)
    print(f"[la] boxes {len(L.boxes)}: " + ", ".join(f"{k} {v}" for k, v in L.boxes.class_final.value_counts().items()),
          f"| classes {L.classes} | pairs {len(L.units)}")
    c0, g0, _ = la_search(L, CHOSEN_BUFFER, ks=(2,), step_u=20.0, step_v=20.0, refine=5.0, verbose=False)
    print(f"[la] guillotine start: {g0.describe()} cost {c0:.3f}")
    cost, st, cs = la_anneal(L, CHOSEN_BUFFER, g0)
    print(f"[la] chosen zoning: {st.describe()}; cost {cost:.3f}; usable boxes (cell x class {L.classes}):\n{cs}")
    protocols, boxfiles, counts, audits = {}, {}, [], []
    zone_of = la_zone_of_boxes(L, st)
    u19 = L.units[L.units.farm == "TN19"].reset_index(drop=True)

    def save(name, rows, boxes_df, buf, strict):
        extra = [c for c in rows.columns if c not in LA_COLS + ["geometry"]]
        rows[LA_COLS + extra].to_csv(OUT / f"label_aware_{name}.csv", index=False)
        boxes_df.to_csv(OUT / f"label_aware_{name}_boxes.csv", index=False)
        protocols[name], boxfiles[name] = rows, boxes_df
        counts.extend(la_counts(boxes_df, L.classes, name))
        audits.append(la_audit(L, name, rows, boxes_df, buf, strict, ctree, cells))
        a = audits[-1]
        print(f"  {name}: pairs {a['pairs_train']}/{a['pairs_val']}/{a['pairs_test']} excl {a['pairs_buffer_excluded']} "
              f"cross-fp {a['cross_split_footprint_pairs']} shared phys {a['shared_phys_module_id_total']} "
              f"code {a['shared_module_code_total']} cell {a['shared_ref_cell_total']} "
              f"visible {a['boxes_visible_in_other_split_images']} -> {a['verdict']}")

    for b in BUFFERS:                                                  # LA-A main
        ps = st.assign(u19, b)
        rows = la_image_rows(L, ps[["pair_id", "zone_id", "split"]])
        rows = rows[rows.farm == "TN19"]
        save(f"main_buf{b:02d}", rows, la_box_rows(L, rows, zone_of), b, True)
    unp = {}
    for kind in ("strict", "shared"):                                  # LA-B 31 -> 35
        ps, unp[kind] = la_xalt_rows(L, st, kind, CHOSEN_BUFFER)
        rows = la_image_rows(L, ps)
        rows = rows[rows.farm == "TN19"]
        save(f"xalt_{kind}_31to35", rows, la_box_rows(L, rows, zone_of), CHOSEN_BUFFER, kind == "strict")
    main10 = st.assign(u19, CHOSEN_BUFFER)                             # LA-C TN19 -> other farm
    main10["split"] = main10.split.replace({"test": "train"})
    for key, fid in LA_TARGET.items():
        tu = L.units[L.units.flight_id == fid]
        ts = pd.DataFrame({"pair_id": tu.pair_id.values, "zone_id": f"{key.upper()}-all", "split": "test"})
        rows = la_image_rows(L, pd.concat([main10[["pair_id", "zone_id", "split"]], ts], ignore_index=True))
        rows = rows[rows.farm.isin(["TN19", key.upper()])]
        save(f"xfarm_tn19_to_{key}", rows, la_box_rows(L, rows, zone_of), CHOSEN_BUFFER, True)
    cnt = pd.DataFrame(counts)
    cnt.to_csv(OUT / "label_aware_box_counts.csv", index=False)
    fl = []                       # boxes (in eval zone) per protocol, flight and split/status: what is used / lost / unusable
    for name, bx_ in boxfiles.items():
        z = bx_[bx_.in_eval_zone]
        t = z.groupby(["flight_id", "split"]).size().unstack(fill_value=0)
        for c_ in ("train", "val", "test", "buffer_excluded", "unused"):
            if c_ not in t:
                t[c_] = 0
        t = t[["train", "val", "test", "buffer_excluded", "unused"]].reset_index()
        t.insert(0, "protocol", name)
        fl.append(t)
    pd.concat(fl).to_csv(OUT / "label_aware_box_counts_by_flight.csv", index=False)
    aud = pd.DataFrame(audits)
    aud["unpaired_35m_pairs_not_used"] = aud.protocol.map({f"xalt_{k}_31to35": v for k, v in unp.items()})
    aud.to_csv(OUT / "label_aware_audit_summary.csv", index=False)
    st.table().to_csv(OUT / "label_aware_zones.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(cnt[cnt.class_final != "ALL"].pivot_table(index=["protocol", "split"], columns="class_final",
                                                    values="boxes_in_eval_zone", aggfunc="sum").to_string())
    print(aud.drop(columns=["strict"]).to_string(index=False))
    if figures:
        la_figures(L, st, protocols, boxfiles, cnt)
    return L, st, protocols, boxfiles, cnt, aud


def _la_zones(ax, st, labels=True):
    from matplotlib.patches import Polygon as MPoly
    for zid, (g, s) in st.polygons().items():
        for p in getattr(g, "geoms", [g]):
            xs, ys = p.exterior.xy
            ax.add_patch(MPoly(np.c_[xs, ys], closed=True, fc=COL[s], ec="k", lw=0.5, alpha=0.18, zorder=0))
        if labels:
            c = g.representative_point()
            ax.text(c.x, c.y, zid.split("-")[-1], fontsize=6, ha="center", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))


def la_figures(L, st, protocols, boxfiles, cnt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    FIG = OUT / "figures"
    FIG.mkdir(exist_ok=True)
    CC = {"SingleHotSpot": "#2ca02c", "MultiHotSpot": "#d62728", "Bypass": "#1f77b4", "ModuleOpenCircuit": "#9467bd"}
    win = box(265330, 1263570, 265880, 1263890)

    def bars(ax, name, title):
        c = cnt[(cnt.protocol == name) & (cnt.class_final != "ALL")]
        bottom = np.zeros(3)
        for cl in L.classes:
            v = np.array([c[(c.split == s) & (c.class_final == cl)].boxes_in_eval_zone.sum() for s in SPLITS])
            if v.sum() == 0:
                continue
            ax.bar(SPLITS, v, bottom=bottom, color=CC.get(cl, "0.5"), label=cl)
            for i, (vv, bb) in enumerate(zip(v, bottom)):
                if vv >= 25:
                    ax.text(i, bb + vv / 2, str(vv), ha="center", va="center", fontsize=7, color="w")
            bottom += v
        for i, t in enumerate(bottom):
            ax.text(i, t, int(t), ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)

    def map_(ax, name, flights, title):
        _la_zones(ax, st)
        r = protocols[name]
        r = r[(r.sensor == "T") & r.flight_id.isin(flights)]
        _draw(ax, r, lw=0.12, every_excl=4)
        b = boxfiles[name]
        b = b[(b.farm == "TN19") & b.in_eval_zone & b.flight_id.isin(flights)]
        for cl in L.classes:
            x = b[b.class_final == cl]
            if len(x):
                ax.scatter(x.centre_E, x.centre_N, s=6, color=CC.get(cl, "0.5"), edgecolors="k", linewidths=0.2, zorder=5)
        ax.set_xlim(265390, 265840); ax.set_ylim(1263625, 1263830)
        ax.set_title(title, fontsize=8)

    h = [Line2D([0], [0], color=COL[s], lw=2, label=s) for s in SPLITS + ["buffer_excluded"]] + \
        [Line2D([0], [0], marker="o", ls="", color=CC[c], label=c) for c in L.classes if c in set(L.boxes.class_final)]
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    map_(fig.add_subplot(gs[0, :]), f"main_buf{CHOSEN_BUFFER:02d}", LA_TN19,
         f"LA-A label-aware main split, buffer {CHOSEN_BUFFER} m: strips (shading = owner), thermal footprints of the 4 labelled "
         "TN19 flights (colour = split), boxes in the eval zone (dots = class)")
    for k, b in enumerate((5, 10, 20)):
        bars(fig.add_subplot(gs[1, k]), f"main_buf{b:02d}", f"boxes in eval zone per split, buffer {b} m")
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.07, top=0.95, hspace=0.28)
    fig.savefig(FIG / "LA_A_main_label_aware.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for row, kind in zip(axs, ("strict", "shared")):
        n = f"xalt_{kind}_31to35"
        map_(row[0], n, LA_TN19[:1], f"LA-B {kind}: 31 m flight (train / val)")
        map_(row[1], n, LA_TN19[1:], f"LA-B {kind}: 35 m flights (test = " +
             ("test-zone strips" if kind == "strict" else "train-zone strips = same modules as 31 m train") + ")")
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.07, top=0.95, hspace=0.22)
    fig.savefig(FIG / "LA_B_xalt_label_aware.png", dpi=120); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    bars(axs[0], "xalt_strict_31to35", "LA-B strict 31->35 m")
    bars(axs[1], "xalt_shared_31to35", "LA-B shared 31->35 m")
    bars(axs[2], "xfarm_tn19_to_tndl", "LA-C TN19 -> TNDL 35M_T2")
    bars(axs[3], "xfarm_tn19_to_tt", "LA-C TN19 -> TT 39M_T1")
    axs[0].legend(fontsize=7)
    fig.suptitle("boxes inside the evaluation zone, by split and class", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "LA_BC_box_counts.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "audit", "figures", "all", "label_aware"])
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-phash", action="store_true")
    a = ap.parse_args()
    if a.cmd == "label_aware":          # independent of build/all: writes only label_aware_* files
        run_label_aware(figures=not a.no_figures)
        return
    units, imgs, cells, strips, protocols = build_all()
    if a.cmd in ("audit", "all"):
        df, nd, bal = audit_all(protocols, units, cells, do_phash=not a.no_phash)
        pd.set_option("display.width", 250, "display.max_columns", 40)
        print(df.drop(columns=["family", "by_split_pair"]).to_string(index=False))
    if a.cmd in ("figures", "all"):
        make_figures(protocols, strips, cells, units)


if __name__ == "__main__":
    main()
