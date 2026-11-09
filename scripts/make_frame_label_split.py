#!/usr/bin/env python
"""Frame-level, leakage-aware split of the labelled thermal frames (3000 frames, thermal_frame_boxes.csv).

    python -X utf8 code/scripts/make_frame_label_split.py                    # COMPONENT split (default): each frame/crop in exactly one split, 0 overlap
    python -X utf8 code/scripts/make_frame_label_split.py --variant B        # component split without crop windows at strip boundaries
    python -X utf8 code/scripts/make_frame_label_split.py --moc-mode strict  # archived fallback: strip + 10 m buffer strict split (-> _strip_buffer_strict/)
    python -X utf8 code/scripts/make_frame_label_split.py --moc-mode train|test|internal|relaxed --no-figures
    options: --no-sensitivity (skip margin sensitivity), --margin-rtk 1.5 --margin-bim2 5.0, --tune N, --out DIR

Component split (default; metadata/splits/README.md section 7):
  1. overlap graph per farm: an edge joins two labelled frames (any flight, any altitude) when their nominal footprints, each buffered by the safety
     margin (1.5 m RTK farms, 5 m BIM2), intersect, i.e. they are closer than 2 x margin; NO extra 10 m buffer;
  2. small farms (TNDL, TT, BIM2): whole connected components are assigned to train/val/test (exhaustive search, ~70/15/15 of the boxes and of each
     class, every class present where possible): no frame is lost;
  3. TN19 is one giant component (99 % of the frames): it is partitioned by strips (box-level strip search so that the balance is the one crop windows
     can reach); a frame is kept whole if footprint + margin lies in one strip run; MOC patch: three map regions with gap columns and crop windows;
     dropped frames are re-admitted greedily when no frame/crop of another split lies within 2 x margin (variant B); frames that still straddle a
     boundary yield CROP WINDOWS inside each strip run eroded by the margin (variant C, default), boxes kept if the centre is in the run and >= 70 %
     of the area inside the crop; only frames that yield no usable crop stay in the buffer;
  4. audit: cross-split pairs intersecting / within margin / within 2 x margin / within 2.3 m, shared reference cells, shared ortho modules, random
     split + archived splits for comparison, margin sensitivity, an independent re-computation of the footprints from the raw pose.
Outputs (dataset/metadata/splits/): frame_label_split.csv (+ crop_window), frame_label_split_crops.csv / _crop_boxes.csv (MOC + boundary crops),
_variants, _components, _component_tuning, _variant_B, _moc_{boxes,regions,layouts,alternatives,feasibility}, _xdomain_*, _{strips,counts,buffer_loss,
strict_audit,margin_sensitivity,independent_check}.csv, figures/fig_split_*.png. The strip + buffer strict split (--moc-mode strict), train/test/internal
and the relaxed split (SUPERSEDED, margin 0) write into _strip_buffer_strict/ and _superseded_relaxed/.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import shapely.affinity
from shapely import STRtree
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_footprints as bf  # noqa: E402
import make_splits as ms  # noqa: E402

ROOT = bf.ROOT
DS = ROOT / "dataset"
OUT = DS / "metadata/splits"
SUB = "annotations/thermal_anomaly_obb"
CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]
SPL = ["train", "val", "test"]
TARGET = np.array([0.60, 0.15, 0.25])
SEED = 20260421
STEP = 5.0
FARMS = ["TN19", "TNDL", "TT", "BIM2"]
MOC = 3
MOC_FLIGHT = "TN19_20260411_42M_T2"
MOC_GAP_COLS = 1          # relaxed rule: modules (cell columns) left unused between two groups
MOC_MIN_FRAMES, MOC_MIN_BOXES = 3, 30
W_ALT = 2.0
FORCED = {"train": 0, "test": 2}   # --moc-mode: the ModuleOpenCircuit block is forced into this split


# ----------------------------------------------------------------------------- data
def load_frames():
    fp = bf.read_gpkg_layer(bf.GPKG, "image_footprints")
    parts = []
    for farm, d in ms.FARM_DIR.items():
        fr = pd.read_csv(d / SUB / "frames_reviewed.csv")
        bx = pd.read_csv(d / SUB / "thermal_frame_boxes.csv")
        bx = bx[bx.kept_in_yolo == 1]
        cnt = bx.groupby(["flight_id", "thermal_file", "class_id"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(columns=range(4), fill_value=0)
        cnt.columns = [f"n_{c}" for c in CLASSES]
        fr = fr.merge(cnt, left_on=["flight_id", "thermal_file"], right_index=True, how="left")
        fr[cnt.columns] = fr[cnt.columns].fillna(0).astype(int)
        fr["farm"] = farm
        parts.append(fr)
    F = pd.concat(parts, ignore_index=True)
    F = F.merge(fp[["flight_id", "file", "geometry"]], left_on=["flight_id", "thermal_file"],
                right_on=["flight_id", "file"], how="left").drop(columns="file")
    assert F.geometry.notna().all()
    F["altitude_m"] = F.flight_id.map(ms.alt_of)
    F["cx"] = [g.centroid.x for g in F.geometry]
    F["cy"] = [g.centroid.y for g in F.geometry]
    return F


class Axis:
    """f(x,y) = (a x + b y + c)/|(a,b)| in metres along the strip-normal direction."""

    def __init__(self, a, b, c):
        self.a, self.b, self.c, self.n = a, b, c, float(np.hypot(a, b))

    def f(self, x, y):
        return (self.a * x + self.b * y + self.c) / self.n

    def extent(self, geoms, buf):
        lo, hi = np.empty(len(geoms)), np.empty(len(geoms))
        for i, g in enumerate(geoms):
            xy = np.asarray(g.exterior.coords)
            v = self.f(xy[:, 0], xy[:, 1])
            lo[i], hi[i] = v.min() - buf[i], v.max() + buf[i]
        return lo, hi


def farm_axis(F):
    if F.farm.iloc[0] == "TN19":
        return Axis(*ms.tn19_frame())
    c = np.c_[F.cx, F.cy]
    c0 = c.mean(0)
    w, v = np.linalg.eigh(np.cov((c - c0).T))
    ax = v[:, 1] if v[0, 1] > 0 else -v[:, 1]
    return Axis(ax[0], ax[1], -float(c0 @ ax))


# ----------------------------------------------------------------------------- search
class Problem:
    def __init__(self, F, buf, moc_mode):
        self.F, self.moc_mode = F.reset_index(drop=True), moc_mode
        self.ax = farm_axis(self.F)
        self.lo, self.hi = self.ax.extent(self.F.geometry.values, buf)
        self.fc = self.ax.f(self.F.cx.values, self.F.cy.values)
        alts = sorted(self.F.altitude_m.unique())
        cls = self.F[[f"n_{c}" for c in CLASSES]].values.astype(float)
        self.X = np.c_[cls, np.ones(len(self.F)), *[(self.F.altitude_m.values == a).astype(float) for a in alts]]
        self.tot = self.X.sum(0)
        self.nalt = len(alts)
        self.minreq = np.array([max(1.0, min(0.08 * t, 80.0)) for t in self.tot[:4]])
        self.ncls = [c for c in range(4) if self.tot[c] > 0]
        self.sp_cache = {}
        self.visited = {}

    def assignments(self, ns):
        if ns not in self.sp_cache:
            O = np.array(list(itertools.product(range(3), repeat=ns)), dtype=np.int8)
            R = np.concatenate([np.zeros((len(O), 1), np.int8), np.cumsum(O[:, 1:] != O[:, :-1], axis=1, dtype=np.int8)], 1)
            self.sp_cache[ns] = (O, R)
        return self.sp_cache[ns]

    def kept_counts(self, cuts):
        """-> O, K[n_assign, 3, F]"""
        ns = len(cuts) + 1
        O, R = self.assignments(ns)
        sl = np.searchsorted(cuts, self.lo, side="right")
        sh = np.searchsorted(cuts, self.hi, side="right")
        B = np.zeros((ns, ns, self.X.shape[1]))
        np.add.at(B, (sl, sh), self.X)
        K = np.zeros((len(O), 3, self.X.shape[1]))
        eye = np.eye(3)
        for a in range(ns):
            for b in range(a, ns):
                if not B[a, b].any():
                    continue
                m = (R[:, a] == R[:, b]).astype(float)
                K += (eye[O[:, a]] * m[:, None])[:, :, None] * B[a, b][None, None, :]
        return O, K

    def cost(self, K):
        n, _, Fd = K.shape
        T = self.tot
        kept_c = K[:, :, :4]
        cost = np.zeros(n)
        for c in self.ncls:
            if c == MOC and self.moc_mode in FORCED:
                continue
            s = kept_c[:, :, c].sum(1)
            share = np.where(s[:, None] > 0, kept_c[:, :, c] / np.maximum(s, 1)[:, None], 0.0)
            dev = np.abs(share - TARGET).sum(1)
            dev = np.where(s > 0, dev, 2.0)
            cost += dev
            pres = np.clip(1 - kept_c[:, :, c] / self.minreq[c], 0, 1).sum(1)
            cost += 6.0 * pres / 1.0
        if MOC in self.ncls and self.moc_mode in FORCED:
            fs = FORCED[self.moc_mode]
            lost = T[MOC] - kept_c[:, :, MOC].sum(1)
            other = kept_c[:, :, MOC].sum(1) - kept_c[:, fs, MOC]
            cost += 40.0 * (other + lost) / T[MOC]
        nb = max(T[:4].sum(), 1)
        cost += 4.0 * (T[:4].sum() - kept_c.sum((1, 2))) / nb
        cost += 2.0 * (T[4] - K[:, :, 4].sum(1)) / T[4]
        if self.nalt > 1:
            overall = T[5:] / T[4]
            fr = K[:, :, 4]
            mix = K[:, :, 5:] / np.maximum(fr, 1)[:, :, None]
            dev = np.abs(mix - overall).sum(2)
            cost += W_ALT * ((fr / T[4]) * dev).sum(1)
        return cost

    def best(self, cuts):
        O, K = self.kept_counts(np.asarray(cuts))
        c = self.cost(K)
        i = int(np.argmin(c))
        self.visited[tuple(np.asarray(cuts).tolist())] = (float(c[i]), O[i].copy())
        return c[i], O[i]

    def search(self, kmin, kmax, restarts=10, min_gap=15.0):
        rng = np.random.default_rng(SEED)
        grid = np.arange(np.floor(self.fc.min()) + STEP, np.ceil(self.fc.max()), STEP)
        best = (1e18, None, None)
        for k in range(kmin, kmax + 1):
            for r in range(restarts):
                cuts = np.sort(rng.choice(grid, k, replace=False))
                if np.any(np.diff(cuts) < min_gap):
                    continue
                cur, own = self.best(cuts)
                improved = True
                while improved:
                    improved = False
                    for i in range(k):
                        lo_b = cuts[i - 1] + min_gap if i else grid[0]
                        hi_b = cuts[i + 1] - min_gap if i < k - 1 else grid[-1]
                        for p in grid[(grid >= lo_b) & (grid <= hi_b)]:
                            if p == cuts[i]:
                                continue
                            t = cuts.copy(); t[i] = p
                            c, o = self.best(t)
                            if c < cur - 1e-12:
                                cur, own, cuts, improved = c, o, t, True
                if cur < best[0] - 1e-12:
                    best = (cur, cuts.copy(), own)
            print(f"   k={k}: best so far cost {best[0]:.3f} cuts {np.round(best[1], 1)}", flush=True)
        return best

    def apply(self, cuts, own, buf, prefix):
        """-> DataFrame(split, strip_id, reason) aligned to self.F, strip table."""
        ns = len(cuts) + 1
        run = np.concatenate([[0], np.cumsum(own[1:] != own[:-1])])
        edges = [self.fc.min() - 1, *cuts, self.fc.max() + 1]
        runs = sorted(set(run))
        sid = {r: f"{prefix}-S{i + 1:02d}" for i, r in enumerate(runs)}
        sl = np.searchsorted(cuts, self.lo, side="right")
        sh = np.searchsorted(cuts, self.hi, side="right")
        sc = np.searchsorted(cuts, self.fc, side="right")
        split, strip, reason = [], [], []
        for i in range(len(self.F)):
            strip.append(sid[run[sc[i]]])
            if run[sl[i]] == run[sh[i]]:
                split.append(SPL[own[sl[i]]])
                reason.append(f"inside strip {sid[run[sl[i]]]} (owner {SPL[own[sl[i]]]}), footprint+{buf[i]:g} m")
            else:
                split.append("buffer")
                a, b = sid[run[sl[i]]], sid[run[sh[i]]]
                reason.append(f"buffer: footprint+{buf[i]:g} m touches strips {a} and {b} of different owners")
        tab = []
        for r in runs:
            segs = [s for s in range(ns) if run[s] == r]
            tab.append(dict(strip_id=sid[r], owner=SPL[own[segs[0]]], f_from_m=round(edges[segs[0]], 1),
                            f_to_m=round(edges[segs[-1] + 1], 1)))
        return pd.DataFrame({"split": split, "strip_id": strip, "reason": reason}), pd.DataFrame(tab)


def split_farm(F, farm, moc_mode, moc_buffer, base_buf=10.0):
    sub = F[F.farm == farm].reset_index(drop=True)
    buf = np.full(len(sub), float(base_buf))
    if moc_mode == "internal" and farm == "TN19":
        buf[sub[f"n_{CLASSES[MOC]}"].values > 0] = moc_buffer
    P = Problem(sub, buf, moc_mode if farm == "TN19" else "none")
    kmin, kmax = (2, 6) if farm == "TN19" else (2, 3)
    print(f"[{farm}] {len(sub)} frames; moc_mode={moc_mode}", flush=True)
    cost, cuts, own = P.search(kmin, kmax)
    res, strips = P.apply(cuts, own, buf, farm)
    strips.insert(0, "farm", farm)
    out = pd.concat([sub[["farm", "flight_id", "thermal_file", "altitude_m"]
                         + [f"n_{c}" for c in CLASSES]], res], axis=1)
    if farm == "TN19" and moc_mode in FORCED:
        m = (out[f"n_{CLASSES[MOC]}"] > 0)
        out.loc[m & (out.split == SPL[FORCED[moc_mode]]), "reason"] += (
            f"; ModuleOpenCircuit block kept in {SPL[FORCED[moc_mode]]} (--moc-mode {moc_mode})")
    print(f"[{farm}] cost {cost:.3f}, cuts {np.round(cuts, 1)}, owners {[SPL[o] for o in own]}")
    return out, strips, P


# ----------------------------------------------------------------------------- audit
def load_assoc():
    try:
        ba = pd.read_csv(DS / "metadata/association/thermal_box_to_cell.csv", usecols=["box_id", "module_id", "status"])
        ba = ba[ba.status == "unique"]
        det = pd.read_csv(DS / ms.FARM_DIR["TN19"].name / SUB / "thermal_box_frame_detections.csv",
                          usecols=["flight_id", "box_id", "thermal_file", "x_px", "y_px"])
        return det.merge(ba[["box_id", "module_id"]], on="box_id")
    except Exception as e:  # pragma: no cover
        print("association unavailable:", e)
        return None


def audit(F, split, name, cells=None, cell_tree=None, assoc=None, dwithin=10.0):
    """F: frames with geometry (aligned to split array). Returns dict of leakage metrics over non-buffer frames."""
    keep = np.where(split != "buffer")[0]
    g = F.geometry.values[keep]
    sp = split[keep]
    tree = STRtree(g)
    ii, jj = tree.query(g, predicate="intersects")
    m = ii < jj
    ii, jj = ii[m], jj[m]
    cross = sp[ii] != sp[jj]
    inter = shapely.area(shapely.intersection(g[ii[cross]], g[jj[cross]]))
    n_pairs = int((inter > 1e-6).sum())
    d1, d2 = tree.query(g, predicate="dwithin", distance=dwithin)
    m2 = d1 < d2
    n_near = int((sp[d1[m2]] != sp[d2[m2]]).sum())
    unions = {s: unary_union(list(g[sp == s])) for s in SPL if (sp == s).any()}
    out = dict(variant=name, n_frames={s: int((sp == s).sum()) for s in SPL}, n_buffer=int((split == "buffer").sum()),
               cross_split_overlapping_pairs=n_pairs, cross_split_pairs_within_10m=n_near)
    ov = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in unions and b in unions:
            x = unions[a].intersection(unions[b]).area
            ov[f"{a}-{b}"] = (round(x, 1), round(100 * x / unions[a].area, 2), round(100 * x / unions[b].area, 2))
    out["shared_area_m2_pct_of_first_pct_of_second"] = ov
    if cells is not None:
        cs = {}
        for s in SPL:
            if (sp == s).any():
                idx = cell_tree.query(unions[s], predicate="intersects")
                cs[s] = set(idx.tolist())
        out["cells_per_split"] = {s: len(v) for s, v in cs.items()}
        out["shared_cells"] = {f"{a}-{b}": len(cs[a] & cs[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
                               if a in cs and b in cs}
    if assoc is not None:
        key = pd.Series(split, index=pd.MultiIndex.from_arrays([F.flight_id.values, F.thermal_file.values]))
        a = assoc.copy()
        a["split"] = [key.get((f, t), None) for f, t in zip(a.flight_id, a.thermal_file)]
        a = a[a.split.isin(SPL)]
        ns = a.groupby("module_id").split.nunique()
        out["assoc_modules_seen"] = int(len(ns))
        out["assoc_modules_in_2plus_splits"] = int((ns > 1).sum())
    return out


def flat(a):
    r = dict(variant=a["variant"], n_train=a["n_frames"].get("train", 0), n_val=a["n_frames"].get("val", 0),
             n_test=a["n_frames"].get("test", 0), n_buffer=a["n_buffer"],
             cross_split_overlapping_frame_pairs=a["cross_split_overlapping_pairs"],
             cross_split_frame_pairs_within_10m=a["cross_split_pairs_within_10m"])
    for k, v in a["shared_area_m2_pct_of_first_pct_of_second"].items():
        r[f"shared_area_m2_{k}"], r[f"shared_pct_of_first_{k}"], r[f"shared_pct_of_second_{k}"] = v
    for k, v in a.get("shared_cells", {}).items():
        r[f"shared_cells_{k}"] = v
    for k, v in a.get("cells_per_split", {}).items():
        r[f"cells_touched_{k}"] = v
    if "assoc_modules_seen" in a:
        r["ortho_modules_seen"], r["ortho_modules_in_2plus_splits"] = a["assoc_modules_seen"], a["assoc_modules_in_2plus_splits"]
    return r


def random_split(n, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(SPL, size=n, p=TARGET)


def counts_table(S, variant):
    """frames / boxes per class per split; ModuleOpenCircuit boxes flagged `ignore` (n_moc_ignored) are not counted."""
    S = S.copy()
    if "n_moc_ignored" not in S:
        S["n_moc_ignored"] = 0
    S[f"k_{CLASSES[MOC]}"] = S[f"n_{CLASSES[MOC]}"] - S.n_moc_ignored
    kc = [f"n_{c}" if c != CLASSES[MOC] else f"k_{c}" for c in CLASSES]
    rows = []
    for (farm), g in list(S.groupby("farm")) + [("ALL", S)]:
        for s in SPL + ["buffer"]:
            h = g[g.split == s]
            nneg = int((h[[f"n_{c}" for c in CLASSES]].sum(1) == 0).sum())
            rows.append(dict(variant=variant, farm=farm, split=s, n_frames=len(h),
                             n_frames_reviewed_negative=nneg,
                             n_boxes=int(h[kc].values.sum()),
                             **{f"n_{c}": int(h[k].sum()) for c, k in zip(CLASSES, kc)},
                             n_moc_ignored=int(h.n_moc_ignored.sum()),
                             **{f"frames_{a}m": int((h.altitude_m == a).sum()) for a in sorted(S.altitude_m.unique())}))
    return pd.DataFrame(rows)


def moc_feasibility(F):
    """Best straight cut through the MOC block: frames fully on each side after buffering (footprint + b)."""
    t = F[F.farm == "TN19"].reset_index(drop=True)
    ax = farm_axis(t)
    mo = t[t[f"n_{CLASSES[MOC]}"] > 0]
    rows = []
    for b in (0, 2, 5, 10, 15, 20):
        lo, hi = ax.extent(mo.geometry.values, np.full(len(mo), float(b)))
        best = (-1, 0, 0, 0, 0, 0)
        for c in np.arange(lo.min(), hi.max(), 0.5):
            L, R = hi <= c, lo >= c
            cand = (min(L.sum(), R.sum()), L.sum(), R.sum(), int(mo[f"n_{CLASSES[MOC]}"][L].sum()),
                    int(mo[f"n_{CLASSES[MOC]}"][R].sum()), c)
            if cand[:5] > best[:5]:
                best = cand
        rows.append(dict(buffer_m=b, n_moc_frames=len(mo), frames_side_a=best[1], frames_side_b=best[2],
                         boxes_side_a=best[3], boxes_side_b=best[4],
                         frames_lost_to_buffer=len(mo) - best[1] - best[2],
                         moc_block_extent_m=round(float(ax.f(mo.cx.values, mo.cy.values).ptp()), 1)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- relaxed ModuleOpenCircuit rule
def moc_ground_boxes(F, cells):
    """MOC boxes of the MOC flight with ground centre (bilinear map of the footprint corners BL,BR,TR,TL), reference cell and
    cell column along the table rows. Cell = the grid cell containing (nearest to) the projected centre."""
    d = ms.FARM_DIR["TN19"] / SUB
    bx = pd.read_csv(d / "thermal_frame_boxes.csv")
    bx = bx[(bx.kept_in_yolo == 1) & (bx.flight_id == MOC_FLIGHT) & (bx.class_id == MOC)].reset_index(drop=True)
    g = F[F.flight_id == MOC_FLIGHT].set_index("thermal_file").geometry
    E, N = np.empty(len(bx)), np.empty(len(bx))
    for i, r in enumerate(bx.itertuples()):
        P = np.asarray(g[r.thermal_file].exterior.coords)[:4]
        sx = np.mean([r.x1_px, r.x2_px, r.x3_px, r.x4_px]) / 640.0
        sy = 1.0 - np.mean([r.y1_px, r.y2_px, r.y3_px, r.y4_px]) / 512.0
        p = (1 - sx) * (1 - sy) * P[0] + sx * (1 - sy) * P[1] + sx * sy * P[2] + (1 - sx) * sy * P[3]
        E[i], N[i] = p
    tree = STRtree(cells.geometry.values)
    idx = tree.nearest(shapely.points(E, N))
    out = bx[["box_uid", "flight_id", "thermal_file"]].copy()
    out["E"], out["N"] = E, N
    out["ref_cell"] = cells.module_id.values[idx]
    cx = cells.x.values[idx]
    out["col"] = np.round((cx - cx.min()) / 1.9).astype(int)      # cell pitch along the rows is 1.9 m
    return out


def moc_frame_split(B):
    """B has `group`; frame -> group holding the majority of its non-gap boxes (ties: first in train, val, test order)."""
    rows = []
    for (fl, tf), g in B.groupby(["flight_id", "thermal_file"]):
        vc = g.group[g.group != "gap"].value_counts().reindex(SPL).fillna(0)
        rows.append((fl, tf, SPL[int(np.argmax(vc.values))] if vc.sum() > 0 else "buffer"))
    return pd.DataFrame(rows, columns=["flight_id", "thermal_file", "frame_split"])


def moc_choose_groups(B):
    """Exhaustive search of two cut columns (a gap of MOC_GAP_COLS columns at each cut) and of the order of the groups."""
    ncol = int(B.col.max()) + 1
    T = np.array([0.6, 0.2, 0.2])
    best = None
    for order in itertools.permutations(SPL):
        for a in range(1, ncol):                    # group 1 = columns [0, a), gap, group 2 = [a+gap, b), gap, group 3 = [b+gap, ncol)
            for b in range(a + MOC_GAP_COLS + 1, ncol - MOC_GAP_COLS):
                grp = np.array(["gap"] * ncol, dtype=object)
                grp[:a] = order[0]
                grp[a + MOC_GAP_COLS:b] = order[1]
                grp[b + MOC_GAP_COLS:] = order[2]
                Bg = B.assign(group=grp[B.col.values])
                fs = moc_frame_split(Bg)
                kept = Bg.merge(fs, on=["flight_id", "thermal_file"])
                kept = kept[kept.group == kept.frame_split]
                nb = kept.group.value_counts().reindex(SPL).fillna(0).values
                nf = fs.frame_split.value_counts().reindex(SPL).fillna(0).values
                mod = np.array([(grp == s_).sum() for s_ in SPL], float)
                if nb.min() < MOC_MIN_BOXES or nf.min() < MOC_MIN_FRAMES:
                    continue
                cost = np.abs(nb / nb.sum() - T).sum() + np.abs(mod / mod.sum() - T).sum() + (1.0 - nb.sum() / len(B))
                if best is None or cost < best[0] - 1e-12:
                    best = (cost, grp.copy(), order, a, b)
    assert best is not None, "no feasible relaxed MOC split"
    return best


def moc_box_table(B, frame_split, mapfn=lambda x: x):
    """box_uid, split, reason for a frame table (flight_id, thermal_file, frame_split). `mapfn` re-maps group names (xdomain)."""
    t = B.merge(frame_split, on=["flight_id", "thermal_file"], how="left")
    sp, rs = [], []
    for r in t.itertuples():
        if pd.isna(r.frame_split) or r.frame_split == "buffer":
            sp.append("buffer"); rs.append("frame is in the buffer")
        elif r.group == "gap":
            sp.append("ignore"); rs.append("module column in the gap between two groups (not used)")
        elif mapfn(r.group) == r.frame_split:
            sp.append(r.frame_split); rs.append(f"module group {r.group}, frame split {r.frame_split}")
        else:
            sp.append("ignore"); rs.append(f"module group {r.group} but frame split {r.frame_split}: box of another group in a shared frame")
    return pd.DataFrame({"box_uid": t.box_uid, "split": sp, "reason": rs, "flight_id": t.flight_id, "thermal_file": t.thermal_file,
                         "frame_split": t.frame_split, "module_group": t.group, "ref_cell": t.ref_cell, "cell_column": t.col})


def apply_relaxed(S, F, cells):
    """S: strip result of --moc-mode train. Re-split the MOC patch by module groups.
    -> S (+ moc_relaxed, n_moc_ignored), per-box table, group info."""
    B = moc_ground_boxes(F, cells)
    cost, grp, order, a, b = moc_choose_groups(B)
    B["group"] = grp[B.col.values]
    cur = S[S.flight_id == MOC_FLIGHT].set_index("thermal_file").split
    fsplit = moc_frame_split(B)
    # only frames the strips put in train are re-assigned; a frame in the buffer stays in the buffer (its boxes are flagged)
    fsplit["frame_split"] = [fs if cur[tf] == "train" else cur[tf] for tf, fs in zip(fsplit.thermal_file, fsplit.frame_split)]
    BT = moc_box_table(B, fsplit)
    BT["E"], BT["N"] = B.E.values, B.N.values
    ign = BT[BT.split == "ignore"].groupby(["flight_id", "thermal_file"]).size()
    S = S.copy()
    S["moc_relaxed"] = 0
    S["n_moc_ignored"] = 0
    key = {(fl, tf): sp for fl, tf, sp in fsplit.itertuples(index=False)}
    for i, r in S[(S.flight_id == MOC_FLIGHT) & (S.n_ModuleOpenCircuit > 0)].iterrows():
        k = (r.flight_id, r.thermal_file)
        if r.split != "train":
            continue
        n_grp = B[B.thermal_file == r.thermal_file].group.value_counts()
        S.loc[i, "moc_relaxed"] = 1
        S.loc[i, "n_moc_ignored"] = int(ign.get(k, 0))
        S.loc[i, "split"] = key[k]
        S.loc[i, "reason"] = (r.reason + f"; ModuleOpenCircuit patch, relaxed rule: frame to {key[k]} by majority of its MOC boxes ("
                              + ", ".join(f"{g_} {n_}" for g_, n_ in n_grp.items()) + ")")
    info = dict(cost=cost, order=order, a=a, b=b, ncol=int(B.col.max()) + 1, groups=grp)
    return S, BT, info


def pair_overlap(geoms_a, geoms_b):
    """intersecting pairs between two polygon arrays: (ia, ib, inter_area, iou, share_of_smaller)."""
    t = STRtree(geoms_b)
    ia, ib = t.query(geoms_a, predicate="intersects")
    inter = shapely.area(shapely.intersection(geoms_a[ia], geoms_b[ib]))
    ok = inter > 1e-6
    ia, ib, inter = ia[ok], ib[ok], inter[ok]
    aa, ab = shapely.area(geoms_a[ia]), shapely.area(geoms_b[ib])
    return ia, ib, inter, inter / (aa + ab - inter), inter / np.minimum(aa, ab)


def moc_leakage(S, BT, cells):
    """Residual overlap of the relaxed ModuleOpenCircuit split (inside the patch; other frames of TN19 for the visibility rows)."""
    keep = S[(S.farm == "TN19") & S.split.isin(SPL)].reset_index(drop=True)
    is_moc = ((keep.flight_id == MOC_FLIGHT) & (keep.n_ModuleOpenCircuit > 0)).values
    mf = keep[is_moc].reset_index(drop=True)
    g = mf.geometry.values
    ia, ib, inter, iou, sh = pair_overlap(g, g)
    m = (ia < ib) & (mf.split.values[ia] != mf.split.values[ib])
    rows = []

    def add(k, v, note=""):
        rows.append(dict(metric=k, value=v, note=note))
    for s_ in SPL:
        add(f"moc_frames_{s_}", int((mf.split == s_).sum()), "frames of the patch (ModuleOpenCircuit boxes) in this split")
    add("moc_frames_in_buffer", int(((S.farm == "TN19") & (S.split == "buffer") & (S.flight_id == MOC_FLIGHT) & (S.n_ModuleOpenCircuit > 0)).sum()),
        "frame of the patch left in the buffer by the strips (not used)")
    cross_total = sum((mf.split == a_).sum() * (mf.split == b_).sum() for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")))
    add("moc_frame_pairs_cross_split_total", int(cross_total), "all pairs of patch frames of different splits")
    add("moc_frame_pairs_cross_split_overlapping", int(m.sum()), "of these, pairs with intersecting footprints")
    add("moc_overlap_mean_IoU", round(float(iou[m].mean()), 3) if m.any() else 0.0, "mean over the overlapping cross-split pairs")
    add("moc_overlap_mean_share_of_smaller_footprint", round(float(sh[m].mean()), 3) if m.any() else 0.0, "mean over the overlapping cross-split pairs")
    add("moc_overlap_max_share_of_smaller_footprint", round(float(sh[m].max()), 3) if m.any() else 0.0)
    un = {s_: unary_union(list(mf.geometry[mf.split == s_])) for s_ in SPL}
    for s_ in ("val", "test"):
        add(f"moc_{s_}_footprint_share_covered_by_train_patch_frames",
            round(un[s_].intersection(un["train"]).area / un[s_].area, 3), "area share of the union of the patch frames of this split")
    add("moc_val_footprint_share_covered_by_test_patch_frames", round(un["val"].intersection(un["test"]).area / un["val"].area, 3))
    other = keep.geometry.values
    ia2, ib2, _, _, _ = pair_overlap(g, other)
    c = (mf.split.values[ia2] != keep.split.values[ib2]) & ~is_moc[ib2]
    add("moc_frames_vs_other_kept_frames_cross_split_overlapping_pairs", int(c.sum()),
        "patch frames vs the other kept TN19 frames (any flight/altitude, not carrying ModuleOpenCircuit) of another split")
    add("moc_frames_with_cross_split_overlap_in_other_frames", int(len(set(ia2[c].tolist()))), f"of {len(mf)} patch frames")
    ctree = STRtree(cells.geometry.values)
    cs = {s_: set(ctree.query(un[s_], predicate="intersects").tolist()) for s_ in SPL}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        add(f"moc_footprint_cells_shared_{a_}_{b_}", len(cs[a_] & cs[b_]),
            "reference cells under footprints of patch frames of both splits (cells seen in both, not labelled modules)")
    used = BT[BT.split.isin(SPL)]
    nsp = used.groupby("ref_cell").split.nunique()
    add("moc_labelled_cells_in_2plus_splits", int((nsp > 1).sum()),
        f"of {len(nsp)} reference cells carrying retained MOC boxes (one group per module column, so 0 by construction)")
    # visibility of retained val/test boxes in frames of another split
    u = used[used.split.isin(["val", "test"])]
    tree = STRtree(keep.geometry.values)
    pts = shapely.points(u.E.values, u.N.values)
    ip, ig = tree.query(pts, predicate="within")
    oth = keep.split.values[ig] != u.split.values[ip]
    hit = pd.Series(oth).groupby(ip).any().reindex(range(len(u)), fill_value=False).values
    hit_f = pd.Series(oth & (keep.flight_id.values[ig] == MOC_FLIGHT)).groupby(ip).any().reindex(range(len(u)), fill_value=False).values
    for s_ in ("val", "test"):
        mm = (u.split == s_).values
        add(f"moc_{s_}_boxes_visible_in_frame_of_another_split", int(hit[mm].sum()),
            f"of {int(mm.sum())} retained {s_} boxes whose centre lies inside the footprint of a kept frame of another split (all flights)")
        add(f"moc_{s_}_boxes_visible_in_other_split_frame_of_same_flight", int(hit_f[mm].sum()), f"of {int(mm.sum())}; frames of {MOC_FLIGHT} only")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- strict ModuleOpenCircuit rule (crop windows)
PITCH = 1.904            # reference-cell pitch along the table rows (m)
MIN_CROP_PX = 64
MIN_BOX_FRAC = 0.70
BIG = 1e9
CLS_COLS = [f"n_{c}" for c in CLASSES]


def g_map(P, px, py):
    """bilinear ground position of pixel (px, py) of a 640x512 thermal frame; P = footprint corners BL, BR, TR, TL."""
    sx, sy = np.asarray(px, float) / 640.0, 1.0 - np.asarray(py, float) / 512.0
    return (((1 - sx) * (1 - sy))[..., None] * P[0] + (sx * (1 - sy))[..., None] * P[1]
            + (sx * sy)[..., None] * P[2] + ((1 - sx) * sy)[..., None] * P[3])


def row_axes(cells):
    c = cells[cells.module_id.str.startswith("block2_string13")]
    al = np.arctan(np.polyfit(c.x, c.y, 1)[0])
    return np.array([np.cos(al), np.sin(al)]), np.array([-np.sin(al), np.cos(al)])


def best_crop(P, eu, ev, ub_lo, ub_hi, vb_lo, vb_hi, step=4, min_px=None):
    """largest axis-aligned pixel rectangle (x0, y0, x1, y1) whose ground image lies in u in [ub_lo, ub_hi], v in [vb_lo, vb_hi]."""
    cons = []                                           # c0 + A x + B y >= R  (all four corners)
    for ax_, lo, hi in ((eu, ub_lo, ub_hi), (ev, vb_lo, vb_hi)):
        f00, f10, f01 = (g_map(P, *q) @ ax_ for q in ((0, 0), (640, 0), (0, 512)))
        A, B = (f10 - f00) / 640.0, (f01 - f00) / 512.0
        if lo > -BIG / 2:
            cons.append((f00, A, B, lo))
        if hi < BIG / 2:
            cons.append((-f00, -A, -B, -hi))
    ys = np.arange(0, 513, step)
    y0, y1 = np.meshgrid(ys, ys, indexing="ij")
    mpx = MIN_CROP_PX if min_px is None else min_px
    ok = (y1 - y0) >= mpx
    lo0 = np.zeros(y0.shape); hi0 = np.full(y0.shape, 640.0); lo1 = lo0.copy(); hi1 = hi0.copy()
    for c0, A, B, R in cons:
        for y in (y0, y1):
            rhs = R - c0 - B * y
            if abs(A) < 1e-9:
                ok &= rhs <= 1e-9
            else:
                bound = rhs / A
                if A > 0:
                    lo0 = np.maximum(lo0, bound); lo1 = np.maximum(lo1, bound)
                else:
                    hi0 = np.minimum(hi0, bound); hi1 = np.minimum(hi1, bound)
    x0, x1 = np.ceil(lo0), np.floor(hi1)
    ok &= (x0 <= hi0) & (x1 >= lo1) & ((x1 - x0) >= mpx)
    area = np.where(ok, (x1 - x0) * (y1 - y0), -1.0)
    k = np.unravel_index(int(np.argmax(area)), area.shape)
    if area[k] <= 0:
        return None
    r = [int(x0[k]), int(y0[k]), int(x1[k]), int(y1[k])]
    for _ in range(40):                                  # exact bilinear check (the search used the affine approximation)
        cs = g_map(P, np.array([r[0], r[2], r[2], r[0]]), np.array([r[1], r[1], r[3], r[3]]))
        u, v = cs @ eu, cs @ ev
        bad = (u.min() < ub_lo - 1e-3) or (u.max() > ub_hi + 1e-3) or (v.min() < vb_lo - 1e-3) or (v.max() > vb_hi + 1e-3)
        if not bad:
            break
        r = [r[0] + 1, r[1] + 1, r[2] - 1, r[3] - 1]
    if r[2] - r[0] < mpx or r[3] - r[1] < mpx:
        return None
    return tuple(r)


def crop_polygon(P, r):
    return shapely.Polygon(g_map(P, np.array([r[0], r[2], r[2], r[0]]), np.array([r[1], r[1], r[3], r[3]])))


class StrictMOC:
    """Region/crop machinery for the ModuleOpenCircuit patch. Regions are bands along the table rows (u axis) with a gap of
    `gap_cols` module columns between neighbours; the outer two are open outwards; val/test are limited across the rows to
    the rows of the patch +- vpad (train is open across the rows). A crop window is the largest pixel rectangle whose ground
    image lies in the region eroded by `margin` m at every boundary facing another region."""

    def __init__(self, S0, F, cells, margin=1.5, gap_cols=1, vpad=3.0):
        self.margin, self.gap, self.vpad = margin, gap_cols, vpad
        self.eu, self.ev = row_axes(cells)
        self.B = moc_ground_boxes(F, cells)
        cx = cells.set_index("module_id").loc[self.B.ref_cell]
        xy = np.c_[cx.x.values, cx.y.values]
        self.B["u"], self.B["v"] = xy @ self.eu, xy @ self.ev
        self.u0 = self.B.u.min()
        self.B["col"] = np.round((self.B.u - self.u0) / PITCH).astype(int)
        self.ncol = int(self.B.col.max()) + 1
        self.vlo, self.vhi = self.B.v.min() - vpad, self.B.v.max() + vpad
        pf = S0[(S0.farm == "TN19") & (S0.flight_id == MOC_FLIGHT) & (S0.n_ModuleOpenCircuit > 0) & (S0.split == "train")]
        self.pf = pf.reset_index(drop=True)
        fb = pd.read_csv(ms.FARM_DIR["TN19"] / SUB / "thermal_frame_boxes.csv")
        fb = fb[(fb.kept_in_yolo == 1) & (fb.flight_id == MOC_FLIGHT)]
        self.fb = {tf: g.reset_index(drop=True) for tf, g in fb.groupby("thermal_file")}
        self.Bi = self.B.set_index("box_uid")
        self.cache = {}
        self._bi = {}

    def bd(self, k):
        return self.u0 + (k - 0.5) * PITCH

    def regions(self, order, a, b):
        G, m = self.gap, self.margin
        ivals = [(-BIG, self.bd(a)), (self.bd(a + G), self.bd(b)), (self.bd(b + G), BIG)]
        cols = [(-1, a), (a + G, b), (b + G, self.ncol + 1)]
        out = {}
        for s, (ua, ub), (c0, c1) in zip(order, ivals, cols):
            e = (ua + m if ua > -BIG / 2 else ua, ub - m if ub < BIG / 2 else ub)
            vb = (-BIG, BIG) if s == "train" else (self.vlo, self.vhi)
            out[s] = dict(u=(ua, ub), ue=e, v=vb, cols=(c0, c1))
        return out

    def _boxinfo(self, tf, P):
        """per-frame box polygons with ground (u, v) of the centre (independent of the region); cached."""
        if tf not in self._bi:
            lst = []
            for _, b in self.fb[tf].iterrows():
                pts = np.array([[b[f"x{j}_clip"] * 640, b[f"y{j}_clip"] * 512] for j in range(1, 5)])
                poly = shapely.Polygon(pts)
                if not poly.is_valid or poly.area <= 0:
                    continue
                cen = poly.centroid
                gc = g_map(P, cen.x, cen.y)
                lst.append((b, poly, cen, float(gc @ self.eu), float(gc @ self.ev)))
            self._bi[tf] = lst
        return self._bi[tf]

    def crop_boxes(self, tf, P, r, reg):
        """boxes of frame tf kept in crop r: list of dict(box_uid, cls, pts in crop pixel coordinates), number dropped
        (centre in the region but < 70 % of the box area inside the crop)."""
        x0, y0, x1, y1 = r
        rect = shapely.box(x0, y0, x1, y1)
        kept, dropped = [], 0
        for b, poly, cen, u, v in self._boxinfo(tf, P):
            in_region = reg["u"][0] <= u <= reg["u"][1] and reg["v"][0] <= v <= reg["v"][1]
            if b.class_id == MOC:
                col = self.Bi.col.get(b.box_uid, None)
                in_region = in_region and col is not None and reg["cols"][0] <= col < reg["cols"][1]
            if not in_region:
                continue
            frac = poly.intersection(rect).area / poly.area
            if frac < MIN_BOX_FRAC or not rect.contains(cen):
                dropped += 1
                continue
            pts = np.asarray(poly.exterior.coords)[:4]
            if frac < 0.999:
                pts = np.asarray(shapely.minimum_rotated_rectangle(poly.intersection(rect)).exterior.coords)[:4]
                pts = np.clip(pts, [x0, y0], [x1, y1])
            kept.append(dict(box_uid=b.box_uid, cls=int(b.class_id), pts=pts - [x0, y0], clipped=bool(frac < 0.999)))
        return kept, dropped

    def evaluate(self, order, a, b):
        regs = self.regions(order, a, b)
        crops = []
        for fi, f in self.pf.iterrows():
            P = np.asarray(f.geometry.exterior.coords)[:4]
            for s in order:
                reg = regs[s]
                key = (fi, reg["ue"], reg["v"])
                if key not in self.cache:
                    self.cache[key] = best_crop(P, self.eu, self.ev, *reg["ue"], *reg["v"])
                r = self.cache[key]
                if r is None:
                    continue
                kept, dropped = self.crop_boxes(f.thermal_file, P, r, reg)
                if not kept:
                    continue
                crops.append(dict(fi=fi, flight_id=f.flight_id, thermal_file=f.thermal_file, split=s, rect=r,
                                  poly=crop_polygon(P, r), boxes=kept, dropped=dropped))
        return regs, crops

    def summarize(self, crops):
        nb = {s: 0 for s in SPL}; cl = {s: set() for s in SPL}
        for c in crops:
            for bx in c["boxes"]:
                if bx["cls"] == MOC:
                    nb[c["split"]] += 1
                    cl[c["split"]].add(self.Bi.ref_cell[bx["box_uid"]])
        return nb, {s: len(v) for s, v in cl.items()}

    def collateral(self, S0, crops):
        """whole frames (train/val/test, not patch frames) closer than 2*margin to a crop of another split -> {(flight, file): (crop split, distance)}."""
        if not crops:
            return {}
        t = shapely.STRtree([c["poly"] for c in crops])
        w = S0[(S0.farm == "TN19") & S0.split.isin(SPL) & ~((S0.flight_id == MOC_FLIGHT) & (S0.n_ModuleOpenCircuit > 0))]
        ia, ic = t.query(w.geometry.values, predicate="dwithin", distance=2 * self.margin)
        out = {}
        for i, j in zip(ia, ic):
            if w.split.values[i] != crops[j]["split"]:
                d = float(shapely.distance(w.geometry.values[i], crops[j]["poly"]))
                key = (w.flight_id.values[i], w.thermal_file.values[i])
                if key not in out or d < out[key][1]:
                    out[key] = (crops[j]["split"], d)
        return out

    def search(self, S0, T=np.array([0.6, 0.2, 0.2]), min_boxes=1, orders=None):
        tot = max(S0[S0.split.isin(SPL)][CLS_COLS].values.sum(), 1)
        patch = S0[(S0.flight_id == MOC_FLIGHT) & (S0.n_ModuleOpenCircuit > 0) & (S0.split == "train")][CLS_COLS].values.sum()
        key_boxes = S0.set_index(["flight_id", "thermal_file"])[CLS_COLS].sum(axis=1)
        rows, best = [], None
        for order in (orders or itertools.permutations(SPL)):
            for a in range(1, self.ncol):
                for b in range(a + self.gap + 1, self.ncol - self.gap):
                    regs, crops = self.evaluate(order, a, b)
                    nb, nc = self.summarize(crops)
                    if min(nb.values()) < min_boxes:
                        continue
                    col = self.collateral(S0, crops)
                    lost = sum(int(key_boxes[k]) for k in col)
                    kept_crop = sum(len(c["boxes"]) for c in crops)
                    loss = (lost + patch - kept_crop) / tot
                    n = np.array([nb[s] for s in SPL], float); m_ = np.array([nc[s] for s in SPL], float)
                    cost = (np.abs(n / n.sum() - T).sum() + np.abs(m_ / m_.sum() - T).sum() + 3.0 * loss
                            + 1.0 * (1 - n.sum() / len(self.B)))
                    rows.append(dict(order="|".join(order), a=a, b=b, moc_train=nb["train"], moc_val=nb["val"], moc_test=nb["test"],
                                     cells_train=nc["train"], cells_val=nc["val"], cells_test=nc["test"], n_crops=len(crops),
                                     frames_to_buffer=len(col), box_loss_share=round(loss, 4), cost=round(cost, 4)))
                    if best is None or cost < best[0] - 1e-12:
                        best = (cost, order, a, b)
        return best, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- strict rule: assembly
MARGIN = {"TN19": 1.5, "TNDL": 1.5, "TT": 1.5, "BIM2": 5.0}   # m, added to the 10 m buffer; BIM2 has no RTK
P95_RESIDUAL = 2.3                                           # m, frame -> map residual p95 (metadata/projection_check)


def build_strict(S, F, cells, margin=1.5, gap_cols=1, orders=None, T=None, layout=None):
    """S: strip result of all farms (mode `train`, with geometry). Returns S2 (patch frames -> `crop`, collateral frames -> buffer,
    + crop_window), crops, crop boxes, per-box MOC table, region table, search table, crop polygons, info."""
    st19 = S[S.farm == "TN19"]
    sm = StrictMOC(st19, F, cells, margin=margin, gap_cols=gap_cols)
    if layout is not None:
        best, tab = (0.0,) + tuple(layout), pd.DataFrame()
    else:
        best, tab = sm.search(st19, orders=orders) if T is None else sm.search(st19, T=np.asarray(T), orders=orders)
    if best is None:
        raise ValueError("no region layout gives every split a ModuleOpenCircuit box")
    cost, order, a, b = best
    regs, cr = sm.evaluate(order, a, b)
    col = sm.collateral(st19, cr)
    S2 = S.copy()
    S2["crop_window"] = 0
    pos = {k: i for i, k in enumerate(zip(S2.flight_id, S2.thermal_file))}
    crop_frames = {(c["flight_id"], c["thermal_file"]) for c in cr}
    ix = S2.index
    for k in crop_frames:
        i = ix[pos[k]]
        S2.loc[i, "crop_window"] = 1
        S2.loc[i, "split"] = "crop"
        S2.loc[i, "reason"] = S2.loc[i, "reason"] + ("; ModuleOpenCircuit patch: not used as a whole frame, only the crop windows of "
                                                   "frame_label_split_moc_crops.csv")
    for fl, tf in zip(sm.pf.flight_id, sm.pf.thermal_file):
        if (fl, tf) not in crop_frames:
            i = ix[pos[(fl, tf)]]
            S2.loc[i, "split"] = "buffer"
            S2.loc[i, "reason"] = "buffer: ModuleOpenCircuit patch frame without a usable crop window"
    for (fl, tf), (cs, d) in col.items():
        i = ix[pos[(fl, tf)]]
        S2.loc[i, "split"] = "buffer"
        S2.loc[i, "reason"] = f"buffer: footprint is {d:.1f} m (< {2 * margin:g} m) from a {cs} crop window of the ModuleOpenCircuit patch"
    rows, brows, polys = [], [], {}
    cnt = {s: 0 for s in SPL}
    for c in sorted(cr, key=lambda c: (c["split"], c["thermal_file"])):
        cnt[c["split"]] += 1
        cid = f"TN19_MOC_{c['split']}_{cnt[c['split']]:02d}"
        x0, y0, x1, y1 = c["rect"]
        ncls = np.zeros(4, int)
        for bx in c["boxes"]:
            ncls[bx["cls"]] += 1
            brows.append(dict(crop_id=cid, box_uid=bx["box_uid"], class_id=bx["cls"], split=c["split"], clipped=int(bx["clipped"]),
                              **{f"{k}{j + 1}": round(float(v), 2) for j, p_ in enumerate(bx["pts"]) for k, v in zip("xy", p_)}))
        polys[cid] = c["poly"]
        rows.append(dict(crop_id=cid, farm="TN19", flight_id=c["flight_id"], thermal_file=c["thermal_file"], split=c["split"],
                         x0=x0, y0=y0, x1=x1, y1=y1,
                         n_boxes_by_class=";".join(f"{n_}:{k}" for n_, k in zip(CLASSES, ncls)), n_boxes=int(ncls.sum()),
                         **{f"n_{n_}": int(k) for n_, k in zip(CLASSES, ncls)}, n_boxes_dropped_partly_outside=c["dropped"],
                         area_m2=round(c["poly"].area, 1),
                         reason=f"largest pixel rectangle whose ground image lies in the {c['split']} map region eroded by {margin:g} m "
                                "at the boundaries facing other splits; boxes kept if centre in the region and >= 70 % of the area inside the crop"))
    crops, cboxes = pd.DataFrame(rows), pd.DataFrame(brows)
    Bt = sm.B.copy()
    inb = cboxes[cboxes.class_id == MOC].set_index("box_uid")
    ck = {(r.flight_id, r.thermal_file) for r in crops.itertuples()}
    grp = np.array(["gap"] * sm.ncol, dtype=object)
    for s_, rg in regs.items():
        grp[max(rg["cols"][0], 0):min(rg["cols"][1], sm.ncol)] = s_
    Bt["module_group"] = grp[Bt.col.values]
    sp, why, cid_ = [], [], []
    for r in Bt.itertuples():
        if r.box_uid in inb.index:
            sp.append(inb.split[r.box_uid]); cid_.append(inb.crop_id[r.box_uid]); why.append(f"inside {inb.split[r.box_uid]} crop window")
        else:
            cid_.append("")
            if (r.flight_id, r.thermal_file) not in ck:
                sp.append("buffer"); why.append("frame without crop window (buffer)")
            elif r.module_group == "gap":
                sp.append("unused"); why.append("module column in the gap between two regions")
            else:
                sp.append("unused"); why.append(f"module group {r.module_group}, but not in a crop window (< 70 % of the box inside the crop, or centre outside it)")
    mb = pd.DataFrame(dict(box_uid=Bt.box_uid, split=sp, reason=why, crop_id=cid_, flight_id=Bt.flight_id, thermal_file=Bt.thermal_file,
                           module_group=Bt.module_group, ref_cell=Bt.ref_cell, cell_column=Bt.col, E=Bt.E.round(2), N=Bt.N.round(2)))
    rr = []
    for s_, rg in regs.items():
        ua, ub = rg["u"]
        rr.append(dict(split=s_, cell_columns=f"{max(rg['cols'][0], 0)}-{min(rg['cols'][1], sm.ncol) - 1}",
                       u_from_m=None if ua < -BIG / 2 else round(ua - sm.u0, 2), u_to_m=None if ub > BIG / 2 else round(ub - sm.u0, 2),
                       v_from_m=None if rg["v"][0] < -BIG / 2 else round(rg["v"][0] - sm.vlo, 2),
                       v_to_m=None if rg["v"][1] > BIG / 2 else round(rg["v"][1] - sm.vlo, 2),
                       u_from_eroded_m=None if rg["ue"][0] < -BIG / 2 else round(rg["ue"][0] - sm.u0, 2),
                       u_to_eroded_m=None if rg["ue"][1] > BIG / 2 else round(rg["ue"][1] - sm.u0, 2),
                       note=f"u along the table rows (axis ({sm.eu[0]:.5f}, {sm.eu[1]:.5f}) in E,N) from the origin at module column 0 minus half a pitch; "
                            f"v across the rows from the south edge of the patch rows minus {sm.vpad:g} m; open ends are empty; gap columns {gap_cols}; margin {margin:g} m"))
    info = dict(order=order, a=a, b=b, cost=cost, margin=margin, gap=gap_cols, ncol=sm.ncol, regs=regs, sm=sm,
                collateral=len(col), n_patch_frames=len(sm.pf))
    return S2, crops, cboxes, mb, pd.DataFrame(rr), tab, polys, info


def region_polys(info, pad=40.0):
    """map polygons of the three ModuleOpenCircuit regions (full and eroded by the margin) for the figures."""
    sm = info["sm"]
    out = {}
    for s_, rg in info["regs"].items():
        res = []
        for ukey in ("u", "ue"):
            ua, ub = rg[ukey]
            ua = max(ua, sm.B.u.min() - pad); ub = min(ub, sm.B.u.max() + pad)
            va = max(rg["v"][0], sm.vlo - pad / 2); vb = min(rg["v"][1], sm.vhi + pad / 2)
            res.append(shapely.Polygon([ua * sm.eu + va * sm.ev, ub * sm.eu + va * sm.ev, ub * sm.eu + vb * sm.ev, ua * sm.eu + vb * sm.ev]))
        out[s_] = res
    return out


def unit_table(S, crops, polys):
    """Audit units: whole frames of train/val/test + crop windows."""
    w = S[S.split.isin(SPL)]
    U = pd.DataFrame(dict(farm=w.farm.values, unit=[f"{f}|{t}" for f, t in zip(w.flight_id, w.thermal_file)], split=w.split.values,
                          geometry=w.geometry.values))
    if len(crops):
        C = pd.DataFrame(dict(farm=crops.farm.values, unit=crops.crop_id.values, split=crops.split.values,
                              geometry=[polys[c] for c in crops.crop_id]))
        U = pd.concat([U, C], ignore_index=True)
    return U


def audit_units(U, name, margin, cells=None, ctree=None, assoc=None, S=None, crops_df=None):
    """Leakage metrics over the units of ONE farm."""
    g = U.geometry.values
    sp = U.split.values
    tree = STRtree(g)
    r = dict(variant=name, n_units=len(U), **{f"n_units_{s}": int((sp == s).sum()) for s in SPL})
    ii, jj = tree.query(g, predicate="intersects")
    m = (ii < jj) & (sp[ii] != sp[jj])
    inter = shapely.area(shapely.intersection(g[ii[m]], g[jj[m]]))
    r["cross_split_pairs_intersecting"] = int((inter > 1e-6).sum())
    for lab, d in (("margin", margin), ("p95_2.3m", P95_RESIDUAL), ("5m", 5.0), ("10m", 10.0)):
        i2, j2 = tree.query(g, predicate="dwithin", distance=d)
        m2 = (i2 < j2) & (sp[i2] != sp[j2])
        r[f"cross_split_pairs_within_{lab}"] = int(m2.sum())
    i3, j3 = tree.query(g, predicate="dwithin", distance=40.0)
    m3 = (i3 < j3) & (sp[i3] != sp[j3])
    r["min_cross_split_distance_m"] = round(float(shapely.distance(g[i3[m3]], g[j3[m3]]).min()), 2) if m3.any() else float("nan")
    un = {s: unary_union(list(g[sp == s])) for s in SPL if (sp == s).any()}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        if a_ in un and b_ in un:
            x = un[a_].intersection(un[b_]).area
            r[f"shared_area_m2_{a_}_{b_}"] = round(x, 1)
            r[f"shared_pct_of_smaller_{a_}_{b_}"] = round(100 * x / min(un[a_].area, un[b_].area), 2)
    if cells is not None:
        cs = {s: set(ctree.query(un[s], predicate="intersects").tolist()) for s in un}
        for s in un:
            r[f"cells_touched_{s}"] = len(cs[s])
        for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
            if a_ in cs and b_ in cs:
                r[f"shared_cells_{a_}_{b_}"] = len(cs[a_] & cs[b_])
    if assoc is not None and S is not None:
        key = {(f, t): s for f, t, s in zip(S.flight_id, S.thermal_file, S.split)}
        a = assoc.copy()
        a["split"] = [key.get((f, t)) for f, t in zip(a.flight_id, a.thermal_file)]
        if crops_df is not None and len(crops_df) and "x_px" in a.columns:       # detections in frames that were cut into crop windows
            cr = {}
            for c in crops_df.itertuples():
                cr.setdefault((c.flight_id, c.thermal_file), []).append((c.x0, c.y0, c.x1, c.y1, c.split))
            sp_ = a.split.values.copy()
            for k, (f, t, x, y) in enumerate(zip(a.flight_id, a.thermal_file, a.x_px, a.y_px)):
                for x0, y0, x1, y1, s_ in cr.get((f, t), []):
                    if x0 <= x < x1 and y0 <= y < y1:
                        sp_[k] = s_
            a["split"] = sp_
        a = a[a.split.isin(SPL)]
        ns = a.groupby("module_id").split.nunique()
        r["ortho_modules_seen"], r["ortho_modules_in_2plus_splits"] = int(len(ns)), int((ns > 1).sum())
    return r


def counts_strict(S, crops, cboxes, variant):
    """Per farm x split: whole frames, crop windows, boxes per class (whole frames + crops), negatives, units per altitude.
    Extra rows: `buffer` (frames not used) and `not_in_crop` (boxes of patch frames outside every crop window)."""
    cc = [f"n_{c}" for c in CLASSES]
    alt = {(f, t): a for f, t, a in zip(S.flight_id, S.thermal_file, S.altitude_m)}
    alts = sorted(S.altitude_m.unique())
    cb = cboxes.merge(crops[["crop_id", "farm", "flight_id", "thermal_file"]], on="crop_id") if len(crops) else cboxes
    rows = []
    for farm, g in list(S.groupby("farm")) + [("ALL", S)]:
        gc = crops if farm == "ALL" else crops[crops.farm == farm]
        gb = cb if farm == "ALL" else cb[cb.farm == farm]
        for s in SPL + ["buffer", "not_in_crop"]:
            if s == "not_in_crop":
                cf = g[g.split == "crop"]
                n_cls = cf[cc].sum().values - (gb.class_id.value_counts().reindex(range(4), fill_value=0).values if len(gb) else 0)
                rows.append(dict(variant=variant, farm=farm, split=s, n_whole_frames=0, n_crops=0, n_frames_reviewed_negative=0,
                                 n_boxes=int(np.sum(n_cls)), **{c: int(k) for c, k in zip(cc, n_cls)}, **{f"units_{a}m": 0 for a in alts}))
                continue
            h = g[g.split == s]
            hc = gc[gc.split == s] if s in SPL and len(gc) else gc.iloc[0:0]
            hb = gb[gb.split == s] if s in SPL and len(gb) else gb.iloc[0:0]
            bcl = hb.class_id.value_counts().reindex(range(4), fill_value=0).values if len(hb) else np.zeros(4, int)
            wcl = h[cc].sum().values
            ualt = list(h.altitude_m) + [alt[(f, t)] for f, t in zip(hc.flight_id, hc.thermal_file)]
            rows.append(dict(variant=variant, farm=farm, split=s, n_whole_frames=len(h), n_crops=len(hc),
                             n_frames_reviewed_negative=int((h[cc].sum(1) == 0).sum()), n_boxes=int((wcl + bcl).sum()),
                             **{c: int(k) for c, k in zip(cc, wcl + bcl)}, **{f"units_{a}m": int(sum(1 for x in ualt if x == a)) for a in alts}))
    return pd.DataFrame(rows)


def loss_table(Cn, S):
    """Per farm: share of frames and boxes not used."""
    rows = []
    for farm in FARMS + ["ALL"]:
        c = Cn[Cn.farm == farm].set_index("split")
        used = c.loc[SPL]
        n_fr = len(S) if farm == "ALL" else int((S.farm == farm).sum())
        tot_b = int(c.n_boxes.sum())
        rows.append(dict(farm=farm, labelled_frames=n_fr, whole_frames_used=int(used.n_whole_frames.sum()), crop_windows_used=int(used.n_crops.sum()),
                         frames_in_buffer=int(c.loc["buffer", "n_whole_frames"]), frames_replaced_by_crops=int((S[S.farm == farm] if farm != "ALL" else S).split.eq("crop").sum()),
                         frames_lost_pct=round(100 * c.loc["buffer", "n_whole_frames"] / n_fr, 1),
                         boxes_total=tot_b, boxes_used=int(used.n_boxes.sum()), boxes_in_buffer_frames=int(c.loc["buffer", "n_boxes"]),
                         boxes_of_patch_frames_outside_crops=int(c.loc["not_in_crop", "n_boxes"]),
                         boxes_lost_pct=round(100 * (1 - used.n_boxes.sum() / tot_b), 1)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- strict: sensitivity and alternatives
def strips_for(F, margins):
    parts, probs = [], {}
    for farm in FARMS:
        o, st, P = split_farm(F, farm, "train", 20.0, 10.0 + margins[farm])
        parts.append(o); probs[farm] = (st, P)
    S = pd.concat(parts, ignore_index=True).merge(F[["flight_id", "thermal_file", "geometry"]], on=["flight_id", "thermal_file"])
    S = S.sort_values(["farm", "flight_id", "thermal_file"]).reset_index(drop=True)
    return S, probs


def margin_sensitivity(F, cells):
    rows = []
    cc = [f"n_{c}" for c in CLASSES]
    for m in (0.0, 1.5, 3.0, 5.0):
        S, _ = strips_for(F, {f: m for f in FARMS})
        try:
            S2, crops, cboxes, mb, rg, tab, polys, info = build_strict(S, F, cells, margin=m, orders=[("train", "val", "test")])
        except ValueError:          # infeasible: the whole patch goes to the buffer
            S2 = S.copy()
            S2["crop_window"] = 0
            pm = (S2.farm == "TN19") & (S2.flight_id == MOC_FLIGHT) & (S2.n_ModuleOpenCircuit > 0) & (S2.split == "train")
            S2.loc[pm, "split"] = "buffer"
            crops, cboxes, polys = pd.DataFrame(columns=["farm", "crop_id", "split"]), pd.DataFrame(columns=["class_id"]), {}
            mb = pd.DataFrame(columns=["split"])
            info = dict(collateral=0)
        U = unit_table(S2, crops, polys)
        for farm in FARMS:
            sub = S2[S2.farm == farm]
            a = audit_units(U[U.farm == farm].reset_index(drop=True), "m", m)
            used_b = sub[sub.split.isin(SPL)][cc].sum().sum() + (len(cboxes) if farm == "TN19" else 0)
            tot = sub[cc].sum().sum()
            r = dict(margin_m=m, farm=farm, buffer_each_side_m=10.0 + m, n_frames=len(sub), frames_in_buffer=int((sub.split == "buffer").sum()),
                     frames_replaced_by_crops=int((sub.split == "crop").sum()),
                     frame_loss_pct=round(100 * (sub.split == "buffer").sum() / len(sub), 1), boxes_total=int(tot), boxes_used=int(used_b),
                     box_loss_pct=round(100 * (1 - used_b / tot), 1), cross_split_pairs_intersecting=a["cross_split_pairs_intersecting"],
                     cross_split_pairs_within_p95_2p3m=a["cross_split_pairs_within_p95_2.3m"],
                     min_cross_split_distance_m=a["min_cross_split_distance_m"])
            if farm == "TN19":
                nb = mb[mb.split.isin(SPL)].split.value_counts().reindex(SPL, fill_value=0)
                r.update(moc_boxes_train=int(nb.train), moc_boxes_val=int(nb.val), moc_boxes_test=int(nb.test), crop_windows=len(crops),
                         patch_collateral_frames_to_buffer=info["collateral"])
            rows.append(r)
        print(f"[margin sensitivity] margin {m} m done", flush=True)
    return pd.DataFrame(rows)


def moc_alternatives(S, F, cells):
    """ModuleOpenCircuit layouts evaluated: gap columns 0/1/2 (order train|val|test; the mirror order is symmetric)."""
    rows = []
    st19 = S[S.farm == "TN19"]
    for gap in (0, 1, 2):
        sm = StrictMOC(st19, F, cells, margin=1.5, gap_cols=gap)
        best, tab = sm.search(st19, orders=[("train", "val", "test")])
        if best is None:
            rows.append(dict(gap_columns=gap, margin_m=1.5, note="infeasible"))
            continue
        t = tab.sort_values("cost").iloc[0]
        rows.append(dict(gap_columns=gap, margin_m=1.5, cut_a=int(t.a), cut_b=int(t.b), moc_train=int(t.moc_train), moc_val=int(t.moc_val),
                         moc_test=int(t.moc_test), n_crops=int(t.n_crops), collateral_frames_to_buffer=int(t.frames_to_buffer),
                         box_loss_share=float(t.box_loss_share), layouts_evaluated=len(tab),
                         layouts_with_val_and_test_ge15=int(((tab.moc_val >= 15) & (tab.moc_test >= 15)).sum()),
                         note="min-cost layout (about 60/20/20 of boxes and cells, small loss)"))
        print(f"[alternatives] gap {gap} done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- legacy modes (relaxed [superseded], train, test, internal)
def main_legacy(a, F, margins, outdir):
    relaxed = a.moc_mode == "relaxed"
    strip_mode = "train" if relaxed else a.moc_mode          # relaxed = strips of `train`, then the patch is re-split
    primary = relaxed
    if primary:
        moc_feasibility(F).to_csv(outdir / "frame_label_split_moc_feasibility.csv", index=False)
        print(pd.read_csv(outdir / "frame_label_split_moc_feasibility.csv").to_string())
    parts, strips, probs = [], [], {}
    for farm in FARMS:
        o, st, P = split_farm(F, farm, strip_mode, a.moc_buffer, 10.0 + margins[farm])
        parts.append(o); strips.append(st); probs[farm] = P
    S = pd.concat(parts, ignore_index=True)
    St = pd.concat(strips, ignore_index=True)
    suffix = "" if primary else f"_moc{a.moc_mode}"
    S = S.merge(F[["flight_id", "thermal_file", "geometry"]], on=["flight_id", "thermal_file"])
    assert len(S) == 3000
    S = S.sort_values(["farm", "flight_id", "thermal_file"]).reset_index(drop=True)
    cells = ms.load_cells()
    BT = None
    if relaxed:
        before = S.copy()
        S, BT, info = apply_relaxed(S, F, cells)
        print(f"[relaxed MOC] order {info['order']}, cut columns {info['a']} / {info['b']} of {info['ncol']}, cost {info['cost']:.3f}")
        chg = S[S.split != before.split]
        same = S.drop(columns=["moc_relaxed", "n_moc_ignored", "reason", "split"]).equals(before.drop(columns=["reason", "split"]))
        print(f"[relaxed MOC] frames whose split changed (moved out of train): {len(chg)}; everything else unchanged: {bool(same)}")
    cols = ["farm", "flight_id", "thermal_file", "split", "strip_id", "reason", "altitude_m"] + [f"n_{c}" for c in CLASSES]
    if relaxed:
        cols += ["moc_relaxed", "n_moc_ignored"]
    outdir.mkdir(parents=True, exist_ok=True)
    S[cols].to_csv(outdir / f"frame_label_split{suffix}.csv", index=False)
    St["moc_mode"] = a.moc_mode
    St.to_csv(outdir / f"frame_label_split_strips{suffix}.csv", index=False)

    # ---- xdomain variant (only for the primary run)
    X = None
    if primary:
        X = S.copy()
        t19 = X.farm == "TN19"
        X.loc[t19 & (X.split == "test"), "split"] = "train"
        X.loc[t19 & (X.split == "train") & S.split.eq("test"), "reason"] = "TN19 test strip moved to train (xdomain variant)"
        X.loc[~t19, "split"] = "test"
        X.loc[~t19, "reason"] = "external-domain test farm (" + X.loc[~t19, "farm"] + ")"
        X.loc[~t19, "strip_id"] = X.loc[~t19, "farm"] + "-all"
        # per-box flags of the xdomain variant: module group `test` now belongs to train
        fx = X[(X.flight_id == MOC_FLIGHT) & (X.n_ModuleOpenCircuit > 0)][["flight_id", "thermal_file", "split"]]
        fx = fx.rename(columns={"split": "frame_split"})
        Bg = BT.drop(columns=["frame_split", "split", "reason"]).rename(columns={"module_group": "group", "cell_column": "col"})
        BX = moc_box_table(Bg, fx, mapfn=lambda g_: "train" if g_ == "test" else g_)
        ign = BX[BX.split == "ignore"].groupby(["flight_id", "thermal_file"]).size()
        X["n_moc_ignored"] = [int(ign.get((f, t), 0)) for f, t in zip(X.flight_id, X.thermal_file)]
        X[cols].to_csv(outdir / "frame_label_split_xdomain_tn19_to_others.csv", index=False)
        bcols = ["box_uid", "split", "reason", "flight_id", "thermal_file", "frame_split", "module_group", "ref_cell", "cell_column"]
        BT[bcols].to_csv(outdir / "frame_label_split_moc_boxes.csv", index=False)
        BX[bcols].to_csv(outdir / "frame_label_split_xdomain_moc_boxes.csv", index=False)
        moc_leakage(S, BT, cells).to_csv(outdir / "frame_label_split_moc_leakage.csv", index=False)
        print(pd.read_csv(outdir / "frame_label_split_moc_leakage.csv").to_string())

    # ---- counts
    C = counts_table(S, f"main{suffix}")
    if primary:
        C = pd.concat([C, counts_table(X, "xdomain_tn19_to_others")], ignore_index=True)
    C.to_csv(outdir / f"frame_label_split_counts{suffix}.csv", index=False)
    print(C[C.farm.isin(["ALL", "TN19"])].to_string())

    # ---- audit
    ctree = STRtree(cells.geometry.values)
    assoc = load_assoc()
    rows = []
    t19 = (S.farm == "TN19").values
    for lab, sel in (("TN19", t19), ("TNDL", (S.farm == "TNDL").values), ("TT", (S.farm == "TT").values),
                     ("BIM2", (S.farm == "BIM2").values)):
        sub = S[sel].reset_index(drop=True)
        use_c = lab == "TN19"
        r = audit(sub, sub.split.values, f"main{suffix}_{lab}", cells if use_c else None, ctree if use_c else None,
                  assoc if use_c else None)
        rows.append(flat(r))
        for seed in (1, 2, 3):
            rs = random_split(len(sub), SEED + seed)
            r = audit(sub, rs, f"random_per_frame_seed{seed}_{lab}", cells if use_c else None, ctree if use_c else None,
                      assoc if use_c else None)
            rows.append(flat(r))
    if primary:
        sub = X[(X.farm == "TN19") | (X.split == "test")].reset_index(drop=True)
        r = audit(sub, sub.split.values, "xdomain_tn19_to_others_ALL", cells, ctree, assoc)
        rows.append(flat(r))
    A = pd.DataFrame(rows)
    A.to_csv(outdir / f"frame_label_split_audit{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(A.T.to_string())
    if not a.no_figures:
        figures_legacy(S, St, probs, suffix, X, C, BT, fd=outdir / "figures")




# ----------------------------------------------------------------------------- strict: figures
COL = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c", "buffer": "#9a9a9a", "crop": "#8e44ad"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def strip_poly(ax, f0, f1, c0, L):
    nv = np.array([ax.a, ax.b]) / ax.n
    t = np.array([-nv[1], nv[0]])
    p = lambda f: np.asarray(c0, float) + nv * (f - ax.f(*c0))
    return shapely.Polygon([p(f0) - L * t, p(f1) - L * t, p(f1) + L * t, p(f0) + L * t])


def _add(ax_, geoms, fc=None, ec="k", lw=0.5, alpha=1.0, ls="-", hatch=None, z=1):
    from matplotlib.patches import Polygon as MP
    for g in geoms:
        if g.is_empty:
            continue
        gs = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for q in gs:
            ax_.add_patch(MP(np.asarray(q.exterior.coords), closed=True, fill=fc is not None, fc=fc if fc else "none", ec=ec, lw=lw,
                             alpha=alpha, ls=ls, hatch=hatch, zorder=z))


def _legend_split(ax_, extra=()):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[s], lw=7, alpha=0.6, label=s + (" (not used)" if s == "buffer" else "")) for s in ("train", "val", "test", "buffer")]
    ax_.legend(handles=h + list(extra), loc="lower right", fontsize=8, framealpha=0.9)


def _strip_layer(ax_, St, probs, farm, c0, L, buf, alpha=0.10):
    P = probs[farm]
    for r in St[St.farm == farm].itertuples():
        _add(ax_, [strip_poly(P.ax, r.f_from_m, r.f_to_m, c0, L)], fc=COL[r.owner], ec="none", alpha=alpha, z=0)
    cuts = sorted(set(St[St.farm == farm].f_from_m.tolist()[1:]))
    for c in cuts:
        _add(ax_, [strip_poly(P.ax, c - buf, c + buf, c0, L)], fc="#555555", ec="none", alpha=0.22, z=0.5)
        _add(ax_, [strip_poly(P.ax, c, c, c0, L).buffer(0.3)], fc="k", ec="none", alpha=0.7, z=0.6)
    return cuts


def _frame_view(S, farm):
    t = S[S.farm == farm]
    b = np.array([g.bounds for g in t.geometry])
    return b[:, 0].min() - 15, b[:, 2].max() + 15, b[:, 1].min() - 15, b[:, 3].max() + 15


def fig_principle(S, St, probs, fd, buf):
    plt = _mpl()
    P = probs["TN19"]
    t = S[(S.farm == "TN19") & (S.altitude_m == 35)]
    x0, x1, y0, y1 = _frame_view(S, "TN19")
    c0 = ((x0 + x1) / 2, (y0 + y1) / 2)
    cuts = sorted(set(St[St.farm == "TN19"].f_from_m.tolist()[1:]))
    fc = np.array([P.ax.f(g.centroid.x, g.centroid.y) for g in t.geometry])
    c_ = cuts[0]
    pick = {}
    tall = S[S.farm == "TN19"]
    cand = tall.assign(f=np.array([P.ax.f(g.centroid.x, g.centroid.y) for g in tall.geometry]))
    bufc = cand[(cand.split == "buffer") & (np.abs(cand.f - c_) < 6) & cand.geometry.apply(lambda g: strip_poly(P.ax, c_, c_, c0, 1000).buffer(0.1).intersects(g))]
    pick["buffer"] = bufc.iloc[(bufc.f - c_).abs().argsort().values[0]] if len(bufc) else None
    own = St[(St.farm == "TN19") & (St.f_from_m == c_)].owner.iloc[0]       # strip after the cut
    own0 = St[(St.farm == "TN19") & (St.f_to_m == c_)].owner.iloc[0]
    a = cand[(cand.split == own0) & (cand.f < c_ - buf - 14)]
    b = cand[(cand.split == own) & (cand.f > c_ + buf + 14)]
    ref = pick["buffer"].geometry.centroid if pick["buffer"] is not None else None
    for nm, d in ((own0, a), (own, b)):
        if len(d):
            d = d.assign(dist=[g.centroid.distance(ref) for g in d.geometry]) if ref is not None else d.assign(dist=0)
            pick[nm + "_ex"] = d.sort_values("dist").iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, ax_ in enumerate(axs):
        _strip_layer(ax_, St, probs, "TN19", c0, 600, buf)
        ax_.set_aspect("equal")
    ax_ = axs[0]
    _add(ax_, list(t[t.split == "buffer"].geometry), fc=COL["buffer"], ec="none", alpha=0.35, z=1)
    for s_ in SPL:
        _add(ax_, list(t[t.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.3, alpha=0.18, z=2)
    ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1)
    ax_.set_title("(a) TN19, 35 m labelled frames: strips = splits,\ndark bands = buffer (no kept footprint may enter)", fontsize=9)
    ax_ = axs[1]
    ex = [(v, k) for k, v in pick.items() if v is not None]
    if ex:
        cx = np.mean([v.geometry.centroid.x for v, _ in ex]); cy = np.mean([v.geometry.centroid.y for v, _ in ex])
        ax_.set_xlim(cx - 75, cx + 75); ax_.set_ylim(cy - 50, cy + 50)
        for v, k in ex:
            sp = v.split
            _add(ax_, [v.geometry], fc=COL[sp], ec=COL[sp], lw=1.8, alpha=0.35, z=3)
            _add(ax_, [v.geometry.buffer(buf, join_style=2)], ec="k", lw=0.9, ls="--", z=4)
            lab = {"buffer": "straddles a boundary\n-> buffer"}.get(sp, f"inside the {sp} strip\n(footprint + {buf:g} m stays inside)")
            ax_.text(v.geometry.centroid.x, v.geometry.centroid.y, lab, ha="center", va="center", fontsize=8, zorder=6,
                     bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
    ax_.set_title(f"(b) example frames; dashed = footprint + {buf:g} m\n({buf - 10:g} m of it is the safety margin)", fontsize=9)
    for a_ in axs:
        a_.set_xlabel("E (m, EPSG:32649, UTM 49N)"); a_.set_ylabel("N (m)"); a_.tick_params(labelsize=8)
    _legend_split(axs[0])
    fig.tight_layout(); fig.savefig(fd / "fig_split_principle.png", dpi=100); plt.close(fig)


def fig_by_altitude(S, crops, polys, St, probs, fd, buf, band="dark band = buffer", per_run=False):
    plt = _mpl()
    x0, x1, y0, y1 = _frame_view(S, "TN19")
    c0 = ((x0 + x1) / 2, (y0 + y1) / 2)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.2))
    t = S[S.farm == "TN19"]
    for ax_, alt in zip(axs, (31, 35, 42)):
        _strip_layer(ax_, St, probs, "TN19", c0, 600, buf, alpha=0.12)
        d = t[t.altitude_m == alt]
        _add(ax_, list(d[d.split == "buffer"].geometry), fc=COL["buffer"], ec="none", alpha=0.4, z=1)
        for s_ in SPL:
            _add(ax_, list(d[d.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.3, alpha=0.2, z=2)
        cp = crops[(crops.split.isin(SPL))]
        cp = cp[[S[(S.flight_id == f) & (S.thermal_file == t_)].altitude_m.iloc[0] == alt for f, t_ in zip(cp.flight_id, cp.thermal_file)]] if len(cp) else cp
        for r in cp.itertuples():
            _add(ax_, [polys[r.crop_id]], fc=COL[r.split], ec="k", lw=0.5, alpha=0.8, z=4)
        n = {s_: len(d[d.split == s_]) for s_ in SPL + ["buffer"]}
        ax_.set_title(f"{alt} m: " + ", ".join(f"{k} {v}" for k, v in n.items()) + f"  (+{len(cp)} crop windows)", fontsize=9)
        ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1); ax_.set_aspect("equal"); ax_.tick_params(labelsize=7)
        ax_.set_xlabel("E (m, UTM 49N)", fontsize=8)
    axs[0].set_ylabel("N (m, UTM 49N)")
    _legend_split(axs[2])
    fig.suptitle("Labelled TN19 thermal frames by altitude: the same ground gets the same split at 31, 35 and 42 m "
                 "(shaded = strip owner, " + band + ")", fontsize=10)
    fig.tight_layout(); fig.savefig(fd / "fig_split_by_altitude.png", dpi=100); plt.close(fig)


def fig_leakage_compare(S, fd, cnt_random, cnt_strict, seed, crops=None, polys=None):
    plt = _mpl()
    t = S[S.farm == "TN19"].reset_index(drop=True)
    x0, x1, y0, y1 = _frame_view(S, "TN19")
    rs = random_split(len(t), seed)
    g = t.geometry.values
    tree = STRtree(g)
    ii, jj = tree.query(g, predicate="intersects")
    m = (ii < jj) & (rs[ii] != rs[jj])
    ii, jj = ii[m], jj[m]
    ov = shapely.intersection(g[ii], g[jj])
    rng = np.random.default_rng(1)
    sel = rng.choice(len(ov), min(2500, len(ov)), replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.8))
    ax_ = axs[0]
    for s_ in SPL:
        _add(ax_, list(g[rs == s_]), fc=COL[s_], ec=COL[s_], lw=0.2, alpha=0.12, z=1)
    _add(ax_, [ov[i] for i in sel], fc="red", ec="none", alpha=0.35, z=3)
    ax_.set_title(f"random per-frame split (all 2,505 TN19 frames): {cnt_random:,} cross-split pairs of overlapping frames\n"
                  "(red = overlap of such pairs, 2,500 of them drawn)", fontsize=9)
    ax_ = axs[1]
    k = t[t.split.isin(SPL)]
    for s_ in SPL:
        _add(ax_, list(k[k.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.2, alpha=0.2, z=1)
    _add(ax_, list(t[t.split == "buffer"].geometry), fc=COL["buffer"], ec="none", alpha=0.3, z=0)
    if crops is not None and len(crops):
        for r in crops.itertuples():
            _add(ax_, [polys[r.crop_id]], fc=COL[r.split], ec=COL[r.split], lw=0.2, alpha=0.25, z=2)
    ax_.set_title(f"strict split (this release): {cnt_strict} cross-split pairs of overlapping frames or crop windows\n"
                  "(grey = buffer frames, not used)", fontsize=9)
    for a_ in axs:
        a_.set_xlim(x0, x1); a_.set_ylim(y0, y1); a_.set_aspect("equal"); a_.tick_params(labelsize=8)
        a_.set_xlabel("E (m, EPSG:32649, UTM 49N)"); a_.set_ylabel("N (m)")
    _legend_split(axs[1])
    fig.tight_layout(); fig.savefig(fd / "fig_split_leakage_compare.png", dpi=100); plt.close(fig)


def fig_moc_patch(S, crops, polys, mb, info, fd):
    plt = _mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sm = info["sm"]
    rp = region_polys(info)
    fig, ax_ = plt.subplots(figsize=(13.5, 8))
    cx, cy = mb.E.mean(), mb.N.mean()
    for s_, (reg, ere) in rp.items():
        _add(ax_, [reg], fc=COL[s_], ec=COL[s_], lw=1.2, alpha=0.10, z=0)
        _add(ax_, [ere], ec=COL[s_], lw=1.2, ls="--", z=2)
    pf = S[(S.farm == "TN19") & (S.flight_id == MOC_FLIGHT) & (S.n_ModuleOpenCircuit > 0)]
    _add(ax_, list(pf.geometry), ec="#bbbbbb", lw=0.5, z=1)
    ccx = (cx, cy)
    near = S[(S.farm == "TN19") & S.reason.str.contains("crop window of the ModuleOpenCircuit")]
    _add(ax_, list(near.geometry), ec="#888888", lw=0.4, ls=":", alpha=0.7, z=1)
    for r in crops.itertuples():
        _add(ax_, [polys[r.crop_id]], fc=COL[r.split], ec="k", lw=0.5, alpha=0.30, z=3)
    for sname in ("train", "val", "test", "gap", "buffer", "unused"):
        q = mb[(mb.split == sname)] if sname in SPL + ["buffer", "unused"] else mb[mb.module_group == "gap"]
        if sname == "gap":
            q = mb[(mb.module_group == "gap") & (~mb.split.isin(SPL))]
        if sname == "unused":
            q = mb[(mb.split == "unused") & (mb.module_group != "gap")]
        col_ = {"gap": "#222222", "unused": "#c0392b"}.get(sname, COL.get(sname))
        ax_.scatter(q.E, q.N, s=16, color=col_, zorder=5, marker={"gap": "x", "unused": "+"}.get(sname, "o"), edgecolors="k" if sname in SPL else None, linewidths=0.25)
    # gap columns drawn as thin hatched strips
    ax_.set_xlim(cx - 24, cx + 22); ax_.set_ylim(cy - 16, cy + 16); ax_.set_aspect("equal")
    h = [Patch(fc=COL[s], ec=COL[s], alpha=0.35, label=f"{s} map region / crop windows (fill)") for s in SPL]
    h += [Line2D([], [], color="k", ls="--", label="region eroded by the margin (crop windows must lie inside)"),
          Line2D([], [], color="#bbbbbb", label="footprints of the 30 patch frames (whole frames not used)"),
          Line2D([], [], color="#666666", ls=":", label="other frames sent to buffer because of the crop windows"),
          Line2D([], [], marker="o", ls="", color="k", label="ModuleOpenCircuit box centres, colour = split"),
          Line2D([], [], marker="x", ls="", color="#222222", label="box in a gap column (unused)"),
          Line2D([], [], marker="+", ls="", color="#c0392b", label="box in a region but outside every crop (unused)")]
    ax_.legend(handles=h, loc="upper left", fontsize=7.5, framealpha=0.92)
    ax_.set_title(f"ModuleOpenCircuit patch ({MOC_FLIGHT}): three map regions (train | gap | val | gap | test along the module rows), crop windows per split",
                  fontsize=9)
    ax_.set_xlabel("E (m, EPSG:32649, UTM 49N)"); ax_.set_ylabel("N (m)"); ax_.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(fd / "fig_split_moc_patch.png", dpi=100); plt.close(fig)


def fig_small_farms(S, St, probs, fd):
    plt = _mpl()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax_, farm in zip(axs, ["TNDL", "TT", "BIM2"]):
        x0, x1, y0, y1 = _frame_view(S, farm)
        c0 = ((x0 + x1) / 2, (y0 + y1) / 2)
        _strip_layer(ax_, St, probs, farm, c0, 3000, 10.0 + MARGIN[farm], alpha=0.12)
        t = S[S.farm == farm]
        _add(ax_, list(t[t.split == "buffer"].geometry), fc=COL["buffer"], ec="none", alpha=0.4, z=1)
        for s_ in SPL:
            _add(ax_, list(t[t.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.4, alpha=0.25, z=2)
        n = {s_: int((t.split == s_).sum()) for s_ in SPL + ["buffer"]}
        ax_.set_title(f"{farm}: " + ", ".join(f"{k} {v}" for k, v in n.items()) + f"; buffer {10 + MARGIN[farm]:g} m", fontsize=9)
        ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1); ax_.set_aspect("equal"); ax_.tick_params(labelsize=7)
        ax_.set_xlabel("E (m, UTM 49N)", fontsize=8)
    axs[0].set_ylabel("N (m, UTM 49N)")
    _legend_split(axs[2])
    fig.suptitle("Small farms: labelled frames by split (shaded = strip owner, dark band = buffer)", fontsize=10)
    fig.tight_layout(); fig.savefig(fd / "fig_split_small_farms.png", dpi=100); plt.close(fig)


def fig_no_rtk(S, St, probs, fd, allfp):
    plt = _mpl()
    from matplotlib.patches import Ellipse
    farm = "BIM2"
    P = probs[farm]
    b = 10.0 + MARGIN[farm]
    t = S[S.farm == farm]
    x0, x1, y0, y1 = _frame_view(S, farm)
    c0 = ((x0 + x1) / 2, (y0 + y1) / 2)
    cuts = sorted(set(St[St.farm == farm].f_from_m.tolist()[1:]))
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.3, 0.7, 1.1]})
    ax_ = axs[0]
    _strip_layer(ax_, St, probs, farm, c0, 3000, b, alpha=0.10)
    _add(ax_, list(allfp.geometry), ec="#cccccc", lw=0.15, z=0.2)
    _add(ax_, list(t[t.split == "buffer"].geometry), fc=COL["buffer"], ec="none", alpha=0.45, z=1)
    for s_ in SPL:
        _add(ax_, list(t[t.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.4, alpha=0.3, z=2)
    ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1); ax_.set_aspect("equal")
    ax_.set_title(f"(a) BIM2: all thermal frames (grey outline), labelled ones coloured; dark band = {b:g} m buffer", fontsize=9)
    ax_.set_xlabel("E (m, EPSG:32649, UTM 49N)"); ax_.set_ylabel("N (m)")
    _legend_split(ax_)
    # (b) buffer widths
    ax_ = axs[1]
    ax_.axvline(0, color="k", lw=2)
    for y, (lab, w, c) in enumerate((("RTK farms (TN19, TNDL, TT)", 10 + MARGIN["TN19"], "#2c7fb8"), ("BIM2 (no RTK)", b, "#d95f0e"))):
        ax_.barh(y, 2 * w, left=-w, color=c, alpha=0.35, height=0.5)
        ax_.barh(y, 2 * MARGIN["TN19" if y == 0 else "BIM2"], left=-MARGIN["TN19" if y == 0 else "BIM2"], color=c, alpha=0.8, height=0.5)
        ax_.text(0, y + 0.36, f"{lab}: {w:g} m = 10 m + {w - 10:g} m margin", ha="center", fontsize=8)
    ax_.set_xlim(-20, 20); ax_.set_ylim(-0.6, 1.8); ax_.set_yticks([])
    ax_.set_xlabel("distance to the strip boundary (m)"); ax_.set_title("(b) no kept footprint may enter the band (light);\nsolid = the safety margin", fontsize=9)
    # (c) GPS uncertainty
    ax_ = axs[2]
    ex = t[(t.split != "buffer")].copy()
    cuts_f = cuts[0] if cuts else None
    if cuts_f is not None:
        ex["d"] = [abs(P.ax.f(g.centroid.x, g.centroid.y) - cuts_f) for g in ex.geometry]
        r0 = ex.sort_values("d").iloc[0]
        cx, cy = r0.geometry.centroid.x, r0.geometry.centroid.y
        _add(ax_, [strip_poly(P.ax, cuts_f - b, cuts_f + b, (cx, cy), 400)], fc="#555555", ec="none", alpha=0.2, z=0)
        _add(ax_, [strip_poly(P.ax, cuts_f - 0.3, cuts_f + 0.3, (cx, cy), 400)], fc="k", ec="none", alpha=0.9, z=5)
        e = MARGIN["BIM2"]
        near = t[[g.centroid.distance(r0.geometry.centroid) < 45 for g in t.geometry]]
        for r in near.itertuples():
            _add(ax_, [r.geometry], fc=COL[r.split], ec=COL[r.split], lw=1.0, alpha=0.25, z=3)
        g = r0.geometry
        for dx, dy in ((e, 0), (-e, 0), (0, e), (0, -e)):
            _add(ax_, [shapely.affinity.translate(g, dx, dy)], ec="#222222", lw=0.8, ls="--", z=4)
        c = g.centroid
        ax_.add_patch(Ellipse((c.x, c.y), 2 * e, 2 * e, fill=False, ec="red", lw=1.2, zorder=6))
        ax_.set_xlim(cx - 40, cx + 40); ax_.set_ylim(cy - 30, cy + 30); ax_.set_aspect("equal")
    ax_.set_title(f"(c) kept frame nearest the boundary (black line); dashed = copies shifted\nby the assumed {MARGIN['BIM2']:g} m GPS error (red circle), grey band = {b:g} m buffer", fontsize=9)
    ax_.set_xlabel("E (m, UTM 49N)"); ax_.set_ylabel("N (m)")
    fig.suptitle(f"No-RTK case (BIM2, plain GPS image positions, nominal footprints): margin {MARGIN['BIM2']:g} m instead of {MARGIN['TN19']:g} m", fontsize=10)
    fig.tight_layout(); fig.savefig(fd / "fig_split_no_rtk.png", dpi=100); plt.close(fig)


# ----------------------------------------------------------------------------- component mode (overlap graph, margin only)
TARGET_C = np.array([0.70, 0.15, 0.15])
CUT_MIN_PX = 96             # smallest side of a crop window at a strip boundary (pixels of the 640 x 512 frame)
CONFLICT_EPS = 0.01          # m: units of different splits must be farther apart than 2 x margin - CONFLICT_EPS


def load_all_boxes():
    """kept YOLO boxes of all farms, per (flight_id, thermal_file)."""
    out = {}
    for farm, d in ms.FARM_DIR.items():
        fb = pd.read_csv(d / SUB / "thermal_frame_boxes.csv")
        fb = fb[fb.kept_in_yolo == 1]
        for k, g in fb.groupby(["flight_id", "thermal_file"]):
            out[k] = g.reset_index(drop=True)
    return out


def graph_components(geoms, d):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(geoms)
    ii, jj = STRtree(geoms).query(geoms, predicate="dwithin", distance=d)
    k = ii < jj
    A = coo_matrix((np.ones(int(k.sum())), (ii[k], jj[k])), shape=(n, n))
    return connected_components(A, directed=False)[1], int(k.sum())


def component_report(F, margins_list=(0.0, 1.5, 3.0, 5.0)):
    """overlap graph per farm: edge when the footprints, each buffered by the margin, intersect (distance <= 2 x margin)."""
    rows = []
    cc = [f"n_{c}" for c in CLASSES]
    for farm in FARMS:
        sub = F[F.farm == farm].reset_index(drop=True)
        nb = sub[cc].sum(1).values
        for mg in margins_list:
            lab, ne = graph_components(sub.geometry.values, 2 * mg)
            sz = np.bincount(lab)
            order = np.argsort(-sz)
            big = order[0]
            rows.append(dict(farm=farm, margin_m=mg, edge_distance_m=2 * mg, n_frames=len(sub), n_edges=ne, n_components=len(sz),
                             largest_component_frames=int(sz[big]), largest_component_pct=round(100 * sz[big] / len(sub), 1),
                             largest_component_boxes=int(nb[lab == big].sum()), second_largest_frames=int(sz[order[1]]) if len(sz) > 1 else 0,
                             isolated_frames=int((sz == 1).sum())))
    return pd.DataFrame(rows)


def assign_components(sub, lab, target=TARGET_C):
    """whole components -> train/val/test by exhaustive search (<= 13 components): box shares per class near the target, every class
    present in every split where possible, frame shares near the target."""
    lab = lab.copy()
    while lab.max() + 1 > 13:                       # exhaustive search is limited to 13 components: merge the two smallest (sensitivity runs only)
        sz = np.bincount(lab)
        a_, b_ = np.argsort(sz)[:2]
        lab[lab == b_] = a_
        _, lab = np.unique(lab, return_inverse=True)
    k = int(lab.max()) + 1
    cc = [f"n_{c}" for c in CLASSES]
    X = np.zeros((k, 5))
    for c in range(k):
        m = lab == c
        X[c, :4] = sub.loc[m, cc].sum().values
        X[c, 4] = m.sum()
    assert k <= 13, k
    O = np.array(list(itertools.product(range(3), repeat=k)), dtype=np.int8)
    K = np.stack([(O == s).astype(np.float32) @ X.astype(np.float32) for s in range(3)], 1)       # (n, 3, 5)
    T = X.sum(0)
    cost = np.zeros(len(O))
    for c in range(4):
        if T[c] <= 0:
            continue
        kept = K[:, :, c]
        share = kept / T[c]
        cost += np.abs(share - target).sum(1)
        cost += 6.0 * np.clip(1 - kept / max(1.0, min(0.08 * T[c], 80.0)), 0, 1).sum(1)
    cost += 1.0 * np.abs(K[:, :, 4] / T[4] - target).sum(1)
    cost += 50.0 * (K[:, :, 4] == 0).sum(1)
    i = int(np.argmin(cost))
    return O[i][lab], float(cost[i]), k, lab


def component_farm(F, farm, margin, target=TARGET_C):
    """small farm: whole overlap-graph components, no frame lost."""
    sub = F[F.farm == farm].reset_index(drop=True)
    lab, ne = graph_components(sub.geometry.values, 2 * margin)
    own, cost, k, lab = assign_components(sub, lab, target)
    sz = np.bincount(lab)
    out = sub[["farm", "flight_id", "thermal_file", "altitude_m"] + [f"n_{c}" for c in CLASSES]].copy()
    out["split"] = [SPL[o] for o in own]
    out["strip_id"] = [f"{farm}-K{int(l) + 1:02d}" for l in lab]
    out["reason"] = [f"overlap-graph component {int(l) + 1} of {k} ({int(sz[l])} frames, edge = footprints within {2 * margin:g} m) assigned to {SPL[o]}"
                     for l, o in zip(lab, own)]
    print(f"[{farm}] components {k}, sizes {sorted(sz.tolist(), reverse=True)}, cost {cost:.3f}")
    return out


# ----------------------------------------------------------------------------- cut crops (strip boundaries other than the MOC patch)
def _frame_box_info(FB, key, P, nv, cn, cache):
    if key not in cache:
        lst = []
        g = FB.get(key)
        if g is not None:
            for _, b in g.iterrows():
                pts = np.array([[b[f"x{j}_clip"] * 640, b[f"y{j}_clip"] * 512] for j in range(1, 5)])
                poly = shapely.Polygon(pts)
                if not poly.is_valid or poly.area <= 0:
                    continue
                cen = poly.centroid
                gc = g_map(P, cen.x, cen.y)
                lst.append((b, poly, cen, float(gc @ nv + cn)))
        cache[key] = lst
    return cache[key]


def cut_crop_candidates(S, FB, ax, runs, margin, eligible):
    """for every eligible (buffer) frame of one farm: one crop window per strip run it overlaps. runs = [(f0, f1, owner)] sorted;
    the first/last run are open outwards; the run boundaries are eroded by `margin`."""
    nv = np.array([ax.a, ax.b]) / ax.n
    cn = ax.c / ax.n
    cands, cache = [], {}
    for i in eligible:
        r_ = S.iloc[i]
        P = np.asarray(r_.geometry.exterior.coords)[:4]
        fr = ax.f(P[:, 0], P[:, 1])
        lo0, hi0 = float(fr.min()), float(fr.max())
        info = _frame_box_info(FB, (r_.flight_id, r_.thermal_file), P, nv, cn, cache)
        for k, (f0, f1, own) in enumerate(runs):
            if not (lo0 < f1 and hi0 > f0):
                continue
            lo_b = -BIG if k == 0 else f0 + margin
            hi_b = BIG if k == len(runs) - 1 else f1 - margin
            if lo_b > -BIG / 2 and hi_b < BIG / 2 and hi_b - lo_b < 1.0:
                continue
            rect = best_crop(P, nv, nv, lo_b - cn if lo_b > -BIG / 2 else -BIG, hi_b - cn if hi_b < BIG / 2 else BIG, -BIG, BIG, min_px=CUT_MIN_PX)
            if rect is None:
                continue
            poly = crop_polygon(P, rect)
            if not poly.within(r_.geometry.buffer(0.05)):
                continue
            x0, y0, x1, y1 = rect
            rr = shapely.box(x0, y0, x1, y1)
            kept, dropped = [], 0
            for b, bp, cen, fv in info:
                if not (f0 <= fv <= f1):
                    continue
                frac = bp.intersection(rr).area / bp.area
                if frac < MIN_BOX_FRAC or not rr.contains(cen):
                    dropped += 1
                    continue
                pts = np.asarray(bp.exterior.coords)[:4]
                if frac < 0.999:
                    pts = np.asarray(shapely.minimum_rotated_rectangle(bp.intersection(rr)).exterior.coords)[:4]
                    pts = np.clip(pts, [x0, y0], [x1, y1])
                kept.append(dict(box_uid=b.box_uid, cls=int(b.class_id), pts=pts - [x0, y0], clipped=bool(frac < 0.999)))
            if kept:
                cands.append(dict(i=i, farm=r_.farm, flight_id=r_.flight_id, thermal_file=r_.thermal_file, split=own, rect=rect, poly=poly,
                                  boxes=kept, dropped=dropped, run=k))
    return cands


def build_component_tn19(S, F, cells, FB, P, St, margin, cut_crops=True, T=TARGET_C, orders=None, layout=None):
    """TN19 (all farms in S): strips with the margin only, MOC patch by crop windows, then (B) greedy re-admission of dropped frames on the overlap
    graph and (C) crop windows of the frames that still straddle a strip boundary."""
    S2, crops, cboxes, mb, rg, tab, polys, info = build_strict(S, F, cells, margin=margin, orders=orders, T=np.asarray(T), layout=layout)
    ti = np.where((S2.farm == "TN19").values)[0]
    sub = S2.iloc[ti]
    G = sub.geometry.values
    n = len(sub)
    dcut = max(2 * margin - CONFLICT_EPS, 0.0)
    Ls = np.array([s if s in SPL else None for s in sub.split.values], dtype=object)
    ftree = STRtree(G)
    pc_poly = [polys[c] for c in crops.crop_id]                       # patch crops (accepted, protected)
    pc_split = list(crops.split)
    ctree_p = STRtree(pc_poly) if pc_poly else None
    strip_buf = sub.reason.str.startswith("buffer: footprint+").values & (sub.split.values == "buffer")
    nbox = sub[[f"n_{c}" for c in CLASSES]].sum(1).values
    ax = P.ax
    runs_df = St[St.farm == "TN19"].sort_values("f_from_m")
    runs = [(r.f_from_m, r.f_to_m, r.owner) for r in runs_df.itertuples()]
    cf = ax.f(np.array([g.centroid.x for g in G]), np.array([g.centroid.y for g in G]))

    def owner_of(f):
        for f0, f1, o in runs:
            if f0 <= f < f1:
                return o
        return runs[-1][2]
    S3 = S2.copy()
    reason = S3.reason.copy()

    def neighbours(i, crop_tree, crop_splits, crop_ok):
        s = set()
        for j in ftree.query(G[i], predicate="dwithin", distance=dcut):
            if j != i and Ls[j] is not None:
                s.add(Ls[j])
        if ctree_p is not None:
            for j in ctree_p.query(G[i], predicate="dwithin", distance=dcut):
                s.add(pc_split[j])
        if crop_tree is not None:
            for j in crop_tree.query(G[i], predicate="dwithin", distance=dcut):
                if crop_ok[j]:
                    s.add(crop_splits[j])
        return s

    # ---- (B) greedy re-admission
    order = [i for i in np.argsort(-nbox, kind="stable") if strip_buf[i]]
    n_re = 0
    for i in order:
        s = neighbours(i, None, None, None)
        if len(s) > 1:
            continue
        sp = next(iter(s)) if s else owner_of(cf[i])
        Ls[i] = sp
        n_re += 1
        reason.iloc[ti[i]] = f"re-admitted whole: no frame or crop of another split within {2 * margin:g} m (overlap-graph rule), split {sp}"
    for i in range(n):
        if Ls[i] is not None and sub.split.values[i] == "buffer":
            S3.loc[S3.index[ti[i]], "split"] = Ls[i]
    S3["reason"] = reason
    out_B = S3.copy()
    all_crops, all_cboxes = crops.copy(), cboxes.copy()
    all_crops["kind"] = "moc"
    polys_all = dict(polys)
    # ---- (C) cut crops for the frames that still straddle a boundary
    n_cut = 0
    if cut_crops:
        elig = [i for i in range(n) if strip_buf[i] and Ls[i] is None]
        cands = cut_crop_candidates(sub, FB, ax, runs, margin, elig)
        cands.sort(key=lambda c: -len(c["boxes"]))
        cpoly = [c["poly"] for c in cands]
        ctree = STRtree(cpoly) if cpoly else None
        ok = np.zeros(len(cands), bool)
        for k, c in enumerate(cands):
            conf = False
            for j in ftree.query(c["poly"], predicate="dwithin", distance=dcut):
                if Ls[j] is not None and Ls[j] != c["split"]:
                    conf = True; break
            if not conf and ctree_p is not None:
                conf = any(pc_split[j] != c["split"] for j in ctree_p.query(c["poly"], predicate="dwithin", distance=dcut))
            if not conf and ctree is not None:
                conf = any(ok[j] and cands[j]["split"] != c["split"] for j in ctree.query(c["poly"], predicate="dwithin", distance=dcut))
            if not conf:
                ok[k] = True
        acc = [c for c, o in zip(cands, ok) if o]
        acc.sort(key=lambda c: (c["split"], c["flight_id"], c["thermal_file"], c["rect"]))
        cnt = {s: 0 for s in SPL}
        rows, brows = [], []
        for c in acc:
            cnt[c["split"]] += 1
            cid = f"TN19_CUT_{c['split']}_{cnt[c['split']]:04d}"
            x0, y0, x1, y1 = c["rect"]
            ncls = np.zeros(4, int)
            for bx in c["boxes"]:
                ncls[bx["cls"]] += 1
                brows.append(dict(crop_id=cid, box_uid=bx["box_uid"], class_id=bx["cls"], split=c["split"], clipped=int(bx["clipped"]),
                                  **{f"{k_}{j + 1}": round(float(v), 2) for j, p_ in enumerate(bx["pts"]) for k_, v in zip("xy", p_)}))
            polys_all[cid] = c["poly"]
            rows.append(dict(crop_id=cid, farm="TN19", flight_id=c["flight_id"], thermal_file=c["thermal_file"], split=c["split"],
                             x0=x0, y0=y0, x1=x1, y1=y1, n_boxes_by_class=";".join(f"{n_}:{k_}" for n_, k_ in zip(CLASSES, ncls)),
                             n_boxes=int(ncls.sum()), **{f"n_{n_}": int(k_) for n_, k_ in zip(CLASSES, ncls)},
                             n_boxes_dropped_partly_outside=c["dropped"], area_m2=round(c["poly"].area, 1), kind="cut",
                             reason=f"frame straddles a strip boundary: largest pixel rectangle inside strip run {c['run'] + 1} (owner {c['split']}) "
                                    f"eroded by {margin:g} m; boxes kept if centre in the run and >= 70 % inside the crop"))
            fi = ti[c["i"]]
            S3.loc[S3.index[fi], "split"] = "crop"
            S3.loc[S3.index[fi], "crop_window"] = 1
            S3.loc[S3.index[fi], "reason"] = "buffer rule avoided: frame straddles a strip boundary, only its crop windows are used (frame_label_split_crops.csv)"
        n_cut = len(acc)
        if rows:
            all_crops = pd.concat([all_crops, pd.DataFrame(rows)], ignore_index=True)
            all_cboxes = pd.concat([all_cboxes, pd.DataFrame(brows)], ignore_index=True)
    info.update(n_readmitted=n_re, n_cut_crops=n_cut, runs=runs)
    return out_B, S3, all_crops, all_cboxes, mb, rg, tab, polys_all, info, crops, cboxes, polys


# ----------------------------------------------------------------------------- component mode: assembly, variants, audit, checks
A_DIRS = [OUT / "_strip_buffer_strict", OUT]


def a_file(name):
    for d in A_DIRS:
        if (d / name).exists():
            return d / name
    return None


def box_level_problem(F, FB, margin):
    """strip-search problem whose units are single boxes (ground square of 1 m, buffered by the margin): the balance seen by the search is the one
    that crop windows can reach, instead of the whole-frame balance."""
    t = F[F.farm == "TN19"]
    rows = []
    for r in t.itertuples():
        g = FB.get((r.flight_id, r.thermal_file))
        if g is None:
            continue
        P = np.asarray(r.geometry.exterior.coords)[:4]
        for _, b in g.iterrows():
            x = np.mean([b[f"x{j}_clip"] for j in range(1, 5)]) * 640
            y = np.mean([b[f"y{j}_clip"] for j in range(1, 5)]) * 512
            e, n = g_map(P, x, y)
            row = dict(farm="TN19", flight_id=r.flight_id, thermal_file=r.thermal_file, altitude_m=r.altitude_m,
                       geometry=shapely.box(e - 0.5, n - 0.5, e + 0.5, n + 0.5), cx=e, cy=n)
            row.update({f"n_{c}": int(int(b.class_id) == k) for k, c in enumerate(CLASSES)})
            rows.append(row)
    sub = pd.DataFrame(rows).reset_index(drop=True)
    return Problem(sub, np.full(len(sub), float(margin)), "train")


def strips_candidate(F, P, cuts, own, margin):
    sub = F[F.farm == "TN19"].reset_index(drop=True)
    res, strips = P.apply(np.asarray(cuts), own, np.full(len(sub), float(margin)), "TN19")
    strips.insert(0, "farm", "TN19")
    out = pd.concat([sub[["farm", "flight_id", "thermal_file", "altitude_m"] + [f"n_{c}" for c in CLASSES]], res], axis=1)
    return out, strips


def tune_cost(S_C, crops_all, cboxes_all):
    """cost of a finished TN19 result: per-class share deviation from 70/15/15 (SingleHotSpot, MultiHotSpot, Bypass) + presence + box loss."""
    t = S_C[S_C.farm == "TN19"]
    Cn = counts_strict(t, crops_all, cboxes_all, "x")
    c = Cn[Cn.farm == "TN19"].set_index("split")
    cost = 0.0
    for cl in CLASSES[:3]:
        k = c.loc[SPL, f"n_{cl}"].values.astype(float)
        tot = c[f"n_{cl}"].sum()
        sh = k / max(k.sum(), 1)
        cost += float(np.abs(sh - TARGET_C).sum()) + 6.0 * float(np.clip(1 - k / max(1.0, min(0.08 * tot, 80.0)), 0, 1).sum())
    cost += 3.0 * (1 - c.loc[SPL, "n_boxes"].sum() / c.n_boxes.sum())
    return cost


def assemble_component(F, cells, FB, margins, a, orders=None, tune=0):
    """-> dict with S_B, S_C, crops (patch only / all), cboxes, mb, rg, tab, polys, info, St, probs, S0 (strip result)."""
    TARGET[:] = TARGET_C
    o, st, P = split_farm(F, "TN19", "train", a.moc_buffer, margins["TN19"])
    cands = [(None, o, st)]
    if tune > 1:
        Pb = box_level_problem(F, FB, margins["TN19"])
        print("[box-level strip search] (a box counts when its centre is >= margin + 0.5 m from a strip boundary: crop windows recover them)", flush=True)
        Pb.search(2, 6)
        P.visited = Pb.visited
        vis = sorted(P.visited.items(), key=lambda kv: kv[1][0])
        seen = set()
        cands = []
        for cuts, (cst, own) in vis[:1500]:
            oo, ss = strips_candidate(F, P, cuts, own, margins["TN19"])
            sig = hash(tuple(oo.split.values))
            if sig in seen:
                continue
            seen.add(sig)
            cands.append((cst, oo, ss))
            if len(cands) >= tune:
                break
    small = [component_farm(F, farm, margins[farm]) for farm in ("TNDL", "TT", "BIM2")]
    best, layout, log = None, None, []
    for k, (cst, o_, st_) in enumerate(cands):
        S = pd.concat([o_] + small, ignore_index=True).merge(F[["flight_id", "thermal_file", "geometry"]], on=["flight_id", "thermal_file"])
        S = S.sort_values(["farm", "flight_id", "thermal_file"]).reset_index(drop=True)
        assert len(S) == 3000
        try:
            res = build_component_tn19(S, F, cells, FB, P, st_, margins["TN19"], cut_crops=True, orders=orders, layout=layout)
        except ValueError:
            raise
        S_B, S_C, crops_all, cboxes_all, mb, rg, tab, polys_all, info, crops_p, cboxes_p, polys_p = res
        if layout is None:
            layout = (info["order"], info["a"], info["b"])
            tab0 = tab
        tc = tune_cost(S_C, crops_all, cboxes_all)
        log.append(dict(candidate=k, strip_search_cost=cst, cuts=";".join(f"{x:g}" for x in st_[st_.farm == "TN19"].f_from_m.iloc[1:]),
                        owners="|".join(st_.owner), full_pipeline_cost=round(tc, 3), frames_in_buffer=int((S_C.split == "buffer").sum()),
                        crop_windows=len(crops_all)))
        if tune > 1:
            print(f"[tune] candidate {k}: search cost {cst:.3f}, full cost {tc:.3f}, buffer {log[-1]['frames_in_buffer']}, crops {len(crops_all)}", flush=True)
        if best is None or tc < best[0] - 1e-9:
            best = (tc, dict(S=S, St=pd.concat([st_], ignore_index=True), probs={"TN19": P}, S_B=S_B, S_C=S_C, crops_all=crops_all,
                             cboxes_all=cboxes_all, crops_p=crops_p, cboxes_p=cboxes_p, mb=mb, rg=rg, tab=tab, polys_all=polys_all,
                             polys_p=polys_p, info=info))
    R = best[1]
    R["tab"] = tab0
    R["tuning"] = pd.DataFrame(log)
    return R


def balance_row(Cn, variant, farm="ALL", target=TARGET_C):
    """deviation of the per-class box shares (train/val/test of the kept boxes) from the target; mean over classes present in 3 splits."""
    c = Cn[(Cn.variant == variant) & (Cn.farm == farm)].set_index("split")
    devs, out = [], {}
    for cl in CLASSES:
        k = c.loc[SPL, f"n_{cl}"].values.astype(float)
        if k.sum() > 0:
            sh = k / k.sum()
            out[f"share_{cl}"] = "/".join(f"{100 * x:.0f}" for x in sh)
            devs.append(float(np.abs(sh - target).sum() / 2))
    k = c.loc[SPL, "n_boxes"].values.astype(float)
    out["share_all_boxes"] = "/".join(f"{100 * x / k.sum():.0f}" for x in k)
    out["mean_class_share_deviation_pp"] = round(100 * float(np.mean(devs)), 1)
    out["classes_missing_in_a_split"] = int(sum(1 for cl in CLASSES if c.loc[SPL, f"n_{cl}"].min() == 0 and c.loc[SPL, f"n_{cl}"].sum() > 0))
    return out


def variants_table(Cn_all):
    """A (strip + 10 m buffer), B (overlap graph, margin only, whole frames), C (B + crop windows): frames and boxes kept/lost per farm."""
    rows = []
    for v in Cn_all.variant.unique():
        for farm in FARMS + ["ALL"]:
            c = Cn_all[(Cn_all.variant == v) & (Cn_all.farm == farm)].set_index("split")
            if c.empty:
                continue
            used = c.loc[SPL]
            boxes_total = int(c.n_boxes.sum())
            r = dict(variant=v, farm=farm, whole_frames_kept=int(used.n_whole_frames.sum()), crop_windows=int(used.n_crops.sum()),
                     frames_in_buffer=int(c.loc["buffer", "n_whole_frames"]), boxes_total=boxes_total, boxes_kept=int(used.n_boxes.sum()),
                     boxes_lost_pct=round(100 * (1 - used.n_boxes.sum() / boxes_total), 1),
                     **{f"{s}_frames+crops": f"{int(c.loc[s, 'n_whole_frames'])}+{int(c.loc[s, 'n_crops'])}" for s in SPL},
                     **{f"{s}_boxes": int(c.loc[s, "n_boxes"]) for s in SPL},
                     **{f"{s}_{cl}": int(c.loc[s, f"n_{cl}"]) for s in SPL for cl in CLASSES})
            r.update(balance_row(Cn_all, v, farm))
            rows.append(r)
    return pd.DataFrame(rows)


def frames_lost_pct(S, farm):
    sub = S[S.farm == farm]
    return round(100 * (sub.split == "buffer").sum() / len(sub), 1)


def independent_check(S2, crops, polys, margin):
    """Second footprint implementation (explicit rotation matrices, ray casting from the raw XMP pose, own code path) for all TN19 frames and
    crop windows in the split; compares the polygons with image_footprints.gpkg and repeats the cross-split pair audit with them."""
    from pyproj import Transformer
    inv = pd.read_csv(bf.INVENTORY, usecols=["flight_id", "file", "sensor", "width", "height", "model", "focal_mm", "GpsLatitude", "GpsLongitude",
                                             "RelativeAltitude", "GimbalYawDegree", "GimbalPitchDegree", "GimbalRollDegree"])
    inv = inv[inv.sensor == "T"].set_index(["flight_id", "file"])
    tr = Transformer.from_crs(4326, 32649, always_xy=True)
    t = S2[(S2.farm == "TN19") & S2.split.isin(SPL + ["crop"])]
    ind, rows = {}, []
    for r in t.itertuples():
        q = inv.loc[(r.flight_id, r.thermal_file)]
        x, y = tr.transform(q.GpsLongitude, q.GpsLatitude)
        pitch_mm, sw0, f = bf.SENSORS[(q.model, "T")]
        sw = sw0 if sw0 is not None else q.width * pitch_mm / 1000.0
        sh = sw * q.height / q.width
        if q.model == "ZH20T" and not np.isnan(q.focal_mm):
            f = q.focal_mm
        psi, p, rl = np.radians(q.GimbalYawDegree), np.radians(q.GimbalPitchDegree), np.radians(q.GimbalRollDegree)
        Rx = np.array([[1, 0, 0], [0, np.cos(p + np.pi / 2), -np.sin(p + np.pi / 2)], [0, np.sin(p + np.pi / 2), np.cos(p + np.pi / 2)]])
        Rz = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])      # heading psi clockwise from north
        Ry = np.array([[np.cos(rl), 0, np.sin(rl)], [0, 1, 0], [-np.sin(rl), 0, np.cos(rl)]])
        R = Rz @ Rx @ Ry
        alt = q.RelativeAltitude
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):          # BL, BR, TR, TL
            d = R @ np.array([sx * sw / 2, sy * sh / 2, -f])
            tt = alt / -d[2]
            pts.append((x + tt * d[0], y + tt * d[1]))
        poly = shapely.Polygon(pts)
        ind[(r.flight_id, r.thermal_file)] = np.array(pts)
        rows.append(dict(flight_id=r.flight_id, thermal_file=r.thermal_file, iou=poly.intersection(r.geometry).area / poly.union(r.geometry).area,
                         centre_offset_m=poly.centroid.distance(r.geometry.centroid)))
    cmp_ = pd.DataFrame(rows)
    # independent polygons of the units
    U = []
    for r in t[t.split.isin(SPL)].itertuples():
        U.append((r.split, shapely.Polygon(ind[(r.flight_id, r.thermal_file)])))
    for c in crops[crops.farm == "TN19"].itertuples():
        P = ind[(c.flight_id, c.thermal_file)]
        U.append((c.split, crop_polygon(P, (c.x0, c.y0, c.x1, c.y1))))
    g = np.array([u[1] for u in U], dtype=object)
    sp = np.array([u[0] for u in U])
    tree = STRtree(g)
    res = dict(n_units=len(U), n_frames_recomputed=len(cmp_), footprint_iou_median=round(float(cmp_.iou.median()), 4),
               footprint_iou_min=round(float(cmp_.iou.min()), 4), centre_offset_median_m=round(float(cmp_.centre_offset_m.median()), 3),
               centre_offset_max_m=round(float(cmp_.centre_offset_m.max()), 3))
    for lab, d in (("intersecting", 0.0), ("within_margin", margin), ("within_p95_2.3m", P95_RESIDUAL)):
        i2, j2 = tree.query(g, predicate="dwithin", distance=d) if d > 0 else tree.query(g, predicate="intersects")
        m2 = (i2 < j2) & (sp[i2] != sp[j2])
        if d == 0.0:
            m2 &= shapely.area(shapely.intersection(g[i2], g[j2])) > 1e-6
        res[f"cross_split_pairs_{lab}"] = int(m2.sum())
    i3, j3 = tree.query(g, predicate="dwithin", distance=40.0)
    m3 = (i3 < j3) & (sp[i3] != sp[j3])
    res["min_cross_split_distance_m"] = round(float(shapely.distance(g[i3[m3]], g[j3[m3]]).min()), 2)
    return res, cmp_


def audit_rows(S2, crops, polys, cells, ctree, assoc, margins, name, S_for_assoc=None):
    rows = []
    U = unit_table(S2, crops, polys)
    for farm in FARMS:
        u = U[U.farm == farm].reset_index(drop=True)
        r = audit_units(u, name, margins[farm], cells if farm == "TN19" else None, ctree if farm == "TN19" else None,
                        assoc if farm == "TN19" else None, S2[S2.farm == farm], crops_df=crops[crops.farm == farm] if len(crops) else None)
        i2, j2 = STRtree(u.geometry.values).query(u.geometry.values, predicate="dwithin", distance=2 * margins[farm] - CONFLICT_EPS)
        sp = u.split.values
        r["cross_split_pairs_within_2x_margin"] = int(((i2 < j2) & (sp[i2] != sp[j2])).sum())
        r["farm"] = farm
        rows.append(r)
    return rows


def polys_for_crops(crops, S):
    """ground polygons of a crop table (rect + frame footprint)."""
    key = {(f, t): g for f, t, g in zip(S.flight_id, S.thermal_file, S.geometry)}
    out = {}
    for c in crops.itertuples():
        P = np.asarray(key[(c.flight_id, c.thermal_file)].exterior.coords)[:4]
        out[c.crop_id] = crop_polygon(P, (c.x0, c.y0, c.x1, c.y1))
    return out


def run_component(a, F, cells):
    outdir = Path(a.out) if a.out else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    margins = {"TN19": a.margin_rtk, "TNDL": a.margin_rtk, "TT": a.margin_rtk, "BIM2": a.margin_bim2}
    MARGIN.update(margins)
    FB = load_all_boxes()
    comp = component_report(F)
    comp.to_csv(outdir / "frame_label_split_components.csv", index=False)
    print(comp[comp.margin_m.isin([1.5, 5.0])].to_string())
    R = assemble_component(F, cells, FB, margins, a, tune=a.tune)
    R["tuning"].to_csv(outdir / "frame_label_split_component_tuning.csv", index=False)
    S, St, probs = R["S"], R["St"], R["probs"]
    info = R["info"]
    print(f"[component] patch order {info['order']} cuts {info['a']}/{info['b']}, readmitted {info['n_readmitted']}, cut crops {info['n_cut_crops']}, "
          f"patch collateral {info['collateral']}")
    variant = a.variant
    S_B, S_C = R["S_B"], R["S_C"]
    crops_p, cboxes_p, polys_p = R["crops_p"], R["cboxes_p"], R["polys_p"]
    crops_all, cboxes_all, polys_all = R["crops_all"], R["cboxes_all"], R["polys_all"]
    # --- variants A / B / C
    CnB = counts_strict(S_B, crops_p, cboxes_p, "B_overlap_graph_margin_only")
    CnC = counts_strict(S_C, crops_all, cboxes_all, "C_overlap_graph_margin_only_with_crops")
    CnA = None
    fA = a_file("frame_label_split_counts.csv")
    if fA is not None:
        cA = pd.read_csv(fA)
        CnA = cA[cA.variant == "main_strict"].copy()
        CnA["variant"] = "A_strip_buffer_10m_strict"
    parts = [x for x in (CnA, CnB, CnC) if x is not None]
    Vt = variants_table(pd.concat(parts, ignore_index=True))
    Vt.to_csv(outdir / "frame_label_split_variants.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(Vt[["variant", "farm", "whole_frames_kept", "crop_windows", "frames_in_buffer", "boxes_total", "boxes_kept", "boxes_lost_pct",
              "share_all_boxes", "mean_class_share_deviation_pp", "classes_missing_in_a_split"]].to_string())
    chosen_S, chosen_crops, chosen_cboxes, chosen_polys = (S_C, crops_all, cboxes_all, polys_all) if variant == "C" else (S_B, crops_p, cboxes_p, polys_p)
    chosen_crops = chosen_crops.copy()
    # --- write chosen variant
    cols = ["farm", "flight_id", "thermal_file", "split", "strip_id", "reason", "altitude_m"] + [f"n_{c}" for c in CLASSES] + ["crop_window"]
    chosen_S[cols].to_csv(outdir / "frame_label_split.csv", index=False)
    S_B.assign(crop_window=S_B.crop_window)[cols].to_csv(outdir / "frame_label_split_variant_B.csv", index=False)
    St["moc_mode"] = "component"
    St["buffer_each_side_m"] = St.farm.map(margins)
    St.to_csv(outdir / "frame_label_split_strips.csv", index=False)
    chosen_crops.to_csv(outdir / "frame_label_split_crops.csv", index=False)
    chosen_cboxes.to_csv(outdir / "frame_label_split_crop_boxes.csv", index=False)
    R["mb"].to_csv(outdir / "frame_label_split_moc_boxes.csv", index=False)
    R["rg"].to_csv(outdir / "frame_label_split_moc_regions.csv", index=False)
    R["tab"].sort_values("cost").to_csv(outdir / "frame_label_split_moc_layouts.csv", index=False)
    moc_feasibility(F).to_csv(outdir / "frame_label_split_moc_feasibility.csv", index=False)
    # --- xdomain
    X = chosen_S.copy()
    t19 = X.farm == "TN19"
    mv = t19 & (X.split == "test")
    X.loc[mv, "split"] = "train"
    X.loc[mv, "reason"] = "TN19 test region moved to train (xdomain variant)"
    X.loc[~t19, "split"] = "test"
    X.loc[~t19, "reason"] = "external-domain test farm (" + X.loc[~t19, "farm"] + ")"
    X.loc[~t19, "strip_id"] = X.loc[~t19, "farm"] + "-all"
    X.loc[~t19, "crop_window"] = 0
    mp = lambda d: d.assign(split=d.split.replace({"test": "train"}))
    crops_x, cboxes_x, mb_x = mp(chosen_crops), mp(chosen_cboxes), mp(R["mb"])
    crops_x["reason"] = crops_x.reason + "; xdomain variant: test crops are used as train"
    X[cols].to_csv(outdir / "frame_label_split_xdomain_tn19_to_others.csv", index=False)
    crops_x.to_csv(outdir / "frame_label_split_xdomain_crops.csv", index=False)
    cboxes_x.to_csv(outdir / "frame_label_split_xdomain_crop_boxes.csv", index=False)
    mb_x.to_csv(outdir / "frame_label_split_xdomain_moc_boxes.csv", index=False)
    Cn = pd.concat([counts_strict(chosen_S, chosen_crops, chosen_cboxes, "main_component"),
                    counts_strict(X, crops_x, cboxes_x, "xdomain_tn19_to_others")], ignore_index=True)
    Cn.to_csv(outdir / "frame_label_split_counts.csv", index=False)
    print(Cn[Cn.farm.isin(["ALL", "TN19"])].drop(columns=[c for c in Cn.columns if c.startswith("units_")]).to_string())
    L = loss_table(Cn[Cn.variant == "main_component"], chosen_S)
    L.to_csv(outdir / "frame_label_split_buffer_loss.csv", index=False)
    print(L.to_string())
    # --- audit
    ctree = STRtree(cells.geometry.values)
    assoc = load_assoc()
    rows = audit_rows(chosen_S, chosen_crops, chosen_polys, cells, ctree, assoc, margins, "component_main")
    for farm in FARMS:
        base = S[S.farm == farm].reset_index(drop=True)
        for seed in (1, 2, 3):
            rs = random_split(len(base), SEED + seed)
            ub = pd.DataFrame(dict(farm=farm, unit=base.thermal_file, split=rs, geometry=base.geometry))
            r = audit_units(ub, f"random_per_frame_seed{seed}", margins[farm], cells if farm == "TN19" else None, ctree if farm == "TN19" else None,
                            assoc if farm == "TN19" else None, base.assign(split=rs))
            r["farm"] = farm
            rows.append(r)
    for tag, fname, cname in (("strip_buffer_strict_A", "frame_label_split.csv", "frame_label_split_moc_crops.csv"),):
        fs, fc = a_file(fname), a_file(cname)
        if fs is not None and fc is not None and CnA is not None:
            SA = pd.read_csv(fs).merge(F[["flight_id", "thermal_file", "geometry"]], on=["flight_id", "thermal_file"])
            if "crop_window" in SA.columns:
                CA = pd.read_csv(fc)
                rows += audit_rows(SA, CA, polys_for_crops(CA, SA), cells, ctree, assoc, margins, tag)
    rf = OUT / "_superseded_relaxed/frame_label_split.csv"
    if rf.exists():
        Rr = pd.read_csv(rf)[["flight_id", "thermal_file", "split"]].merge(S[["farm", "flight_id", "thermal_file", "geometry"]], on=["flight_id", "thermal_file"])
        for farm in FARMS:
            rr_ = Rr[(Rr.farm == farm) & Rr.split.isin(SPL)].reset_index(drop=True)
            ub = pd.DataFrame(dict(farm=farm, unit=rr_.thermal_file, split=rr_.split, geometry=rr_.geometry))
            r = audit_units(ub, "relaxed_superseded", margins[farm], cells if farm == "TN19" else None, ctree if farm == "TN19" else None,
                            assoc if farm == "TN19" else None, Rr[Rr.farm == farm])
            r["farm"] = farm
            rows.append(r)
    ux = unit_table(X, crops_x, chosen_polys)
    r = audit_units(ux.reset_index(drop=True), "xdomain_tn19_to_others", margins["TN19"], cells, ctree, assoc, X)
    r["farm"] = "ALL"
    rows.append(r)
    A_ = pd.DataFrame(rows)
    A_.insert(0, "farm", A_.pop("farm"))
    A_.to_csv(outdir / "frame_label_split_strict_audit.csv", index=False)
    print(A_[["farm", "variant", "n_units", "cross_split_pairs_intersecting", "cross_split_pairs_within_margin", "cross_split_pairs_within_p95_2.3m",
              "min_cross_split_distance_m"]].to_string())
    # --- independent check
    ic, cmp_ = independent_check(chosen_S, chosen_crops, chosen_polys, margins["TN19"])
    pd.DataFrame([ic]).to_csv(outdir / "frame_label_split_independent_check.csv", index=False)
    print("[independent check]", ic)
    R_ = dict(R=R, chosen_S=chosen_S, chosen_crops=chosen_crops, chosen_polys=chosen_polys, X=X, A=A_, margins=margins, Cn=Cn, Vt=Vt, FB=FB)
    # --- sensitivity
    if not a.no_sensitivity:
        ms_ = margin_sensitivity_component(F, cells, FB, a)
        ms_.to_csv(outdir / "frame_label_split_margin_sensitivity.csv", index=False)
        print(ms_.to_string())
        alt = moc_alternatives(R["S"], F, cells)
        alt.to_csv(outdir / "frame_label_split_moc_alternatives.csv", index=False)
    # --- figures
    if not a.no_figures:
        fd = outdir / "figures"
        fd.mkdir(exist_ok=True)
        figures_component(R_, fd, a, variant)
    return R_


def margin_sensitivity_component(F, cells, FB, a):
    rows = []
    cc = [f"n_{c}" for c in CLASSES]
    for m in (0.0, 1.5, 2.3, 3.0, 5.0):
        mg = {f: m for f in FARMS}
        try:
            R = assemble_component(F, cells, FB, mg, a, orders=[("train", "val", "test")], tune=3)
            S2, crops, cboxes, polys = R["S_C"], R["crops_all"], R["cboxes_all"], R["polys_all"]
            mb = R["mb"]
            info = R["info"]
        except ValueError:
            R = None
        if R is None:
            rows.append(dict(margin_m=m, farm="TN19", note="infeasible: no ModuleOpenCircuit region layout gives val and test a box; the patch is unusable"))
            print(f"[component margin sensitivity] {m} m: patch infeasible", flush=True)
            continue
        U = unit_table(S2, crops, polys)
        for farm in FARMS:
            sub = S2[S2.farm == farm]
            u = U[U.farm == farm].reset_index(drop=True)
            au = audit_units(u, "m", m)
            i2, j2 = STRtree(u.geometry.values).query(u.geometry.values, predicate="dwithin", distance=max(2 * m - CONFLICT_EPS, 0.0))
            sp = u.split.values
            n2 = int(((i2 < j2) & (sp[i2] != sp[j2])).sum()) if m > 0 else au["cross_split_pairs_intersecting"]
            cb = cboxes.merge(crops[["crop_id", "farm"]], on="crop_id") if len(cboxes) else cboxes
            cbf = int((cb.farm == farm).sum()) if len(cb) else 0
            used_b = int(sub[sub.split.isin(SPL)][cc].sum().sum()) + cbf
            tot = int(sub[cc].sum().sum())
            r = dict(margin_m=m, farm=farm, n_frames=len(sub), whole_frames_kept=int(sub.split.isin(SPL).sum()), crop_windows=int((crops.farm == farm).sum()),
                     frames_in_buffer=int((sub.split == "buffer").sum()), frame_loss_pct=round(100 * (sub.split == "buffer").sum() / len(sub), 1),
                     boxes_total=tot, boxes_used=used_b, box_loss_pct=round(100 * (1 - used_b / tot), 1),
                     cross_split_pairs_intersecting=au["cross_split_pairs_intersecting"], cross_split_pairs_within_margin=au["cross_split_pairs_within_margin"],
                     cross_split_pairs_within_p95_2p3m=au["cross_split_pairs_within_p95_2.3m"], cross_split_pairs_within_2x_margin=n2,
                     min_cross_split_distance_m=au["min_cross_split_distance_m"])
            if farm == "TN19":
                nb = mb[mb.split.isin(SPL)].split.value_counts().reindex(SPL, fill_value=0)
                r.update(moc_boxes_train=int(nb.train), moc_boxes_val=int(nb.val), moc_boxes_test=int(nb.test),
                         readmitted_frames=info["n_readmitted"], cut_crops=info["n_cut_crops"], patch_collateral_frames=info["collateral"])
            rows.append(r)
        print(f"[component margin sensitivity] {m} m done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- component mode: figures
def fig_principle_component(S, crops, polys, St, probs, fd, m):
    plt = _mpl()
    P = probs["TN19"]
    x0, x1, y0, y1 = _frame_view(S, "TN19")
    c0 = ((x0 + x1) / 2, (y0 + y1) / 2)
    t = S[S.farm == "TN19"]
    cuts = sorted(set(St[St.farm == "TN19"].f_from_m.tolist()[1:]))
    fc = np.array([P.ax.f(g.centroid.x, g.centroid.y) for g in t.geometry])
    cp = crops[crops.kind == "cut"] if "kind" in crops else crops.iloc[0:0]
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax_ in axs:
        _strip_layer(ax_, St, probs, "TN19", c0, 600, m)
        ax_.set_aspect("equal")
    ax_ = axs[0]
    _add(ax_, list(t[t.split == "buffer"].geometry), fc=COL["buffer"], ec="none", alpha=0.4, z=1)
    for s_ in SPL:
        _add(ax_, list(t[t.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.3, alpha=0.15, z=2)
    for r in cp.itertuples():
        _add(ax_, [polys[r.crop_id]], fc=COL[r.split], ec="none", alpha=0.35, z=3)
    ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1)
    ax_.set_title(f"(a) TN19 (all altitudes): strips = splits; dark band = {2 * m:g} m gap (margin {m:g} m on each side);\n"
                  "light fill = whole frames, darker fill = crop windows of frames that straddle a boundary", fontsize=9)
    ax_ = axs[1]
    c_ = cuts[0]
    own0 = St[(St.farm == "TN19") & (St.f_to_m == c_)].owner.iloc[0]
    own1 = St[(St.farm == "TN19") & (St.f_from_m == c_)].owner.iloc[0]
    # straddling frame with two crops of different splits (or one) near the first cut
    cand = None
    if len(cp):
        g = cp.groupby(["flight_id", "thermal_file"]).split.nunique()
        two = [k for k, v in g.items() if v >= 2]
        if two:
            kk = min(two, key=lambda k: abs(P.ax.f(*t[(t.flight_id == k[0]) & (t.thermal_file == k[1])].geometry.iloc[0].centroid.coords[0]) - c_))
            cand = kk
    if cand is not None:
        fr = t[(t.flight_id == cand[0]) & (t.thermal_file == cand[1])].geometry.iloc[0]
        cx, cy = fr.centroid.x, fr.centroid.y
        ax_.set_xlim(cx - 60, cx + 60); ax_.set_ylim(cy - 45, cy + 45)
        _add(ax_, [fr], ec="k", lw=1.6, z=5)
        for r in cp[(cp.flight_id == cand[0]) & (cp.thermal_file == cand[1])].itertuples():
            _add(ax_, [polys[r.crop_id]], fc=COL[r.split], ec=COL[r.split], lw=1.2, alpha=0.5, z=4)
            c = polys[r.crop_id].centroid
            ax_.text(c.x, c.y, f"{r.split} crop", ha="center", va="center", fontsize=8, zorder=6, bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
        near = t[[g_.distance(fr) < 70 for g_ in t.geometry]]
        for r in near.itertuples():
            if r.split in SPL:
                _add(ax_, [r.geometry], fc=COL[r.split], ec=COL[r.split], lw=0.8, alpha=0.18, z=2)
        ax_.text(fr.bounds[0] + 1, fr.bounds[3] - 3, "one frame straddling the boundary (black outline)", fontsize=8, zorder=7,
                 bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
    ax_.set_title(f"(b) example: a straddling frame is not discarded, it yields one crop window per side\n(each eroded by {m:g} m); frames fully inside a strip stay whole", fontsize=9)
    for a_ in axs:
        a_.set_xlabel("E (m, EPSG:32649, UTM 49N)"); a_.set_ylabel("N (m)"); a_.tick_params(labelsize=8)
    _legend_split(axs[0])
    fig.tight_layout(); fig.savefig(fd / "fig_split_principle.png", dpi=100); plt.close(fig)


def fig_small_farms_component(S, fd):
    plt = _mpl()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax_, farm in zip(axs, ["TNDL", "TT", "BIM2"]):
        x0, x1, y0, y1 = _frame_view(S, farm)
        t = S[S.farm == farm]
        for s_ in SPL:
            _add(ax_, list(t[t.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.4, alpha=0.25, z=2)
        k = t.strip_id.nunique()
        n = {s_: int((t.split == s_).sum()) for s_ in SPL}
        ax_.set_title(f"{farm}: {k} overlap-graph components, margin {MARGIN[farm]:g} m; " + ", ".join(f"{a} {b}" for a, b in n.items()) + ", buffer 0", fontsize=8.5)
        ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1); ax_.set_aspect("equal"); ax_.tick_params(labelsize=7)
        ax_.set_xlabel("E (m, UTM 49N)", fontsize=8)
    axs[0].set_ylabel("N (m, UTM 49N)")
    _legend_split(axs[2])
    fig.suptitle("Small farms: whole connected components of the footprint-overlap graph are assigned to one split (no frame is lost)", fontsize=10)
    fig.tight_layout(); fig.savefig(fd / "fig_split_small_farms.png", dpi=100); plt.close(fig)


def fig_no_rtk_component(S, fd, allfp):
    plt = _mpl()
    from matplotlib.patches import Ellipse
    t = S[S.farm == "BIM2"]
    x0, x1, y0, y1 = _frame_view(S, "BIM2")
    e = MARGIN["BIM2"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.3, 0.7, 1.1]})
    ax_ = axs[0]
    _add(ax_, list(allfp.geometry), ec="#cccccc", lw=0.15, z=0.2)
    for s_ in SPL:
        _add(ax_, list(t[t.split == s_].geometry), fc=COL[s_], ec=COL[s_], lw=0.4, alpha=0.3, z=2)
    ax_.set_xlim(x0, x1); ax_.set_ylim(y0, y1); ax_.set_aspect("equal")
    ax_.set_title(f"(a) BIM2: all thermal frames (grey outline), labelled frames coloured by split; "
                  f"{t.strip_id.nunique()} components, all kept", fontsize=9)
    ax_.set_xlabel("E (m, EPSG:32649, UTM 49N)"); ax_.set_ylabel("N (m)")
    _legend_split(ax_)
    ax_ = axs[1]
    for y, (lab, w, c) in enumerate((("RTK farms (TN19, TNDL, TT)", MARGIN["TN19"], "#2c7fb8"), ("BIM2 (no RTK)", e, "#d95f0e"))):
        ax_.barh(y, 4 * w, left=-2 * w, color=c, alpha=0.35, height=0.5)
        ax_.text(0, y + 0.36, f"{lab}: margin {w:g} m -> keep-out {2 * w:g} m between splits", ha="center", fontsize=8)
    ax_.axvline(0, color="k", lw=2)
    ax_.set_xlim(-12, 12); ax_.set_ylim(-0.6, 1.8); ax_.set_yticks([])
    ax_.set_xlabel("distance (m) across the boundary between two splits"); ax_.set_title("(b) footprints of different splits must be\nfarther apart than 2 x margin", fontsize=9)
    ax_ = axs[2]
    g = t.geometry.values
    tree = STRtree(g)
    ii, jj = tree.query(g, predicate="dwithin", distance=40.0)
    best = None
    for i, j in zip(ii, jj):
        if i < j and t.split.values[i] != t.split.values[j]:
            d = g[i].distance(g[j])
            if best is None or d < best[0]:
                best = (d, i, j)
    if best is not None:
        d, i, j = best
        cx = (g[i].centroid.x + g[j].centroid.x) / 2; cy = (g[i].centroid.y + g[j].centroid.y) / 2
        for k in (i, j):
            sp = t.split.values[k]
            _add(ax_, [g[k]], fc=COL[sp], ec=COL[sp], lw=1.4, alpha=0.3, z=3)
            for dx, dy in ((e, 0), (-e, 0), (0, e), (0, -e)):
                _add(ax_, [shapely.affinity.translate(g[k], dx, dy)], ec="#222222", lw=0.7, ls="--", z=4)
            c = g[k].centroid
            ax_.add_patch(Ellipse((c.x, c.y), 2 * e, 2 * e, fill=False, ec="red", lw=1.1, zorder=6))
        ax_.set_xlim(cx - 45, cx + 45); ax_.set_ylim(cy - 35, cy + 35); ax_.set_aspect("equal")
        ax_.set_title(f"(c) closest frames of different splits ({d:.1f} m apart);\ndashed = shifted by the assumed {e:g} m GPS error (red circle)", fontsize=9)
    ax_.set_xlabel("E (m, UTM 49N)"); ax_.set_ylabel("N (m)")
    fig.suptitle(f"No-RTK case (BIM2, plain GPS positions, nominal footprints): margin {e:g} m instead of {MARGIN['TN19']:g} m", fontsize=10)
    fig.tight_layout(); fig.savefig(fd / "fig_split_no_rtk.png", dpi=100); plt.close(fig)


def figures_component(R_, fd, a, variant):
    S, crops, polys = R_["chosen_S"], R_["chosen_crops"], R_["chosen_polys"]
    St, probs, margins = R_["R"]["St"], R_["R"]["probs"], R_["margins"]
    m = margins["TN19"]
    fig_principle_component(S, crops, polys, St, probs, fd, m)
    fig_by_altitude(S, crops, polys, St, probs, fd, m, band=f"dark band = {2 * m:g} m gap", per_run=True)
    A = R_["A"]
    n_rand = int(A[(A.variant == "random_per_frame_seed1") & (A.farm == "TN19")].cross_split_pairs_intersecting.iloc[0])
    n_c = int(A[(A.variant == "component_main") & (A.farm == "TN19")].cross_split_pairs_intersecting.iloc[0])
    fig_leakage_compare(S, fd, n_rand, n_c, SEED + 1, crops=crops, polys=polys)
    R = R_["R"]
    fig_moc_patch(S, R["crops_p"], R["polys_p"], R["mb"], R["info"], fd)
    fig_small_farms_component(S, fd)
    fp = bf.read_gpkg_layer(bf.GPKG, "image_footprints")
    fig_no_rtk_component(S, fd, fp[(fp.farm == "BIM2") & (fp.sensor == "T")])


# ----------------------------------------------------------------------------- main
def run_strict(a, F, cells):
    outdir = OUT / "_strip_buffer_strict"           # archived fallback: strip + 10 m buffer strict split
    outdir.mkdir(parents=True, exist_ok=True)
    margins = {"TN19": a.margin_rtk, "TNDL": a.margin_rtk, "TT": a.margin_rtk, "BIM2": a.margin_bim2}
    MARGIN.update(margins)
    parts, strips, probs = [], [], {}
    for farm in FARMS:
        o, st, P = split_farm(F, farm, "train", a.moc_buffer, 10.0 + margins[farm])
        parts.append(o); strips.append(st); probs[farm] = P
    St = pd.concat(strips, ignore_index=True)
    S = pd.concat(parts, ignore_index=True).merge(F[["flight_id", "thermal_file", "geometry"]], on=["flight_id", "thermal_file"])
    S = S.sort_values(["farm", "flight_id", "thermal_file"]).reset_index(drop=True)
    assert len(S) == 3000
    St["moc_mode"] = "strict"
    St["buffer_each_side_m"] = St.farm.map(lambda f: 10.0 + margins[f])
    moc_feasibility(F).to_csv(outdir / "frame_label_split_moc_feasibility.csv", index=False)
    S2, crops, cboxes, mb, rg, tab, polys, info = build_strict(S, F, cells, margin=margins["TN19"])
    print(f"[strict] order {info['order']}, cut columns {info['a']}/{info['b']} of {info['ncol']}; crops {len(crops)}; "
          f"collateral frames to buffer {info['collateral']}; patch frames {info['n_patch_frames']}")
    cols = ["farm", "flight_id", "thermal_file", "split", "strip_id", "reason", "altitude_m"] + [f"n_{c}" for c in CLASSES] + ["crop_window"]
    S2[cols].to_csv(outdir / "frame_label_split.csv", index=False)
    St.to_csv(outdir / "frame_label_split_strips.csv", index=False)
    crops.to_csv(outdir / "frame_label_split_moc_crops.csv", index=False)
    cboxes.to_csv(outdir / "frame_label_split_moc_crop_boxes.csv", index=False)
    mb.to_csv(outdir / "frame_label_split_moc_boxes.csv", index=False)
    rg.to_csv(outdir / "frame_label_split_moc_regions.csv", index=False)
    tab.sort_values("cost").to_csv(outdir / "frame_label_split_moc_layouts.csv", index=False)
    # ---- xdomain variant
    X = S2.copy()
    t19 = X.farm == "TN19"
    mv = t19 & (X.split == "test")
    X.loc[mv, "split"] = "train"
    X.loc[mv, "reason"] = "TN19 test strip moved to train (xdomain variant)"
    X.loc[~t19, "split"] = "test"
    X.loc[~t19, "reason"] = "external-domain test farm (" + X.loc[~t19, "farm"] + ")"
    X.loc[~t19, "strip_id"] = X.loc[~t19, "farm"] + "-all"
    X.loc[~t19, "crop_window"] = 0
    mp = lambda d: d.assign(split=d.split.replace({"test": "train"}))
    crops_x, cboxes_x, mb_x = mp(crops), mp(cboxes), mp(mb)
    crops_x["reason"] = crops_x.reason + "; xdomain variant: test crops are used as train"
    X[cols].to_csv(outdir / "frame_label_split_xdomain_tn19_to_others.csv", index=False)
    crops_x.to_csv(outdir / "frame_label_split_xdomain_moc_crops.csv", index=False)
    cboxes_x.to_csv(outdir / "frame_label_split_xdomain_moc_crop_boxes.csv", index=False)
    mb_x.to_csv(outdir / "frame_label_split_xdomain_moc_boxes.csv", index=False)
    # ---- counts and loss
    Cn = pd.concat([counts_strict(S2, crops, cboxes, "main_strict"), counts_strict(X, crops_x, cboxes_x, "xdomain_tn19_to_others")], ignore_index=True)
    Cn.to_csv(outdir / "frame_label_split_counts.csv", index=False)
    pd.set_option("display.width", 250)
    print(Cn[Cn.farm.isin(["ALL", "TN19"])].to_string())
    L = loss_table(Cn[Cn.variant == "main_strict"], S2)
    L.to_csv(outdir / "frame_label_split_buffer_loss.csv", index=False)
    print(L.to_string())
    # ---- audit
    ctree = STRtree(cells.geometry.values)
    assoc = load_assoc()
    rows = []
    U = unit_table(S2, crops, polys)
    for farm in FARMS:
        u = U[U.farm == farm].reset_index(drop=True)
        sub = S2[S2.farm == farm]
        r = audit_units(u, "strict_main", margins[farm], cells if farm == "TN19" else None, ctree if farm == "TN19" else None,
                        assoc if farm == "TN19" else None, sub)
        r["farm"] = farm
        rows.append(r)
        if farm != "BIM2" and r["cross_split_pairs_within_p95_2.3m"] > 0:
            print("WARNING: leakage at the p95 residual for", farm)
        base = S[S.farm == farm].reset_index(drop=True)
        for seed in (1, 2, 3):
            rs = random_split(len(base), SEED + seed)
            ub = pd.DataFrame(dict(farm=farm, unit=base.thermal_file, split=rs, geometry=base.geometry))
            r = audit_units(ub, f"random_per_frame_seed{seed}", margins[farm], cells if farm == "TN19" else None,
                            ctree if farm == "TN19" else None, assoc if farm == "TN19" else None, base.assign(split=rs))
            r["farm"] = farm
            rows.append(r)
    rel_f = OUT / "_superseded_relaxed/frame_label_split.csv"
    if rel_f.exists():
        R = pd.read_csv(rel_f)[["flight_id", "thermal_file", "split"]].merge(S[["farm", "flight_id", "thermal_file", "geometry"]],
                                                                          on=["flight_id", "thermal_file"])
        for farm in FARMS:
            rr_ = R[(R.farm == farm) & R.split.isin(SPL)].reset_index(drop=True)
            ub = pd.DataFrame(dict(farm=farm, unit=rr_.thermal_file, split=rr_.split, geometry=rr_.geometry))
            r = audit_units(ub, "relaxed_superseded", margins[farm], cells if farm == "TN19" else None,
                            ctree if farm == "TN19" else None, assoc if farm == "TN19" else None, R[R.farm == farm])
            r["farm"] = farm
            rows.append(r)
    ux = unit_table(X, crops_x, polys)
    r = audit_units(ux.reset_index(drop=True), "xdomain_tn19_to_others", margins["TN19"], cells, ctree, assoc, X)
    r["farm"] = "ALL"
    rows.append(r)
    A = pd.DataFrame(rows)
    A.insert(0, "farm", A.pop("farm"))
    A.to_csv(outdir / "frame_label_split_strict_audit.csv", index=False)
    print(A.T.to_string())
    # ---- sensitivity / alternatives
    if not a.no_sensitivity:
        ms_ = margin_sensitivity(F, cells)
        ms_.to_csv(outdir / "frame_label_split_margin_sensitivity.csv", index=False)
        print(ms_.to_string())
        alt = moc_alternatives(S, F, cells)
        alt.to_csv(outdir / "frame_label_split_moc_alternatives.csv", index=False)
        print(alt.to_string())
    # ---- figures
    if not a.no_figures:
        fd = outdir / "figures"
        fd.mkdir(exist_ok=True)
        buf = 10.0 + margins["TN19"]
        n_rand = int(A[(A.variant == "random_per_frame_seed1") & (A.farm == "TN19")].cross_split_pairs_intersecting.iloc[0])
        n_strict = int(A[(A.variant == "strict_main") & (A.farm == "TN19")].cross_split_pairs_intersecting.iloc[0])
        fig_principle(S2, St, probs, fd, buf)
        fig_by_altitude(S2, crops, polys, St, probs, fd, buf)
        fig_leakage_compare(S2, fd, n_rand, n_strict, SEED + 1)
        fig_moc_patch(S2, crops, polys, mb, info, fd)
        fig_small_farms(S2, St, probs, fd)
        fp = bf.read_gpkg_layer(bf.GPKG, "image_footprints")
        fig_no_rtk(S2, St, probs, fd, fp[(fp.farm == "BIM2") & (fp.sensor == "T")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moc-mode", choices=["component", "strict", "relaxed", "train", "test", "internal"], default="component",
                    help="strict (default): regions are map areas, the ModuleOpenCircuit patch is split with CROP WINDOWS, 0 cross-split overlap; "
                         "relaxed: SUPERSEDED (overlapping frames across splits; tables in metadata/splits/_superseded_relaxed/); "
                         "train: the patch stays in train; test: the patch is test-only; internal: strips may cut through the patch "
                         "(--moc-buffer, infeasible)")
    ap.add_argument("--moc-buffer", type=float, default=20.0)
    ap.add_argument("--margin-rtk", type=float, default=None, help="safety margin (m) for the RTK farms: default 2.3 m (component split, the measured p95 frame-to-map error), 1.5 m for the archived strip + buffer modes")
    ap.add_argument("--margin-bim2", type=float, default=5.0, help="margin (m) for BIM2 (no RTK, plain GPS positions)")
    ap.add_argument("--variant", choices=["B", "C"], default="C", help="component mode: B = whole frames only, C = B + crop windows at strip boundaries")
    ap.add_argument("--tune", type=int, default=12, help="component mode: number of candidate strip layouts run through the full pipeline (best full-pipeline cost wins)")
    ap.add_argument("--out", default=None, help="output directory (default dataset/metadata/splits)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the margin sensitivity table and the layout alternatives (~5 min)")
    a = ap.parse_args()
    if a.margin_rtk is None:
        a.margin_rtk = 2.3 if a.moc_mode == "component" else 1.5
    F = load_frames()
    if a.moc_mode == "strict":
        run_strict(a, F, ms.load_cells())
    elif a.moc_mode == "component":
        run_component(a, F, ms.load_cells())
    else:
        legacy = a.moc_mode != "relaxed"
        margins = {f: (MARGIN[f] if legacy and f != "BIM2" else (a.margin_bim2 if legacy else 0.0)) for f in FARMS}
        if legacy:
            margins = {"TN19": a.margin_rtk, "TNDL": a.margin_rtk, "TT": a.margin_rtk, "BIM2": a.margin_bim2}
        outdir = OUT / "_strip_buffer_strict" if legacy else OUT / "_superseded_relaxed"
        outdir.mkdir(parents=True, exist_ok=True)
        main_legacy(a, F, margins, outdir)


def figures_legacy(S, St, probs, suffix, X, C, BT=None, fd=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    fd = fd or OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    col = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c", "buffer": "#bbbbbb"}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    t = S[S.farm == "TN19"]
    for s in ["buffer", "train", "val", "test"]:
        for g in t[t.split == s].geometry:
            xy = np.asarray(g.exterior.coords)
            ax.add_patch(MP(xy, closed=True, fill=s != "buffer", fc=col[s], ec=col[s], alpha=0.12 if s != "buffer" else 1, lw=0.4))
    P = probs["TN19"]
    ax.set_xlim(t.geometry.apply(lambda g: g.bounds[0]).min() - 20, t.geometry.apply(lambda g: g.bounds[2]).max() + 20)
    ax.set_ylim(t.geometry.apply(lambda g: g.bounds[1]).min() - 20, t.geometry.apply(lambda g: g.bounds[3]).max() + 20)
    xs = np.array(ax.get_xlim()); ys = np.array(ax.get_ylim())
    for _, r in St[St.farm == "TN19"].iterrows():      # strip boundaries: draw f = const lines
        for fv in (r.f_from_m,):
            # f = (x - 0.4*(y-V0))/n  ->  x = f*n + 0.4*(y-V0)
            yy = ys; xx = fv * P.ax.n + 0.4 * (yy - 1263600.0)
            ax.plot(xx, yy, color="k", lw=0.5, ls=":")
        xm = (r.f_from_m + r.f_to_m) / 2 * P.ax.n + 0.4 * (ys[1] - 1263600.0)
        ax.text(xm, ys[1] - 4, r.strip_id.split("-")[1], ha="center", va="top", fontsize=8, color=col[r.owner])
    ax.set_aspect("equal"); ax.set_title(f"TN19 labelled thermal frames by split (footprints; grey = buffer)")
    ax.set_xlabel("E (m, EPSG:32649)"); ax.set_ylabel("N (m)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=col[s], lw=6, alpha=0.5, label=s) for s in col], loc="lower right", fontsize=8)
    ax = axs[1]
    cc = C[(C.variant == f"main{suffix}") & (C.farm == "ALL")].set_index("split")
    w = 0.2
    for i, s in enumerate(SPL):
        ax.bar(np.arange(4) + (i - 1) * w, [cc.loc[s, f"n_{c}"] for c in CLASSES], w, color=col[s], label=s)
    ax.set_xticks(range(4)); ax.set_xticklabels([c.replace("Module", "Module\n") for c in CLASSES], fontsize=8)
    ax.set_yscale("log"); ax.set_title("boxes per class and split (all farms)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fd / f"frame_label_split{suffix}_TN19.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, farm in zip(axs, ["TNDL", "TT", "BIM2"]):
        t = S[S.farm == farm]
        for s in ["buffer", "train", "val", "test"]:
            for g in t[t.split == s].geometry:
                xy = np.asarray(g.exterior.coords)
                ax.add_patch(MP(xy, closed=True, fill=s != "buffer", fc=col[s], ec=col[s], alpha=0.15 if s != "buffer" else 1, lw=0.4))
        b = np.array([g.bounds for g in t.geometry])
        ax.set_xlim(b[:, 0].min() - 10, b[:, 2].max() + 10); ax.set_ylim(b[:, 1].min() - 10, b[:, 3].max() + 10)
        ax.set_aspect("equal"); ax.set_title(f"{farm}: {len(t)} labelled frames", fontsize=9)
    fig.tight_layout(); fig.savefig(fd / f"frame_label_split{suffix}_small_farms.png", dpi=120); plt.close(fig)

    if BT is not None:        # relaxed ModuleOpenCircuit patch: footprints of the patch frames by split, box centres by module group
        fig, ax = plt.subplots(figsize=(11, 6.5))
        mf = S[(S.farm == "TN19") & (S.flight_id == MOC_FLIGHT) & (S.n_ModuleOpenCircuit > 0)]
        for r in mf.itertuples():
            xy = np.asarray(r.geometry.exterior.coords)
            ax.add_patch(MP(xy, closed=True, fill=False, ec=col[r.split], lw=0.9, alpha=0.8))
        gcol = dict(col, gap="#444444")
        for gname in ["train", "val", "test", "gap"]:
            q = BT[BT.module_group == gname]
            ax.scatter(q.E, q.N, s=7, color=gcol[gname], label=f"boxes of module group {gname}" + (" (unused)" if gname == "gap" else ""))
        ax.set_xlim(265485, 265542); ax.set_ylim(1263765, 1263810); ax.set_aspect("equal")
        ax.set_title("ModuleOpenCircuit patch (TN19_20260411_42M_T2): frame footprints by split, box centres by module group", fontsize=9)
        ax.set_xlabel("E (m, EPSG:32649)"); ax.set_ylabel("N (m)"); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(fd / "frame_label_split_moc_patch.png", dpi=130); plt.close(fig)



if __name__ == "__main__":
    main()
