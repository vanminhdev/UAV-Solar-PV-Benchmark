#!/usr/bin/env python
"""Frame-level, leakage-aware split of the labelled thermal frames (3000 frames, thermal_frame_boxes.csv).

    python -X utf8 code/scripts/make_frame_label_split.py                    # primary (--moc-mode relaxed): ModuleOpenCircuit patch split by module groups
    python -X utf8 code/scripts/make_frame_label_split.py --moc-mode train     # former default: MOC block kept in train (strict 10 m buffer)
    python -X utf8 code/scripts/make_frame_label_split.py --moc-mode test      # MOC block held out as test-only
    python -X utf8 code/scripts/make_frame_label_split.py --moc-mode internal  # cut through the MOC block with a larger buffer (infeasible)

Basis: the farm-level spatial-strip machinery of make_splits.py (sheared strip axis of TN19, contiguous strips, buffer
recommendation 10 m, rule "a frame is kept in split s iff its footprint buffered by b metres lies inside strips owned by s,
otherwise split = buffer"). Differences to make_splits.py (written before any label existed):
  * unit = one labelled THERMAL frame (640x512 footprint, image_footprints.gpkg), not an RGB/thermal pair;
  * more than three strips are allowed (a split may own several strips) so that every class can be present in every split;
  * strips (cut positions, 5 m grid, 2..6 cuts for TN19, 2..3 for the small farms) and their owners are chosen by a
    deterministic multi-start coordinate descent on a cost = class-share deviation from 60/15/25 + class-presence penalty
    + loss to the buffer + altitude-mix deviation (+ ModuleOpenCircuit constraint);
  * a strip contains every flight and altitude at that location, so the same ground surveyed at 31/35/42 m is in one split.
Variants written to dataset/metadata/splits/:
  frame_label_split.csv                          main (all farms; TN19 / TNDL / TT / BIM2 each split by strips)
  frame_label_split_xdomain_tn19_to_others.csv   train/val = TN19 (TN19 test strip moved to train), test = TNDL + TT + BIM2
  frame_label_split_moc{train,test,internal}.csv (only with --moc-mode train|test|internal) alternative main splits
  frame_label_split_moc_boxes.csv                (relaxed) per-box table of the 416 ModuleOpenCircuit boxes: box_uid, split, reason
                                                 (+ frame split, module group, reference cell); split = ignore for boxes of another
                                                 module group inside a shared frame or in the gap between groups
  frame_label_split_xdomain_moc_boxes.csv        same for the xdomain variant
  frame_label_split_moc_leakage.csv              (relaxed) residual overlap inside the ModuleOpenCircuit patch
  frame_label_split_moc_feasibility.csv          how many ModuleOpenCircuit frames can sit on each side of a straight cut, per buffer
  frame_label_split_{strips,counts,audit}.csv    strip table, images/boxes per class per split, leakage audit (+ random baseline)

--moc-mode relaxed (default). All 416 ModuleOpenCircuit boxes lie in 31 frames of one flight (TN19_20260411_42M_T2), one row-aligned
patch of 16 module columns (~29 m); a 42 m frame covers ~22 x 18 m, so a 3-way strip split with the 10 m buffer is geometrically
impossible. The strips are computed exactly as in `--moc-mode train` (the rest of the split is therefore unchanged), then the patch
is re-split by MODULE GROUPS: box centres are projected to the ground (bilinear map of the footprint), assigned to the TN19
reference cell they fall in, and the 16 cell columns along the table rows are cut into three contiguous groups train/val/test
separated by a gap of one module column (boxes in the gap are not used). Cut positions and the order of the groups are chosen by
exhaustive search (cost: deviation of retained box and module shares from 60/20/20 + share of boxes lost; each split needs >= 3 frames and >= 30 boxes).
A frame goes to the group holding the majority of its MOC boxes; its MOC boxes of other groups (and gap boxes) are flagged
`ignore` in frame_label_split_moc_boxes.csv (make_yolo_dataset_view.py drops them from the label files). Frames overlap, so the
three groups still share image content; the residual overlap is measured in frame_label_split_moc_leakage.csv.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_footprints as bf  # noqa: E402
import make_splits as ms  # noqa: E402

ROOT = bf.ROOT
DS = ROOT / "dataset"
OUT = DS / "metadata/splits"
SUB = "annotations/model2_thermal_anomaly_obb"
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


def split_farm(F, farm, moc_mode, moc_buffer):
    sub = F[F.farm == farm].reset_index(drop=True)
    buf = np.full(len(sub), 10.0)
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
                          usecols=["flight_id", "box_id", "thermal_file"])
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


def best_crop(P, eu, ev, ub_lo, ub_hi, vb_lo, vb_hi, step=4):
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
    ok = (y1 - y0) >= MIN_CROP_PX
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
    ok &= (x0 <= hi0) & (x1 >= lo1) & ((x1 - x0) >= MIN_CROP_PX)
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
    if r[2] - r[0] < MIN_CROP_PX or r[3] - r[1] < MIN_CROP_PX:
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

    def crop_boxes(self, tf, P, r, reg):
        """boxes of frame tf kept in crop r: list of dict(box_uid, cls, pts in crop pixel coordinates), number dropped."""
        g = self.fb[tf]
        x0, y0, x1, y1 = r
        rect = shapely.box(x0, y0, x1, y1)
        kept, dropped = [], 0
        for _, b in g.iterrows():
            pts = np.array([[b[f"x{j}_clip"] * 640, b[f"y{j}_clip"] * 512] for j in range(1, 5)])
            poly = shapely.Polygon(pts)
            if not poly.is_valid or poly.area <= 0:
                continue
            cen = poly.centroid
            gc = g_map(P, cen.x, cen.y)
            u, v = float(gc @ self.eu), float(gc @ self.ev)
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

    def search(self, S0, T=np.array([0.6, 0.2, 0.2]), min_boxes=1):
        tot = max(S0[S0.split.isin(SPL)][CLS_COLS].values.sum(), 1)
        patch = S0[(S0.flight_id == MOC_FLIGHT) & (S0.n_ModuleOpenCircuit > 0) & (S0.split == "train")][CLS_COLS].values.sum()
        key_boxes = S0.set_index(["flight_id", "thermal_file"])[CLS_COLS].sum(axis=1)
        rows, best = [], None
        for order in itertools.permutations(SPL):
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moc-mode", choices=["relaxed", "train", "test", "internal"], default="relaxed",
                    help="relaxed (default): the 31-frame ModuleOpenCircuit patch is split by module groups into train/val/test with a "
                         "one-module gap (strips as in `train`; boxes of other groups in a shared frame are flagged in "
                         "frame_label_split_moc_boxes.csv); train: the block stays in train (strict 10 m buffer, former default); "
                         "test: the block is held out as test-only; internal: strips may cut through it, MOC frames get --moc-buffer "
                         "(infeasible for buffer >= ~10 m, see frame_label_split_moc_feasibility.csv)")
    ap.add_argument("--moc-buffer", type=float, default=20.0)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    relaxed = a.moc_mode == "relaxed"
    strip_mode = "train" if relaxed else a.moc_mode          # relaxed = strips of `train`, then the patch is re-split
    primary = relaxed
    F = load_frames()
    if primary:
        moc_feasibility(F).to_csv(OUT / "frame_label_split_moc_feasibility.csv", index=False)
        print(pd.read_csv(OUT / "frame_label_split_moc_feasibility.csv").to_string())
    parts, strips, probs = [], [], {}
    for farm in FARMS:
        o, st, P = split_farm(F, farm, strip_mode, a.moc_buffer)
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
    OUT.mkdir(parents=True, exist_ok=True)
    S[cols].to_csv(OUT / f"frame_label_split{suffix}.csv", index=False)
    St["moc_mode"] = a.moc_mode
    St.to_csv(OUT / f"frame_label_split_strips{suffix}.csv", index=False)

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
        X[cols].to_csv(OUT / "frame_label_split_xdomain_tn19_to_others.csv", index=False)
        bcols = ["box_uid", "split", "reason", "flight_id", "thermal_file", "frame_split", "module_group", "ref_cell", "cell_column"]
        BT[bcols].to_csv(OUT / "frame_label_split_moc_boxes.csv", index=False)
        BX[bcols].to_csv(OUT / "frame_label_split_xdomain_moc_boxes.csv", index=False)
        moc_leakage(S, BT, cells).to_csv(OUT / "frame_label_split_moc_leakage.csv", index=False)
        print(pd.read_csv(OUT / "frame_label_split_moc_leakage.csv").to_string())

    # ---- counts
    C = counts_table(S, f"main{suffix}")
    if primary:
        C = pd.concat([C, counts_table(X, "xdomain_tn19_to_others")], ignore_index=True)
    C.to_csv(OUT / f"frame_label_split_counts{suffix}.csv", index=False)
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
    A.to_csv(OUT / f"frame_label_split_audit{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(A.T.to_string())
    if not a.no_figures:
        figures(S, St, probs, suffix, X, C, BT)


def figures(S, St, probs, suffix, X, C, BT=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    fd = OUT / "figures"
    fd.mkdir(exist_ok=True)
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
