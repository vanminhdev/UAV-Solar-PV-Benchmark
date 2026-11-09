#!/usr/bin/env python
"""Monte Carlo of the residual leakage of a frame-level split under measured footprint errors.

    python -X utf8 code/scripts/leakage_mc.py --variant C_margin1.5 --scenario S1 [--draws 1000] [--bim2-sigma 2.0]
    python -X utf8 code/scripts/leakage_mc.py --variant C_margin1.5 --scenario S3        # deterministic registration-corrected footprints

Units = whole frames of train/val/test + crop windows of the split table (variants in leakage_common.VARIANTS). Every draw perturbs the footprint of every
parent frame (crops of a frame move with it) and counts, per farm and split pair, the cross-split pairs whose footprints intersect, the overlapped area as a
share of the smaller footprint, and the share of boxes that fall into the footprint of a unit of another split.
Scenarios
  S1  independent frame errors: a residual vector drawn from the measured frame -> ortho residuals of the farm (projection_check, variant V0, centred on the
      farm mean, scaled by altitude / altitude of the measured row), plus a rotation about the frame centre (sd 0.5 deg). BIM2 (no measured residuals): isotropic
      Gaussian with sd --bim2-sigma per axis (assumption, default 2.0 m).
  S2  S1 + a shift shared by all frames of a flight (sd per axis = relative offset between overlapping flights measured by ortho_registration; BIM2: 3 m).
  S3  deterministic: footprints corrected with the measured per-flight offsets (mean residual per flight, then the registration transform to the base ortho).
Output: dataset/metadata/splits/leakage_check/mc_<variant>_<scenario>.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakage_common as lc  # noqa: E402

DMAX = 30.0          # candidate unit pairs: nominal distance below this (m)
DBOX = 18.0          # candidate (box, unit) pairs
ROT_SD = np.radians(0.5)


def ring(c):
    return np.concatenate([c, c[:, :1]], 1)


def flight_bias_sd():
    """per-axis sd of the shared per-flight shift of the RTK farms: relative offsets of TN19 flights whose registration origins are within 60 m
    (same ground, different flight/altitude), RMS / sqrt(2)."""
    t, _ = lc.registration()
    t = t[t.index.str.startswith("TN19") & ~t.index.str.contains("MAP")]
    ids = list(t.index)
    d = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = t.loc[ids[i]], t.loc[ids[j]]
            if np.hypot(a.origin_E - b.origin_E, a.origin_N - b.origin_N) < 60:
                d.append([a.shift_E_at_origin_m - b.shift_E_at_origin_m, a.shift_N_at_origin_m - b.shift_N_at_origin_m])
    d = np.array(d)
    return float(np.sqrt((d ** 2).mean()) / np.sqrt(2)), len(d)


class Setup:
    def __init__(self, variant):
        self.U, self.B = lc.load_units(variant)
        U = self.U
        self.n = len(U)
        self.corners = np.stack(U.corners.values)
        self.split = U.split.values
        self.farm = U.farm.values
        self.fidx = U.frame_idx.values
        keys, cen = lc.frame_centres(U)
        self.keys, self.cen = keys, cen
        self.nf = len(keys)
        self.f_farm = np.array([k[0].split("_")[0] for k in keys])
        self.f_flight = np.array([k[0] for k in keys])
        fr = lc.frames()
        self.f_alt = np.array([fr[k].altitude_m for k in keys], float)
        self.nom = shapely.polygons(ring(self.corners))
        self.area = shapely.area(self.nom)
        # candidate unit pairs per farm (cross-split, different parent frame)
        ia, ib = [], []
        for farm in lc.mf.FARMS:
            idx = np.where(self.farm == farm)[0]
            if len(idx) == 0:
                continue
            tree = STRtree(self.nom[idx])
            ii, jj = tree.query(self.nom[idx], predicate="dwithin", distance=DMAX)
            k = (ii < jj) & (self.split[idx][ii] != self.split[idx][jj]) & (self.fidx[idx][ii] != self.fidx[idx][jj])
            ia.append(idx[ii[k]]); ib.append(idx[jj[k]])
        self.ia, self.ib = np.concatenate(ia), np.concatenate(ib)
        sa, sb = self.split[self.ia], self.split[self.ib]
        pair_of = {("train", "val"): 0, ("val", "train"): 0, ("train", "test"): 1, ("test", "train"): 1, ("val", "test"): 2, ("test", "val"): 2}
        self.pcode = np.array([pair_of[(a, b)] for a, b in zip(sa, sb)])
        self.pfarm = self.farm[self.ia]
        # boxes
        B = self.B
        self.bx, self.by, self.bu = B.E.values, B.N.values, B.unit.values
        pts = shapely.points(self.bx, self.by)
        bi, bj = [], []
        for farm in lc.mf.FARMS:
            idx = np.where(self.farm == farm)[0]
            if len(idx) == 0:
                continue
            tree = STRtree(self.nom[idx])
            pi, pj = tree.query(pts, predicate="dwithin", distance=DBOX)
            gj = idx[pj]
            k = (self.farm[self.bu[pi]] == farm) & (self.split[self.bu[pi]] != self.split[gj]) & (self.fidx[self.bu[pi]] != self.fidx[gj])
            bi.append(pi[k]); bj.append(gj[k])
        self.bi, self.bj = np.concatenate(bi), np.concatenate(bj)
        self.bfarm = self.farm[self.bu]
        self.n_boxes_farm = {f: int((self.bfarm == f).sum()) for f in lc.mf.FARMS}
        self.bpair = None

    def transform(self, shift, theta):
        """shift (nf, 2), theta (nf,) -> corners (n,4,2) and box points."""
        c = self.cen[self.fidx][:, None, :]
        R = np.stack([np.stack([np.cos(theta), -np.sin(theta)], -1), np.stack([np.sin(theta), np.cos(theta)], -1)], -2)[self.fidx]   # (n,2,2)
        d = self.corners - c
        corn = c + np.einsum("nij,nkj->nki", R, d) + shift[self.fidx][:, None, :]
        cb = self.cen[self.fidx[self.bu]]
        Rb = np.stack([np.stack([np.cos(theta), -np.sin(theta)], -1), np.stack([np.sin(theta), np.cos(theta)], -1)], -2)[self.fidx[self.bu]]
        pb = cb + np.einsum("nij,nj->ni", Rb, np.c_[self.bx, self.by] - cb) + shift[self.fidx[self.bu]]
        return corn, pb

    def evaluate(self, corn, pb):
        polys = shapely.polygons(ring(corn))
        a, b = polys[self.ia], polys[self.ib]
        hit = shapely.intersects(a, b)
        frac = np.zeros(len(self.ia))
        if hit.any():
            ar = shapely.area(shapely.intersection(a[hit], b[hit]))
            frac[hit] = ar / np.minimum(shapely.area(a[hit]), shapely.area(b[hit]))
        inside = shapely.contains_xy(polys[self.bj], pb[self.bi, 0], pb[self.bi, 1])
        bhit = np.zeros(len(self.bx), bool)
        bhit[self.bi[inside]] = True
        return hit & (frac > 1e-6), frac, bhit


def draw_shift(rng, S, scen, pools, sd_bias, bim2_sigma):
    shift = np.zeros((S.nf, 2))
    for farm in lc.mf.FARMS:
        m = S.f_farm == farm
        if not m.any():
            continue
        if farm in pools and farm != "BIM2":
            p = pools[farm]
            res = p["res"] - p["res"].mean(0)
            k = rng.integers(len(res), size=m.sum())
            shift[m] = res[k] * (S.f_alt[m] / p["alt"][k])[:, None]
        else:
            shift[m] = rng.normal(0, bim2_sigma, (m.sum(), 2))
        if scen == "S2":
            fl = S.f_flight[m]
            u, inv = np.unique(fl, return_inverse=True)
            sd = 3.0 if farm == "BIM2" else sd_bias
            shift[m] += rng.normal(0, sd, (len(u), 2))[inv]
    return shift


def s3_shift(S):
    """registration-corrected footprints: per flight, corners -> T_f(corners - mean residual). Returns corners (n,4,2) and box points."""
    t, mres = lc.registration()
    corn = S.corners.copy()
    pb = np.c_[S.bx, S.by].copy()
    ok = np.zeros(S.n, bool)
    for fl in np.unique(S.U.flight_id):
        if fl not in t.index:
            continue
        tr = t.loc[fl]
        r = mres.loc[fl].values if fl in mres.index else np.zeros(2)
        m = (S.U.flight_id == fl).values
        corn[m] = lc.apply_transform(S.corners[m] - r, tr)
        mb = m[S.bu]
        pb[mb] = lc.apply_transform(pb[mb] - r, tr)
        ok[m] = True
    return corn, pb, ok


def summarise(S, variant, scen, hits, fracs, bhits, extra):
    """hits (D, npairs) bool, fracs (D, npairs), bhits (D, nboxes) bool."""
    rows = []
    D = hits.shape[0]
    for farm in lc.mf.FARMS + ["ALL"]:
        for pc, name in [(0, "train-val"), (1, "train-test"), (2, "val-test"), (-1, "any")]:
            m = np.ones(len(S.ia), bool) if farm == "ALL" else (S.pfarm == farm)
            if pc >= 0:
                m &= S.pcode == pc
            cnt = hits[:, m].sum(1)
            fr = fracs[:, m][hits[:, m]]
            bm = np.ones(len(S.bx), bool) if farm == "ALL" else (S.bfarm == farm)
            nb = int(bm.sum())
            bshare = bhits[:, bm].sum(1) / max(nb, 1) if pc < 0 else np.full(D, np.nan)
            rows.append(dict(variant=variant, scenario=scen, farm=farm, split_pair=name, n_draws=D, n_candidate_pairs=int(m.sum()),
                             pairs_intersecting_mean=round(float(cnt.mean()), 3), pairs_lo95=float(np.percentile(cnt, 2.5)), pairs_hi95=float(np.percentile(cnt, 97.5)),
                             prob_any_pair=round(float((cnt > 0).mean()), 4), n_pair_draws_overlapping=int(len(fr)),
                             overlap_share_of_smaller_mean=round(float(fr.mean()), 4) if len(fr) else 0.0,
                             overlap_share_p95=round(float(np.percentile(fr, 95)), 4) if len(fr) else 0.0,
                             overlap_share_max=round(float(fr.max()), 4) if len(fr) else 0.0,
                             boxes_in_other_split_footprint_mean_pct=round(100 * float(np.nanmean(bshare)), 4) if pc < 0 else np.nan,
                             boxes_lo95_pct=round(100 * float(np.nanpercentile(bshare, 2.5)), 4) if pc < 0 else np.nan,
                             boxes_hi95_pct=round(100 * float(np.nanpercentile(bshare, 97.5)), 4) if pc < 0 else np.nan,
                             boxes_total=nb if pc < 0 else np.nan, **extra))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="C_margin1.5")
    ap.add_argument("--scenario", choices=["S1", "S2", "S3"], default="S1")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--bim2-sigma", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    t0 = time.time()
    S = Setup(a.variant)
    print(f"[{a.variant}/{a.scenario}] units {S.n}, frames {S.nf}, candidate pairs {len(S.ia)}, boxes {len(S.bx)}, box candidates {len(S.bi)}", flush=True)
    pools, _ = lc.residual_pools()
    sd_bias, nrel = flight_bias_sd()
    extra = dict(bim2_sigma_m=a.bim2_sigma, flight_bias_sd_m=round(sd_bias, 3), rot_sd_deg=0.5)
    if a.scenario == "S3":
        corn, pb, ok = s3_shift(S)
        hit, frac, bhit = S.evaluate(corn, pb)
        out = summarise(S, a.variant, "S3", hit[None], frac[None], bhit[None], dict(n_units_corrected=int(ok.sum()), **extra))
    else:
        rng = np.random.default_rng(a.seed)
        D = a.draws
        hits = np.zeros((D, len(S.ia)), bool)
        fracs = np.zeros((D, len(S.ia)), np.float32)
        bhits = np.zeros((D, len(S.bx)), bool)
        for d in range(D):
            shift = draw_shift(rng, S, a.scenario, pools, sd_bias, a.bim2_sigma)
            theta = rng.normal(0, ROT_SD, S.nf)
            corn, pb = S.transform(shift, theta)
            hits[d], fracs[d], bhits[d] = S.evaluate(corn, pb)
            if d % 100 == 0:
                print(f"  draw {d} ({time.time() - t0:.0f} s)", flush=True)
        out = summarise(S, a.variant, a.scenario, hits, fracs, bhits, extra)
    name = f"mc_{a.variant}_{a.scenario}{a.tag}.csv"
    lc.OUTD.mkdir(parents=True, exist_ok=True)
    out.to_csv(lc.OUTD / name, index=False)
    pd.set_option("display.width", 250)
    print(out[out.split_pair == "any"][["farm", "pairs_intersecting_mean", "pairs_lo95", "pairs_hi95", "prob_any_pair", "overlap_share_of_smaller_mean",
                                        "overlap_share_p95", "overlap_share_max", "boxes_in_other_split_footprint_mean_pct", "boxes_lo95_pct", "boxes_hi95_pct"]].to_string())
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
