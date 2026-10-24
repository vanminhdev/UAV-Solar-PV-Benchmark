#!/usr/bin/env python
"""Build the RGB (W) <-> thermal (T) pairing table of the UAV PV dataset.

Read-only: uses only the image inventory CSV (never opens images).

Rule (see dataset/metadata/pairing/README.md), applied per flight_id:
  Stage 1 (seq): candidate = W,T with
        H20T flights (TN19/TNDL/TT): identical seq          (method seq_equal)
        Mavic 2 (BIM2):              |seq difference| == 1  (method seq_adjacent)
      cost = |dt| + 0.1 * dist_m, one-to-one by min-cost assignment.
  Stage 2 (time_position): leftovers matched if
        (|dt| <= 3 s and dist <= 20 m) or (dist <= 3 m and |dt| <= 10 s).
  Status: exact_pair  = stage-1 pair with |dt| <= 1 s and |dseq| <= 1
          nearest_pair= any other accepted pair (|dt| <= 3 s or dist <= 3 m)
          unpaired_rgb / unpaired_thermal otherwise.
Sign convention: delta = thermal - rgb.

Usage:
  python -X utf8 code/scripts/build_pairing.py \
      --inventory dataset/metadata/inventory/image_inventory.csv \
      --out dataset/metadata/pairing/rgb_thermal_pairing.csv
A per-flight summary CSV is written next to --out (pairing_summary_by_flight.csv).
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import linear_sum_assignment

BIG = 1e6
T_NEAR_S, D_NEAR_M = 3.0, 20.0     # time-based fallback
D_POS_M, T_POS_S = 3.0, 10.0       # position-based fallback
EXACT_DT_S = 1.0

COLS = ["farm", "flight_id", "pair_id", "rgb_file", "thermal_file", "rgb_timestamp",
        "thermal_timestamp", "delta_t_s", "delta_seq", "delta_pos_m", "pairing_status",
        "pair_method"]


def match(W, T, cand_fn, cost_fn):
    """One-to-one min-cost assignment restricted to candidates; returns list of (i, j)."""
    if len(W) == 0 or len(T) == 0:
        return []
    dt = (T["t"].values[None, :] - W["t"].values[:, None]) / np.timedelta64(1, "s")
    dx = T["x"].values[None, :] - W["x"].values[:, None]
    dy = T["y"].values[None, :] - W["y"].values[:, None]
    dist = np.hypot(dx, dy)
    ds = T["seq"].values[None, :] - W["seq"].values[:, None]
    ok = cand_fn(dt, dist, ds)
    cost = np.where(ok, cost_fn(dt, dist), BIG)
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if ok[i, j]]


def pair_flight(g, h20t):
    W = g[g.sensor == "W"].sort_values(["t", "seq"]).reset_index(drop=True)
    T = g[g.sensor == "T"].sort_values(["t", "seq"]).reset_index(drop=True)
    cost = lambda dt, dist: np.abs(dt) + 0.1 * dist
    if h20t:
        seq_ok = lambda dt, dist, ds: ds == 0
        m1 = "seq_equal"
    else:
        seq_ok = lambda dt, dist, ds: np.abs(ds) == 1
        m1 = "seq_adjacent"
    pairs = [(i, j, m1) for i, j in match(W, T, seq_ok, cost)]
    # a stage-1 pair that is neither time- nor position-compatible is rejected
    def accepted(i, j):
        dt = abs((T.t[j] - W.t[i]).total_seconds())
        d = np.hypot(T.x[j] - W.x[i], T.y[j] - W.y[i])
        return dt <= T_NEAR_S or d <= D_POS_M
    pairs = [p for p in pairs if accepted(p[0], p[1])]
    uw = sorted(set(range(len(W))) - {p[0] for p in pairs})
    ut = sorted(set(range(len(T))) - {p[1] for p in pairs})
    Wl, Tl = W.loc[uw].reset_index(drop=True), T.loc[ut].reset_index(drop=True)
    fb_ok = lambda dt, dist, ds: ((np.abs(dt) <= T_NEAR_S) & (dist <= D_NEAR_M)) | \
                                 ((dist <= D_POS_M) & (np.abs(dt) <= T_POS_S))
    for i, j in match(Wl, Tl, fb_ok, cost):
        pairs.append((uw[i], ut[j], "time_position"))
    rows = []
    for i, j, m in pairs:
        w, t = W.loc[i], T.loc[j]
        dts = (t.t - w.t).total_seconds()
        dseq = int(t.seq - w.seq)
        d = float(np.hypot(t.x - w.x, t.y - w.y))
        exact = m != "time_position" and abs(dts) <= EXACT_DT_S and abs(dseq) <= 1
        rows.append(dict(farm=w.farm, flight_id=w.flight_id, rgb_file=w.file, thermal_file=t.file,
                         rgb_timestamp=w.timestamp, thermal_timestamp=t.timestamp, delta_t_s=dts,
                         delta_seq=dseq, delta_pos_m=round(d, 3),
                         pairing_status="exact_pair" if exact else "nearest_pair", pair_method=m,
                         _k=w.t, _s=w.seq))
    paired_w = {p[0] for p in pairs}
    paired_t = {p[1] for p in pairs}
    for i in range(len(W)):
        if i not in paired_w:
            w = W.loc[i]
            rows.append(dict(farm=w.farm, flight_id=w.flight_id, rgb_file=w.file, thermal_file="",
                             rgb_timestamp=w.timestamp, pairing_status="unpaired_rgb",
                             pair_method="no_thermal_in_flight" if len(T) == 0 else "none",
                             _k=w.t, _s=w.seq))
    for j in range(len(T)):
        if j not in paired_t:
            t = T.loc[j]
            rows.append(dict(farm=t.farm, flight_id=t.flight_id, rgb_file="", thermal_file=t.file,
                             thermal_timestamp=t.timestamp, pairing_status="unpaired_thermal",
                             pair_method="none", _k=t.t, _s=t.seq))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="dataset/metadata/inventory/image_inventory.csv")
    ap.add_argument("--out", default="dataset/metadata/pairing/rgb_thermal_pairing.csv")
    a = ap.parse_args()

    inv = pd.read_csv(a.inventory)
    inv["t"] = pd.to_datetime(inv.timestamp.astype("int64").astype(str), format="%Y%m%d%H%M%S")
    zone = Transformer.from_crs(4326, 32649, always_xy=True)  # UTM 49N
    inv["x"], inv["y"] = zone.transform(inv.GpsLongitude.values, inv.GpsLatitude.values)

    out = []
    for fid, g in inv.groupby("flight_id", sort=True):
        out += pair_flight(g, h20t=(g.model.iloc[0] == "ZH20T"))
    df = pd.DataFrame(out).sort_values(["flight_id", "_k", "_s"], kind="stable")
    df["pair_id"] = [f"{f}_P{n:04d}" for f, n in zip(df.flight_id, df.groupby("flight_id").cumcount() + 1)]
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    df = df[COLS]
    for c in ("rgb_timestamp", "thermal_timestamp", "delta_seq"):
        df[c] = df[c].astype("Int64")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8", lineterminator="\n")

    # integrity: every inventory image appears exactly once
    used = pd.concat([df.rgb_file[df.rgb_file != ""], df.thermal_file[df.thermal_file != ""]])
    assert len(used) == len(inv), "image count mismatch"
    n_img = inv.groupby("flight_id").size()
    npair = df.pairing_status.isin(["exact_pair", "nearest_pair"]).groupby(df.flight_id).sum()
    unp = df.pairing_status.str.startswith("unpaired").groupby(df.flight_id).sum()
    assert ((npair * 2 + unp) == n_img).all(), "image count mismatch"

    rows = []
    for fid, g in df.groupby("flight_id"):
        p = g[g.pairing_status.isin(["exact_pair", "nearest_pair"])]
        dt = p.delta_t_s.astype(float)
        rows.append(dict(
            farm=g.farm.iloc[0], flight_id=fid, n_rgb=int((g.rgb_file != "").sum()),
            n_thermal=int((g.thermal_file != "").sum()), n_exact=int((g.pairing_status == "exact_pair").sum()),
            n_nearest=int((g.pairing_status == "nearest_pair").sum()),
            n_unpaired_rgb=int((g.pairing_status == "unpaired_rgb").sum()),
            n_unpaired_thermal=int((g.pairing_status == "unpaired_thermal").sum()),
            dt_median_s=dt.median() if len(p) else np.nan,
            dt_p95_s=dt.quantile(.95) if len(p) else np.nan,
            dt_abs_max_s=dt.abs().max() if len(p) else np.nan,
            dpos_median_m=p.delta_pos_m.median() if len(p) else np.nan,
            dpos_max_m=p.delta_pos_m.max() if len(p) else np.nan,
            n_fallback=int((p.pair_method == "time_position").sum())))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(os.path.dirname(os.path.abspath(a.out)), "pairing_summary_by_flight.csv"),
             index=False, lineterminator="\n")
    print(s.to_string(index=False))
    pp = df[df.pairing_status.isin(["exact_pair", "nearest_pair"])]
    print("\nALL pairs:", len(pp), "delta_t median/p95/absmax:", pp.delta_t_s.astype(float).median(),
          pp.delta_t_s.astype(float).quantile(.95), pp.delta_t_s.astype(float).abs().max())
    print(df.pairing_status.value_counts().to_string())
    print(pd.crosstab(pp.pair_method, pp.pairing_status).to_string())


if __name__ == "__main__":
    main()
