#!/usr/bin/env python
"""Link thermal fault boxes (TN19) to the TN19 module reference grid and study the `ma_tam` code.

Run:  python -X utf8 code/scripts/associate_boxes_to_grid.py [--no-figures]

Reads (never writes):
  dataset/farm_01_thuan_nam_19/annotations/model2_thermal_anomaly_obb/thermal_boxes.csv
  dataset/farm_01_thuan_nam_19/module_reference/TN19_modules_master.gpkg   (table `modules`, EPSG:32649)
  dataset/metadata/grid_audit/registration_windows.csv   (thermal ortho vs 87 m ortho shift windows)
Writes dataset/metadata/association/ :
  thermal_box_to_cell.csv, association_summary.csv, registration_used.csv,
  code_analysis.json, code_pairs.csv, figures/*.png

Method
------
1. The thermal orthos (31/35 m flights) are NOT registered to the 87 m ortho on which the grid was
   drawn: the grid audit measured ortho-to-ortho shifts of 1.3-4.2 m (more than the 0.96 m row pitch and
   about 2 module widths).  Raw box-vs-grid overlap is therefore at chance level (see `raw_*`
   statistics).  We apply a LOCAL registration shift taken from the audit windows (inverse-distance
   interpolation of the shifts measured by ortho correlation, independent of the boxes; the sign is
   the one that makes the boxes land on the grid, checked on the data for both signs), then a small
   per-flight RESIDUAL translation fitted on the boxes (bounded to +-0.95 m along the row and +-0.5 m
   across, i.e. less than one module, so it cannot hop to another module).
2. For every box the corrected quadrilateral is intersected with all grid cells within 3 m; intersection
   area, IoU, centroid distance and offset (E/N and along/across the rows) are stored.
3. Status from the IoU of the best and second-best cells (thresholds in THRESH, justified in README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_footprints import read_gpkg_layer  # noqa: E402

ROOT = HERE.parent.parent
FARM = ROOT / "dataset" / "farm_01_thuan_nam_19"
BOXES = FARM / "annotations" / "model2_thermal_anomaly_obb" / "thermal_boxes.csv"
GPKG = FARM / "module_reference" / "TN19_modules_master.gpkg"
REGWIN = ROOT / "dataset" / "metadata" / "grid_audit" / "registration_windows.csv"
OUT = ROOT / "dataset" / "metadata" / "association"

# Status thresholds (see README): perfect overlap of a 1.827x0.905 box on a 1.904x0.96 cell is IoU 0.90.
THRESH = dict(unassigned_iou=0.30,   # best IoU below this: no cell explains the box
              unique_iou=0.50,       # best IoU at least this ...
              unique_ratio=2.0)      # ... and best >= ratio * second (or second < 0.5*best)
LOCAL_K, LOCAL_R = 15, 60.0
GRID_THETA = None  # filled from the data (row direction, radians)


# ------------------------------------------------------------------------------------------- loading
def load():
    g = read_gpkg_layer(GPKG, "modules")
    g["cx"] = [p.centroid.x for p in g.geometry]
    g["cy"] = [p.centroid.y for p in g.geometry]
    b = pd.read_csv(BOXES)
    b = b[b.farm_dir == "farm_01_thuan_nam_19"].reset_index(drop=True)
    b["block"] = None
    return g, b


def row_direction(g):
    s = g[(g.layer == "Block2") & (g.Row_ID == 1)].sort_values(["String_ID", "ID_Pin"])
    dx = np.diff(s.cx.values)
    dy = np.diff(s.cy.values)
    ok = (np.hypot(dx, dy) < 2.2)
    return float(np.median(np.arctan2(dy[ok], dx[ok])))


def box_polys(b):
    return [Polygon([(r.E1, r.N1), (r.E2, r.N2), (r.E3, r.N3), (r.E4, r.N4)]) for r in b.itertuples()]


# ------------------------------------------------------------------------------------- registration
def local_prior(rw, ortho, P, k=4):
    """IDW interpolation of the ortho-vs-87m shift (dE,dN) of `ortho` at points P (n,2)."""
    w = rw[(rw.ortho == ortho) & (rw.zncc > rw.second_peak)]
    if len(w) == 0:
        return np.zeros_like(P), 0
    tt = cKDTree(w[["E", "N"]].values)
    d, i = tt.query(P, k=min(k, len(w)))
    d = d.reshape(len(P), -1)
    i = i.reshape(len(P), -1)
    wt = 1.0 / (d + 10.0)
    sh = np.c_[(w.dE.values[i] * wt).sum(1) / wt.sum(1), (w.dN.values[i] * wt).sum(1) / wt.sum(1)]
    return sh, len(w)


def fast_iou(P, G, tree, theta, a=1.827, bb=0.905, ca=1.904, cb=0.96):
    """IoU of an axis-aligned a x bb box centred at P with the nearest cell (rotated by theta); a
    <1 deg rotation is ignored.  Used only for the residual search."""
    d, i = tree.query(P)
    o = P - G[i]
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])
    ox, oy = o @ u, o @ v
    ix = np.clip(np.minimum(ox + a / 2, ca / 2) - np.maximum(ox - a / 2, -ca / 2), 0, None)
    iy = np.clip(np.minimum(oy + bb / 2, cb / 2) - np.maximum(oy - bb / 2, -cb / 2), 0, None)
    inter = ix * iy
    return inter / (a * bb + ca * cb - inter)


# ----------------------------------------------------------------------------------- association
def associate(b, polys, g, tree_poly, shift_xy):
    """Candidate cells for each corrected box. Returns per-box dict list sorted by IoU."""
    out = []
    cx, cy = g.cx.values, g.cy.values
    th = GRID_THETA
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    from shapely.affinity import translate
    for n, (p, s) in enumerate(zip(polys, shift_xy)):
        q = translate(p, s[0], s[1])
        cand = tree_poly.query(q.buffer(1.5))
        rows = []
        for j in cand:
            gp = g.geometry.iloc[j]
            inter = q.intersection(gp).area
            if inter <= 0:
                continue
            iou = inter / (q.area + gp.area - inter)
            o = np.array([q.centroid.x - cx[j], q.centroid.y - cy[j]])
            rows.append((iou, j, inter, o))
        rows.sort(key=lambda r: -r[0])
        out.append(rows)
    return out


def classify(i1, i2):
    if i1 < THRESH["unassigned_iou"]:
        return "unassigned"
    if i1 >= THRESH["unique_iou"] and i2 <= i1 / THRESH["unique_ratio"]:
        return "unique"
    return "ambiguous"


def neighbour_kind(g, j1, j2):
    """Relation of second-best cell j2 to best cell j1."""
    a, c = g.iloc[j1], g.iloc[j2]
    if a.layer != c.layer or a.String_ID != c.String_ID:
        if a.layer == c.layer and a.Row_ID == c.Row_ID and abs(int(a.ID_Pin) - int(c.ID_Pin)) == 1:
            return "same_row_next_string"
        return "other_string"
    if a.Row_ID == c.Row_ID:
        return "same_row_adjacent" if abs(a.ID_Pin - c.ID_Pin) == 1 else "same_row_farther"
    if abs(a.Row_ID - c.Row_ID) == 1:
        return "adjacent_row_same_table"
    return "other_row_same_table"


def pct(x, q):
    return float(np.nanpercentile(x, q)) if len(x) else float("nan")


def main():
    global GRID_THETA
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    g, b = load()
    GRID_THETA = row_direction(g)
    G = g[["cx", "cy"]].values
    ctree = cKDTree(G)
    polys = box_polys(b)
    gpoly = STRtree(list(g.geometry))
    rw = pd.read_csv(REGWIN)
    C = b[["centre_E", "centre_N"]].values
    u = np.array([np.cos(GRID_THETA), np.sin(GRID_THETA)])
    v = np.array([-np.sin(GRID_THETA), np.cos(GRID_THETA)])

    # ---- raw (uncorrected) overlap, for reference
    raw_iou = fast_iou(C, G, ctree, GRID_THETA)
    b["raw_best_iou"] = raw_iou

    # ---- registration: local prior (sign -1 : box position minus the ortho shift) + residual
    shift = np.zeros_like(C)
    reg_rows = []
    sign_check = {}
    for f, idx in b.groupby("flight_id").groups.items():
        idx = np.array(list(idx))
        ortho = f.split("_", 2)[2]
        pr, nwin = local_prior(rw, ortho, C[idx])
        for sg in (-1, 1):
            sign_check[f"{f}|sign{sg:+d}"] = float(fast_iou(C[idx] + sg * pr, G, ctree, GRID_THETA).mean())
        base = C[idx] - pr
        best = (-1, 0, 0)
        for dE in np.arange(-0.95, 0.951, 0.05):
            for dN in np.arange(-0.5, 0.501, 0.05):
                s = fast_iou(base + [dE, dN], G, ctree, GRID_THETA).mean()
                if s > best[0]:
                    best = (s, dE, dN)
        shift[idx] = -pr + [best[1], best[2]]
        reg_rows.append(dict(flight_id=f, ortho=ortho, n_boxes=len(idx), n_windows=nwin,
                             prior_shift_E_mean=(-pr[:, 0]).mean(), prior_shift_N_mean=(-pr[:, 1]).mean(),
                             prior_shift_E_sd=pr[:, 0].std(), prior_shift_N_sd=pr[:, 1].std(),
                             resid_dE=best[1], resid_dN=best[2],
                             mean_fast_iou_raw=float(fast_iou(C[idx], G, ctree, GRID_THETA).mean()),
                             mean_fast_iou_prior=float(fast_iou(base, G, ctree, GRID_THETA).mean()),
                             mean_fast_iou_final=float(best[0])))
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / "registration_used.csv", index=False)
    b["reg_dE"], b["reg_dN"] = shift[:, 0], shift[:, 1]

    # ---- stage 1 result (ortho-audit prior + per-flight residual only), kept for comparison
    cands0 = associate(b, polys, g, gpoly, shift)
    iou_prior = np.array([r[0][0] if r else 0.0 for r in cands0])
    j_prior = [g.module_id.iloc[r[0][1]] if r else "" for r in cands0]

    # ---- stage 2: smooth local drift field.  The ortho windows are 30 m apart and 1-4 m wide errors
    # remain; each box is shifted by the median residual offset (box - best cell) of its K nearest
    # boxes of the same flight (self excluded, only boxes with IoU >= 0.3).  Two passes.
    # The correction is bounded by construction to the residual of already well-overlapping boxes
    # (<0.5 m), so it cannot move a box to a different module by itself.
    shift1 = shift.copy()
    for _ in range(2):
        cs = associate(b, polys, g, gpoly, shift)
        off = np.array([r[0][3] if r else [np.nan, np.nan] for r in cs])
        iou = np.array([r[0][0] if r else 0.0 for r in cs])
        for f, idx in b.groupby("flight_id").groups.items():
            idx = np.array(list(idx))
            good = idx[iou[idx] >= 0.3]
            tr = cKDTree(C[good])
            kk = min(LOCAL_K + 1, len(good))
            dist, ii = tr.query(C[idx], k=kk)
            upd = np.zeros((len(idx), 2))
            for n in range(len(idx)):
                nb = [good[m] for m, dd in zip(ii[n], dist[n]) if good[m] != idx[n] and dd < LOCAL_R]
                if len(nb) >= 4:
                    upd[n] = np.median(off[nb], axis=0)
            shift[idx] = shift[idx] - upd
    b["reg_dE"], b["reg_dN"] = shift[:, 0], shift[:, 1]
    b["drift_dE"], b["drift_dN"] = shift[:, 0] - shift1[:, 0], shift[:, 1] - shift1[:, 1]

    # ---- candidate cells
    cands = associate(b, polys, g, gpoly, shift)
    recs = []
    for n, rows in enumerate(cands):
        r = b.iloc[n]
        i1, j1, inter1, o1 = rows[0] if rows else (0.0, None, 0.0, np.array([np.nan, np.nan]))
        i2, j2 = (rows[1][0], rows[1][1]) if len(rows) > 1 else (0.0, None)
        st = classify(i1, i2)
        rec = dict(box_id=r.box_id, flight_id=r.flight_id, module_code=r.module_code, class_final=r.class_final,
                   in_eval_zone=r.in_eval_zone,
                   module_id=g.module_id.iloc[j1] if j1 is not None else "",
                   iou=round(i1, 4), intersection_m2=round(inter1, 4),
                   centroid_offset_m=round(float(np.hypot(*o1)), 3),
                   offset_E_m=round(float(o1[0]), 3), offset_N_m=round(float(o1[1]), 3),
                   offset_along_row_m=round(float(o1 @ u), 3), offset_across_row_m=round(float(o1 @ v), 3),
                   status=st,
                   second_best_module_id=g.module_id.iloc[j2] if j2 is not None else "",
                   second_best_iou=round(i2, 4),
                   second_best_relation=neighbour_kind(g, j1, j2) if (j1 is not None and j2 is not None) else "",
                   block=g.layer.iloc[j1] if j1 is not None else "",
                   string_id=g.String_ID.iloc[j1] if j1 is not None else "",
                   row_id=int(g.Row_ID.iloc[j1]) if j1 is not None else -1,
                   id_pin=int(g.ID_Pin.iloc[j1]) if j1 is not None else -1,
                   cell_E=round(float(g.cx.iloc[j1]), 3) if j1 is not None else np.nan,
                   cell_N=round(float(g.cy.iloc[j1]), 3) if j1 is not None else np.nan,
                   reg_dE=round(float(r.reg_dE), 3), reg_dN=round(float(r.reg_dN), 3),
                   raw_best_iou=round(float(r.raw_best_iou), 4),
                   iou_prior_only=round(float(iou_prior[n]), 4), module_id_prior_only=j_prior[n],
                   drift_dE=round(float(r.drift_dE), 3), drift_dN=round(float(r.drift_dN), 3))
        recs.append(rec)
    a = pd.DataFrame(recs)
    cols = ["box_id", "module_code", "module_id", "iou", "centroid_offset_m", "status",
            "second_best_module_id", "second_best_iou"]
    a = a[cols + [c for c in a.columns if c not in cols]]
    a.to_csv(OUT / "thermal_box_to_cell.csv", index=False)

    # ---- summaries
    def summ(df, label, key):
        rows = []
        for k, d in df.groupby(key):
            rows.append(dict(group=label, value=k, n=len(d),
                             n_unique=int((d.status == "unique").sum()),
                             n_ambiguous=int((d.status == "ambiguous").sum()),
                             n_unassigned=int((d.status == "unassigned").sum()),
                             iou_median=d.iou.median(), iou_p05=pct(d.iou, 5), iou_p95=pct(d.iou, 95),
                             offset_median_m=d.centroid_offset_m.median(), offset_p95_m=pct(d.centroid_offset_m, 95),
                             along_median_m=d.offset_along_row_m.median(), across_median_m=d.offset_across_row_m.median(),
                             along_sd_m=d.offset_along_row_m.std(), across_sd_m=d.offset_across_row_m.std(),
                             raw_iou_median=d.raw_best_iou.median()))
        return rows
    a["all"] = "all"
    srows = summ(a, "overall", "all") + summ(a, "flight", "flight_id") + summ(a, "block", "block") \
        + summ(a, "class_final", "class_final") + summ(a, "in_eval_zone", "in_eval_zone") + summ(a, "status", "status")
    pd.DataFrame(srows).to_csv(OUT / "association_summary.csv", index=False)

    # adjacent confusion (second best exists and IoU>0.05)
    sec = a[(a.second_best_iou > 0.05)]
    conf = {k: int(v) for k, v in sec.second_best_relation.value_counts().items()}
    conf_amb = {k: int(v) for k, v in a[a.status == "ambiguous"].second_best_relation.value_counts().items()}

    # ---- module code analysis
    info = code_analysis(a, g)
    info["sign_check_mean_fastiou"] = sign_check
    info["status_counts"] = {k: int(v) for k, v in a.status.value_counts().items()}
    info["second_best_relation_all"] = conf
    info["second_best_relation_ambiguous"] = conf_amb
    info["grid_theta_deg"] = float(np.degrees(GRID_THETA))
    info["thresholds"] = THRESH
    (OUT / "code_analysis.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")

    print(json.dumps({k: info[k] for k in ("status_counts", "second_best_relation_all",
                                          "second_best_relation_ambiguous", "sign_check_mean_fastiou")},
                     indent=1))
    print(reg.round(3).to_string())
    print(pd.DataFrame(srows)[lambda d: d.group.isin(["overall", "flight", "block"])].round(3).to_string())

    if not args.no_figures:
        figures(a, g, b, polys, shift)


# ---------------------------------------------------------------------------------- code analysis
def code_analysis(a, g):
    """Tests of how module_code (Dxx-Tyyy) relates to the assigned cell.  Fills code_pairs.csv."""
    from itertools import combinations
    info = {}
    ok = a[a.status == "unique"].copy()
    ok[["Dn", "Tn"]] = ok.module_code.str.extract(r"D(\d+)-T(\d+)").astype(float)
    a2 = a.copy()
    a2[["Dn", "Tn"]] = a2.module_code.str.extract(r"D(\d+)-T(\d+)").astype(float)
    info["n_codes_total"] = int(a.module_code.nunique())
    info["D_range"] = [int(a2.Dn.min()), int(a2.Dn.max())]
    info["T_range"] = [int(a2.Tn.min()), int(a2.Tn.max())]
    info["n_unique_D"] = int(a2.Dn.nunique())
    # repeated codes: same code, different flights -> same cell?
    rep = ok.groupby("module_code").filter(lambda d: d.flight_id.nunique() > 1)
    pairs = []
    for code, d in rep.groupby("module_code"):
        for (_, r1), (_, r2) in combinations(d.iterrows(), 2):
            if r1.flight_id == r2.flight_id:
                continue
            same = r1.module_id == r2.module_id
            dist = float(np.hypot(r1.cell_E - r2.cell_E, r1.cell_N - r2.cell_N))
            pairs.append(dict(module_code=code, flight_1=r1.flight_id, flight_2=r2.flight_id,
                              module_id_1=r1.module_id, module_id_2=r2.module_id, same_cell=bool(same),
                              cell_distance_m=round(dist, 2),
                              dE=round(r2.cell_E - r1.cell_E, 2), dN=round(r2.cell_N - r1.cell_N, 2)))
    pr = pd.DataFrame(pairs)
    pr.to_csv(OUT / "code_pairs.csv", index=False)
    info["repeat_pairs"] = int(len(pr))
    info["repeat_codes"] = int(pr.module_code.nunique()) if len(pr) else 0
    if len(pr):
        info["repeat_same_cell_frac"] = float(pr.same_cell.mean())
        info["repeat_within_1p5m_frac"] = float((pr.cell_distance_m < 1.5).mean())
        info["repeat_cell_distance_median"] = float(pr.cell_distance_m.median())
        info["repeat_cell_distance_p90"] = float(pr.cell_distance_m.quantile(.9))
    # purity: for each D prefix, how concentrated are cells in (block, string, table) ?
    ok["table_key"] = ok.block + "|" + ok.string_id
    ok["tablerow_key"] = ok.table_key + "|" + ok.row_id.astype(str)
    for key in ("block", "table_key", "tablerow_key", "row_id"):
        gb = ok.groupby("Dn")[key].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        w = ok.groupby("Dn").size()
        info[f"D_purity_{key}"] = float((gb * w).sum() / w.sum())
        gb = ok.groupby("Tn")[key].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        w = ok.groupby("Tn").size()
        info[f"T_purity_{key}"] = float((gb * w).sum() / w.sum())
    # D vs spatial location: spread (m) of the cells with same D
    sp = ok.groupby("Dn").agg(n=("cell_E", "size"), sdE=("cell_E", "std"), sdN=("cell_N", "std"),
                             rE=("cell_E", lambda s: s.max() - s.min()), rN=("cell_N", lambda s: s.max() - s.min()))
    info["D_cell_spread_median_m"] = {c: float(sp[c].median()) for c in ("sdE", "sdN", "rE", "rN")}
    info["overall_cell_sd_m"] = [float(ok.cell_E.std()), float(ok.cell_N.std())]
    st = ok.groupby("Tn").agg(n=("cell_E", "size"), sdE=("cell_E", "std"), sdN=("cell_N", "std"))
    info["T_cell_spread_median_m"] = {"sdE": float(st.sdE.median()), "sdN": float(st.sdN.median())}
    # correlations
    for k in ("Tn", "Dn"):
        info[f"corr_{k}_vs"] = {c: float(ok[[k, c]].corr(method="spearman").iloc[0, 1])
                                for c in ("cell_E", "cell_N", "id_pin", "row_id")}
    ok.drop(columns=["all"], errors="ignore").to_csv(OUT / "code_unique_cells.csv", index=False)
    info.update(code_rule_tests(ok, g))
    return info


def code_rule_tests(ok, g):
    """Per-flight tests of the Dxx-Tyyy code: D -> string / row band, T -> position along the row,
    cross-flight consistency on cells hit in two flights, within-flight duplicates."""
    from scipy.stats import spearmanr
    info = {}
    gg = g.copy()
    gg["k"] = gg.groupby(["layer", "String_ID", "Row_ID"]).ID_Pin.rank().astype(int)
    ok = ok.merge(gg[["module_id", "k", "Tam_X", "Tam_Y"]], on="module_id")
    ok["str_key"] = ok.block + "|" + ok.string_id
    # (i) D -> physical string, per flight
    per = {}
    for f, d in ok.groupby("flight_id"):
        pur = d.groupby("Dn").str_key.agg(lambda s: s.value_counts(normalize=True).iloc[0])
        nrows = d.groupby("Dn").row_id.nunique()
        multi = d.groupby("Dn").size()
        # D monotone with N: Spearman of D vs mean N of its cells
        mN = d.groupby("Dn").cell_N.mean()
        rho = spearmanr(mN.index, mN.values)[0]
        # string -> number of D values
        dps = d.groupby("str_key").Dn.nunique()
        # table-level D: > 1 row hit AND T range > 130 (cannot be one 90-module row)
        tr = d.groupby("Dn").Tn.agg(lambda x: x.max() - x.min())
        tab = ((nrows > 1) & (d.groupby("Dn").Tn.max() > 130)).sum()
        per[f] = dict(n_boxes=len(d), n_D=int(d.Dn.nunique()), D_to_single_string_frac=float((pur == 1).mean()),
                      D_to_string_weighted_purity=float((pur * multi).sum() / multi.sum()),
                      spearman_D_vs_meanN=float(rho), median_D_per_string=float(dps.median()),
                      max_D_per_string=int(dps.max()), n_D_spanning_multiple_rows=int((nrows > 1).sum()),
                      n_D_table_level=int(tab))
    info["D_tests_per_flight"] = per
    # (ii) T vs k within (flight, D, row) for groups with >= 4 distinct cells; row-level D
    rows = []
    for (f, D, r, sk), d in ok.groupby(["flight_id", "Dn", "row_id", "str_key"]):
        d = d.drop_duplicates("module_id")
        if len(d) < 4:
            continue
        b, a0 = np.polyfit(d.k, d.Tn, 1)
        res = d.Tn - (a0 + b * d.k)
        rows.append(dict(flight_id=f, D=D, row=r, n=len(d), slope=b, resid_max=float(res.abs().max()),
                         rho=spearmanr(d.k, d.Tn)[0], Tn_minus_k_range=float((d.Tn - d.k).max() - (d.Tn - d.k).min())))
    tk = pd.DataFrame(rows)
    # which of those D groups are single-row D (only one row ever hit) -> clean row-level numbering
    nrowsD = ok.groupby(["flight_id", "Dn"]).row_id.nunique().rename("nrowsD").reset_index()
    tk = tk.merge(nrowsD, left_on=["flight_id", "D"], right_on=["flight_id", "Dn"])
    sing = tk[tk.nrowsD == 1]
    info["T_vs_k_single_row_D"] = dict(n_groups=int(len(sing)), median_slope=float(sing.slope.median()),
                                       slope_within_0p9_1p1=float(((sing.slope > .9) & (sing.slope < 1.1)).mean()),
                                       median_spearman=float(sing.rho.median()),
                                       resid_max_le_2_frac=float((sing.resid_max <= 2).mean()),
                                       median_Tminusk_range=float(sing.Tn_minus_k_range.median()))
    mult = tk[tk.nrowsD > 1]
    info["T_vs_k_multi_row_D"] = dict(n_groups=int(len(mult)), median_slope=float(mult.slope.median()) if len(mult) else None,
                                      slope_3_to_5=float(((mult.slope > 3) & (mult.slope < 5)).mean()) if len(mult) else None)
    # whole-D (all rows together) T ~ 4k for table-level D
    rows = []
    for (f, D), d in ok.groupby(["flight_id", "Dn"]):
        d = d.drop_duplicates("module_id")
        if d.row_id.nunique() >= 3 and len(d) >= 8 and d.Tn.max() > 130:
            b, a0 = np.polyfit(d.k, d.Tn, 1)
            rows.append(dict(flight_id=f, D=D, n=len(d), slope=b, rho=spearmanr(d.k, d.Tn)[0],
                             resid_sd=float((d.Tn - (a0 + b * d.k)).std())))
    tl = pd.DataFrame(rows)
    info["T_vs_k_table_level_D"] = dict(n_groups=int(len(tl)), median_slope=float(tl.slope.median()),
                                        median_spearman=float(tl.rho.median()), median_resid_sd=float(tl.resid_sd.median()))
    tk.to_csv(OUT / "code_T_vs_k_groups.csv", index=False)
    # (iii) cells hit in two different flights: code differences
    x = ok.drop_duplicates(["module_id", "flight_id"])
    m = x.merge(x, on="module_id", suffixes=("1", "2"))
    m = m[m.flight_id1 < m.flight_id2].copy()
    m["dD"] = m.Dn1 - m.Dn2
    m["dT"] = m.Tn1 - m.Tn2
    keep = ["module_id", "flight_id1", "flight_id2", "module_code1", "module_code2", "dD", "dT", "block1", "string_id1", "row_id1", "k1"]
    m = m.merge(gg[["module_id"]].assign(), on="module_id")
    m[["module_id", "flight_id1", "flight_id2", "module_code1", "module_code2", "dD", "dT", "block1", "string_id1", "row_id1"]]         .to_csv(OUT / "code_shared_cells.csv", index=False)
    info["shared_cells"] = dict(n_pairs=int(len(m)), n_cells=int(m.module_id.nunique()),
                                identical_code_frac=float(((m.dD == 0) & (m.dT == 0)).mean()),
                                dD_zero_frac=float((m.dD == 0).mean()),
                                dD_median=float(m.dD.median()), dD_range=[int(m.dD.min()), int(m.dD.max())])
    # dD constant within a string?  fraction of pairs equal to the string's modal dD
    mode = m.groupby(["flight_id1", "flight_id2", "string_id1", "block1"]).dD.agg(lambda s: s.value_counts().index[0])
    mm = m.join(mode.rename("modal"), on=["flight_id1", "flight_id2", "string_id1", "block1"])
    info["shared_cells"]["dD_equals_string_mode_frac"] = float((mm.dD == mm.modal).mean())
    # dT constant within (string,row,dD) up to +-3
    gm = m.groupby(["flight_id1", "flight_id2", "string_id1", "block1", "row_id1", "dD"]).dT
    info["shared_cells"]["dT_range_within_row_median"] = float(gm.agg(lambda s: s.max() - s.min()).median())
    # (iv) same code, different flights vs same flight
    same_flight = ok.groupby(["flight_id", "module_code"]).module_id.nunique()
    info["within_flight_codes_repeated"] = int((ok.groupby(["flight_id", "module_code"]).size() > 1).sum())
    info["within_flight_repeated_same_cell_frac"] = float(
        (same_flight[ok.groupby(["flight_id", "module_code"]).size() > 1] == 1).mean())
    # (v) Spearman T vs Tam_X within D (row-level, single row) and D vs Tam_Y within flight
    r = []
    for (f, D), d in ok.groupby(["flight_id", "Dn"]):
        d = d.drop_duplicates("module_id")
        if len(d) >= 5:
            r.append(spearmanr(d.Tn, d.Tam_X)[0])
    info["spearman_T_vs_TamX_within_D_median"] = float(np.nanmedian(r))
    info["codes_total_rows"] = int(len(ok))
    # (vi) accuracy of the partial rule.  A (flight, D) is "table-level" when any of its T exceeds 110
    # (a 90-module row cannot give that), otherwise "row-level".
    tmax = ok.groupby(["flight_id", "Dn"]).Tn.transform("max")
    ok = ok.assign(level=np.where(tmax > 110, "table", "row"))
    ok = ok.drop_duplicates(["flight_id", "module_id"])
    rl = ok[ok.level == "row"].copy()
    rl["c"] = rl.Tn - rl.k
    info["rule_row_level"] = dict(n_boxes=int(len(rl)), frac_of_boxes=float(len(rl) / len(ok)),
                                  c_median=float(rl.c.median()), c_p05_p95=[float(rl.c.quantile(.05)), float(rl.c.quantile(.95))],
                                  T_eq_k_plus_10_pm3=float(((rl.c - 10).abs() <= 3).mean()),
                                  T_eq_k_plus_group_median_pm2=float(
                                      ((rl.c - rl.groupby(["flight_id", "str_key", "row_id"]).c.transform("median")).abs() <= 2).mean()))
    # D - (4 - row) constant within (flight,string) for row-level boxes
    rl["base"] = rl.Dn - (4 - rl.row_id)
    mode_share = rl.groupby(["flight_id", "str_key"]).base.agg(lambda x: (x == x.value_counts().index[0]).mean())
    n_in = rl.groupby(["flight_id", "str_key"]).size()
    info["rule_row_level"]["D_minus_(4-row)_const_in_string_frac_boxes"] = float((mode_share * n_in).sum() / n_in.sum())
    info["rule_row_level"]["D_minus_(4-row)_within_pm1_of_string_median_frac"] = float(
        ((rl.base - rl.groupby(["flight_id", "str_key"]).base.transform("median")).abs() <= 1).mean())
    tl = ok[ok.level == "table"].copy()
    b, a0 = np.polyfit(tl.k, tl.Tn, 1) if len(tl) > 3 else (np.nan, np.nan)
    info["rule_table_level"] = dict(n_boxes=int(len(tl)), frac_of_boxes=float(len(tl) / len(ok)),
                                    pooled_slope=float(b),
                                    corr_T_vs_4k_plus_rowterm=float(np.corrcoef(tl.Tn, 4 * (tl.k - 1) + (5 - tl.row_id))[0, 1]),
                                    abs_err_median_vs_group_fit=float(np.median(np.abs(
                                        (tl.Tn - 4 * (tl.k - 1) - (5 - tl.row_id)) -
                                        (tl.Tn - 4 * (tl.k - 1) - (5 - tl.row_id)).groupby([tl.flight_id, tl.Dn]).transform("median")))))
    return info


# ----------------------------------------------------------------------------------------- figures
def figures(a, g, b, polys, shift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    from shapely.affinity import translate

    F = OUT / "figures"
    flights = sorted(a.flight_id.unique())
    col = dict(zip(flights, ["#1b6ca8", "#d95f02", "#1b9e77", "#7570b3"]))

    # 1. IoU histogram: raw vs registered, + second best
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(a.raw_best_iou, bins, color="#999", label="raw (no registration)")
    ax[0].hist(a.iou, bins, alpha=.75, color="#1b6ca8", label="registered")
    ax[0].axvline(THRESH["unassigned_iou"], c="r", ls="--", lw=1)
    ax[0].axvline(THRESH["unique_iou"], c="g", ls="--", lw=1)
    ax[0].set_xlabel("best-cell IoU"); ax[0].set_ylabel("boxes"); ax[0].legend()
    ax[1].scatter(a.iou, a.second_best_iou, s=5, c=[col[f] for f in a.flight_id], alpha=.6)
    ax[1].set_xlabel("best IoU"); ax[1].set_ylabel("second-best IoU"); ax[1].plot([0, 1], [0, 1], "k:", lw=.7)
    for f in flights:
        ax[2].hist(a[a.flight_id == f].iou, bins, histtype="step", color=col[f], label=f.split("_", 2)[2], lw=1.5)
    ax[2].legend(); ax[2].set_xlabel("best IoU (registered), per flight")
    fig.tight_layout(); fig.savefig(F / "iou_histogram.png", dpi=130); plt.close(fig)

    # 2. offset map: residual vectors after registration, on the layout; + along/across scatter
    fig, ax = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw=dict(width_ratios=[2.1, 1]))
    ok = a[a.status != "unassigned"]
    ax[0].scatter(g.cx[::40], g.cy[::40], s=.3, c="#ccc")
    q = ax[0].quiver(ok.cell_E, ok.cell_N, ok.offset_E_m, ok.offset_N_m, ok.centroid_offset_m,
                     angles="xy", scale_units="xy", scale=0.02, cmap="viridis", width=.002)
    plt.colorbar(q, ax=ax[0], label="residual centroid offset (m)", shrink=.6)
    ax[0].set_aspect("equal"); ax[0].set_title("Residual offset box - cell after registration (arrow length = offset x 50)")
    ax[0].set_xlabel("E"); ax[0].set_ylabel("N")
    for f in flights:
        d = ok[ok.flight_id == f]
        ax[1].scatter(d.offset_along_row_m, d.offset_across_row_m, s=6, color=col[f], alpha=.5, label=f.split("_", 2)[2])
    ax[1].axhline(0, c="k", lw=.5); ax[1].axvline(0, c="k", lw=.5)
    ax[1].set_xlabel("offset along rows (m)"); ax[1].set_ylabel("offset across rows (m)"); ax[1].legend()
    ax[1].set_aspect("equal"); ax[1].set_title("residual after registration")
    fig.tight_layout(); fig.savefig(F / "offset_map.png", dpi=130); plt.close(fig)

    # 3. registration shift map (what had to be applied) + raw offset of nearest cell
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for f in flights:
        d = a[a.flight_id == f]
        ax[0].scatter(d.cell_E, d.cell_N, s=7, c=np.hypot(d.reg_dE, d.reg_dN), cmap="viridis", vmin=0, vmax=6)
    ax[0].set_aspect("equal"); ax[0].set_title("magnitude of registration shift applied (m)")
    for f in flights:
        d = a[a.flight_id == f]
        ax[1].scatter(d.reg_dE, d.reg_dN, s=6, color=col[f], label=f.split("_", 2)[2], alpha=.6)
    ax[1].set_xlabel("shift E (m)"); ax[1].set_ylabel("shift N (m)"); ax[1].legend(); ax[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig(F / "registration_shift_map.png", dpi=130); plt.close(fig)

    # 4. zoom: boxes over cells (raw and registered) around the best-populated spot
    ok = a[a.status == "unique"]
    cE, cN = ok.cell_E.median(), ok.cell_N.median()
    dd = np.hypot(ok.cell_E - cE, ok.cell_N - cN)
    k = ok.index[dd.argmin()]
    cE, cN = a.cell_E[k], a.cell_N[k]
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    sel = g[(abs(g.cx - cE) < 14) & (abs(g.cy - cN) < 7)]
    for t, axx in zip(("raw box positions (no registration)", "registered boxes"), ax):
        for r in sel.itertuples():
            axx.add_patch(MPoly(np.array(r.geometry.exterior.coords), fc="none", ec="#999", lw=.5))
        axx.set_xlim(cE - 14, cE + 14); axx.set_ylim(cN - 7, cN + 7); axx.set_aspect("equal"); axx.set_title(t)
    stc = {"unique": "#d62728", "ambiguous": "#ff7f0e", "unassigned": "#9467bd"}
    for n, p in enumerate(polys):
        s = shift[n]
        for axx, pp in ((ax[0], p), (ax[1], translate(p, s[0], s[1]))):
            c = pp.centroid
            if abs(c.x - cE) < 14 and abs(c.y - cN) < 7:
                axx.add_patch(MPoly(np.array(pp.exterior.coords), fc="none",
                                    ec=stc[a.status[n]] if axx is ax[1] else "#1b6ca8", lw=1.4))
    fig.tight_layout(); fig.savefig(F / "zoom_boxes_over_cells.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
