#!/usr/bin/env python
"""Inter-annotator agreement metrics for YOLO-OBB thermal labels (A1 = Chi, A2, A3).

Implements .local/annotation_agreement/AGREEMENT_PLAN.md.  Inputs
  --ann-dir      agreement folder (default <project>/.local/annotation_agreement) with _key/sample_key.csv
                 and _key/chi_labels/<frame_uid>.txt (annotator A1)
  --labels-a2/a3 folders with the returned label files named like the package (0001.txt ...)
                 (default <ann-dir>/returned/A2/labels, .../A3/labels)
Outputs (CSV tables + summary.md) go to <ann-dir>/results/ (or --out).

  --simulate     build SIMULATED annotators A2/A3 by perturbing A1 labels and run everything into
                 results/_simulation_test/ (software test only; the numbers are NOT findings).

Geometry is evaluated in pixel space (640x512) with shapely polygons; box matching is the Hungarian
assignment on IoU restricted to pairs with IoU >= threshold.  Bootstrap CIs resample frames.
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

W, H = 640, 512
CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]
NONE = 4
ANN = ["A1", "A2", "A3"]
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIRN = [f"{ANN[a]}-{ANN[b]}" for a, b in PAIRS]
THRS = [0.3, 0.5, 0.7]
PRIMARY = 0.5
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAN = float("nan")


# ------------------------------------------------------------------ reading / geometry
def read_labels(path, errors, tag):
    if not os.path.exists(path):
        return None
    boxes = []
    for i, ln in enumerate(open(path, encoding="utf8").read().splitlines(), 1):
        s = ln.split()
        if not s:
            continue
        try:
            if len(s) != 9:
                raise ValueError("need 9 fields")
            c = int(float(s[0]))
            v = np.array(s[1:], float)
            if c not in range(4):
                raise ValueError("class id outside 0-3")
            if v.min() < -1e-3 or v.max() > 1 + 1e-3:
                raise ValueError("coordinates outside [0,1]")
            pts = np.clip(v.reshape(4, 2), 0, 1) * [W, H]
            if Polygon(pts).convex_hull.area < 1.0:
                raise ValueError("degenerate box (<1 px2)")
        except ValueError as e:
            errors.append((tag, i, str(e)))
            continue
        boxes.append((c, pts))
    return boxes


def poly(p):
    g = Polygon(p)
    return g if g.is_valid else g.convex_hull


def long_axis_deg(p):
    c = np.cov((p - p.mean(0)).T, bias=True)
    w, v = np.linalg.eigh(c)
    vx, vy = v[:, 1]
    return math.degrees(math.atan2(vy, vx)) % 180.0


def angdiff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def iou_matrix(pa, pb):
    M = np.zeros((len(pa), len(pb)))
    for i, g in enumerate(pa):
        for j, h in enumerate(pb):
            if g.intersects(h):
                u = g.union(h).area
                M[i, j] = g.intersection(h).area / u if u > 0 else 0.0
    return M


def match(M, thr):
    if M.size == 0:
        return np.zeros((0, 2), int)
    S = np.where(M >= thr, M, 0.0)
    r, c = linear_sum_assignment(-S)
    k = S[r, c] > 0
    return np.stack([r[k], c[k]], 1)


# ------------------------------------------------------------------ per-frame records
def build_rec(boxes3):
    polys = [[poly(p) for _, p in bl] for bl in boxes3]
    cls = [np.array([c for c, _ in bl], int) for bl in boxes3]
    cen = [np.array([g.centroid.coords[0] for g in pl]).reshape(-1, 2) for pl in polys]
    ang = [np.array([long_axis_deg(p) for _, p in bl]) for bl in boxes3]
    iou = {pi: iou_matrix(polys[a], polys[b]) for pi, (a, b) in enumerate(PAIRS)}
    rec = {"cls": cls, "cen": cen, "ang": ang, "polys": polys, "pts": [[p for _, p in bl] for bl in boxes3],
           "iou": iou, "t": {}}
    cnt = np.zeros((3, 5), int)
    for a in range(3):
        for c in range(4):
            cnt[a, c] = (cls[a] == c).sum()
        cnt[a, 4] = len(cls[a])
    rec["count"] = cnt
    for t in THRS:
        rec["t"][t] = per_thr(rec, t)
    rec["cons"] = consensus(rec, rec["t"][PRIMARY]["comps"])
    return rec


def components(rec, thr):
    """union-find over boxes (annotator, index) linked by pairwise matches."""
    par = {}

    def f(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a in range(3):
        for i in range(len(rec["cls"][a])):
            f((a, i))
    mm = {}
    for pi, (a, b) in enumerate(PAIRS):
        m = match(rec["iou"][pi], thr)
        mm[pi] = m
        for i, j in m:
            par[f((a, int(i)))] = f((b, int(j)))
    comp = {}
    for x in list(par):
        comp.setdefault(f(x), []).append(x)
    return list(comp.values()), mm


def per_thr(rec, t):
    comps, mm = components(rec, t)
    d = {"pair": [], "comps": comps}
    for pi, (a, b) in enumerate(PAIRS):
        m = mm[pi]
        M = rec["iou"][pi]
        ca, cb = rec["cls"][a], rec["cls"][b]
        x = {"nA": len(ca), "nB": len(cb), "tp": len(m),
             "ca": ca[m[:, 0]] if len(m) else np.zeros(0, int), "cb": cb[m[:, 1]] if len(m) else np.zeros(0, int),
             "iou": M[m[:, 0], m[:, 1]] if len(m) else np.zeros(0)}
        x["dc"] = np.linalg.norm(rec["cen"][a][m[:, 0]] - rec["cen"][b][m[:, 1]], axis=1) if len(m) else np.zeros(0)
        x["da"] = np.array([angdiff(rec["ang"][a][i], rec["ang"][b][j]) for i, j in m]) if len(m) else np.zeros(0)
        ctp = np.zeros(4); cA = np.zeros(4); cB = np.zeros(4)
        for c in range(4):
            ia, ib = np.where(ca == c)[0], np.where(cb == c)[0]
            cA[c], cB[c] = len(ia), len(ib)
            if len(ia) and len(ib):
                ctp[c] = len(match(M[np.ix_(ia, ib)], t))
        x["ctp"], x["cA"], x["cB"] = ctp, cA, cB
        ua = np.full(len(ca), NONE); ub = np.full(len(cb), NONE)
        rows = [(ca[i], cb[j]) for i, j in m]
        mi = set(int(i) for i in m[:, 0]) if len(m) else set()
        mj = set(int(j) for j in m[:, 1]) if len(m) else set()
        rows += [(ca[i], NONE) for i in range(len(ca)) if i not in mi]
        rows += [(NONE, cb[j]) for j in range(len(cb)) if j not in mj]
        x["u2"] = np.array(rows, int).reshape(-1, 2)
        d["pair"].append(x)
    u3, conflicts = [], 0
    for comp in comps:
        row = [NONE] * 3
        for a in range(3):
            mem = [i for (aa, i) in comp if aa == a]
            if len(mem) > 1:
                conflicts += 1
            if mem:
                row[a] = int(rec["cls"][a][sorted(mem)[0]])
        u3.append(row)
    d["u3"] = np.array(u3, int).reshape(-1, 3)
    d["conflicts"] = conflicts
    return d


def consensus(rec, comps):
    """majority-of-3 consensus at PRIMARY IoU; returns boxes and per-annotator comparison counts."""
    boxes = []  # (pts, cls or -1 for tie, support)
    minority = 0
    for comp in comps:
        mem = {}
        for a, i in comp:
            mem.setdefault(a, i)
        sup = len(mem)
        if sup < 2:
            minority += 1
            continue
        cl = [int(rec["cls"][a][i]) for a, i in mem.items()]
        vals, cnts = np.unique(cl, return_counts=True)
        top = cnts.max()
        winners = vals[cnts == top]
        tie = top < 2 or len(winners) > 1
        # geometry: member with highest summed IoU to the others (medoid); first member if 2
        keys = list(mem.items())
        best, bs = keys[0], -1
        if len(keys) > 2:
            for a, i in keys:
                s = 0.0
                for b, j in keys:
                    if b != a:
                        g, h = rec["polys"][a][i], rec["polys"][b][j]
                        s += g.intersection(h).area / g.union(h).area
                if s > bs:
                    best, bs = (a, i), s
        pts = rec["pts"][best[0]][best[1]]
        boxes.append((pts, -1 if tie else int(winners[0]), sup))
    cpoly = [poly(b[0]) for b in boxes]
    ccl = np.array([b[1] for b in boxes], int)
    out = {"nC": len(boxes), "ties": int((ccl < 0).sum()), "minority": minority, "boxes": boxes, "ann": []}
    for a in range(3):
        M = iou_matrix(rec["polys"][a], cpoly)
        m = match(M, PRIMARY)
        ca = rec["cls"][a]
        tpc = int(sum(1 for i, j in m if ccl[j] >= 0 and ccl[j] == ca[i]))
        out["ann"].append({"nA": len(ca), "tp": len(m), "tpc": tpc})
    return out


# ------------------------------------------------------------------ statistics
def cohen_stats(x, y, Q):
    n = len(x)
    if n == 0:
        return dict(n=0, po=NAN, kappa=NAN, ac1=NAN, pabak=NAN)
    conf = np.zeros((Q, Q))
    np.add.at(conf, (x, y), 1)
    po = np.trace(conf) / n
    pa, pb = conf.sum(1) / n, conf.sum(0) / n
    pe = (pa * pb).sum()
    kap = (po - pe) / (1 - pe) if pe < 1 else NAN
    pi = (pa + pb) / 2
    pe1 = (pi * (1 - pi)).sum() / (Q - 1)
    ac1 = (po - pe1) / (1 - pe1) if pe1 < 1 else NAN
    return dict(n=n, po=po, kappa=kap, ac1=ac1, pabak=(Q * po - 1) / (Q - 1))


def fleiss_stats(U, Q):
    k, r = U.shape
    if k == 0:
        return dict(n=0, po=NAN, kappa=NAN, ac1=NAN, pabak=NAN)
    nij = np.stack([(U == q).sum(1) for q in range(Q)], 1).astype(float)
    po = (nij * (nij - 1)).sum(1).mean() / (r * (r - 1))
    pj = nij.sum(0) / (k * r)
    pe = (pj ** 2).sum()
    kap = (po - pe) / (1 - pe) if pe < 1 else NAN
    pe1 = (pj * (1 - pj)).sum() / (Q - 1)
    ac1 = (po - pe1) / (1 - pe1) if pe1 < 1 else NAN
    return dict(n=k, po=po, kappa=kap, ac1=ac1, pabak=(Q * po - 1) / (Q - 1))


def icc_stats(X):
    """two-way random, absolute agreement: ICC(A,1) and ICC(A,k).  X: n x k"""
    n, k = X.shape
    if n < 2:
        return NAN, NAN
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    d1 = msr + (k - 1) * mse + k * (msc - mse) / n
    dk = msr + (msc - mse) / n
    return ((msr - mse) / d1 if d1 > 0 else NAN), ((msr - mse) / dk if dk > 0 else NAN)


def prf(tp, nA, nB):
    p = tp / nB if nB else NAN
    r = tp / nA if nA else NAN
    f = 2 * tp / (nA + nB) if (nA + nB) else NAN
    return p, r, f


def compute(R):
    out = {}

    def put(tab, thr, subj, cat, metric, val):
        out[(tab, thr, subj, cat, metric)] = float(val)

    def putd(tab, thr, subj, cat, d):
        for k, v in d.items():
            put(tab, thr, subj, cat, k, v)
    for t in THRS:
        T = [r["t"][t] for r in R]
        for pi, pn in enumerate(PAIRN):
            pp = [x["pair"][pi] for x in T]
            nA, nB, tp = (sum(x[k] for x in pp) for k in ("nA", "nB", "tp"))
            ctp, cA, cB = (sum(x[k] for x in pp) for k in ("ctp", "cA", "cB"))
            for cat, (a_, b_, c_) in [("any_class", (tp, nA, nB))] + \
                    [(CLASSES[c], (ctp[c], cA[c], cB[c])) for c in range(4)] + \
                    [("class_aware_all", (ctp.sum(), nA, nB))]:
                p, r, f = prf(a_, b_, c_)
                putd("detection", t, pn, cat, dict(n_ref=b_, n_pred=c_, tp=a_, precision=p, recall=r, f1=f))
            ca = np.concatenate([x["ca"] for x in pp]); cb = np.concatenate([x["cb"] for x in pp])
            putd("class_agreement", t, pn, "matched_only|all", cohen_stats(ca, cb, 4))
            u2 = np.concatenate([x["u2"] for x in pp])
            putd("class_agreement", t, pn, "with_none|all", cohen_stats(u2[:, 0], u2[:, 1], 5))
            for c in range(4):
                putd("class_agreement", t, pn, f"with_none|{CLASSES[c]}",
                     cohen_stats((u2[:, 0] == c).astype(int), (u2[:, 1] == c).astype(int), 2))
            iou = np.concatenate([x["iou"] for x in pp]); dc = np.concatenate([x["dc"] for x in pp])
            da = np.concatenate([x["da"] for x in pp])
            putd("localisation", t, pn, "matched", dict(
                n_matched=len(iou), mean_iou=iou.mean() if len(iou) else NAN,
                mean_centre_px=dc.mean() if len(dc) else NAN, median_centre_px=np.median(dc) if len(dc) else NAN,
                mean_angle_deg=da.mean() if len(da) else NAN))
        U = np.concatenate([x["u3"] for x in T])
        full = U[(U < NONE).all(1)]
        putd("class_agreement", t, "fleiss3", "matched_only|all", fleiss_stats(full, 4))
        putd("class_agreement", t, "fleiss3", "with_none|all", fleiss_stats(U, 5))
        for c in range(4):
            putd("class_agreement", t, "fleiss3", f"with_none|{CLASSES[c]}", fleiss_stats((U == c).astype(int), 2))
        put("class_agreement", t, "fleiss3", "units", "n_conflicts", sum(x["conflicts"] for x in T))
    # frame level + ICC
    C = np.stack([r["count"] for r in R]) if R else np.zeros((0, 3, 5), int)
    names = CLASSES + ["any_defect"]
    for k in range(5):
        pres = (C[:, :, k] > 0).astype(int)
        put("frame_level", "-", "all", names[k], "prevalence", pres.mean() if len(pres) else NAN)
        for pi, (a, b) in enumerate(PAIRS):
            putd("frame_level", "-", PAIRN[pi], names[k], cohen_stats(pres[:, a], pres[:, b], 2))
        putd("frame_level", "-", "fleiss3", names[k], fleiss_stats(pres, 2))
        i1, ik = icc_stats(C[:, :, k].astype(float)) if len(C) > 1 else (NAN, NAN)
        put("icc_n_boxes", "-", "all3", names[k] if k < 4 else "total", "icc_a1", i1)
        put("icc_n_boxes", "-", "all3", names[k] if k < 4 else "total", "icc_ak", ik)
        put("icc_n_boxes", "-", "all3", names[k] if k < 4 else "total", "mean_boxes_per_frame",
            C[:, :, k].mean() if len(C) else NAN)
    # consensus
    K = [r["cons"] for r in R]
    put("consensus", PRIMARY, "all", "consensus_boxes", "n_consensus", sum(c["nC"] for c in K))
    put("consensus", PRIMARY, "all", "consensus_boxes", "n_class_ties", sum(c["ties"] for c in K))
    put("consensus", PRIMARY, "all", "consensus_boxes", "n_minority_single_support", sum(c["minority"] for c in K))
    for a in range(3):
        nA = sum(c["ann"][a]["nA"] for c in K); tp = sum(c["ann"][a]["tp"] for c in K)
        tpc = sum(c["ann"][a]["tpc"] for c in K); nC = sum(c["nC"] for c in K)
        p, r, f = prf(tp, nC, nA)
        pc, rc, fc = prf(tpc, nC, nA)
        putd("consensus", PRIMARY, ANN[a], "vs_consensus", dict(
            n_annotator=nA, n_consensus=nC, tp=tp, precision=p, recall=r, f1=f, f1_class_aware=fc))
    # mean over pairs
    grp = {}
    for (tab, thr, subj, cat, met), v in out.items():
        if subj in PAIRN and met in ("f1", "precision", "recall", "kappa", "po", "ac1", "pabak", "mean_iou",
                                     "mean_centre_px", "mean_angle_deg"):
            grp.setdefault((tab, thr, cat, met), []).append(v)
    for (tab, thr, cat, met), vs in grp.items():
        with np.errstate(all="ignore"):
            out[(tab, thr, "mean_pairs", cat, met)] = float(np.nanmean(vs)) if not np.all(np.isnan(vs)) else NAN
    return out


def run_bootstrap(recs, scopes, B, seed):
    res = {}
    for si, (sc, idx) in enumerate(scopes.items()):
        R = [recs[i] for i in idx]
        full = compute(R)
        rng = np.random.default_rng([seed, si])
        store = {k: np.empty(B) for k in full}
        n = len(R)
        for b in range(B):
            o = compute([R[i] for i in rng.integers(0, n, n)])
            for k in full:
                store[k][b] = o.get(k, NAN)
        for k, v in full.items():
            s = store[k][~np.isnan(store[k])]
            lo, hi = (np.percentile(s, [2.5, 97.5]) if len(s) >= max(20, B // 10) else (NAN, NAN))
            res[(sc,) + k] = (v, lo, hi, len(s))
    return res


# ------------------------------------------------------------------ output tables
def to_tables(res, out):
    rows = [dict(scope=k[0], table=k[1], thr=k[2], subject=k[3], category=k[4], metric=k[5], value=v[0],
                 ci_lo=v[1], ci_hi=v[2], n_boot_valid=v[3]) for k, v in res.items()]
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/all_metrics_long.csv", index=False)
    for tab, g in df.groupby("table"):
        idx = ["scope", "thr", "subject", "category"]
        w = g.pivot_table(index=idx, columns="metric", values="value", aggfunc="first")
        lo = g.pivot_table(index=idx, columns="metric", values="ci_lo", aggfunc="first").add_suffix("_ci_lo")
        hi = g.pivot_table(index=idx, columns="metric", values="ci_hi", aggfunc="first").add_suffix("_ci_hi")
        wide = pd.concat([w, lo, hi], axis=1)
        order = []
        for m in w.columns:
            order += [m, m + "_ci_lo", m + "_ci_hi"]
        wide[order].reset_index().to_csv(f"{out}/{tab}.csv", index=False, float_format="%.4f")
    return df


def acceptance(df, key, thr=PRIMARY, level=0.6):
    def get(scope, table, thr_, subj, cat, met):
        q = df[(df.scope == scope) & (df.table == table) & (df.thr.astype(str) == str(thr_)) & (df.subject == subj)
               & (df.category == cat) & (df.metric == met)]
        return (q.value.iloc[0], q.ci_lo.iloc[0], q.ci_hi.iloc[0]) if len(q) else (NAN, NAN, NAN)
    rows = []

    def add(level_, scope, cls, measure, vals):
        v, lo, hi = vals
        absent = cls in CLASSES and (key["n_" + cls] if scope == "ALL" else key.loc[key.farm == scope, "n_" + cls]).sum() == 0
        st = "NOT_EVALUABLE_no_A1_box_in_sample" if absent else "UNDEFINED" if np.isnan(v) else ("PASS" if lo > level else ("PASS_point_only" if v > level else "BELOW"))
        rows.append(dict(level=level_, scope=scope, class_or_all=cls, measure=measure, kappa=v, ci_lo=lo, ci_hi=hi,
                         status=st))
    for sc in ["ALL", "TN19", "TT", "BIM2", "TNDL"]:
        lv = "overall" if sc == "ALL" else "farm"
        add(lv, sc, "all", "box class kappa, Fleiss, matched boxes only", get(sc, "class_agreement", thr, "fleiss3", "matched_only|all", "kappa"))
        add(lv, sc, "all", "box class kappa, Fleiss, with no-match category", get(sc, "class_agreement", thr, "fleiss3", "with_none|all", "kappa"))
        add(lv, sc, "any_defect", "frame presence kappa, Fleiss", get(sc, "frame_level", "-", "fleiss3", "any_defect", "kappa"))
        for c in CLASSES:
            lv2 = "class" if sc == "ALL" else "farm x class"
            add(lv2, sc, c, "box one-vs-rest kappa (with no-match), Fleiss", get(sc, "class_agreement", thr, "fleiss3", f"with_none|{c}", "kappa"))
            add(lv2, sc, c, "frame presence kappa, Fleiss", get(sc, "frame_level", "-", "fleiss3", c, "kappa"))
    return pd.DataFrame(rows)


def write_summary(df, acc, out, meta, banner):
    def g(scope, table, thr, subj, cat, met):
        q = df[(df.scope == scope) & (df.table == table) & (df.thr.astype(str) == str(thr)) & (df.subject == subj)
               & (df.category == cat) & (df.metric == met)]
        return q.iloc[0] if len(q) else None

    def fmt(r):
        return "n/a" if r is None or np.isnan(r.value) else f"{r.value:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
    L = [banner, "", "# Agreement summary", "",
         f"frames {meta['n_frames']}; boxes per annotator A1/A2/A3 = {meta['n_boxes']}; bootstrap {meta['boot']} resamples (frames), seed {meta['seed']}; "
         f"primary IoU threshold {PRIMARY}.", ""]
    if meta["errors"]:
        L += [f"WARNING: {meta['errors']} malformed label lines were ignored (see format_errors.csv).", ""]
    L += ["## Detection F1 (class-agnostic, IoU>=0.5), ALL scope", "", "| pair | precision (2nd vs 1st) | recall | F1 |", "|---|---|---|---|"]
    for pn in PAIRN + ["mean_pairs"]:
        L.append(f"| {pn} | {fmt(g('ALL','detection',PRIMARY,pn,'any_class','precision'))} | {fmt(g('ALL','detection',PRIMARY,pn,'any_class','recall'))} | {fmt(g('ALL','detection',PRIMARY,pn,'any_class','f1'))} |")
    L += ["", "## Class agreement on boxes (IoU>=0.5), ALL scope", "", "| rater set | variant | kappa | PABAK | AC1 | % agreement |", "|---|---|---|---|---|---|"]
    for sj in PAIRN + ["fleiss3"]:
        for v in ("matched_only|all", "with_none|all"):
            L.append(f"| {sj} | {v} | {fmt(g('ALL','class_agreement',PRIMARY,sj,v,'kappa'))} | {fmt(g('ALL','class_agreement',PRIMARY,sj,v,'pabak'))} | {fmt(g('ALL','class_agreement',PRIMARY,sj,v,'ac1'))} | {fmt(g('ALL','class_agreement',PRIMARY,sj,v,'po'))} |")
    L += ["", "## Acceptance (kappa > 0.6; PASS = CI lower bound > 0.6, PASS_point_only = point estimate only)", "",
          "| level | scope | class | measure | kappa [95% CI] | status |", "|---|---|---|---|---|---|"]
    for r in acc.itertuples():
        k = "n/a" if np.isnan(r.kappa) else f"{r.kappa:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
        L.append(f"| {r.level} | {r.scope} | {r.class_or_all} | {r.measure} | {k} | {r.status} |")
    L += ["", "Cells with few frames or near-zero prevalence give wide CIs or UNDEFINED kappa; use AC1/PABAK in the CSV tables there.",
          "Below-threshold classes: report them; optional consensus meeting + clearly marked second pass; released labels are not silently changed.", ""]
    open(f"{out}/summary.md", "w", encoding="utf8").write("\n".join(L))


# ------------------------------------------------------------------ simulation
def simulate(key, chi_dir, out, seed, rates):
    rng = np.random.default_rng(seed)
    truth = {}
    for ai, ann in enumerate(("A2", "A3")):
        d = f"{out}/sim_{ann}"
        os.makedirs(d, exist_ok=True)
        tr = dict(n_ref=0, n_miss=0, n_flip=0, n_kept=0, n_extra=0)
        for r in key.itertuples():
            errs = []
            bl = read_labels(f"{chi_dir}/{r.frame_uid}.txt", errs, "chi") or []
            lines = []
            for c, p in bl:
                tr["n_ref"] += 1
                if rng.random() < rates["miss"]:
                    tr["n_miss"] += 1
                    continue
                tr["n_kept"] += 1
                cen = p.mean(0)
                th = math.radians(rng.normal(0, rates["rot_deg"]))
                R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                s = math.exp(rng.normal(0, rates["scale"]))
                q = (p - cen) @ R.T * s + cen + rng.normal(0, rates["shift_px"], 2)
                cc = c
                if rng.random() < rates["flip"]:
                    cc = int(rng.choice([k for k in range(4) if k != c]))
                    tr["n_flip"] += 1
                lines.append((cc, q))
            lam = rates["extra_per_box"] * len(bl) + (rates["extra_neg_frame"] if len(bl) == 0 else 0)
            for _ in range(rng.poisson(lam)):
                tr["n_extra"] += 1
                w, h = rng.normal(65, 5), rng.normal(32, 3)
                cx, cy = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
                base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
                th = math.radians(rng.choice([0, 90]) + rng.normal(0, 3))
                R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                lines.append((int(rng.integers(0, 4)), base @ R.T + [cx, cy]))
            name = getattr(r, f"name_{ann}")
            with open(f"{d}/{name}.txt", "w") as fh:
                for c, q in lines:
                    v = np.clip(q / [W, H], 0, 1).reshape(-1)
                    fh.write(f"{c} " + " ".join(f"{x:.6f}" for x in v) + "\n")
        truth[ann] = tr
    json.dump(dict(rates=rates, truth=truth), open(f"{out}/sim_truth.json", "w"), indent=1)
    return truth


def sim_check(df, key, chi_dir, out, truth, rates):
    def val(table, subj, cat, met, thr=PRIMARY):
        q = df[(df.scope == "ALL") & (df.table == table) & (df.thr.astype(str) == str(thr)) & (df.subject == subj)
               & (df.category == cat) & (df.metric == met)]
        return float(q.value.iloc[0])
    # Chi class marginals
    cl = []
    for r in key.itertuples():
        cl += [c for c, _ in (read_labels(f"{chi_dir}/{r.frame_uid}.txt", [], "chi") or [])]
    p = np.bincount(cl, minlength=4) / len(cl)
    f = rates["flip"]
    q = (1 - f) * p + f * (1 - p) / 3
    po12 = 1 - f
    pe12 = (p * q).sum()
    k12 = (po12 - pe12) / (1 - pe12)
    po23 = (1 - f) ** 2 + f ** 2 / 3
    rows = []
    for ann, pn in (("A2", "A1-A2"), ("A3", "A1-A3")):
        t = truth[ann]
        rec_exp = t["n_kept"] / t["n_ref"]
        prec_exp = t["n_kept"] / (t["n_kept"] + t["n_extra"])
        rows += [(f"{pn} recall (=1-miss)", rec_exp, val("detection", pn, "any_class", "recall"), 0.04),
                 (f"{pn} precision", prec_exp, val("detection", pn, "any_class", "precision"), 0.04),
                 (f"{pn} class % agreement (=1-flip)", 1 - t["n_flip"] / t["n_kept"], val("class_agreement", pn, "matched_only|all", "po"), 0.03),
                 (f"{pn} class Cohen kappa (analytic)", k12, val("class_agreement", pn, "matched_only|all", "kappa"), 0.05),
                 (f"{pn} mean centre offset px", rates["shift_px"] * math.sqrt(math.pi / 2), val("localisation", pn, "matched", "mean_centre_px"), 0.25),
                 (f"{pn} mean angle diff deg", rates["rot_deg"] * math.sqrt(2 / math.pi), val("localisation", pn, "matched", "mean_angle_deg"), 0.25)]
    rows += [("A2-A3 class % agreement (analytic)", po23, val("class_agreement", "A2-A3", "matched_only|all", "po"), 0.04),
             ("A2-A3 mean centre offset px", rates["shift_px"] * math.sqrt(math.pi), val("localisation", "A2-A3", "matched", "mean_centre_px"), 0.35),
             ("A2-A3 mean angle diff deg", rates["rot_deg"] * math.sqrt(4 / math.pi), val("localisation", "A2-A3", "matched", "mean_angle_deg"), 0.35)]
    # independent frame-level any-defect check (A1 vs A2) from raw files
    a1, a2 = [], []
    for r in key.itertuples():
        a1.append(len(read_labels(f"{chi_dir}/{r.frame_uid}.txt", [], "x") or []) > 0)
        a2.append(len(read_labels(f"{out}/sim_A2/{r.name_A2}.txt", [], "x") or []) > 0)
    a1, a2 = np.array(a1), np.array(a2)
    po = (a1 == a2).mean(); pe = a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean())
    rows.append(("A1-A2 any-defect frame kappa (independent recomputation)", (po - pe) / (1 - pe),
                 val("frame_level", "A1-A2", "any_defect", "kappa", "-"), 0.001))
    L = ["# SIMULATION, NOT RESULTS", "",
         "Software test of agreement_metrics.py on annotators A2/A3 simulated by perturbing A1 labels. These numbers say nothing about real annotators.", "",
         f"perturbation rates: {json.dumps(rates)}", "", f"known counts: {json.dumps(truth)}", "",
         "| quantity | expected | recovered | tolerance | ok |", "|---|---|---|---|---|"]
    allok = True
    for n, e, g_, tol in rows:
        ok = abs(e - g_) <= (tol * abs(e) if tol in (0.25, 0.35) else tol)
        allok &= ok
        L.append(f"| {n} | {e:.4f} | {g_:.4f} | {tol} | {'yes' if ok else 'NO'} |")
    L += ["", f"overall: {'all recovered within tolerance' if allok else 'SOME CHECKS FAILED'}"]
    open(f"{out}/simulation_check.md", "w", encoding="utf8").write("\n".join(L))
    return allok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default=f"{ROOT}/.local/annotation_agreement")
    ap.add_argument("--labels-a2"); ap.add_argument("--labels-a3")
    ap.add_argument("--out"); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="treat missing label files as empty")
    a = ap.parse_args()
    key = pd.read_csv(f"{a.ann_dir}/_key/sample_key.csv", dtype={"name_A2": str, "name_A3": str})
    chi = f"{a.ann_dir}/_key/chi_labels"
    banner = "REAL DATA"
    if a.simulate:
        out = a.out or f"{a.ann_dir}/results/_simulation_test"
        os.makedirs(out, exist_ok=True)
        rates = dict(miss=0.10, flip=0.10, extra_per_box=0.05, extra_neg_frame=0.10, shift_px=1.5, rot_deg=1.5, scale=0.03)
        truth = simulate(key, chi, out, a.seed, rates)
        l2, l3 = f"{out}/sim_A2", f"{out}/sim_A3"
        banner = "# SIMULATION, NOT RESULTS (simulated annotators; software test only)"
    else:
        out = a.out or f"{a.ann_dir}/results"
        os.makedirs(out, exist_ok=True)
        l2 = a.labels_a2 or f"{a.ann_dir}/returned/A2/labels"
        l3 = a.labels_a3 or f"{a.ann_dir}/returned/A3/labels"
        banner = "Real annotation results"
    errors, recs, miss = [], [], []
    for r in key.itertuples():
        b3 = []
        for tag, path in (("A1", f"{chi}/{r.frame_uid}.txt"), ("A2", f"{l2}/{r.name_A2}.txt"), ("A3", f"{l3}/{r.name_A3}.txt")):
            bl = read_labels(path, errors, f"{tag}:{r.frame_uid}")
            if bl is None:
                miss.append((tag, r.frame_uid, path))
                bl = []
            b3.append(bl)
        recs.append(build_rec(b3))
    if miss and not a.allow_missing:
        sys.exit(f"{len(miss)} label files missing (first: {miss[0]}); use --allow-missing to treat as empty")
    pd.DataFrame(errors, columns=["annotator:frame", "line", "error"]).to_csv(f"{out}/format_errors.csv", index=False)
    scopes = {"ALL": list(range(len(key)))}
    for f in ["TN19", "TT", "BIM2", "TNDL"]:
        scopes[f] = list(np.where(key.farm == f)[0])
    res = run_bootstrap(recs, scopes, a.boot, a.seed)
    df = to_tables(res, out)
    acc = acceptance(df, key)
    acc.to_csv(f"{out}/acceptance.csv", index=False, float_format="%.4f")
    # consensus outputs and disagreement list
    os.makedirs(f"{out}/consensus_labels", exist_ok=True)
    cb, dis = [], []
    for r, rec in zip(key.itertuples(), recs):
        lines = []
        for pts, c, sup in rec["cons"]["boxes"]:
            v = (pts / [W, H]).reshape(-1)
            cb.append(dict(frame_uid=r.frame_uid, cls=c, tie=int(c < 0), support=sup, **{f"v{i}": x for i, x in enumerate(v)}))
            if c >= 0:
                lines.append(f"{c} " + " ".join(f"{x:.6f}" for x in v))
        open(f"{out}/consensus_labels/{r.frame_uid}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
        for row in rec["t"][PRIMARY]["u3"]:
            if len(set(row)) > 1:
                dis.append(dict(frame_uid=r.frame_uid, farm=r.farm, flight_id=r.flight_id, **{ANN[i]: ("none" if row[i] == NONE else CLASSES[row[i]]) for i in range(3)}))
    pd.DataFrame(cb).to_csv(f"{out}/consensus_boxes.csv", index=False)
    pd.DataFrame(dis).to_csv(f"{out}/disagreements_for_review.csv", index=False)
    nb = [int(sum(len(r["cls"][i]) for r in recs)) for i in range(3)]
    write_summary(df, acc, out, dict(n_frames=len(key), n_boxes=nb, boot=a.boot, seed=a.seed, errors=len(errors)), banner)
    if a.simulate:
        ok = sim_check(df, key, chi, out, truth, rates)
        open(f"{out}/README_SIMULATION.txt", "w").write("SIMULATION, NOT RESULTS. Software test only; do not cite.\n")
        print("simulation check:", "OK" if ok else "FAILED", "->", f"{out}/simulation_check.md")
    print("done ->", out)


if __name__ == "__main__":
    main()
