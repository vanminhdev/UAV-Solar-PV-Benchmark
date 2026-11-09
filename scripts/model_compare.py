"""Leave-one-out comparison of translation / similarity / affine transform models per orthomosaic. Run from the dataset root parent: python -X utf8 code/scripts/model_compare.py"""
import numpy as np
import pandas as pd

D = "dataset/metadata/ortho_registration/"
cp = pd.read_csv(D + "ortho_control_points.csv")
tr = pd.read_csv(D + "ortho_registration_transforms.csv")
good = set(tr[(tr.is_primary == 1) & (tr.quality == "good")].flight_id)
cp = cp[(cp.status != "rejected") & cp.flight_id.isin(good)]


def design(model, x, y):
    one = np.ones_like(x)
    zero = np.zeros_like(x)
    if model == "translation":
        return np.r_[np.c_[one, zero], np.c_[zero, one]]
    if model == "similarity":  # E' = a x - b y + tE ; N' = b x + a y + tN
        return np.r_[np.c_[x, -y, one, zero], np.c_[y, x, zero, one]]
    if model == "affine":
        return np.r_[np.c_[x, y, one, zero, zero, zero], np.c_[zero, zero, zero, x, y, one]]


def fit_predict(model, tr_pts, te_pts):
    x0 = tr_pts.E_src.values.mean(); y0 = tr_pts.N_src.values.mean()
    def prep(p):
        return p.E_src.values - x0, p.N_src.values - y0
    x, y = prep(tr_pts)
    if model == "translation":
        t = np.array([(tr_pts.E_ref - tr_pts.E_src).mean(), (tr_pts.N_ref - tr_pts.N_src).mean()])
        return te_pts.E_src.values + t[0], te_pts.N_src.values + t[1]
    A = design(model, x, y)
    b = np.r_[tr_pts.E_ref.values - x0 * 0, tr_pts.N_ref.values]
    # reference positions relative to the same origin as the source
    b = np.r_[tr_pts.E_ref.values - x0, tr_pts.N_ref.values - y0]
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    xt, yt = prep(te_pts)
    At = design(model, xt, yt)
    pr = At @ p
    n = len(xt)
    return pr[:n] + x0, pr[n:] + y0


rows = []
for fl, g in cp.groupby("flight_id"):
    g = g.reset_index(drop=True)
    n = len(g)
    for model, npar in (("translation", 2), ("similarity", 4), ("affine", 6)):
        if n < npar // 2 + 3:
            continue
        res = []
        for i in range(n):
            trn = g.drop(index=i)
            te = g.iloc[[i]]
            e, nn = fit_predict(model, trn, te)
            res.append(float(np.hypot(e[0] - te.E_ref.values[0], nn[0] - te.N_ref.values[0])))
        rows.append(dict(flight_id=fl, model=model, n=n, loo_median=np.median(res), loo_p95=np.percentile(res, 95), loo_rms=float(np.sqrt(np.mean(np.square(res))))))
r = pd.DataFrame(rows)
piv = r.pivot(index="flight_id", columns="model", values="loo_rms")
print("flights:", len(piv))
print("median over flights of LOO rms (m):", piv.median().round(3).to_dict())
print("flights where model has the lowest LOO rms:", piv.idxmin(axis=1).value_counts().to_dict())
print("mean ratio vs similarity:", (piv.div(piv.similarity, axis=0)).median().round(3).to_dict())
r.to_csv(D + "model_comparison_loo.csv", index=False)
