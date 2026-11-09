"""Check pixel alignment between R-JPEG (palette render) and decoded temperature matrices.
Spearman correlation of JPG luminance vs temp_c for 5 orientations, stratified sample of frames.
Output: .local/annotation_review/alignment_check.csv + printed summary."""
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from review_common import *

ORI = {"identity": lambda a: a, "transpose": lambda a: a.T, "flipud": lambda a: a[::-1],
       "fliplr": lambda a: a[:, ::-1], "rot180": lambda a: a[::-1, ::-1]}


def main(n_per_flight=9, seed=0):
    fr = load_frames()
    rng = np.random.default_rng(seed)
    rows = []
    for fid, g in fr.groupby("flight_id"):
        k = min(len(g), n_per_flight)
        for i in rng.choice(len(g), k, replace=False):
            r = g.iloc[i]
            im = Image.open(r.jpg_path).convert("L")
            t = load_temp(r.npz_path)
            lum = np.asarray(im, dtype=np.float32)
            rec = dict(farm=r.farm, flight_id=fid, frame_id=r.frame_id, jpg_size=f"{im.size[0]}x{im.size[1]}", tmp_shape=f"{t.shape[1]}x{t.shape[0]}")
            for name, f in ORI.items():
                tt = f(t)
                if tt.shape != lum.shape:
                    rec[name] = np.nan
                    continue
                rec[name] = spearmanr(lum.ravel()[::3], tt.ravel()[::3])[0]
            rows.append(rec)
    d = pd.DataFrame(rows)
    others = d[["transpose", "flipud", "fliplr", "rot180"]].max(axis=1)
    d["margin"] = d["identity"] - others
    d.to_csv(OUT / "alignment_check.csv", index=False)
    print("frames", len(d), "flights", d.flight_id.nunique(), d.farm.value_counts().to_dict())
    print("size mismatch:", (d.jpg_size != d.tmp_shape).sum())
    print(d[["identity", "transpose", "flipud", "fliplr", "rot180", "margin"]].describe().round(3))
    print("identity best:", (d.margin > 0).mean(), " margin>0.2:", (d.margin > 0.2).mean(), " margin>0.1:", (d.margin > 0.1).mean())
    print("identity rho<0.8:", (d.identity < 0.8).sum())
    print(d.groupby("farm").agg(n=("identity", "size"), rho_med=("identity", "median"), rho_min=("identity", "min"), margin_min=("margin", "min")).round(3))
    print(d.sort_values("identity").head(5)[["frame_id", "identity", "margin"]])


if __name__ == "__main__":
    main()
