"""Build the second-pass review package for ALL thermal frames (see .local/annotation_review/README.md).
Needs box_temperature.csv (run box_temperature.py first). Outputs in .local/annotation_review/:
  review_queue.csv, images/<farm>/<flight>/<stem>.jpg (+ <stem>_cand.jpg), box_flags.csv,
  review_decisions_template.csv, review_frames_template.csv, review_added_boxes_template.csv
Run: python -X utf8 code/scripts/build_review_package.py [--limit N] [--no-images]
Original labels are only read, never modified."""
import argparse
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from review_common import *
from box_temperature import poly_mask

BG_KSIZE = 61          # large median filter (px) for the local background
BLOB_DELTA = 10.0      # degC above local background
BLOB_MIN, BLOB_MAX = 5, 300
SCALE = 1.5            # review image upscaling
JPG_Q = 80
N_CAND_IMAGES = 1000
W_REASON = {"small_mask": 4, "overlap_diff_class": 5, "duplicate_like": 5, "low_delta_ring": 3,
            "rare_class": 2, "clipped": 1.5, "box_too_large": 2, "box_too_small": 2}


def analyse_frame(args):
    """Candidate blobs (all frames) + box-box overlaps (frames with boxes)."""
    fid, npz, boxrows = args
    t = load_temp(npz)
    lo, hi = float(t.min()), float(t.max())
    sc = 255.0 / max(hi - lo, 1e-3)
    u8 = np.clip(np.round((t - lo) * sc), 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(u8, BG_KSIZE).astype(np.float32) / sc + lo
    local = t - bg
    res = dict(frame_id=fid, max_local_delta=float(local.max()))
    n, lab, stats, cent = cv2.connectedComponentsWithStats((local > BLOB_DELTA).astype(np.uint8), connectivity=8)
    best = None; nb = 0
    if n > 1:
        peaks = ndi.maximum(local, lab, index=np.arange(1, n))
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if BLOB_MIN <= a <= BLOB_MAX:
                nb += 1
                if best is None or peaks[i - 1] > best[0]:
                    best = (float(peaks[i - 1]), i)
    res["n_blobs"] = nb
    if best:
        i = best[1]
        m = lab == i
        res.update(blob_contrast_c=best[0], blob_x=float(cent[i][0]), blob_y=float(cent[i][1]),
                   blob_n_px=int(stats[i, cv2.CC_STAT_AREA]), blob_peak_c=float(t[m].max()))
    # box overlaps
    flags = []
    if boxrows:
        masks = [poly_mask(box_polygon(r, clipped=True)) for r in boxrows]
        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                inter = (masks[a] & masks[b]).sum()
                if inter == 0:
                    continue
                iou = inter / (masks[a] | masks[b]).sum()
                same = boxrows[a]["class_name"] == boxrows[b]["class_name"]
                if not same and iou > 0.3:
                    flags += [(boxrows[a]["box_uid"], "overlap_diff_class", iou), (boxrows[b]["box_uid"], "overlap_diff_class", iou)]
                if same and iou > 0.7:
                    flags += [(boxrows[a]["box_uid"], "duplicate_like", iou), (boxrows[b]["box_uid"], "duplicate_like", iou)]
    res["overlap_flags"] = flags
    return res


def get_font():
    try:
        return ImageFont.load_default(size=11)
    except TypeError:
        return ImageFont.load_default()


def render(args):
    jpg, out, boxes, cand = args
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(jpg).convert("RGB").resize((int(W * SCALE), int(H * SCALE)), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    f = get_font()
    for cls, uid, poly in boxes:
        col = CLASS_RGB[cls]
        pts = [(x * SCALE, y * SCALE) for x, y in poly]
        d.line(pts + [pts[0]], fill=col, width=1)
        tx, ty = min(pts, key=lambda p: (p[1], p[0]))
        d.text((tx + 1, max(ty - 12, 0)), SHORT[cls] + uid.rsplit("__", 1)[-1], fill=col, font=f)
    im.save(out, "JPEG", quality=JPG_Q, optimize=True)
    if cand is not None:
        im2 = Image.open(out).convert("RGB")
        d = ImageDraw.Draw(im2)
        cx, cy = cand[0] * SCALE, cand[1] * SCALE
        r = 16
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=1)
        im2.save(out.with_name(out.stem + "_cand.jpg"), "JPEG", quality=JPG_Q, optimize=True)
    return out.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N frames")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fr = load_frames()
    if a.limit:
        fr = fr.sample(a.limit, random_state=0).reset_index(drop=True)
    b = load_boxes()
    bt = pd.read_csv(OUT / "box_temperature.csv")
    b = b.merge(bt[["box_uid", "n_pixels", "delta_ring", "delta_max", "flag_small_mask"]], on="box_uid", how="left")
    b["frame_id"] = b.flight_id + "/" + b.thermal_file.str[:-4]
    b["alt"] = b.flight_id.str.extract(r"_(\d+)M_")[0].astype(int)
    # ---- reviewed-status per frame
    rev = pd.concat([pd.read_csv(f) for f in (DATASET.glob("farm_*/annotations/thermal_anomaly_obb/frames_reviewed.csv"))])
    rev["frame_id"] = rev.flight_id + "/" + rev.thermal_file.str[:-4]
    status = dict(zip(rev.frame_id, rev.status))
    # ---- box flags
    b["reasons"] = [[] for _ in range(len(b))]
    reasons = {u: [] for u in b.box_uid}
    p5 = b.groupby("class_name").delta_ring.quantile(0.05)
    for _, r in b.iterrows():
        u = r.box_uid
        if r.delta_ring <= p5[r.class_name]: reasons[u].append("low_delta_ring")
        if r.flag_small_mask == 1: reasons[u].append("small_mask")
        if int(r.clipped_flag) == 1: reasons[u].append("clipped")
        if r.class_name in ("ModuleOpenCircuit", "Bypass"): reasons[u].append("rare_class")
    for (c, alt), g in b.groupby(["class_name", "alt"]):
        gg = g if len(g) >= 30 else b[b.class_name == c]
        lo, hi = gg.n_pixels.quantile(0.02), gg.n_pixels.quantile(0.98)
        for _, r in g.iterrows():
            if r.n_pixels > hi: reasons[r.box_uid].append("box_too_large")
            if r.n_pixels < lo: reasons[r.box_uid].append("box_too_small")
    # ---- frame analysis
    byframe = {k: [r for _, r in g.iterrows()] for k, g in b.groupby("frame_id")}
    jobs = [(r.frame_id, r.npz_path, byframe.get(r.frame_id, [])) for r in fr.itertuples()]
    if a.limit:
        keep = set(fr.frame_id)
    with Pool(6) as p:
        an = list(p.imap_unordered(analyse_frame, jobs, chunksize=16))
    an = {x["frame_id"]: x for x in an}
    for x in an.values():
        for u, why, iou in x["overlap_flags"]:
            reasons[u].append(why)
    b["flag_reasons"] = [";".join(dict.fromkeys(reasons[u])) for u in b.box_uid]
    b["n_flags"] = [len(set(reasons[u])) for u in b.box_uid]
    b["flag_score"] = [sum(W_REASON[w] for w in set(reasons[u])) for u in b.box_uid]
    b[["box_uid", "frame_id", "class_name", "n_pixels", "delta_ring", "flag_reasons", "n_flags", "flag_score"]].to_csv(out / "box_flags.csv", index=False)
    # ---- queue
    rows = []
    for r in fr.itertuples():
        x = an[r.frame_id]
        g = b[b.frame_id == r.frame_id] if r.frame_id in byframe else b.iloc[0:0]
        nflagged = int((g.n_flags > 0).sum())
        rs = sorted({w for s in g.flag_reasons for w in s.split(";") if w})
        rec = dict(frame_id=r.frame_id, flight_id=r.flight_id, farm=r.farm, farm_dir=r.farm_dir, thermal_file=r.file,
                   altitude_m=r.altitude_m, status=("positive" if len(g) else status.get(r.frame_id, "unlisted_no_box")),
                   n_boxes=len(g), n_flagged_boxes=nflagged, n_flags=int(g.n_flags.sum()) if len(g) else 0,
                   flag_reasons=";".join(rs),
                   class_counts=" ".join(f"{SHORT[c]}{n}" for c, n in g.class_name.value_counts().items()),
                   n_blobs=x["n_blobs"], blob_contrast_c=x.get("blob_contrast_c"), blob_x=x.get("blob_x"), blob_y=x.get("blob_y"),
                   blob_n_px=x.get("blob_n_px"), blob_peak_c=x.get("blob_peak_c"), max_local_delta_c=x["max_local_delta"])
        if len(g) and nflagged:
            rec["tier"] = 1; rec["priority_score"] = 1000 + float(g.flag_score.sum())
        elif len(g):
            rec["tier"] = 2; rec["priority_score"] = 500 + (10 - min(max(float(g.delta_ring.min()), 0), 10))   # weakest box first
        else:
            rec["tier"] = 3
            rec["priority_score"] = x["blob_contrast_c"] if x.get("blob_contrast_c") is not None and not np.isnan(x.get("blob_contrast_c", np.nan)) else 0.01 * x["max_local_delta"]
        rows.append(rec)
    q = pd.DataFrame(rows).sort_values(["tier", "priority_score", "frame_id"], ascending=[True, False, True]).reset_index(drop=True)
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    q["review_image"] = ("images/" + q.farm_dir + "/" + q.flight_id + "/" + q.thermal_file.str[:-4] + ".jpg")
    q["cand_image"] = ""
    t3 = q[(q.tier == 3) & q.blob_x.notna()].head(N_CAND_IMAGES)
    q.loc[t3.index, "cand_image"] = ("images/" + t3.farm_dir + "/" + t3.flight_id + "/" + t3.thermal_file.str[:-4] + "_cand.jpg")
    q.round(2).to_csv(out / "review_queue.csv", index=False)
    print("queue", len(q), q.tier.value_counts().sort_index().to_dict())
    # ---- templates
    d = b.merge(q[["frame_id", "rank", "farm"]], on="frame_id").sort_values(["rank", "box_uid"])
    dec = pd.DataFrame(dict(box_uid=d.box_uid, frame_id=d.frame_id, farm=d.farm, thermal_file=d.thermal_file, current_class=d.class_name,
                            box_flags=d.flag_reasons, review_rank=d["rank"], decision="", new_class="", note=""))
    dec.to_csv(out / "review_decisions_template.csv", index=False)
    fq = q.sort_values("rank")
    pd.DataFrame(dict(frame_id=fq.frame_id, farm=fq.farm, thermal_file=fq.thermal_file, review_rank=fq["rank"], n_boxes=fq.n_boxes,
                      missed_anomaly="", note="", reviewed_date="")).to_csv(out / "review_frames_template.csv", index=False)
    pd.DataFrame(columns=["frame_id", "class_name", "x1_px", "y1_px", "x2_px", "y2_px", "x3_px", "y3_px", "x4_px", "y4_px", "note"]).to_csv(out / "review_added_boxes_template.csv", index=False)
    print("templates written; boxes", len(dec), "frames", len(fq))
    fl = b[b.n_flags > 0]
    print("flagged boxes", len(fl), "of", len(b), "; flagged frames", int((q.tier == 1).sum()))
    cnt = pd.Series([w for s in b.flag_reasons for w in s.split(";") if w]).value_counts()
    print(cnt.to_string())
    print("by class:", b[b.n_flags > 0].class_name.value_counts().to_dict())
    # ---- images
    if a.no_images:
        return
    qi = q.set_index("frame_id")
    cand_ids = set(t3.frame_id)
    jj = []
    for r in fr.itertuples():
        boxes = [(x.class_name, x.box_uid, box_polygon(x, clipped=False)) for x in byframe.get(r.frame_id, [])]
        cand = None
        if r.frame_id in cand_ids:
            cand = (qi.loc[r.frame_id, "blob_x"], qi.loc[r.frame_id, "blob_y"])
        jj.append((r.jpg_path, out / "images" / r.farm_dir / r.flight_id / (r.file[:-4] + ".jpg"), boxes, cand))
    with Pool(6) as p:
        sizes = list(p.imap_unordered(render, jj, chunksize=32))
    tot = sum(sizes) + sum(1 for j in jj if j[3] is not None) * np.mean(sizes) * 1.0
    print("images", len(jj), "main bytes", sum(sizes), "mean KB", np.mean(sizes) / 1024)


if __name__ == "__main__":
    main()
