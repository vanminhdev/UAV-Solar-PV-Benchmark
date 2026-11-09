"""Apply the second-pass review decisions: CHANGE REPORT + PROPOSED label tables (originals are never overwritten).

Inputs (filled by the annotator, same columns as the templates):
  --decisions  review_decisions_template.csv  (decision = keep | change_class | delete ; new_class for change_class)
  --frames     review_frames_template.csv     (missed_anomaly yes/no)
  --added      review_added_boxes_template.csv (optional; one row per box to add; x1_px..y4_px = OBB corners in the 640x512 image)
Outputs in --out (default .local/annotation_review/proposed/):
  CHANGE_REPORT.md, change_summary.csv (counts per class / farm), class_transition_matrix.csv, problems.csv,
  <farm_dir>/thermal_frame_boxes_proposed.csv, <farm_dir>/frames_reviewed_proposed.csv, <farm_dir>/labels_yolo_obb_proposed/<flight>/<stem>.txt
  (YOLO files only for frames whose box set changed).
Refuses to write into the dataset folders. Boxes with an empty decision are treated as NOT YET REVIEWED (kept unchanged, counted separately).
Run: python -X utf8 code/scripts/apply_review_decisions.py --decisions D.csv --frames F.csv [--added A.csv] [--out DIR]"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from review_common import *

CID = {c: i for i, c in enumerate(CLASSES)}
DEC_OK = {"keep", "change_class", "delete"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--added")
    ap.add_argument("--out", default=str(OUT / "proposed"))
    ap.add_argument("--dataset", default=str(DATASET), help="dataset root holding the ORIGINAL label tables (read only)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    ds = Path(a.dataset).resolve()
    if ds in out.parents or out == ds:
        sys.exit("refusing to write inside the dataset folder")
    out.mkdir(parents=True, exist_ok=True)

    dec = pd.read_csv(a.decisions, dtype=str).fillna("")
    for c in ("decision", "new_class"):
        dec[c] = dec[c].str.strip()
    dec["decision_l"] = dec.decision.str.lower()
    frm = pd.read_csv(a.frames, dtype=str).fillna("")
    add = pd.read_csv(a.added, dtype=str).fillna("") if a.added and Path(a.added).exists() else pd.DataFrame(columns=["frame_id", "class_name"])

    boxes = []
    for fd in sorted(ds.glob("farm_*")):
        f = fd / "annotations" / "thermal_anomaly_obb" / "thermal_frame_boxes.csv"
        if f.exists():
            d = pd.read_csv(f); d["farm_dir"] = fd.name; boxes.append(d)
    b = pd.concat(boxes, ignore_index=True)
    b["frame_id"] = b.flight_id + "/" + b.thermal_file.str[:-4]

    # ---- validate
    prob = []
    known = set(b.box_uid)
    dec = dec.drop_duplicates("box_uid", keep="last")
    d = dec.set_index("box_uid")
    for u, r in d.iterrows():
        if u not in known:
            prob.append((u, "unknown box_uid"))
        elif r.decision_l and r.decision_l not in DEC_OK:
            prob.append((u, f"invalid decision '{r.decision}'"))
        elif r.decision_l == "change_class" and r.new_class not in CID:
            prob.append((u, f"change_class needs a valid new_class, got '{r.new_class}'"))
        elif r.decision_l == "change_class" and r.new_class == b.loc[b.box_uid == u, "class_name"].iloc[0]:
            prob.append((u, "change_class to the same class"))
    bad = {p[0] for p in prob}
    b["decision"] = b.box_uid.map(d.decision_l).fillna("")
    b["new_class"] = b.box_uid.map(d.new_class).fillna("")
    b["note"] = b.box_uid.map(d.note).fillna("")
    b.loc[b.box_uid.isin(bad), "decision"] = ""          # invalid rows are not applied
    b["outcome"] = np.where(b.decision == "", "not_reviewed", b.decision)
    b["final_class"] = np.where(b.outcome == "change_class", b.new_class, b.class_name)

    # ---- added boxes
    new_rows = []
    nadd_pending = 0
    for i, r in add.reset_index(drop=True).iterrows():
        fid, cls = r.get("frame_id", ""), r.get("class_name", "")
        fl, stem = (fid.split("/") + [""])[:2]
        if cls not in CID or not fl or not stem:
            prob.append((f"added_row_{i + 2}", "invalid frame_id/class_name")); continue
        pts = [r.get(f"{k}{j}_px", "") for j in range(1, 5) for k in "xy"]
        farm_dir = b.loc[b.flight_id == fl, "farm_dir"].head(1)
        if not len(farm_dir):
            ff = [p for p in ds.glob("farm_*") if any(True for _ in (p / "raw_images").glob(f"*/{fl}"))]
            farm_dir = pd.Series([ff[0].name]) if ff else pd.Series([""])
        if any(p == "" for p in pts):
            nadd_pending += 1; continue                      # counted, but no geometry -> cannot be written
        xy = np.array([float(p) for p in pts]).reshape(4, 2)
        n = int((b.frame_id == fid).sum()) + sum(1 for n_ in new_rows if n_["frame_id"] == fid)
        rec = dict(box_uid=f"{fl}__{stem}__A{n:02d}", flight_id=fl, thermal_file=stem + ".JPG", class_id=CID[cls], class_name=cls,
                   clipped_flag=0, frac_area_inside_frame=1.0, kept_in_yolo=1, source_label_name="review_added", farm_dir=farm_dir.iloc[0],
                   frame_id=fid, outcome="added", final_class=cls, decision="added", new_class="", note=r.get("note", ""))
        for j in range(4):
            rec[f"x{j + 1}_px"], rec[f"y{j + 1}_px"] = xy[j]
            rec[f"x{j + 1}"], rec[f"y{j + 1}"] = np.clip(xy[j, 0] / W, 0, 1), np.clip(xy[j, 1] / H, 0, 1)
            rec[f"x{j + 1}_clip"], rec[f"y{j + 1}_clip"] = rec[f"x{j + 1}"], rec[f"y{j + 1}"]
        new_rows.append(rec)
    addf = pd.DataFrame(new_rows)

    # ---- change report tables
    allr = pd.concat([b, addf], ignore_index=True) if len(addf) else b
    farm = allr.farm_dir.str[5:7]
    summ = []
    for lvl, key in (("class", allr.class_name), ("farm", farm)):
        t = pd.crosstab(key, allr.outcome)
        for c in ["keep", "change_class", "delete", "not_reviewed", "added"]:
            if c not in t.columns: t[c] = 0
        t = t[["keep", "change_class", "delete", "not_reviewed", "added"]].reset_index()
        t.columns = ["group"] + list(t.columns[1:])
        t.insert(0, "level", lvl)
        summ.append(t)
    s = pd.concat(summ, ignore_index=True)
    s.to_csv(out / "change_summary.csv", index=False)
    ch = b[b.outcome == "change_class"]
    trans = pd.crosstab(ch.class_name, ch.final_class) if len(ch) else pd.DataFrame()
    trans.to_csv(out / "class_transition_matrix.csv")
    pd.DataFrame(prob, columns=["row", "problem"]).to_csv(out / "problems.csv", index=False)
    tot = b.outcome.value_counts().to_dict()
    nrev = len(b) - tot.get("not_reviewed", 0)
    fm = frm.copy()
    fm["missed"] = fm.missed_anomaly.str.lower().str.strip()
    nframes_done = int((fm.reviewed_date != "").sum())

    # ---- proposed tables
    final = b[b.outcome != "delete"].copy()
    final["class_name"] = final.final_class
    final["class_id"] = final.class_name.map(CID)
    cols = [c for c in boxes[0].columns if c in final.columns]
    final = pd.concat([final, addf], ignore_index=True) if len(addf) else final
    changed_frames = set(b.loc[b.outcome.isin(["change_class", "delete"]), "frame_id"]) | (set(addf.frame_id) if len(addf) else set())
    for farm_dir, g in final.groupby("farm_dir"):
        od = out / farm_dir; od.mkdir(exist_ok=True)
        g[[c for c in cols if c in g.columns]].to_csv(od / "thermal_frame_boxes_proposed.csv", index=False)
        fr0 = pd.read_csv(ds / farm_dir / "annotations" / "thermal_anomaly_obb" / "frames_reviewed.csv")
        fr0["frame_id"] = fr0.flight_id + "/" + fr0.thermal_file.str[:-4]
        nb = g.groupby("frame_id").size()
        fr0["n_boxes_proposed"] = fr0.frame_id.map(nb).fillna(0).astype(int)
        fr0["status_proposed"] = np.where(fr0.n_boxes_proposed > 0, "positive", "reviewed_negative")
        fr0.drop(columns="frame_id").to_csv(od / "frames_reviewed_proposed.csv", index=False)
        for fid in sorted(changed_frames & set(g.frame_id) | (changed_frames & set(fr0.frame_id))):
            fl, stem = fid.split("/")
            sub = g[(g.frame_id == fid) & (g.kept_in_yolo == 1)]
            ld = od / "labels_yolo_obb_proposed" / fl; ld.mkdir(parents=True, exist_ok=True)
            with open(ld / (stem + ".txt"), "w") as fh:
                for _, r in sub.iterrows():
                    fh.write(f"{int(r.class_id)} " + " ".join(f"{r[f'{k}{j}']:.6f}" for j in range(1, 5) for k in "xy") + "\n")

    # ---- report
    L = ["# Second-pass review: CHANGE REPORT", "",
         f"Boxes in original tables: {len(b)}; reviewed (non-empty valid decision): {nrev}; not yet reviewed: {tot.get('not_reviewed', 0)}.", "",
         "| outcome | boxes | share of reviewed |", "|---|---|---|"]
    for k in ("keep", "change_class", "delete"):
        L.append(f"| {k} | {tot.get(k, 0)} | {tot.get(k, 0) / max(nrev, 1):.1%} |")
    L += [f"| added (with geometry) | {len(addf)} | |", f"| added rows without geometry (pending) | {nadd_pending} | |", "",
          f"Frames: {len(frm)} listed, {nframes_done} with a reviewed_date, missed_anomaly=yes: {(fm.missed == 'yes').sum()}, no: {(fm.missed == 'no').sum()}.", "",
          "Quality indicator (box level, among reviewed): agreement with first pass = kept / reviewed = "
          f"{tot.get('keep', 0) / max(nrev, 1):.1%}; class changed = {tot.get('change_class', 0) / max(nrev, 1):.1%}; deleted = {tot.get('delete', 0) / max(nrev, 1):.1%}.", "",
          "## Per class and farm", "", s.to_markdown(index=False) if hasattr(s, "to_markdown") and _has_tab() else s.to_string(index=False), "",
          "## Class transitions (rows: old, columns: new)", "", trans.to_string() if len(trans) else "none", "",
          f"## Problems ({len(prob)})", "", "see problems.csv (invalid rows were NOT applied)", ""]
    (out / "CHANGE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print("written to", out)


def _has_tab():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
