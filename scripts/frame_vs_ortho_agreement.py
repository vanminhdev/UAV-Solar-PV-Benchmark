#!/usr/bin/env python
"""Consistency check between the ortho-level thermal labels (thermal_box_frame_detections.csv: ortho box centre projected
into each thermal frame, x_px,y_px) and the new frame-level boxes (thermal_frame_boxes.csv).

    python -X utf8 code/scripts/frame_vs_ortho_agreement.py

Writes dataset/metadata/qc_validation/frame_vs_ortho_label_agreement.csv. This is a CONSISTENCY check only: both label sets
come from the same project and are not independent ground truth.
Rows (column `analysis`):
  det_to_new_box    ortho detection centres in labelled frames: how many fall inside a new box (any class / same class)
  new_box_to_det    new boxes (in frames that carry >=1 ortho detection): how many contain >=1 detection centre
  frames_no_ortho   labelled frames of ortho-labelled flights that carry no ortho detection: n frames, n new boxes in them
"""
from pathlib import Path
import pandas as pd
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
SUB = "annotations/model2_thermal_anomaly_obb"
FARMS = {"TN19": "farm_01_thuan_nam_19", "TNDL": "farm_03_thuan_nam_duc_long", "TT": "farm_04_thien_tan"}


def main():
    out = []
    for code, d in FARMS.items():
        p = DS / d / SUB
        det = pd.read_csv(p / "thermal_box_frame_detections.csv")
        det = det.merge(pd.read_csv(p / "thermal_boxes.csv", usecols=["box_id", "class_final"]), on="box_id", how="left")
        nb = pd.read_csv(p / "thermal_frame_boxes.csv")
        fr = pd.read_csv(p / "frames_reviewed.csv")
        nb = nb[nb.kept_in_yolo == 1]
        polys = {}
        for r in nb.itertuples(index=False):
            polys.setdefault((r.flight_id, r.thermal_file), []).append(
                (r.class_name, Polygon([(getattr(r, f"x{i}_px"), getattr(r, f"y{i}_px")) for i in range(1, 5)])))
        lab = set(zip(fr.flight_id, fr.thermal_file))
        det_frames = set(zip(det.flight_id, det.thermal_file))
        for fl in sorted(det.flight_id.unique()):
            dd = det[det.flight_id == fl].copy()
            dd["in_labelled"] = [(fl, t) in lab for t in dd.thermal_file]
            hit_any, hit_same, hit_cls = [], [], []
            for r in dd.itertuples(index=False):
                pt = Point(r.x_px, r.y_px)
                ps = polys.get((fl, r.thermal_file), [])
                hits = [c for c, pg in ps if pg.covers(pt)]
                hit_any.append(bool(hits)); hit_same.append(r.class_final in hits)
            dd["hit_any"], dd["hit_same"] = hit_any, hit_same
            for cls, g in list(dd.groupby("class_final")) + [("ALL", dd)]:
                gl = g[g.in_labelled]
                out.append(dict(analysis="det_to_new_box", farm=code, flight_id=fl, class_name=cls,
                                n_total=len(g), n_in_labelled_frames=len(gl), n_hit_any_class=int(gl.hit_any.sum()),
                                n_hit_same_class=int(gl.hit_same.sum()),
                                pct_hit_any=round(100 * gl.hit_any.mean(), 1) if len(gl) else None,
                                pct_hit_same=round(100 * gl.hit_same.mean(), 1) if len(gl) else None))
            # reverse
            fl_fr = fr[fr.flight_id == fl]
            with_det = fl_fr[[(fl, t) in det_frames for t in fl_fr.thermal_file]]
            no_det = fl_fr[[(fl, t) not in det_frames for t in fl_fr.thermal_file]]
            rows = []
            for r in nb[nb.flight_id == fl].itertuples(index=False):
                pg = Polygon([(getattr(r, f"x{i}_px"), getattr(r, f"y{i}_px")) for i in range(1, 5)])
                dets = det[(det.flight_id == fl) & (det.thermal_file == r.thermal_file)]
                has_frame_det = len(dets) > 0
                inside = [pg.covers(Point(x, y)) for x, y in zip(dets.x_px, dets.y_px)]
                rows.append((r.class_name, has_frame_det, any(inside)))
            rb = pd.DataFrame(rows, columns=["cls", "frame_has_det", "has_det_inside"])
            for cls, g in list(rb.groupby("cls")) + [("ALL", rb)]:
                gg = g[g.frame_has_det]
                out.append(dict(analysis="new_box_to_det", farm=code, flight_id=fl, class_name=cls,
                                n_total=len(g), n_in_labelled_frames=len(gg), n_hit_any_class=int(gg.has_det_inside.sum()),
                                pct_hit_any=round(100 * gg.has_det_inside.mean(), 1) if len(gg) else None))
            out.append(dict(analysis="frames_no_ortho", farm=code, flight_id=fl, class_name="ALL",
                            n_total=len(fl_fr), n_in_labelled_frames=len(no_det),
                            n_hit_any_class=int(nb[(nb.flight_id == fl)].set_index(["flight_id", "thermal_file"]).index
                                                .isin(list(zip(no_det.flight_id, no_det.thermal_file))).sum())))
    o = pd.DataFrame(out)
    # clarify the semantics of the generic count columns per analysis in one place
    o["note"] = o.analysis.map({
        "det_to_new_box": "n_total=detections; n_in_labelled_frames=those in a labelled frame; n_hit_*=centre inside a new box",
        "new_box_to_det": "n_total=new boxes; n_in_labelled_frames=new boxes in frames that carry >=1 ortho detection; n_hit_any_class=of those, boxes containing >=1 detection centre",
        "frames_no_ortho": "n_total=labelled frames of the flight; n_in_labelled_frames=frames with no ortho detection; n_hit_any_class=new boxes in those frames"})
    f = DS / "metadata/qc_validation/frame_vs_ortho_label_agreement.csv"
    o.to_csv(f, index=False)
    print(o[o.class_name == "ALL"].drop(columns="note").to_string())


if __name__ == "__main__":
    main()
