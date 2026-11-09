#!/usr/bin/env python
"""Integrate frame-level thermal anomaly labels (YOLO-OBB, one txt per 640x512 thermal frame) into dataset/.

    python -X utf8 code/scripts/integrate_frame_labels.py \
        --labels-dir .local/cache/d5_labels_only \
        --assignment .local/tmp/viz/newlabel_flight_assignment.csv

Inputs
  --labels-dir   folder with <FARM>__<ALT>__<stem>.txt, lines `cls x1 y1 x2 y2 x3 y3 x4 y4` (normalised to the frame)
  --assignment   csv label_image (<FARM>__<ALT>__<stem>.jpg) -> flight_id (resolved earlier by pixel comparison with the
                 raw R-JPEGs; BIM2 file names such as DJI_0161_T.JPG repeat across flights so the name alone is ambiguous)
No image is copied: the labels refer to the ORIGINAL raw R-JPEGs already in dataset/<farm>/raw_images/.

Outputs, per farm, in dataset/<farm_dir>/annotations/thermal_anomaly_obb/
  thermal_frame_boxes.csv     one row per source box (all of them, also those not written to the YOLO files)
  labels_yolo_obb/<flight_id>/<thermal stem>.txt   YOLO-OBB with coordinates CLIPPED to [0,1]; boxes with < MIN_FRAC of
                              their area inside the frame are not written (kept_in_yolo = 0 in the csv)
  frames_reviewed.csv         every labelled frame; status positive | reviewed_negative
  data.yaml                   class names (NOT a ready-to-use ultralytics layout; see make_yolo_dataset_view.py)
and dataset/metadata/annotations_summary/frame_label_counts_by_flight_class.csv, frame_label_counts_by_farm_altitude.csv

Clipping: each corner is clamped to [0,1] (box cut by the frame edge); `frac_area_inside_frame` is the area of the
source quadrilateral inside the frame divided by its total area (shapely).
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon, box as sbox

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset"
CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]
FARM_DIR = {"ThuanNam19": "farm_01_thuan_nam_19", "BIM2": "farm_02_bim_2",
            "ThuanNamDucLong": "farm_03_thuan_nam_duc_long", "ThienTan": "farm_04_thien_tan"}
FARM_CODE = {"ThuanNam19": "TN19", "BIM2": "BIM2", "ThuanNamDucLong": "TNDL", "ThienTan": "TT"}
MIN_FRAC = 0.30
SUB = "annotations/thermal_anomaly_obb"
FRAME = sbox(0, 0, 1, 1)

YAML = """# Frame-level thermal anomaly labels (YOLO-OBB), {farm}
# NOT a ready-to-use ultralytics layout. The images are the ORIGINAL raw thermal R-JPEGs in
#   ../../raw_images/<altitude folder>/<flight_id>/<thermal_file>
# (not copied here; the dataset keeps only the original files). A label file is
#   labels_yolo_obb/<flight_id>/<thermal_file stem>.txt
# and is matched to its image by (flight_id, thermal_file) -- see frames_reviewed.csv. BIM2 file names repeat across
# flights, hence the per-flight sub-folders. Use code/scripts/make_yolo_dataset_view.py to build a symlink/copy view
# (images/ + labels/ + train/val/test lists) in a directory of your choice from metadata/splits/frame_label_split.csv.
# Coordinates: x1 y1 x2 y2 x3 y3 x4 y4, normalised to the 640x512 frame, clipped to [0,1].
path: .
nc: 4
names:
  0: SingleHotSpot
  1: MultiHotSpot
  2: Bypass
  3: ModuleOpenCircuit
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--dataset", default=str(DS))
    a = ap.parse_args()
    ds = Path(a.dataset)
    lab = Path(a.labels_dir)
    asg = pd.read_csv(a.assignment)
    inv = pd.read_csv(ds / "metadata/inventory/image_inventory.csv",
                      usecols=["flight_id", "file", "width", "height"]).set_index(["flight_id", "file"])
    rows, frames = [], []
    for r in asg.itertuples(index=False):
        farm_tok, alt_tok, stem = r.label_image[:-4].split("__")
        assert farm_tok in FARM_DIR, farm_tok
        txt = lab / (r.label_image[:-4] + ".txt")
        tf = stem + ".JPG"
        W, H = inv.loc[(r.flight_id, tf), ["width", "height"]]
        lines = [l.split() for l in txt.read_text().splitlines() if l.strip()]
        for k, t in enumerate(lines):
            assert len(t) == 9, (txt, t)
            c = int(t[0]); v = [float(x) for x in t[1:]]
            xs, ys = v[0::2], v[1::2]
            poly = Polygon(list(zip(xs, ys)))
            area = poly.area
            frac = (poly.intersection(FRAME).area / area) if area > 0 else 0.0
            clip = [min(1.0, max(0.0, x)) for x in v]
            rows.append(dict(
                box_uid=f"{r.flight_id}__{Path(tf).stem}__{k:02d}", farm=FARM_CODE[farm_tok], flight_id=r.flight_id,
                thermal_file=tf, class_id=c, class_name=CLASSES[c],
                **{f"{n}{i+1}": v[2 * i + j] for i in range(4) for j, n in enumerate("xy")},
                **{f"{n}{i+1}_px": v[2 * i + j] * (W, H)[j] for i in range(4) for j, n in enumerate("xy")},
                **{f"{n}{i+1}_clip": clip[2 * i + j] for i in range(4) for j, n in enumerate("xy")},
                clipped_flag=int(any(abs(x - y) > 1e-9 for x, y in zip(v, clip))),
                frac_area_inside_frame=frac, kept_in_yolo=int(frac >= MIN_FRAC),
                source_label_name=txt.name))
        frames.append(dict(farm=FARM_CODE[farm_tok], flight_id=r.flight_id, thermal_file=tf, n_boxes_source=len(lines),
                           source_label_name=txt.name))
    B = pd.DataFrame(rows)
    F = pd.DataFrame(frames)
    kept = B[B.kept_in_yolo == 1].groupby(["flight_id", "thermal_file"]).size().rename("n_boxes")
    F = F.merge(kept, left_on=["flight_id", "thermal_file"], right_index=True, how="left")
    F["n_boxes"] = F.n_boxes.fillna(0).astype(int)
    F["status"] = F.n_boxes_source.map(lambda n: "positive" if n > 0 else "reviewed_negative")
    F["altitude_m"] = F.flight_id.str.extract(r"_(\d+)M_")[0].astype(int)
    B["altitude_m"] = B.flight_id.str.extract(r"_(\d+)M_")[0].astype(int)

    cols = ["box_uid", "flight_id", "thermal_file", "class_id", "class_name"] + \
        [f"{n}{i}{s}" for s in ("", "_px", "_clip") for i in range(1, 5) for n in "xy"] + \
        ["clipped_flag", "frac_area_inside_frame", "kept_in_yolo", "source_label_name"]
    fcols = ["flight_id", "thermal_file", "n_boxes", "n_boxes_source", "status", "source_label_name"]
    for code, fdir in ((FARM_CODE[k], v) for k, v in FARM_DIR.items()):
        out = ds / fdir / SUB
        (out / "labels_yolo_obb").mkdir(parents=True, exist_ok=True)
        b = B[B.farm == code]; f = F[F.farm == code]
        b[cols].round(6).to_csv(out / "thermal_frame_boxes.csv", index=False)
        f.sort_values(["flight_id", "thermal_file"])[fcols].to_csv(out / "frames_reviewed.csv", index=False)
        (out / "data.yaml").write_text(YAML.format(farm=fdir), encoding="utf-8", newline="\n")
        for (fl, tf), g in b.groupby(["flight_id", "thermal_file"]):
            g = g[g.kept_in_yolo == 1]
            d = out / "labels_yolo_obb" / fl
            d.mkdir(exist_ok=True)
            with open(d / (Path(tf).stem + ".txt"), "w", newline="\n") as fh:
                for r in g.itertuples(index=False):
                    co = [getattr(r, f"{n}{i}_clip") for i in range(1, 5) for n in "xy"]
                    fh.write(f"{r.class_id} " + " ".join(f"{x:.6f}" for x in co) + "\n")
        # empty label files for negatives / frames whose boxes were all dropped (explicit "reviewed, nothing")
        for r in f.itertuples(index=False):
            d = out / "labels_yolo_obb" / r.flight_id
            d.mkdir(exist_ok=True)
            p = d / (Path(r.thermal_file).stem + ".txt")
            if not p.exists():
                p.write_text("", newline="\n")
        print(f"{fdir}: {len(f)} frames, {len(b)} boxes ({int(b.kept_in_yolo.sum())} written)")

    # ---- stats tables
    S = ds / "metadata/annotations_summary"
    S.mkdir(parents=True, exist_ok=True)

    def counts(keys):
        Bk = B[B.kept_in_yolo == 1]
        o = F.groupby(keys).agg(n_frames=("thermal_file", "size"),
                                n_frames_positive=("status", lambda s: int((s == "positive").sum())),
                                n_frames_reviewed_negative=("status", lambda s: int((s == "reviewed_negative").sum())))
        o["n_boxes_source"] = F.groupby(keys).n_boxes_source.sum()
        o["n_boxes_in_yolo"] = Bk.groupby(keys).size().reindex(o.index).fillna(0).astype(int)
        for i, c in enumerate(CLASSES):
            o[f"n_{c}"] = Bk[Bk.class_id == i].groupby(keys).size().reindex(o.index).fillna(0).astype(int)
        o["n_boxes_dropped_lt30pct_inside"] = o.n_boxes_source - o.n_boxes_in_yolo
        o["n_boxes_clipped"] = B[B.clipped_flag == 1].groupby(keys).size().reindex(o.index).fillna(0).astype(int)
        return o.reset_index()

    counts(["farm", "flight_id", "altitude_m"]).to_csv(S / "frame_label_counts_by_flight_class.csv", index=False)
    counts(["farm", "altitude_m"]).to_csv(S / "frame_label_counts_by_farm_altitude.csv", index=False)
    print("frames", len(F), "boxes", len(B), "dropped", int((B.kept_in_yolo == 0).sum()),
          "clipped", int(B.clipped_flag.sum()), "negatives", int((F.status == "reviewed_negative").sum()))
    print(B.groupby("class_name").size().to_string())


if __name__ == "__main__":
    main()
