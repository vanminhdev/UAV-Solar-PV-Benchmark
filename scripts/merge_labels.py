"""Merge the per-flight `nhan_tay.json` thermal box files into dataset/<farm>/annotations/.

    python merge_labels.py <labels_dir> <dataset_root>

<labels_dir> holds one sub-folder per flight (31M_T3, 35M_T3, ..., TNDL_GSD3_T2, TT_GSD3_T1) with nhan_tay.json.
Per flight the JSON stores module-sized boxes drawn on the thermal orthomosaic (pixels of a down-scaled 'nen'
image), created and verified by one human annotator:
    tt = 'dung' -> confirmed, 'sai' -> rejected, null -> not individually flagged.
Project decision: boxes not individually flagged are accepted like confirmed ones and released as one set (no review_status
column). The released tables carry ONE class column (class_final); no preliminary class or score fields are written.
Outputs (per farm, in annotations/thermal_anomaly_obb/):
    thermal_boxes.csv                  one row per box, corners in EPSG:32649
    thermal_box_frame_detections.csv   per box and per raw thermal frame: centre pixel
Only released boxes are written: confirmed + unreviewed (rejected boxes are dropped;
the source JSON stays outside dataset/).
Pixel -> map: nen px / s + (c0, r0) = ortho px; lon = lon0 + (x - c0) / gw * (lon1 - lon0),
lat = lat0 + (y - r0) / gh * (lat1 - lat0); then WGS 84 -> EPSG:32649 (pyproj).
"""
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

FLIGHTS = {  # label folder -> (farm dir, flight_id)
    "31M_T3": ("farm_01_thuan_nam_19", "TN19_20260411_31M_T3"),
    "35M_T3": ("farm_01_thuan_nam_19", "TN19_20260411_35M_T3"),
    "35M_T4": ("farm_01_thuan_nam_19", "TN19_20260411_35M_T4"),
    "35M_T5": ("farm_01_thuan_nam_19", "TN19_20260411_35M_T5"),
    "TNDL_GSD3_T2": ("farm_03_thuan_nam_duc_long", "TNDL_20260411_35M_T2"),
    "TT_GSD3_T1": ("farm_04_thien_tan", "TT_20260412_39M_T1"),
}
STATUS = {"dung": "confirmed", "sai": "rejected", None: "accepted_unreviewed"}
TR = Transformer.from_crs(4326, 32649, always_xy=True)


def to_en(q, w):
    s, c0, r0 = w["s"], w["c0"], w["r0"]
    pts = []
    for i in range(0, 8, 2):
        ox, oy = q[i] / s + c0, q[i + 1] / s + r0
        lon = w["lon0"] + (ox - c0) / w["gw"] * (w["lon1"] - w["lon0"])
        lat = w["lat0"] + (oy - r0) / w["gh"] * (w["lat1"] - w["lat0"])
        pts.append(TR.transform(lon, lat))
    return pts


def main():
    src, ds = Path(sys.argv[1]), Path(sys.argv[2])
    inv = pd.read_csv(ds / "metadata/inventory/image_inventory.csv", usecols=["farm", "flight_id", "file", "sensor", "seq"])
    inv = inv[inv.sensor == "T"].copy()
    inv["seq"] = inv.seq.astype(int)
    seqfile = {(r.flight_id, r.seq): r.file for r in inv.itertuples()}
    boxes, frames = {}, {}
    for folder, (farm, fid) in FLIGHTS.items():
        j = json.loads((src / folder / "nhan_tay.json").read_text(encoding="utf-8"))
        w = j["cua_so"]
        for h in j["hop"]:
            e = to_en(h["q_px_nen"], w)
            cx, cy = sum(p[0] for p in e) / 4, sum(p[1] for p in e) / 4
            edge1 = ((e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2) ** 0.5
            edge2 = ((e[2][0] - e[1][0]) ** 2 + (e[2][1] - e[1][1]) ** 2) ** 0.5
            cls_final = h["lop_nguoi"] or h["lop_model"]
            st = STATUS[h["tt"]]
            row = dict(
                farm_dir=farm, flight_id=fid, box_id=f"{fid}_{h['id']}", module_code=h["ma_tam"],
                class_final=cls_final,
                self_drawn=bool(h["tu_ve"]),
                shape_edited=bool(h["sua_hinh"]), in_eval_zone=bool(h["trong_vung"]),
                n_frames=h["n_luot"],
                side_a_m=round(edge1, 3), side_b_m=round(edge2, 3), centre_E=round(cx, 3), centre_N=round(cy, 3),
            )
            for k, (x, y) in enumerate(e, 1):
                row[f"E{k}"], row[f"N{k}"] = round(x, 3), round(y, 3)
            if st == "rejected":  # rejected boxes are not released
                continue
            boxes.setdefault(farm, []).append(row)
            for fr, x, y, _conf, _cls in (h["luot"] or []):  # self-drawn boxes have no frame list
                frames.setdefault(farm, []).append(dict(
                    flight_id=fid, box_id=row["box_id"], seq=int(fr), thermal_file=seqfile.get((fid, int(fr)), ""),
                    x_px=x, y_px=y))
    for farm, rows in boxes.items():
        out = ds / farm / "annotations" / "thermal_anomaly_obb"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "thermal_boxes.csv", index=False)
        pd.DataFrame(frames[farm]).to_csv(out / "thermal_box_frame_detections.csv", index=False)
        print(farm, len(rows), "boxes", len(frames[farm]), "frame detections")


if __name__ == "__main__":
    main()
