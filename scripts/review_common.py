"""Shared helpers for the second-pass annotation review tools (paths, loaders, class colours)."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # solar_uav_project
DATASET = ROOT / "dataset"
TEMP_DIR = DATASET / "derived" / "thermal_temperature_celsius"
OUT = ROOT / ".local" / "annotation_review"
W, H = 640, 512
CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]
CLASS_RGB = {"SingleHotSpot": (0, 255, 0), "MultiHotSpot": (0, 200, 255),
             "Bypass": (255, 0, 255), "ModuleOpenCircuit": (255, 255, 0)}
SHORT = {"SingleHotSpot": "S", "MultiHotSpot": "M", "Bypass": "B", "ModuleOpenCircuit": "MOC"}


def farm_dirs():
    return sorted(p for p in DATASET.glob("farm_*") if p.is_dir())


def load_boxes():
    dfs = []
    for fd in farm_dirs():
        f = fd / "annotations" / "thermal_anomaly_obb" / "thermal_frame_boxes.csv"
        d = pd.read_csv(f)
        d["farm_dir"] = fd.name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def load_frames():
    """All 9,839 thermal frames: inventory (sensor T) + farm dir + jpg path + npz path."""
    inv = pd.read_csv(DATASET / "metadata" / "inventory" / "image_inventory.csv", low_memory=False)
    inv = inv[inv.sensor == "T"].copy()
    jpg = {}
    for fd in farm_dirs():
        for p in (fd / "raw_images").glob("*/*/*_T.JPG"):
            jpg[(p.parent.name, p.name)] = p
    inv["farm_dir"] = [str(jpg[(a, b)].relative_to(DATASET).parts[0]) if (a, b) in jpg else "" for a, b in zip(inv.flight_id, inv.file)]
    inv["jpg_path"] = [jpg.get((a, b)) for a, b in zip(inv.flight_id, inv.file)]
    inv["npz_path"] = [TEMP_DIR / a / (b[:-4] + ".npz") for a, b in zip(inv.flight_id, inv.file)]
    inv["frame_id"] = inv.flight_id + "/" + inv.file.str[:-4]
    inv["altitude_m"] = inv.flight_id.str.extract(r"_(\d+)M_")[0].astype(float)
    return inv.reset_index(drop=True)


def load_temp(npz_path):
    return np.load(npz_path)["temp_c"].astype(np.float32)


def box_polygon(row, clipped=True):
    """Polygon [(x,y)*4] in pixel coords of the 640x512 grid; clipped coords when the box leaves the frame."""
    if clipped and int(row["clipped_flag"]) == 1:
        xs = [row[f"x{i}_clip"] * W for i in range(1, 5)]
        ys = [row[f"y{i}_clip"] * H for i in range(1, 5)]
    else:
        xs = [row[f"x{i}_px"] for i in range(1, 5)]
        ys = [row[f"y{i}_px"] for i in range(1, 5)]
    return list(zip(xs, ys))
