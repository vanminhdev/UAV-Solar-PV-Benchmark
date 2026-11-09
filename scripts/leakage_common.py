#!/usr/bin/env python
"""Shared helpers of the leakage check (leakage_mc.py, leakage_images.py): units (whole frames + crop windows) of a split table, their ground
corners, boxes, measured footprint-error distributions and the per-flight registration transforms."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_footprints as bf  # noqa: E402
import make_frame_label_split as mf  # noqa: E402

ROOT = bf.ROOT
DS = ROOT / "dataset"
SPLITS = DS / "metadata/splits"
OUTD = SPLITS / "leakage_check"
SPL = ["train", "val", "test"]
PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]

# split tables: name -> (directory, crops file, crop-box file)
VARIANTS = {
    "C_margin2.3": (SPLITS, "frame_label_split_crops.csv", "frame_label_split_crop_boxes.csv"),                      # main split
    "C_margin1.5": (SPLITS / "variant_C_margin1p5", "frame_label_split_crops.csv", "frame_label_split_crop_boxes.csv"),
    "C_margin3.0": (SPLITS / "variant_C_margin3p0", "frame_label_split_crops.csv", "frame_label_split_crop_boxes.csv"),
    "C_margin0.0": (SPLITS / "variant_C_margin0p0", "frame_label_split_crops.csv", "frame_label_split_crop_boxes.csv"),
    "A_strip_buffer": (SPLITS / "_strip_buffer_strict", "frame_label_split_moc_crops.csv", "frame_label_split_moc_crop_boxes.csv"),
}


def corners_of(P, rect):
    """ground corners (4 x 2) of a pixel rectangle (x0, y0, x1, y1) of a 640 x 512 frame with footprint corners P (BL, BR, TR, TL)."""
    x0, y0, x1, y1 = rect
    return mf.g_map(P, np.array([x0, x1, x1, x0], float), np.array([y0, y0, y1, y1], float))


_FRAMES = None
_FB = None


def frames():
    global _FRAMES
    if _FRAMES is None:
        F = mf.load_frames()
        _FRAMES = {(r.flight_id, r.thermal_file): r for r in F.itertuples()}
    return _FRAMES


def frame_boxes():
    global _FB
    if _FB is None:
        _FB = mf.load_all_boxes()
    return _FB


def load_units(variant):
    """-> dict(units DataFrame, boxes DataFrame). units: uid, farm, flight_id, thermal_file, split, kind, rect, corners (4,2), alt;
    boxes: unit index, E, N (nominal ground centre of the box)."""
    d, cf, cbf = VARIANTS[variant]
    S = pd.read_csv(d / "frame_label_split.csv")
    crops = pd.read_csv(d / cf)
    cb = pd.read_csv(d / cbf)
    Fr, FB = frames(), frame_boxes()
    rows, brow = [], []

    def add_boxes(ui, key, P, uids=None):
        g = FB.get(key)
        if g is None:
            return
        for _, b in g.iterrows():
            if uids is not None and b.box_uid not in uids:
                continue
            x = np.mean([b[f"x{j}_clip"] for j in range(1, 5)]) * 640
            y = np.mean([b[f"y{j}_clip"] for j in range(1, 5)]) * 512
            e, n = mf.g_map(P, x, y)
            brow.append((ui, float(e), float(n)))

    for r in S[S.split.isin(SPL)].itertuples():
        f = Fr[(r.flight_id, r.thermal_file)]
        P = np.asarray(f.geometry.exterior.coords)[:4]
        rows.append(dict(uid=f"{r.flight_id}|{r.thermal_file}", farm=r.farm, flight_id=r.flight_id, thermal_file=r.thermal_file, split=r.split,
                         kind="frame", rect=(0, 0, 640, 512), corners=P, alt=r.altitude_m))
        add_boxes(len(rows) - 1, (r.flight_id, r.thermal_file), P)
    cbg = {k: set(v.box_uid) for k, v in cb.groupby("crop_id")}
    for c in crops[crops.split.isin(SPL)].itertuples():
        f = Fr[(c.flight_id, c.thermal_file)]
        P = np.asarray(f.geometry.exterior.coords)[:4]
        rows.append(dict(uid=c.crop_id, farm=c.farm, flight_id=c.flight_id, thermal_file=c.thermal_file, split=c.split, kind="crop",
                         rect=(c.x0, c.y0, c.x1, c.y1), corners=corners_of(P, (c.x0, c.y0, c.x1, c.y1)), alt=f.altitude_m))
        add_boxes(len(rows) - 1, (c.flight_id, c.thermal_file), P, cbg.get(c.crop_id, set()))
    U = pd.DataFrame(rows)
    B = pd.DataFrame(brow, columns=["unit", "E", "N"])
    # parent frame index (crops move with their frame)
    key = {k: i for i, k in enumerate(sorted(set(zip(U.flight_id, U.thermal_file))))}
    U["frame_idx"] = [key[(a, b)] for a, b in zip(U.flight_id, U.thermal_file)]
    return U, B


def frame_centres(U):
    """per parent frame: centroid of the frame footprint (rotation centre)."""
    Fr = frames()
    keys = sorted(set(zip(U.flight_id, U.thermal_file)))
    return keys, np.array([[Fr[k].geometry.centroid.x, Fr[k].geometry.centroid.y] for k in keys])


def residual_pools():
    """measured nominal -> ortho residual vectors (projection_check, variant V0) per farm: arrays (n, 2) in m with the altitude of each row."""
    r = pd.read_csv(DS / "metadata/projection_check/projection_residuals.csv", usecols=["flight_id", "RelativeAltitude", "resE_V0_nominal", "resN_V0_nominal"])
    r["farm"] = r.flight_id.str.split("_").str[0]
    out = {}
    for farm, g in r.groupby("farm"):
        out[farm] = dict(res=g[["resE_V0_nominal", "resN_V0_nominal"]].values, alt=g.RelativeAltitude.values, flight=g.flight_id.values)
    return out, r


def registration():
    """per-flight similarity transform flight map -> base ortho (primary row) and the mean nominal residual per flight."""
    t = pd.read_csv(DS / "metadata/ortho_registration/ortho_registration_transforms.csv")
    t = t[(t.is_primary == 1) & t.tE.notna()].set_index("flight_id")
    _, r = residual_pools()
    mres = r.groupby("flight_id")[["resE_V0_nominal", "resN_V0_nominal"]].mean()
    return t, mres


def apply_transform(xy, tr):
    """E_ref = oE + tE + m11 (E - oE) + m12 (N - oN); N_ref = oN + tN + m21 (E - oE) + m22 (N - oN)."""
    dx, dy = xy[..., 0] - tr.origin_E, xy[..., 1] - tr.origin_N
    return np.stack([tr.origin_E + tr.tE + tr.m11 * dx + tr.m12 * dy, tr.origin_N + tr.tN + tr.m21 * dx + tr.m22 * dy], -1)
