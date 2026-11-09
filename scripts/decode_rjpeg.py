#!/usr/bin/env python
"""Decode DJI radiometric R-JPEG images with the official DJI Thermal SDK (libdirp) via ctypes.

The SDK is NOT redistributable and is never copied into this repository; pass its location
with --sdk (the folder that contains libdirp.dll, e.g.
<sdk>/tsdk-core/lib/windows/release_x64). Both the SDK root and the lib folder are accepted.

Output temperatures are *apparent temperatures* returned by dirp_measure_ex() using the
measurement parameters embedded in each R-JPEG (emissivity, reflected temperature, object
distance, humidity, ambient). They are not calibrated surface temperatures.

Modes
  inventory : decode every thermal image listed in image_inventory.csv (sensor == T)
              -> thermal_parameters.csv
  files     : decode explicit files / folders (used for the SDK sample images)
  export    : save decoded matrices of listed files as float16 .npy
  exportall : resumable float16 .npz export of ALL inventory thermal images
              -> dataset/derived/thermal_temperature_celsius/<flight_id>/<stem>.npz + temperature_index.csv
  summary   : per-flight / per-hour summaries from the CSVs (SDK not needed)
  samples   : 12 example frames (.npy float16) + 4-panel PNG per farm

Examples (see dataset/metadata/thermal/README.md)
  python -X utf8 code/scripts/decode_rjpeg.py --sdk <SDK> files <SDK>/dataset/H20T --print
  python -X utf8 code/scripts/decode_rjpeg.py --sdk <SDK> inventory --workers 4
"""
import argparse
import csv
import ctypes as C
import os
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- SDK binding
class Resolution(C.Structure):
    _pack_ = 1
    _fields_ = [("width", C.c_int32), ("height", C.c_int32)]


class MeasParams(C.Structure):
    _pack_ = 1
    _fields_ = [("distance", C.c_float), ("humidity", C.c_float), ("emissivity", C.c_float),
                ("reflection", C.c_float), ("ambient_temp", C.c_float)]


class ApiVersion(C.Structure):
    _pack_ = 1
    _fields_ = [("api", C.c_uint32), ("magic", C.c_char * 8)]


class RjpegVersion(C.Structure):
    _pack_ = 1
    _fields_ = [("rjpeg", C.c_uint32), ("header", C.c_uint32), ("curve", C.c_uint32)]


class Range2(C.Structure):
    _fields_ = [("min", C.c_float), ("max", C.c_float)]


class MeasRange(C.Structure):
    _pack_ = 1
    _fields_ = [("distance", Range2), ("humidity", Range2), ("emissivity", Range2),
                ("reflection", Range2), ("ambient_temp", Range2)]


ERRORS = {0: "SUCCESS", -1: "MALLOC", -2: "POINTER_NULL", -3: "INVALID_PARAMS", -4: "INVALID_RAW",
          -5: "INVALID_HEADER", -6: "INVALID_CURVE", -7: "RJPEG_PARSE", -8: "SIZE",
          -9: "INVALID_HANDLE", -10: "FORMAT_INPUT", -11: "FORMAT_OUTPUT", -12: "UNSUPPORTED_FUNC",
          -13: "NOT_READY", -14: "ACTIVATION", -15: "INVALID_INI", -16: "INVALID_SUB_DLL",
          -32: "ADVANCED", -64: "SUPER_MODE"}


def err_name(code):
    return f"{code}:{ERRORS.get(code, 'UNKNOWN')}"


class Dirp:
    def __init__(self, sdk):
        sdk = Path(sdk).resolve()
        cands = [sdk, sdk / "tsdk-core" / "lib" / "windows" / "release_x64",
                 sdk / "tsdk-core" / "lib" / "linux" / "release_x64"]
        libdir = next((c for c in cands if (c / "libdirp.dll").exists() or (c / "libdirp.so").exists()), None)
        if libdir is None:
            raise FileNotFoundError(f"libdirp not found under {sdk}")
        self.libdir = libdir
        if sys.platform == "win32":
            os.add_dll_directory(str(libdir))
            os.environ["PATH"] = str(libdir) + os.pathsep + os.environ.get("PATH", "")
            lib = C.CDLL(str(libdir / "libdirp.dll"))
        else:
            lib = C.CDLL(str(libdir / "libdirp.so"))
        H = C.c_void_p
        lib.dirp_create_from_rjpeg.argtypes = [C.c_char_p, C.c_int32, C.POINTER(H)]
        lib.dirp_destroy.argtypes = [H]
        lib.dirp_get_api_version.argtypes = [H, C.POINTER(ApiVersion)]
        lib.dirp_get_rjpeg_version.argtypes = [H, C.POINTER(RjpegVersion)]
        lib.dirp_get_rjpeg_resolution.argtypes = [H, C.POINTER(Resolution)]
        lib.dirp_get_original_raw.argtypes = [H, C.c_void_p, C.c_int32]
        lib.dirp_measure_ex.argtypes = [H, C.c_void_p, C.c_int32]
        lib.dirp_get_measurement_params.argtypes = [H, C.POINTER(MeasParams)]
        lib.dirp_get_measurement_params_range.argtypes = [H, C.POINTER(MeasRange)]
        lib.dirp_get_infared_density_filter_installed.argtypes = [H]
        lib.dirp_get_infared_density_filter_installed.restype = C.c_uint32
        for n in ("create_from_rjpeg", "destroy", "get_api_version", "get_rjpeg_version",
                  "get_rjpeg_resolution", "get_original_raw", "measure_ex", "get_measurement_params",
                  "get_measurement_params_range"):
            getattr(lib, "dirp_" + n).restype = C.c_int32
        self.lib = lib

    def decode(self, path, want_matrix=False):
        """Returns (record dict, temperature matrix float32 or None). Never raises for SDK errors."""
        lib = self.lib
        rec = {"decode_ok": False, "error": ""}
        data = Path(path).read_bytes()
        h = C.c_void_p()
        rc = lib.dirp_create_from_rjpeg(data, len(data), C.byref(h))
        if rc != 0:
            rec["error"] = "create_from_rjpeg " + err_name(rc)
            return rec, None
        try:
            res = Resolution()
            rc = lib.dirp_get_rjpeg_resolution(h, C.byref(res))
            if rc != 0:
                rec["error"] = "get_rjpeg_resolution " + err_name(rc)
                return rec, None
            rec["width"], rec["height"] = res.width, res.height
            av, rv = ApiVersion(), RjpegVersion()
            if lib.dirp_get_api_version(h, C.byref(av)) == 0:
                rec["sdk_version"] = f"libdirp_api{av.api}"
            if lib.dirp_get_rjpeg_version(h, C.byref(rv)) == 0:
                rec["rjpeg_version"], rec["header_version"], rec["curve_version"] = rv.rjpeg, rv.header, rv.curve
            mp = MeasParams()
            rc = lib.dirp_get_measurement_params(h, C.byref(mp))
            if rc == 0:
                rec.update(emissivity=mp.emissivity, reflected_temp_c=mp.reflection,
                           distance_m=mp.distance, humidity_pct=mp.humidity,
                           ambient_temp_c=mp.ambient_temp)
            else:
                rec["error"] = "get_measurement_params " + err_name(rc)
            rng = MeasRange()
            if lib.dirp_get_measurement_params_range(h, C.byref(rng)) == 0:
                rec["distance_range"] = f"{rng.distance.min:g}-{rng.distance.max:g}"
            rec["ir_density_filter"] = int(lib.dirp_get_infared_density_filter_installed(h))
            n = res.width * res.height
            raw = np.empty(n, np.uint16)
            if lib.dirp_get_original_raw(h, raw.ctypes.data, n * 2) == 0:
                rec["raw_min"], rec["raw_max"] = int(raw.min()), int(raw.max())
            temp = np.empty(n, np.float32)
            rc = lib.dirp_measure_ex(h, temp.ctypes.data, n * 4)
            if rc != 0:
                rec["error"] = (rec["error"] + "; " if rec["error"] else "") + "measure_ex " + err_name(rc)
                return rec, None
            temp = temp.reshape(res.height, res.width)
            fin = np.isfinite(temp)
            rec["n_nonfinite"] = int((~fin).sum())
            t = temp[fin]
            if t.size == 0:
                rec["error"] = "all non-finite"
                return rec, None
            p = np.percentile(t, [1, 50, 99])
            rec.update(temp_min_c=float(t.min()), temp_max_c=float(t.max()), temp_mean_c=float(t.mean()),
                       temp_median_c=float(p[1]), temp_p01=float(p[0]), temp_p99=float(p[2]))
            rec["decode_ok"] = True
            return rec, (temp if want_matrix else None)
        finally:
            lib.dirp_destroy(h)


# ----------------------------------------------------------------------------- workers
_D = None


def _init(sdk):
    global _D
    _D = Dirp(sdk)


def _work(job):
    farm, flight_id, path = job
    t0 = time.time()
    try:
        rec, _ = _D.decode(path)
    except Exception as e:  # keep going on any failure
        rec = {"decode_ok": False, "error": f"exception {type(e).__name__}: {e}"}
    rec.update(farm=farm, flight_id=flight_id, file=Path(path).name, _t=time.time() - t0)
    return rec


COLS = ["farm", "flight_id", "file", "decode_ok", "error", "width", "height", "sdk_version",
        "rjpeg_version", "header_version", "curve_version", "ir_density_filter",
        "emissivity", "reflected_temp_c", "distance_m", "humidity_pct", "ambient_temp_c",
        "distance_range", "raw_min", "raw_max", "n_nonfinite",
        "temp_min_c", "temp_max_c", "temp_mean_c", "temp_median_c", "temp_p01", "temp_p99"]


def find_images(root, inv_csv):
    """Map (farm, flight_id, file) -> path from the inventory (sensor == T)."""
    import pandas as pd
    inv = pd.read_csv(inv_csv)
    inv = inv[inv.sensor == "T"]
    index = {}
    for farm_dir in sorted(Path(root).glob("farm_*")):
        for p in farm_dir.glob("raw_images/*/*/*_T.JPG"):
            index[(p.parent.name, p.name)] = p
    jobs, missing = [], []
    for r in inv.itertuples():
        p = index.get((r.flight_id, r.file))
        (jobs.append((r.farm, r.flight_id, str(p))) if p else missing.append((r.farm, r.flight_id, r.file)))
    return jobs, missing


def cmd_inventory(a):
    from multiprocessing import Pool
    root = Path(a.dataset)
    inv = a.inventory or root / "metadata" / "inventory" / "image_inventory.csv"
    jobs, missing = find_images(root, inv)
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} thermal images to decode, {len(missing)} inventory rows without a file on disk")
    out = Path(a.out or root / "metadata" / "thermal" / "thermal_parameters.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.sdk,)) as pool:
        for i, rec in enumerate(pool.imap(_work, jobs, chunksize=8), 1):
            rows.append(rec)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    for farm, fl, f in missing:
        rows.append(dict(farm=farm, flight_id=fl, file=f, decode_ok=False, error="file missing on disk"))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(r)
    ok = sum(r["decode_ok"] for r in rows)
    print(f"done in {time.time() - t0:.0f}s: ok={ok} failed={len(rows) - ok} -> {out}")


def cmd_files(a):
    d = Dirp(a.sdk)
    files = []
    for p in a.paths:
        p = Path(p)
        files += sorted(x for x in p.iterdir() if x.suffix.lower() == ".jpg") if p.is_dir() else [p]
    for f in files:
        rec, _ = d.decode(f)
        if rec["decode_ok"]:
            print(f"{f.parent.name}/{f.name}: {rec['width']}x{rec['height']} T[{rec['temp_min_c']:.1f},{rec['temp_max_c']:.1f}] "
                  f"mean {rec['temp_mean_c']:.1f} p01-p99 [{rec['temp_p01']:.1f},{rec['temp_p99']:.1f}] "
                  f"E={rec['emissivity']:.2f} Trefl={rec['reflected_temp_c']:.1f} dist={rec['distance_m']:.1f} "
                  f"RH={rec['humidity_pct']:.0f} Tamb={rec['ambient_temp_c']:.1f} sdk={rec.get('sdk_version')}")
        else:
            print(f"{f.parent.name}/{f.name}: FAILED {rec['error']}")


def cmd_export(a):
    d = Dirp(a.sdk)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.paths:
        rec, m = d.decode(f, want_matrix=True)
        if m is None:
            print("FAILED", f, rec["error"])
            continue
        np.save(out / (Path(f).stem + "_temp_c_f16.npy"), m.astype(np.float16))
        print("saved", Path(f).name, m.shape)


INDEX_COLS = ["image_id", "farm", "flight_id", "rel_path", "dtype", "shape", "min_c", "p01_c", "median_c",
              "p99_c", "max_c", "emissivity", "reflected_temp_c", "distance_m", "humidity_pct",
              "ambient_temp_c", "sdk_version", "decode_status"]


def _export_work(job):
    farm, flight_id, path, outroot = job
    p = Path(path)
    rel = f"{flight_id}/{p.stem}.npz"
    row = dict(image_id=p.stem, farm=farm, flight_id=flight_id, rel_path=rel, dtype="float16")
    try:
        rec, m = _D.decode(path, want_matrix=True)
        row.update({k: rec.get(k, "") for k in ("emissivity", "reflected_temp_c", "distance_m",
                                                 "humidity_pct", "ambient_temp_c", "sdk_version")})
        if m is None:
            row["decode_status"] = "FAILED " + rec.get("error", "")
            return row
        h = m.astype(np.float16)
        s = h.astype(np.float32)   # statistics describe the STORED (float16) values
        if not np.isfinite(s).all():
            row["decode_status"] = "WARN non-finite pixels in float16"
        q = np.nanpercentile(s, [1, 50, 99])
        row.update(shape=f"{s.shape[0]}x{s.shape[1]}", min_c=float(np.nanmin(s)), p01_c=float(q[0]),
                   median_c=float(q[1]), p99_c=float(q[2]), max_c=float(np.nanmax(s)))
        dst = Path(outroot) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp.npz")
        np.savez_compressed(tmp, temp_c=h)
        os.replace(tmp, dst)
        row.setdefault("decode_status", "OK")
    except Exception as e:
        row["decode_status"] = f"FAILED exception {type(e).__name__}: {e}"
    return row


def cmd_exportall(a):
    """Resumable export of every inventory thermal image as compressed float16 .npz
    (<outdir>/<flight_id>/<image_stem>.npz, key 'temp_c', shape 512x640) + temperature_index.csv."""
    from multiprocessing import Pool
    root = Path(a.dataset)
    inv = a.inventory or root / "metadata" / "inventory" / "image_inventory.csv"
    jobs, missing = find_images(root, inv)
    out = Path(a.outdir or root / "derived" / "thermal_temperature_celsius")
    out.mkdir(parents=True, exist_ok=True)
    idx = out / "temperature_index.csv"
    done = set()
    if idx.exists():
        with open(idx, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["decode_status"].startswith("OK") and (out / r["rel_path"]).exists():
                    done.add((r["flight_id"], r["image_id"]))
    todo = [(f, fl, p, str(out)) for f, fl, p in jobs if (fl, Path(p).stem) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(jobs)} thermal images, {len(done)} already done, {len(todo)} to export, "
          f"{len(missing)} missing on disk", flush=True)
    new = not idx.exists()
    t0 = time.time()
    with open(idx, "a", newline="", encoding="utf-8") as fh, \
            Pool(min(a.workers, 4), initializer=_init, initargs=(a.sdk,)) as pool:
        w = csv.DictWriter(fh, INDEX_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap(_export_work, todo, chunksize=8), 1):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
            if i % 200 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        for farm, fl, f in missing:
            w.writerow(dict(image_id=Path(f).stem, farm=farm, flight_id=fl, decode_status="FAILED file missing on disk"))
    print(f"done in {time.time() - t0:.0f}s -> {idx}", flush=True)


def cmd_summary(a):
    """Per-flight and per-hour summaries from thermal_parameters.csv + image_inventory.csv (no SDK needed)."""
    import pandas as pd
    root = Path(a.dataset)
    tdir = root / "metadata" / "thermal"
    par = pd.read_csv(tdir / "thermal_parameters.csv")
    inv = pd.read_csv(root / "metadata" / "inventory" / "image_inventory.csv")
    inv = inv[inv.sensor == "T"][["flight_id", "file", "timestamp", "GimbalPitchDegree", "RelativeAltitude",
                                  "LRFTargetDistance"]]
    d = par.merge(inv, on=["flight_id", "file"], how="left")
    ts = pd.to_datetime(d.timestamp.astype("Int64").astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    d["local_time"] = ts
    d["hour"] = ts.dt.hour
    rows = []
    pcols = ["emissivity", "reflected_temp_c", "distance_m", "humidity_pct", "ambient_temp_c"]
    for (farm, fl), g in d.groupby(["farm", "flight_id"], sort=False):
        ok = g[g.decode_ok.astype(bool)]
        r = dict(farm=farm, flight_id=fl, n_images=len(g), n_ok=len(ok), decode_rate=round(len(ok) / len(g), 4),
                 first_image_local=g.local_time.min(), last_image_local=g.local_time.max(),
                 width=ok.width.mode().iat[0] if len(ok) else "", height=ok.height.mode().iat[0] if len(ok) else "")
        for c in pcols:
            vals = sorted(ok[c].dropna().round(3).unique())
            r[c + "_n_unique"] = len(vals)
            r[c + "_values"] = "|".join(f"{v:g}" for v in vals[:5])
        r["rel_alt_median_m"] = round(g.RelativeAltitude.median(), 2)
        r["lrf_target_distance_median_m"] = round(g.LRFTargetDistance.replace(0, float("nan")).median(), 2)
        for c in ["temp_min_c", "temp_max_c", "temp_mean_c", "temp_median_c", "temp_p01", "temp_p99"]:
            r[c + "_flight_median"] = round(ok[c].median(), 2)
        r["temp_min_c_flight_min"] = round(ok.temp_min_c.min(), 2)
        r["temp_max_c_flight_max"] = round(ok.temp_max_c.max(), 2)
        r["temp_max_c_p99_over_images"] = round(ok.temp_max_c.quantile(0.99), 2)
        r["errors"] = ";".join(sorted(g.error.dropna().astype(str).unique()))
        rows.append(r)
    pd.DataFrame(rows).to_csv(tdir / "decode_summary.csv", index=False)
    ok = d[d.decode_ok.astype(bool)]
    h = ok.groupby(["farm", "hour"]).agg(
        n_images=("file", "size"), mean_of_image_mean_c=("temp_mean_c", "mean"),
        median_of_image_median_c=("temp_median_c", "median"), median_of_image_max_c=("temp_max_c", "median"),
        p99_of_image_max_c=("temp_max_c", lambda x: x.quantile(0.99)), min_of_image_min_c=("temp_min_c", "min"),
        max_of_image_max_c=("temp_max_c", "max")).round(2).reset_index()
    h.to_csv(tdir / "decode_summary_by_hour.csv", index=False)
    f = ok.groupby("farm").agg(
        n_ok=("file", "size"), min_c=("temp_min_c", "min"), p01_of_min=("temp_min_c", lambda x: x.quantile(.01)),
        median_of_median=("temp_median_c", "median"), mean_of_mean=("temp_mean_c", "mean"),
        median_of_max=("temp_max_c", "median"), p99_of_max=("temp_max_c", lambda x: x.quantile(.99)),
        max_c=("temp_max_c", "max")).round(2)
    print(f.to_string())
    print(h.to_string())


def cmd_samples(a):
    """12 example frames (3 per farm): best hot-spot-looking nadir frame, a second one from another
    flight, and a typical frame. Saves float16 .npy + one 4-panel PNG per farm."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(a.dataset)
    par = pd.read_csv(root / "metadata" / "thermal" / "thermal_parameters.csv")
    inv = pd.read_csv(root / "metadata" / "inventory" / "image_inventory.csv")
    inv = inv[inv.sensor == "T"][["flight_id", "file", "GimbalPitchDegree"]]
    d = par.merge(inv, on=["flight_id", "file"])
    d = d[d.decode_ok.astype(bool) & (d.GimbalPitchDegree < -85)].copy()
    d["spread"] = d.temp_max_c - d.temp_p99     # isolated extremes above the bulk of the image
    d["excess"] = d.temp_p99 - d.temp_median_c
    d = d[d.temp_max_c < 120]
    index = {(p.parent.name, p.name): p for p in root.glob("farm_*/raw_images/*/*/*_T.JPG")}
    dec = Dirp(a.sdk)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for farm, g in d.groupby("farm"):
        h1 = g.sort_values("spread", ascending=False).iloc[0]
        h2 = g[g.flight_id != h1.flight_id].sort_values("spread", ascending=False).iloc[0]
        typ = g.iloc[(g.temp_mean_c - g.temp_mean_c.median()).abs().argsort().iloc[0]]
        picks = [("hotspot_1", h1), ("hotspot_2", h2), ("typical", typ)]
        fig, ax = plt.subplots(1, 4, figsize=(22, 5))
        for k, (tag, r) in enumerate(picks):
            path = index[(r.flight_id, r.file)]
            _, m = dec.decode(path, want_matrix=True)
            np.save(out / f"{farm}_{tag}_{Path(r.file).stem}_temp_c_f16.npy", m.astype(np.float16))
            im = ax[k].imshow(m, cmap="inferno")
            ax[k].set_title("\n".join([f"{farm} {tag}", r.flight_id, r.file,
                                       f"{m.min():.1f}..{m.max():.1f} C (apparent)"]), fontsize=8)
            ax[k].axis("off")
            fig.colorbar(im, ax=ax[k], fraction=0.046, label="apparent T [C]")
            ax[3].hist(m.ravel(), bins=120, histtype="step", label=tag, log=True)
        ax[3].set_xlabel("apparent temperature [C] (SDK, embedded params)")
        ax[3].set_ylabel("pixels (log)")
        ax[3].legend()
        ax[3].set_title("histograms")
        fig.tight_layout()
        fig.savefig(out / f"{farm}_samples_4panel.png", dpi=110)
        plt.close(fig)
        print(farm, [(t, r.flight_id, r.file) for t, r in picks])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk", required=True, help="DJI Thermal SDK root or folder containing libdirp.dll")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inventory")
    s.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "dataset"))
    s.add_argument("--inventory")
    s.add_argument("--out")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_inventory)
    s = sub.add_parser("files")
    s.add_argument("paths", nargs="+")
    s.add_argument("--print", action="store_true")
    s.set_defaults(fn=cmd_files)
    s = sub.add_parser("export")
    s.add_argument("outdir")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("exportall")
    s.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "dataset"))
    s.add_argument("--inventory")
    s.add_argument("--outdir")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_exportall)
    s = sub.add_parser("summary")
    s.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "dataset"))
    s.set_defaults(fn=cmd_summary)
    s = sub.add_parser("samples")
    s.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "dataset"))
    s.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / ".local" / "tmp" / "thermal_samples"))
    s.set_defaults(fn=cmd_samples)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
