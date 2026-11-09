#!/usr/bin/env python
"""Design the inter-annotator agreement sample and build the blind annotator packages.

Local-only tool (outputs go to <project>/.local/annotation_agreement/, never into dataset/).

Sample design (seeded, reproducible)
  * ~200 thermal frames, farm quotas TN19 90 / TT 40 / BIM2 40 / TNDL 30 (CLI overridable).
  * Per farm ~70% frames from the labelled positives, ~30% negatives. Negatives: all frames
    without a box (one pool, no distinction between explicitly reviewed and other frames),
    drawn uniformly at random within the farm (stratified by flight, sqrt allocation).
  * Inside each farm and each group the frames are spread over flights with allocation
    proportional to sqrt(pool size) (largest remainder), so small flights are not ignored.
  * Frames inside a flight are drawn at random with a minimum index gap (default 3 frames in
    time order); the gap is relaxed (2, 1) only when a flight cannot supply enough frames.
  * Class coverage: MOC frames (all in one patch, flight 42M_T2) are picked first, evenly spread
    along the patch (one random frame per time bin, gap >= min gap); then Multi/Bypass/Single
    minima (default 25 frames each) are enforced by swapping frames inside the same flight/farm.
  * Reference labels = the released YOLO-OBB label files (annotator A1 = Chi).

Outputs
  <out>/_key/sample_key.csv            frame table incl. reference boxes (NEVER send)
  <out>/_key/chi_labels/<frame_uid>.txt  copy of A1 labels
  <out>/SAMPLE_DESIGN.md               counts + parameters (local report)
  <out>/package_A2/, package_A3/ (+ .zip)  blind packages (with --build-packages)

Usage:  python -X utf8 code/scripts/make_agreement_sample.py [--build-packages]
"""
import argparse, glob, hashlib, io, os, shutil, sys, zipfile
import numpy as np
import pandas as pd

CLASSES = ["SingleHotSpot", "MultiHotSpot", "Bypass", "ModuleOpenCircuit"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FARM_DIRS = {"TN19": "farm_01_thuan_nam_19", "BIM2": "farm_02_bim_2",
             "TNDL": "farm_03_thuan_nam_duc_long", "TT": "farm_04_thien_tan"}


# ----------------------------------------------------------------------------- loading
def load_universe(root):
    inv = pd.read_csv(f"{root}/dataset/metadata/inventory/image_inventory.csv")
    t = inv[inv.sensor == "T"].copy()
    t = t.sort_values(["flight_id", "timestamp", "file"]).reset_index(drop=True)
    t["rank"] = t.groupby("flight_id").cumcount()
    fl = pd.read_csv(f"{root}/dataset/metadata/inventory/image_flags.csv")
    fl = fl[fl.sensor == "T"][["flight_id", "file", "flag_outside_ortho", "flag_altitude_outlier",
                               "flag_tilt_gt2deg", "flag_thermal_extreme"]]
    t = t.merge(fl, on=["flight_id", "file"], how="left")
    t["bad_flag"] = t[["flag_outside_ortho", "flag_altitude_outlier", "flag_tilt_gt2deg",
                       "flag_thermal_extreme"]].fillna(0).sum(axis=1) > 0
    rev = []
    for f, d in FARM_DIRS.items():
        r = pd.read_csv(f"{root}/dataset/{d}/annotations/thermal_anomaly_obb/frames_reviewed.csv")
        r["label_dir"] = f"{root}/dataset/{d}/annotations/thermal_anomaly_obb/labels_yolo_obb"
        rev.append(r)
    rev = pd.concat(rev)[["flight_id", "thermal_file", "status", "label_dir"]]
    t = t.merge(rev, left_on=["flight_id", "file"], right_on=["flight_id", "thermal_file"], how="left")
    t["status"] = np.where(t.status == "positive", "positive", "no_box")
    cnt = np.zeros((len(t), 4), int)
    for i, r in enumerate(t.itertuples()):
        if r.status == "positive":
            p = f"{r.label_dir}/{r.flight_id}/{os.path.splitext(r.file)[0]}.txt"
            for ln in open(p, encoding="utf8").read().split("\n"):
                if ln.strip():
                    cnt[i, int(float(ln.split()[0]))] += 1
    for k, c in enumerate(CLASSES):
        t[c] = cnt[:, k]
    t["n_boxes"] = cnt.sum(axis=1)
    t["altitude_m"] = t.RelativeAltitude.round(1)
    return t


# ----------------------------------------------------------------------------- sampling
def alloc_sqrt(sizes, n, forced=None):
    """Allocate n over groups proportional to sqrt(size), capped by size, largest remainder."""
    forced = forced or {}
    keys = list(sizes)
    w = {k: np.sqrt(sizes[k]) for k in keys}
    a = {k: 0 for k in keys}
    rem = n
    active = set(keys)
    while rem > 0 and active:
        tw = sum(w[k] for k in active)
        raw = {k: rem * w[k] / tw for k in active}
        fl = {k: min(int(raw[k]), sizes[k] - a[k]) for k in active}
        for k in active:
            a[k] += fl[k]
        rem = n - sum(a.values())
        order = sorted(active, key=lambda k: (-(raw[k] - int(raw[k])), k))
        for k in order:
            if rem <= 0:
                break
            if a[k] < sizes[k]:
                a[k] += 1
                rem -= 1
        active = {k for k in active if a[k] < sizes[k]}
        if rem > 0 and not active:
            break
    for k, v in forced.items():              # forced picks count inside their flight
        if a.get(k, 0) < v:
            a[k] = v
    while sum(a.values()) > n:               # trim the largest non-forced allocation
        k = max((k for k in keys if a[k] > forced.get(k, 0)), key=lambda k: (a[k], k))
        a[k] -= 1
    return a


def draw_in_flight(cands, k, rng, taken_ranks, gap):
    """Random draw of k rows from cands (DataFrame) keeping index gap >= gap to taken ranks."""
    chosen, relax = [], 0
    for g in [gap] + [x for x in (2, 1) if x < gap]:
        if len(chosen) >= k:
            break
        order = rng.permutation(len(cands))
        for i in order:
            if len(chosen) >= k:
                break
            r = cands.iloc[i]
            rk = int(r["rank"])
            if r.name in {c.name for c in chosen}:
                continue
            if all(abs(rk - x) >= g for x in taken_ranks):
                chosen.append(r)
                taken_ranks.append(rk)
        if len(chosen) < k:
            relax = max(relax, g - 1) if g > 1 else relax
        if g != gap and len(chosen) > 0:
            relax = g
    return chosen


def pick_group(pool, n, rng, gap, taken, forced_per_flight=None):
    """Stratified (by flight) draw of n frames from pool. taken: dict flight -> list of ranks."""
    forced_per_flight = forced_per_flight or {}
    sizes = pool.groupby("flight_id").size().to_dict()
    al = alloc_sqrt(sizes, n, {f: v for f, v in forced_per_flight.items() if f in sizes})
    out = []
    for f in sorted(al):
        need = al[f] - forced_per_flight.get(f, 0)
        if need <= 0:
            continue
        got = draw_in_flight(pool[pool.flight_id == f], need, rng, taken.setdefault(f, []), gap)
        out.extend(got)
    return out


def select(t, a):
    rng = np.random.default_rng(a.seed)
    quotas = {"TN19": a.n_tn19, "TT": a.n_tt, "BIM2": a.n_bim2, "TNDL": a.n_tndl}
    sel = {}                     # (flight_id,file) -> dict(reason)
    taken = {}

    def add(rows, reason):
        for r in rows:
            sel[(r.flight_id, r.file)] = reason
    # 1. MOC patch: evenly spread frames
    moc = t[t.ModuleOpenCircuit > 0].sort_values("rank")
    assert moc.flight_id.nunique() == 1, "MOC expected in one flight"
    mf = moc.flight_id.iloc[0]
    bins = np.array_split(np.arange(len(moc)), a.n_moc)
    moc_rows = []
    for b in bins:
        cand = moc.iloc[b].sample(frac=1, random_state=int(rng.integers(1 << 31)))
        for i in range(len(cand)):
            r = cand.iloc[i]
            if all(abs(int(r["rank"]) - x) >= a.gap for x in taken.get(mf, [])):
                moc_rows.append(r)
                taken.setdefault(mf, []).append(int(r["rank"]))
                break
        else:
            r = cand.iloc[0]
            moc_rows.append(r)
            taken.setdefault(mf, []).append(int(r["rank"]))
    add(moc_rows, "moc_patch")
    # 2. per farm positives and negatives
    for farm, q in quotas.items():
        nneg = int(round(q * a.neg_frac))
        npos = q - nneg
        tf = t[t.farm == farm]
        pos = tf[tf.status == "positive"]
        forced = {mf: len(moc_rows)} if farm == "TN19" else {}
        got = pick_group(pos[~pos.set_index(["flight_id", "file"]).index.isin(list(sel))], npos
                         - (len(moc_rows) if farm == "TN19" else 0), rng, a.gap, taken)
        add(got, "positive_random")
        # negatives: ALL frames without a box (reviewed negatives and the rest) form ONE pool;
        # uniform random within the farm, stratified by flight (sqrt allocation) like the positives
        neg = tf[tf.status == "no_box"]
        add(pick_group(neg, nneg, rng, a.gap, taken), "neg_random")
    # 3. class coverage repair by within-farm swaps. Requirements: global minimum per class plus a
    #    per-farm minimum (min(a.min_class_farm, pool//2)) so that per-farm/per-class statistics exist.
    t = t.set_index(["flight_id", "file"], drop=False)
    posall = t[t.status == "positive"]
    reqs = [(None, c, a.min_class) for c in CLASSES[:3]]
    for farm in quotas:
        for c in CLASSES[:3]:
            avail = int(((posall.farm == farm) & (posall[c] > 0)).sum())
            reqs.append((farm, c, min(a.min_class_farm, max(1, avail // 2))))
    log = []

    def counts(frame):
        return [int(((frame[c] > 0) & ((frame.farm == f) if f else True)).sum()) for f, c, _ in reqs]

    for _ in range(1000):
        cur = t.loc[list(sel)]
        cnt = counts(cur)
        short = [i for i, (f, c, m) in enumerate(reqs) if cnt[i] < m]
        if not short:
            break
        f, c, m = reqs[short[0]]
        cands = t[(t[c] > 0) & (t.status == "positive") & (~t.index.isin(list(sel)))]
        if f:
            cands = cands[cands.farm == f]
        victims = cur[(cur.status == "positive") & (~cur.index.map(lambda k: sel[k] == "moc_patch"))]
        done = False
        for _try in range(400):
            if cands.empty or victims.empty:
                break
            cand = cands.iloc[int(rng.integers(len(cands)))]
            g_eff = a.gap if _try < 200 else 1      # relax the gap only after 200 failed tries
            if not all(abs(int(cand["rank"]) - x) >= g_eff for x in taken.get(cand.flight_id, [])):
                continue
            vv = victims[victims.farm == cand.farm]
            same = vv[vv.flight_id == cand.flight_id]
            vv = same if len(same) else vv
            if vv.empty:
                continue
            v = vv.iloc[int(rng.integers(len(vv)))]
            new = pd.concat([cur.drop(index=(v.flight_id, v.file)), cand.to_frame().T.astype(cur.dtypes.to_dict(), errors="ignore")])
            nc = counts(new)
            if all(nc[i] >= reqs[i][2] or nc[i] >= cnt[i] for i in range(len(reqs))) and nc[short[0]] > cnt[short[0]]:
                del sel[(v.flight_id, v.file)]
                if int(v["rank"]) in taken.get(v.flight_id, []):
                    taken[v.flight_id].remove(int(v["rank"]))
                sel[(cand.flight_id, cand.file)] = "class_repair"
                taken.setdefault(cand.flight_id, []).append(int(cand["rank"]))
                log.append((f, c, cand.flight_id, cand.file))
                done = True
                break
        if not done:
            print("WARNING: cannot reach minimum", reqs[short[0]], file=sys.stderr)
            reqs[short[0]] = (f, c, 0)
    return t, sel, log


# ----------------------------------------------------------------------------- packages
def find_image(root, flight_id, name):
    g = glob.glob(f"{root}/dataset/farm_*/raw_images/*/{flight_id}/{name}")
    assert len(g) == 1, (flight_id, name, g)
    return g[0]


def strip_jpeg_metadata(b):
    """Losslessly drop APP1..APP15 and COM segments (EXIF/XMP/GPS/radiometric blocks); the
    entropy-coded image data is copied verbatim, so decoded pixels are identical."""
    assert b[:2] == b"\xff\xd8"
    out = bytearray(b"\xff\xd8")
    i = 2
    while True:
        assert b[i] == 0xFF
        m = b[i + 1]
        if m == 0xDA:                        # start of scan: copy until EOI
            j = b.find(b"\xff\xd9", i)
            out += b[i:j] + b"\xff\xd9"
            return bytes(out)
        ln = int.from_bytes(b[i + 2:i + 4], "big")
        seg = b[i:i + 2 + ln]
        if not (0xE1 <= m <= 0xEF and m != 0xEE) and m != 0xFE:
            out += seg
        i += 2 + ln


def build_packages(root, out, key, a):
    from PIL import Image
    pk = {}
    meta_warn = 0
    for ann, seed_off in (("A2", 101), ("A3", 202)):
        rr = np.random.default_rng(a.seed + seed_off)
        order = rr.permutation(len(key))
        names = np.empty(len(key), object)
        for pos, idx in enumerate(order):
            names[idx] = f"{pos + 1:04d}"
        key[f"name_{ann}"] = names
        pk[ann] = names
    hashes_src, hashes_pkg = [], []
    cache = {}
    for r in key.itertuples():
        src = find_image(root, r.flight_id, r.thermal_file)
        b = open(src, "rb").read()
        nb = b if a.keep_metadata else strip_jpeg_metadata(b)
        if not a.keep_metadata:
            ia = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
            ib = np.asarray(Image.open(io.BytesIO(nb)).convert("RGB"))
            assert ia.shape == (512, 640, 3) and np.array_equal(ia, ib), src
        cache[r.frame_uid] = nb
        hashes_src.append(hashlib.sha256(b).hexdigest())
        hashes_pkg.append(hashlib.sha256(nb).hexdigest())
    key["src_sha256"] = hashes_src
    key["package_sha256"] = hashes_pkg
    guide = f"{out}/HUONG_DAN_GAN_NHAN.md"
    assert os.path.exists(guide), "write HUONG_DAN_GAN_NHAN.md first"
    for ann in ("A2", "A3"):
        pdir = f"{out}/package_{ann}"
        if os.path.exists(pdir):
            shutil.rmtree(pdir)                 # only ever our own generated folder
        os.makedirs(f"{pdir}/images"); os.makedirs(f"{pdir}/labels")
        for r in key.itertuples():
            nm = getattr(r, f"name_{ann}")
            open(f"{pdir}/images/{nm}.jpg", "wb").write(cache[r.frame_uid])
            open(f"{pdir}/labels/{nm}.txt", "w").close()
        open(f"{pdir}/classes.txt", "w", newline="\n").write("\n".join(CLASSES) + "\n")
        open(f"{pdir}/notes.csv", "w", encoding="utf8", newline="\n").write("image,note\n")
        open(f"{pdir}/time_log.csv", "w", encoding="utf8", newline="\n").write(
            "date,start_time,end_time,first_image,last_image,break_minutes,comment\n")
        shutil.copy(guide, f"{pdir}/HUONG_DAN_GAN_NHAN.md")
        zp = f"{pdir}.zip"
        if os.path.exists(zp):
            os.remove(zp)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, _, fs in sorted(os.walk(pdir)):
                for f in sorted(fs):
                    full = os.path.join(dp, f)
                    arc = os.path.relpath(full, out).replace("\\", "/")
                    zi = zipfile.ZipInfo(arc, (2026, 1, 1, 0, 0, 0))
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zi.external_attr = 0o644 << 16
                    z.writestr(zi, open(full, "rb").read())
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                assert "_key" not in n and "ANNOTATION_GUIDELINE" not in n, n
    return key


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--n-tn19", dest="n_tn19", type=int, default=90)
    ap.add_argument("--n-tt", dest="n_tt", type=int, default=40)
    ap.add_argument("--n-bim2", dest="n_bim2", type=int, default=40)
    ap.add_argument("--n-tndl", dest="n_tndl", type=int, default=30)
    ap.add_argument("--neg-frac", type=float, default=0.30)
    ap.add_argument("--min-class", type=int, default=25)
    ap.add_argument("--min-class-farm", type=int, default=8)
    ap.add_argument("--n-moc", type=int, default=11)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--build-packages", action="store_true")
    ap.add_argument("--keep-metadata", action="store_true",
                    help="copy JPEGs byte-for-byte (EXIF/XMP incl. GPS stays: NOT blind)")
    a = ap.parse_args()
    out = a.out or f"{a.root}/.local/annotation_agreement"
    os.makedirs(f"{out}/_key/chi_labels", exist_ok=True)
    t = load_universe(a.root)
    t, sel, log = select(t, a)
    key = t.loc[list(sel)].copy()
    key["selection_reason"] = [sel[k] for k in key.index]
    key = key.reset_index(drop=True)
    key = key.sort_values(["farm", "flight_id", "rank"]).reset_index(drop=True)
    key.insert(0, "frame_uid", [f"S{i + 1:03d}" for i in range(len(key))])
    rng = np.random.default_rng(a.seed + 7)
    key["random_key"] = rng.random(len(key)).round(6)
    key["stratum"] = key.farm + "|" + key.flight_id.str.split("_").str[-2:].str.join("_") + "|" \
        + key.status.map({"positive": "pos", "no_box": "neg"})
    key["ref_status"] = key.status
    # copy A1 labels
    for r in key.itertuples():
        dst = f"{out}/_key/chi_labels/{r.frame_uid}.txt"
        if r.status == "positive":
            shutil.copy(f"{r.label_dir}/{r.flight_id}/{os.path.splitext(r.thermal_file_x if hasattr(r,'thermal_file_x') else r.file)[0]}.txt", dst)
        else:
            open(dst, "w").close()
    key["thermal_file"] = key["file"]
    key["n_boxes_ref"] = key.n_boxes
    for c in CLASSES:
        key["n_" + c] = key[c]
    if a.build_packages:
        key = build_packages(a.root, out, key, a)
    cols = ["frame_uid", "farm", "flight_id", "altitude_m", "thermal_file", "ref_status", "n_boxes_ref"] \
        + ["n_" + c for c in CLASSES] + ["stratum", "selection_reason", "rank", "random_key"] \
        + [c for c in ("name_A2", "name_A3", "src_sha256", "package_sha256") if c in key.columns]
    key[cols].to_csv(f"{out}/_key/sample_key.csv", index=False)
    report(key, out, a, log)
    print("frames:", len(key), "->", f"{out}/_key/sample_key.csv")


def report(key, out, a, log):
    L = ["# Agreement sample design (LOCAL, do not send)", "",
         f"seed={a.seed}; quotas TN19 {a.n_tn19}, TT {a.n_tt}, BIM2 {a.n_bim2}, TNDL {a.n_tndl}; "
         f"negative fraction {a.neg_frac}; min frames per class {a.min_class} (per farm {a.min_class_farm} where available); MOC frames {a.n_moc}; min gap {a.gap} frames.", ""]
    farms = ["TN19", "TT", "BIM2", "TNDL"]
    L += ["## Frames and reference boxes (annotator A1) per farm x class", "",
          "| farm | frames | neg frames | " + " | ".join(f"{c} frames / boxes" for c in CLASSES) + " | boxes |",
          "|---|---|---|" + "---|" * (len(CLASSES) + 1)]
    for f in farms + ["ALL"]:
        k = key if f == "ALL" else key[key.farm == f]
        L.append(f"| {f} | {len(k)} | {(k.n_boxes_ref == 0).sum()} | " + " | ".join(
            f"{(k['n_' + c] > 0).sum()} / {k['n_' + c].sum()}" for c in CLASSES) + f" | {k.n_boxes_ref.sum()} |")
    L += ["", "## Negatives by origin", "", key[key.n_boxes_ref == 0].groupby(["farm", "selection_reason"]).size().to_string(), "",
          "## Frames per flight", "", key.groupby(["farm", "flight_id", "ref_status"]).size().to_string(), "",
          "## Class repair swaps", "", str(len(log)) + " swaps", ""]
    gaps = []
    for f, g in key.groupby("flight_id"):
        r = sorted(g["rank"].astype(int))
        gaps += [y - x for x, y in zip(r, r[1:])]
    L += ["## Temporal spacing", "", f"adjacent selected frames inside a flight: min gap {min(gaps)} frames; "
          f"pairs with gap < {a.gap}: {sum(g < a.gap for g in gaps)}; consecutive (gap 1): {sum(g == 1 for g in gaps)}.", "",
          "MOC frame ranks (position in flight 42M_T2): " + ", ".join(
              str(int(x)) for x in key[key.n_ModuleOpenCircuit > 0]["rank"].sort_values()), ""]
    nb = key.n_boxes_ref
    L += ["## Workload per annotator", "",
          f"frames {len(key)}; boxes (reference) {nb.sum()}; median {nb.median()} per frame, max {nb.max()}; "
          f"frames with >=10 boxes: {(nb >= 10).sum()}.", ""]
    open(f"{out}/SAMPLE_DESIGN.md", "w", encoding="utf8").write("\n".join(L))


if __name__ == "__main__":
    main()
