#!/usr/bin/env python
"""SHA-256 checksums of the dataset.

1. Farm files: every file under raw_images/, photogrammetry/, annotations/, module_reference/ and splits/ of each farm.
   Output: dataset/metadata/checksums/sha256_<farm_dir>.txt in `sha256sum` format
   ("<hash>  <path relative to farm dir>", forward slashes), so that
       cd dataset/<farm_dir> && sha256sum -c ../metadata/checksums/sha256_<farm_dir>.txt
   verifies the release. After hashing, 20 random files are re-hashed and compared.
2. Metadata: dataset/provenance/sha256_metadata.txt = every non-hidden file under metadata/ and provenance/ (not itself),
   paths relative to dataset/, byte-order sorted, `sha256sum` format (includes the farm checksum files, so run 1 first).

Usage: python -X utf8 make_checksums.py [--workers 4] [--verify-only] [--sample 20]   # full run: all farm files (slow, ~120 GB)
       python -X utf8 make_checksums.py --annotations   # re-hash only annotations/, module_reference/, splits/ (keep other lines)
       python -X utf8 make_checksums.py --metadata      # rewrite provenance/sha256_metadata.txt
"""
import argparse, hashlib, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "dataset"
OUT = ROOT / "metadata" / "checksums"
SUBDIRS = ("raw_images", "photogrammetry", "annotations", "module_reference", "splits")
LIGHT = ("annotations", "module_reference", "splits")
CHUNK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_files(farm, subdirs=SUBDIRS):
    files = []
    for sub in subdirs:
        d = farm / sub
        if d.is_dir():
            files += [p for p in d.rglob("*") if p.is_file()]
    return sorted(files, key=lambda p: p.relative_to(farm).as_posix())


def hash_farm(farm, workers):
    files = list_files(farm)
    sizes = {p: p.stat().st_size for p in files}
    total = sum(sizes.values())
    print(f"[{farm.name}] {len(files)} files, {total/1e9:.2f} GB", flush=True)
    res, done_b, t0, last = {}, 0, time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(sha256, p): p for p in files}
        for i, fu in enumerate(as_completed(futs), 1):
            p = futs[fu]
            res[p] = fu.result()
            done_b += sizes[p]
            now = time.time()
            if now - last > 30 or i == len(files):
                last = now
                el = now - t0
                print(f"[{farm.name}] {i}/{len(files)} files  {done_b/1e9:.1f}/{total/1e9:.1f} GB  "
                      f"{done_b/1e6/max(el,1e-9):.0f} MB/s  {el:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"sha256_{farm.name}.txt"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for p in files:
            f.write(f"{res[p]}  {p.relative_to(farm).as_posix()}\n")
    return len(files), total


def hash_light(farm):
    """Re-hash only the small folders and keep the existing lines of raw_images/ and photogrammetry/."""
    out = OUT / f"sha256_{farm.name}.txt"
    keep = [l for l in out.read_text(encoding="utf-8").splitlines() if l.split("  ", 1)[1].split("/", 1)[0] not in LIGHT]
    new = [f"{sha256(p)}  {p.relative_to(farm).as_posix()}" for p in list_files(farm, LIGHT)]
    lines = sorted(keep + new, key=lambda l: l.split("  ", 1)[1])
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[{farm.name}] {len(new)} light files re-hashed, {len(keep)} lines kept, {len(lines)} total", flush=True)


def hash_metadata():
    files = []
    for top in ("metadata", "provenance"):
        for p in (ROOT / top).rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and not any(part.startswith(".") for part in rel.split("/")) and rel != "provenance/sha256_metadata.txt":
                files.append(rel)
    files.sort(key=lambda r: r.encode("utf-8"))
    with open(ROOT / "provenance" / "sha256_metadata.txt", "w", encoding="utf-8", newline="\n") as fh:
        for r in files:
            fh.write(f"{sha256(ROOT / r)}  {r}\n")
    print(f"sha256_metadata.txt: {len(files)} files", flush=True)


def verify(farms, n):
    entries = []
    for farm in farms:
        txt = OUT / f"sha256_{farm.name}.txt"
        for line in txt.read_text(encoding="utf-8").splitlines():
            h, rel = line.split("  ", 1)
            entries.append((farm, h, rel))
    random.seed(20260420)
    sample = random.sample(entries, min(n, len(entries)))
    bad = 0
    for farm, h, rel in sample:
        ok = sha256(farm / rel) == h
        bad += not ok
        print(("OK   " if ok else "FAIL ") + f"{farm.name}/{rel}", flush=True)
    print(f"VERIFY: {len(sample)-bad}/{len(sample)} sampled files match", flush=True)
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--annotations", action="store_true", help="re-hash annotations/, module_reference/, splits/ only")
    ap.add_argument("--metadata", action="store_true", help="rewrite provenance/sha256_metadata.txt only")
    a = ap.parse_args()
    farms = sorted(p for p in ROOT.glob("farm_*") if p.is_dir())
    if a.annotations or a.metadata:
        if a.annotations:
            for farm in farms:
                hash_light(farm)
        if a.metadata:
            hash_metadata()
        return
    t0 = time.time()
    nf = nb = 0
    if not a.verify_only:
        for farm in farms:
            f, b = hash_farm(farm, a.workers)
            nf += f; nb += b
        print(f"TOTAL hashed {nf} files, {nb/1e9:.2f} GB ({nb/2**30:.2f} GiB) in {time.time()-t0:.0f}s", flush=True)
    ok = verify(farms, a.sample)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
