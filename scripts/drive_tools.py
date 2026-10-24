"""Crawl and download public Google Drive folders (no API key needed).

    python drive_tools.py crawl <folder_id> <tree.json>
    python drive_tools.py download <tree.json> <out_dir> [--skip-ext .zip .laz] [--only substr ...] [--max-mb N]

Downloads are resumable: existing non-empty files are skipped, results are
appended to <out_dir>/_manifest.csv (path, drive_id, bytes, status).
Original names are kept; renaming/normalising happens later in the dataset.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def list_folder(folder_id, retries=3):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            m = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'([^']+)'", html)
            if not m:
                return []
            data = json.loads(m.group(1).encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8"))
            items = data[0] if data else []
            return [{"id": i[0], "name": i[2], "folder": "folder" in i[3], "mime": i[3]} for i in items]
        except Exception as e:  # noqa: BLE001
            print("retry", folder_id, e, file=sys.stderr)
            time.sleep(2)
    return []


def crawl(folder_id, name="ROOT"):
    node = {"id": folder_id, "name": name, "folder": True, "children": []}
    for it in list_folder(folder_id):
        if it["folder"]:
            node["children"].append(crawl(it["id"], it["name"]))
        else:
            node["children"].append(it)
    return node


def walk(node, prefix=""):
    for c in node["children"]:
        p = f"{prefix}/{c['name']}" if prefix else c["name"]
        if c["folder"]:
            yield from walk(c, p)
        else:
            yield p, c


def _download_with_watchdog(file_id, dst, stall_s=180):
    """Run gdown in a subprocess; kill it if the partial file stops growing for stall_s seconds."""
    import subprocess
    import time

    proc = subprocess.Popen([sys.executable, "-c",
                             f"import gdown;gdown.download(id={file_id!r},output={str(dst)!r},quiet=True)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last, t_last = -1, time.time()
    while proc.poll() is None:
        time.sleep(5)
        cur = sum(f.stat().st_size for f in dst.parent.glob(dst.name + "*"))
        if cur != last:
            last, t_last = cur, time.time()
        elif time.time() - t_last > stall_s:
            proc.kill()
            for f in dst.parent.glob(dst.name + "*"):
                f.unlink(missing_ok=True)
            return False
    return proc.returncode == 0


def download(tree, out_dir, skip_ext, only, max_mb):
    import gdown

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "_manifest.csv"
    new = not mf.exists()
    with open(mf, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["path", "drive_id", "bytes", "status"])
        for rel, it in walk(tree):
            if Path(rel).suffix.lower() in skip_ext or ".files/" in rel or "/Files/" in rel:
                continue  # Metashape project internals
            if only and not any(s.lower() in rel.lower() for s in only):
                continue
            dst = out / rel
            if dst.exists() and dst.stat().st_size > 0:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                r = _download_with_watchdog(it["id"], dst, stall_s=180)
                ok = r and dst.exists()
                size = dst.stat().st_size if ok else 0
                w.writerow([rel, it["id"], size, "ok" if ok else "FAILED"])
            except Exception as e:  # noqa: BLE001
                w.writerow([rel, it["id"], 0, f"FAILED: {str(e)[:80]}"])
            fh.flush()
            print(rel, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl")
    c.add_argument("folder_id")
    c.add_argument("out_json")
    d = sub.add_parser("download")
    d.add_argument("tree_json")
    d.add_argument("out_dir")
    d.add_argument("--skip-ext", nargs="*", default=[])
    d.add_argument("--only", nargs="*", default=[])
    d.add_argument("--max-mb", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "crawl":
        Path(a.out_json).write_text(json.dumps(crawl(a.folder_id), ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        download(json.loads(Path(a.tree_json).read_text(encoding="utf-8")), a.out_dir, set(a.skip_ext), a.only, a.max_mb)
