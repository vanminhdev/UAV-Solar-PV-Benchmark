"""Upload the zip archives to the Zenodo draft records from a Colab (CPU) session, with several parallel streams.

Designed to survive Colab disconnects: it can be re-run at any time. A file that is already in the draft with the same
size is skipped; a file that was only partly sent is simply sent again (Zenodo only lists a file after a complete PUT).
After each upload the MD5 returned by Zenodo is compared with the MD5 computed while the bytes were read.

Usage in a Colab cell (after mounting Google Drive and putting the token in the Colab Secret `ZENODO_TOKEN`):

    from google.colab import userdata
    import os; os.environ["ZENODO_TOKEN"] = userdata.get("ZENODO_TOKEN")
    !python colab_upload_to_zenodo.py /content/drive/MyDrive/zenodo_staging --workers 5

The token is read only from the environment and is never printed or written. Records are NOT published by this script.
"""
import argparse
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://zenodo.org/api"
DRAFTS = {"R1": "23942610", "R2": "23942651", "R3": "23942696", "R4": "23942722", "R5": "23942747", "R6": "23942782"}
LOCK = threading.Lock()


def log(*a):
    with LOCK:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


class HashingReader:
    """File-like object for requests: streams the file, updates an MD5 and reports progress."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.size = os.path.getsize(path)
        self.md5 = hashlib.md5()
        self.sent = 0

    def __len__(self):
        return self.size

    def read(self, n=1 << 20):
        b = self.f.read(min(n, 1 << 22))
        self.md5.update(b)
        self.sent += len(b)
        return b

    def close(self):
        self.f.close()


def upload_one(token, rec, bucket, path, name):
    size = os.path.getsize(path)
    for attempt in range(1, 9):
        rd = HashingReader(path)
        t0 = time.time()
        try:
            log(rec, name, f"start ({size / 1e9:.1f} GB), attempt {attempt}")
            r = requests.put(f"{bucket}/{name}", data=rd, params={"access_token": token}, timeout=(60, 7200))
            r.raise_for_status()
            remote = r.json().get("checksum", "").replace("md5:", "")
            local = rd.md5.hexdigest()
            if remote and remote != local:
                raise RuntimeError(f"md5 mismatch remote={remote} local={local}")
            log(rec, name, f"OK {size / 1e6 / max(time.time() - t0, 1):.1f} MB/s md5 {local}")
            return True
        except Exception as e:  # noqa: BLE001
            log(rec, name, f"FAILED attempt {attempt}: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(min(30 * attempt, 300))
        finally:
            rd.close()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", help="folder with R1 ... R6 sub-folders of zip files")
    ap.add_argument("--records", nargs="*", default=["R2", "R3", "R4", "R5", "R6"])
    ap.add_argument("--workers", type=int, default=5, help="parallel file uploads")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("environment variable ZENODO_TOKEN is not set")
    jobs = []
    for rec in a.records:
        d = os.path.join(a.staging, rec)
        if not os.path.isdir(d):
            log(rec, "folder not found, skipped")
            continue
        r = requests.get(f"{API}/deposit/depositions/{DRAFTS[rec]}", params={"access_token": token}, timeout=60)
        r.raise_for_status()
        info = r.json()
        have = {f["filename"]: f["filesize"] for f in info.get("files", [])}
        for name in sorted(os.listdir(d)):
            if not name.endswith(".zip"):
                continue
            p = os.path.join(d, name)
            if have.get(name) == os.path.getsize(p):
                log(rec, name, "already uploaded")
                continue
            jobs.append((rec, info["links"]["bucket"], p, name))
    log(f"{len(jobs)} files to upload with {a.workers} streams")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(upload_one, token, *j) for j in jobs]
        for f in as_completed(futs):
            if f.result():
                ok += 1
            else:
                fail += 1
    log(f"done: {ok} uploaded, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
