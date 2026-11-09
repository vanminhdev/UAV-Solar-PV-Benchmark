"""Upload the archives built by build_zenodo_archives.py into the Zenodo draft records.

The personal access token is read from the environment variable ZENODO_TOKEN (scope: deposit:write). It is never stored.
Create it at https://zenodo.org/account/settings/applications/ and run, for example (PowerShell):

    $env:ZENODO_TOKEN = "<token>"
    python upload_to_zenodo.py <staging_dir> --record R2            # one record
    python upload_to_zenodo.py <staging_dir>                        # all records

A file already present in the draft with the same size is skipped, so an interrupted run can simply be started again.
After each upload the MD5 returned by Zenodo is compared with the local file. Records are NOT published by this script.
"""
import argparse
import hashlib
import os
import sys
import time

import requests

API = "https://zenodo.org/api"
DRAFTS = {  # record part -> id of the draft created on Zenodo
    "R1": "23942610",
    "R2": "23942651",
    "R3": "23942696",
    "R4": "23942722",
    "R5": "23942747",
    "R6": "23942782",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("staging")
    ap.add_argument("--record", help="R1 ... R6 (default: all)")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token and os.environ.get("ZENODO_TOKEN_FILE") and os.path.exists(os.environ["ZENODO_TOKEN_FILE"]):
        token = open(os.environ["ZENODO_TOKEN_FILE"], encoding="utf-8").read().strip()
    if not token:
        sys.exit("set ZENODO_TOKEN (or ZENODO_TOKEN_FILE, a text file holding the token) first")
    s = requests.Session()
    s.params = {"access_token": token}
    for rec, dep in sorted(DRAFTS.items()):
        if a.record and a.record != rec:
            continue
        d = os.path.join(a.staging, rec)
        if not os.path.isdir(d):
            continue
        r = s.get(f"{API}/deposit/depositions/{dep}")
        r.raise_for_status()
        info = r.json()
        bucket = info["links"]["bucket"]
        have = {f["filename"]: f["filesize"] for f in info.get("files", [])}
        for name in sorted(os.listdir(d)):
            if not name.endswith(".zip"):
                continue
            p = os.path.join(d, name)
            size = os.path.getsize(p)
            if have.get(name) == size:
                print(rec, name, "already uploaded")
                continue
            for attempt in range(5):
                try:
                    t0 = time.time()
                    with open(p, "rb") as f:
                        up = s.put(f"{bucket}/{name}", data=f, timeout=(60, 3600))
                    up.raise_for_status()
                    remote = up.json().get("checksum", "").replace("md5:", "")
                    local = md5(p)
                    if remote and remote != local:
                        raise RuntimeError(f"md5 mismatch {remote} != {local}")
                    print(rec, name, "uploaded %.1f GB in %.0f s" % (size / 1e9, time.time() - t0), flush=True)
                    break
                except Exception as e:  # network errors are retried
                    print(rec, name, "attempt", attempt + 1, "failed:", e, flush=True)
                    time.sleep(30)
            else:
                sys.exit(f"giving up on {name}")


if __name__ == "__main__":
    main()
