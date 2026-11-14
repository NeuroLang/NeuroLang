"""
Download the HCP high-resolution structural connectomes of Mansour et al.
(2021), and their Glasser matrices, into a staging directory, a few at a
time, for the structural worker, which takes each away once
parcellated.

At most ``--most`` connectomes (about 500 MB each) wait in the staging
directory; a subject whose parcellation is done (in a part of OUT) is not
downloaded again, so it resumes where it stopped. Run it where the internet
is reachable (the login node of a cluster).
"""

import json
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .parcels import done

BUCKET = ("https://swift.rc.nectar.org.au/v1/"
          "AUTH_ee5989f4c9184ea29012bb124cd3dff0/connectome_storage")


def context():
    """The SSL context, with the certificates of SSL_CERT_FILE if set (a
    cluster whose Python does not find its own)."""
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))


def listing(prefix: str) -> list:
    """The objects of the bucket under *prefix*: their names and sizes."""
    objects, marker = [], ""
    while True:
        url = (f"{BUCKET}?format=json&limit=10000&prefix={prefix}"
               f"&marker={urllib.request.quote(marker)}")
        with urllib.request.urlopen(url, context=context(), timeout=120) as r:
            page = json.load(r)
        if not page:
            return objects
        objects += page
        marker = page[-1]["name"]


def download(name: str, size: int, path: Path, tries: int = 5) -> None:
    """Download the object *name* to *path*, whole (a .part until then)."""
    part = path.with_name(path.name + ".part")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{name}", context=context(),
                                        timeout=600) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            if part.stat().st_size != size:
                raise OSError(f"{part.stat().st_size} bytes, not {size}")
            part.rename(path)
            return
        except OSError as error:
            print(f"{name}: {error} (try {attempt + 1})", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"{name} could not be downloaded")


def run(staging: Path, out: Path, jobs: int = 6, most: int = 40) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "glasser").mkdir(exist_ok=True)
    objects = listing("HCP_1200/")
    connectomes = {o["name"].split("/")[1]: o for o in objects
                   if "/high_resolution/structural_connectivity/" in o["name"]}
    glasser = [o for o in objects
               if "/glasser_atlas/structural_connectivity/" in o["name"]]
    print(f"{len(connectomes)} connectomes, {len(glasser)} Glasser matrices",
          flush=True)
    # the Glasser matrices (1 MB each) come alongside, two at a time
    small = ThreadPoolExecutor(2)
    for o in glasser:
        path = staging / "glasser" / o["name"].rsplit("/", 1)[1]
        if not path.exists():
            small.submit(download, o["name"], o["bytes"], path)
    finished = done(out)
    todo = [(s, o) for s, o in sorted(connectomes.items())
            if s not in finished]
    print(f"{len(todo)} connectomes to download", flush=True)

    def waiting():
        return len(list(staging.glob("*.npz"))) + \
            len(list(staging.glob("*.part")))

    with ThreadPoolExecutor(jobs) as pool:
        futures = []
        for subject, o in todo:
            path = staging / f"{subject}.npz"
            if path.exists():
                continue
            while waiting() >= most:
                time.sleep(20)
            # the .part counts at once, as the next one waits on it
            path.with_name(path.name + ".part").touch()
            futures.append(pool.submit(download, o["name"], o["bytes"], path))
        for future in futures:
            future.result()
    small.shutdown()
    (staging / "DONE").touch()
    print("all downloaded", flush=True)
