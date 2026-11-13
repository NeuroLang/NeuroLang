"""
Download the HCP high-resolution structural connectomes of Mansour et al.
(2021), and their Glasser matrices, into a staging directory, a few at a
time, for parcellate.py, which takes each away once parcellated.

At most ``--most`` connectomes (about 500 MB each) wait in the staging
directory; a subject whose parcellation is done (``OUT/<subject>.npz``) is
not downloaded again, so the script resumes where it stopped. Run it where
the internet is reachable (the login node of a cluster)::

    python download.py STAGING OUT [--jobs 6] [--most 40]
"""

import argparse
import json
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = ("https://swift.rc.nectar.org.au/v1/"
          "AUTH_ee5989f4c9184ea29012bb124cd3dff0/connectome_storage")
CA = "/etc/pki/tls/certs/ca-bundle.crt"


def context():
    try:
        return ssl.create_default_context(cafile=CA)
    except OSError:
        return ssl.create_default_context()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("staging", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--most", type=int, default=40)
    args = parser.parse_args()
    args.staging.mkdir(parents=True, exist_ok=True)
    (args.staging / "glasser").mkdir(exist_ok=True)
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
        path = args.staging / "glasser" / o["name"].rsplit("/", 1)[1]
        if not path.exists():
            small.submit(download, o["name"], o["bytes"], path)
    todo = [(s, o) for s, o in sorted(connectomes.items())
            if not (args.out / f"{s}.npz").exists()]
    print(f"{len(todo)} connectomes to download", flush=True)

    def waiting():
        return len(list(args.staging.glob("*.npz"))) + \
            len(list(args.staging.glob("*.part")))

    with ThreadPoolExecutor(args.jobs) as pool:
        futures = []
        for subject, o in todo:
            path = args.staging / f"{subject}.npz"
            if path.exists() or (args.out / f"{subject}.npz").exists():
                continue
            while waiting() >= args.most:
                time.sleep(20)
            # the .part counts at once, as the next one waits on it
            path.with_name(path.name + ".part").touch()
            futures.append(pool.submit(download, o["name"], o["bytes"], path))
        for future in futures:
            future.result()
    small.shutdown()
    (args.staging / "DONE").touch()
    print("all downloaded", flush=True)


if __name__ == "__main__":
    main()
