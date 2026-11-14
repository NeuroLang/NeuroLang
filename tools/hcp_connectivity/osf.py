"""
Download the files of an OSF project (https://osf.io/PROJECT), a few at a
time, each checked against the md5 OSF gives (a ``.part`` until whole).
A file already there and whole is not downloaded again, so it resumes.
Run it where the internet is reachable (the login node of a cluster).
"""

import hashlib
import json
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.osf.io/v2/nodes/{project}/files/osfstorage/?page[size]=100"


def context():
    """The SSL context, with the certificates of SSL_CERT_FILE if set."""
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))


def listing(project: str) -> list:
    """The files of the root of the project's OSF storage: their names,
    sizes, md5s and download addresses."""
    files, url = [], API.format(project=project)
    while url:
        with urllib.request.urlopen(url, context=context(), timeout=120) as r:
            page = json.load(r)
        for item in page["data"]:
            a = item["attributes"]
            if a["kind"] == "file":
                files.append({"name": a["name"], "size": a["size"],
                              "md5": a["extra"]["hashes"]["md5"],
                              "url": item["links"]["download"]})
        url = page["links"].get("next")
    return files


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry: dict, folder: Path, tries: int = 5) -> None:
    path = folder / entry["name"]
    if path.exists() and path.stat().st_size == entry["size"]:
        return
    part = path.with_name(path.name + ".part")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(entry["url"], context=context(),
                                        timeout=600) as r, \
                    open(part, "wb") as file:
                while chunk := r.read(1 << 22):
                    file.write(chunk)
            if md5(part) != entry["md5"]:
                raise OSError("its md5 is not OSF's")
            part.rename(path)
            print(f"{entry['name']}", flush=True)
            return
        except OSError as error:
            print(f"{entry['name']}: {error} (try {attempt + 1})", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"{entry['name']} could not be downloaded")


def run(project: str, folder: Path, jobs: int = 6) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    files = listing(project)
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.1f} GB",
          flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        for future in [pool.submit(download, f, folder) for f in files]:
            future.result()
    print("all downloaded", flush=True)
