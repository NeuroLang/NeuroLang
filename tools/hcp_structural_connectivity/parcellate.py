"""
Parcellate the HCP high-resolution structural connectomes as download.py
brings them: each subject's matrix C (59,412 cortical vertices) becomes
P_a' C P_a for each atlas a of atlases.npz (P_a the vertices' indicator of
their parcels), the sum of C between the vertices of two parcels, written
to ``OUT/<subject>.npz``; the downloaded file is then deleted.

Several of these may run at once (a Slurm array): each claims a file by
renaming it. It ends when download.py is done and nothing waits::

    python parcellate.py WORK STAGING OUT
"""

import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def indicators(path: Path) -> dict:
    """The sparse vertex-by-parcel indicator of each atlas of *path*."""
    atlases = np.load(path)
    matrices = {}
    for name in atlases.files:
        if name.endswith(":names"):
            continue
        parcels = atlases[name].astype(int)
        inside = np.flatnonzero(parcels)
        matrices[name] = sp.csr_matrix(
            (np.ones(len(inside)), (inside, parcels[inside] - 1)),
            shape=(len(parcels), len(atlases[f"{name}:names"])))
    return matrices


def parcellate(path: Path, atlases: dict) -> dict:
    connectome = sp.load_npz(path).tocsr()
    result = {"total": np.float64(connectome.sum()),
              "diagonal": np.float64(connectome.diagonal().sum())}
    for name, P in atlases.items():
        result[name] = (P.T @ (connectome @ P)).toarray()
    return result


def main(work: Path, staging: Path, out: Path) -> None:
    atlases = indicators(work / "atlases.npz")
    out.mkdir(parents=True, exist_ok=True)
    me = f"{socket.gethostname()}.{os.getpid()}"
    while True:
        waiting = sorted(staging.glob("*.npz"))
        if not waiting:
            if (staging / "DONE").exists():
                return
            time.sleep(30)
            continue
        claimed = waiting[0].with_name(f"{waiting[0].stem}.{me}.claimed")
        try:
            waiting[0].rename(claimed)
        except FileNotFoundError:  # another took it
            continue
        subject = waiting[0].stem
        start = time.time()
        try:
            result = parcellate(claimed, atlases)
        except Exception as error:  # a broken file: told, and kept
            print(f"{subject}: {type(error).__name__}: {error}", flush=True)
            claimed.rename(claimed.with_suffix(".failed"))
            continue
        np.savez(out / f"{subject}.part.npz", **result)
        (out / f"{subject}.part.npz").rename(out / f"{subject}.npz")
        claimed.unlink()
        print(f"{subject}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
