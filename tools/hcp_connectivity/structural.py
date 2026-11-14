"""
The structural connectivity of the HCP subjects, by atlas: a worker that
parcellates the high-resolution connectomes of Mansour et al. (2021) as
they are downloaded into a staging directory (download.py, relay.sh).

Each subject's matrix C (59,412 cortical vertices, smoothed streamline
counts) becomes P_a' C P_a for each atlas a of ``WORK/atlases.npz`` (P_a
the vertices' indicator of their parcels): the streamlines between the
vertices of two parcels. They are summed into the worker's part
(parcels.Part), with, for each subject, its streamlines (counted twice)
and, when the authors' own Glasser matrix of the subject is in
``STAGING/glasser``, the correlation of the streamlines between distinct
parcels of theirs and of ours and the fraction on which they differ. The
downloaded file is then deleted. Several workers may run at once: each
claims a file by renaming it; each ends when the downloads are done
(``STAGING/DONE``) and nothing waits.
"""

import os
import socket
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .atlases import cortical
from .parcels import Part, indicator


def glasser_check(subject: str, glasser_dir: Path, ours: np.ndarray):
    """The correlation of the streamlines between distinct Glasser parcels
    of the authors' matrix of *subject* and of *ours*, and the fraction of
    them on which they differ (NaN without their matrix)."""
    path = glasser_dir / f"{subject}_test_structural_connectivity_glasser.npy"
    if not path.exists():
        return np.nan, np.nan
    theirs = np.load(path)
    upper = np.triu_indices_from(theirs, 1)
    return (np.corrcoef(ours[upper], theirs[upper])[0, 1],
            np.abs(ours - theirs)[upper].sum() / theirs[upper].sum())


def run(work: Path, staging: Path, out: Path, task: int) -> None:
    atlases = {a: indicator(parcels, len(names))
               for a, (parcels, names) in cortical(work).items()}
    out.mkdir(parents=True, exist_ok=True)
    part = Part(out / f"part-{task}.npz")
    me = f"{socket.gethostname()}.{os.getpid()}"
    while True:
        waiting = sorted(staging.glob("*.npz"))
        if not waiting:
            if (staging / "DONE").exists():
                part.save()
                return
            time.sleep(30)
            continue
        subject = waiting[0].stem
        claimed = waiting[0].with_name(f"{subject}.{me}.claimed")
        try:
            waiting[0].rename(claimed)
        except FileNotFoundError:  # another took it
            continue
        start = time.time()
        try:
            connectome = sp.load_npz(claimed).tocsr()
            matrices = {a: (P.T @ (connectome @ P)).toarray()
                        for a, P in atlases.items()}
        except Exception as error:  # a broken file: told, and kept
            print(f"{subject}: {type(error).__name__}: {error}", flush=True)
            claimed.rename(claimed.with_suffix(".failed"))
            continue
        r, differ = glasser_check(subject, staging / "glasser",
                                  matrices["glasser"])
        part.add(subject, matrices, total=float(connectome.sum()),
                 glasser_r=r, glasser_differ=differ)
        claimed.unlink()
        print(f"{subject}: {time.time() - start:.0f} s", flush=True)
