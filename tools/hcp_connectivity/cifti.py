"""
The grayordinates of the HCP's CIFTI files, and the names of regions.

The 91,282 grayordinates of the HCP are the 59,412 vertices of the cortex
of the fs_LR 32k mesh, the left hemisphere's then the right's, without the
medial wall, then the voxels of the subcortex on a 2 mm MNI grid. The
connectomes of Mansour et al. (2021) are over the cortical ones.
"""

import re
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
from nibabel import analyze

N_CORTEX = 59412
HEMISPHERES = {"L": "CIFTI_STRUCTURE_CORTEX_LEFT",
               "R": "CIFTI_STRUCTURE_CORTEX_RIGHT"}
# The hemisphere in a region's name: Schaefer's ' LH ', Destrieux's 'L ',
# Glasser's 'L_', Tian's ' lh'.
HEMISPHERE = re.compile(r"(^|\s)(L|R)H?(\s|_)|^(L|R)_")


@contextmanager
def _positive_zooms():
    """Let nibabel read a NIfTI-2 header whose voxel sizes are 0 (some of
    the HCP's and of CBIG's CIFTI files): they mean nothing to a CIFTI
    file, read as 1; the file is left as it is."""
    set_zooms = analyze.AnalyzeHeader.set_zooms
    analyze.AnalyzeHeader.set_zooms = lambda self, zooms: set_zooms(
        self, tuple(z if z > 0 else 1.0 for z in zooms))
    try:
        yield
    finally:
        analyze.AnalyzeHeader.set_zooms = set_zooms


def load(path: Path):
    """The CIFTI image of *path*."""
    with _positive_zooms():
        return nib.load(str(path))


def cortex_vertices(axis) -> dict:
    """The mesh vertices of each hemisphere in the rows of the brain models
    *axis*, in their order."""
    return {side: axis.vertex[axis.name == structure]
            for side, structure in HEMISPHERES.items()}


def engine_name(name) -> str:
    """The name of a region as the NeuroLang engines name it
    (neurolang.atlases.label_name): '-' and '_' as spaces."""
    if isinstance(name, bytes):
        name = name.decode()
    return str(name).replace("-", " ").replace("_", " ")


def hemisphere(name: str) -> str:
    """The hemisphere of a cortical region, by its name: L or R."""
    match = HEMISPHERE.search(name)
    return (match.group(2) or match.group(4)) if match else ""


def homologue(name: str) -> str:
    """The name of a region, its hemisphere's letter made X."""
    match = HEMISPHERE.search(name)
    start = match.start(2) if match.group(2) else match.start(4)
    return name[:start] + "X" + name[start + 1:]
