"""
The atlases on the grayordinates of the HCP.

``build`` gives each of the 59,412 cortical vertices its parcel in each
cortical atlas, and writes ``WORK/atlases.npz``: for an atlas ``a``, ``a``
(the parcel of each vertex, from 1; 0 for none) and ``a:names`` (the name
of parcel i at i - 1), named as the NeuroLang engines name them
(``17Networks LH VisCent ExStr 1``, ``L G and S frontomargin``):

- Schaefer 2018, 100 to 1000 parcels, 7 and 17 networks: its fs_LR 32k
  label files (CBIG);
- Glasser 2016 (HCP-MMP1), in the order of the authors' Glasser matrices
  of the structural connectomes, to check against them;
- Destrieux 2009: nilearn's labels of fsaverage5, brought to fs_LR 32k by
  the nearest vertex on the sphere, through the HCP's registration of
  fs_LR to fsaverage (HCPpipelines' resample_fsaverage).

:func:`tian` gives the subcortical grayordinates of a CIFTI file their
region of Tian's subcortex atlas (scale 2, 32 regions; its 3T image is on
the grid of those voxels). Run ``build`` where the internet is reachable.
"""

import json
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .cifti import HEMISPHERES, N_CORTEX, cortex_vertices, engine_name, load

CBIG = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/HCP/fslr32k/cifti/"
    "Schaefer2018_{n}Parcels_{k}Networks_order.dlabel.nii"
)
GLASSER = (
    "https://raw.githubusercontent.com/sina-mansour/UKB-connectomics/main/"
    "data/templates/atlases/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_"
    "Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
)
# A dense file of Mansour et al.: its brain models are the connectomes'.
VERTICES = (
    "https://raw.githubusercontent.com/sina-mansour/neural-identity/master/"
    "data/uniqueness/High%20resolution/"
    "structural_connectivity_uniqueness.dscalar.nii"
)
GLASSER_ORDER = (
    "https://raw.githubusercontent.com/sina-mansour/neural-identity/master/"
    "data/supplementary/glasser_atlas_labels.json"
)
SPHERES = (
    "https://raw.githubusercontent.com/Washington-University/HCPpipelines/"
    "master/global/templates/standard_mesh_atlases/resample_fsaverage/"
)
TIAN = ("https://raw.githubusercontent.com/sina-mansour/UKB-connectomics/"
        "main/data/templates/atlases/Tian_Subcortex_S2_3T")


def fetch(url: str, folder: Path) -> Path:
    """The file of *url* in *folder*, downloaded once."""
    path = folder / urllib.request.unquote(url.rsplit("/", 1)[1])
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)
    return path


def dlabel(path: Path, vertices: dict):
    """The parcel of each cortical vertex (*vertices*, by hemisphere) in the
    label file *path*, and the names of the parcels (the keys of its label
    table, renumbered from 1)."""
    image = load(path)
    table = image.header.get_axis(0).label[0]
    axis = image.header.get_axis(1)
    data = np.asarray(image.get_fdata()[0], dtype=int)
    keys = sorted(k for k in table if k != 0)
    renumber = np.zeros(max(table) + 1, dtype=int)
    renumber[keys] = np.arange(1, len(keys) + 1)
    parcels = []
    for side, structure in HEMISPHERES.items():
        rows = axis.name == structure
        mesh = np.zeros(32492, dtype=int)
        mesh[axis.vertex[rows]] = data[rows]
        parcels.append(renumber[mesh[vertices[side]]])
    return np.concatenate(parcels), [table[k][0] for k in keys]


def destrieux(folder: Path, vertices: dict):
    """The Destrieux parcel of each cortical vertex: nilearn's labels of
    fsaverage5 at the nearest vertex, on the sphere of fsaverage, of each
    vertex of fs_LR 32k."""
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(folder))
    labels = [engine_name(x) for x in atlas["labels"]]
    kept = [i for i, name in enumerate(labels)
            if name not in ("Unknown", "Medial wall")]
    names = [f"{side} {labels[i]}" for side in HEMISPHERES for i in kept]
    parcels = []
    for offset, (side, hemi) in enumerate((("L", "left"), ("R", "right"))):
        fsaverage5 = nib.load(fetch(
            f"{SPHERES}fsaverage5_std_sphere.{side}.10k_fsavg_{side}"
            ".surf.gii", folder)).darrays[0].data
        fs_lr = nib.load(fetch(
            f"{SPHERES}fs_LR-deformed_to-fsaverage.{side}.sphere."
            "32k_fs_LR.surf.gii", folder)).darrays[0].data
        unit = (lambda xyz: xyz / np.linalg.norm(xyz, axis=1)[:, None])
        _, nearest = cKDTree(unit(fsaverage5)).query(unit(fs_lr))
        label = np.asarray(atlas[f"map_{hemi}"])[nearest]
        number = np.zeros(len(labels), dtype=int)
        number[kept] = np.arange(1, len(kept) + 1) + offset * len(kept)
        parcels.append(number[label[vertices[side]]])
    return np.concatenate(parcels), names


def tian_names(folder: Path) -> list:
    """The names of the regions of Tian's atlas."""
    return [engine_name(line.strip()) for line in
            fetch(f"{TIAN}_label.txt", folder).read_text().splitlines()
            if line.strip()]


def tian(folder: Path, axis):
    """The region of Tian's atlas of each subcortical grayordinate of the
    brain models *axis* (from 1; 0 for none), and the regions' names."""
    image = nib.load(fetch(f"{TIAN}.nii.gz", folder))
    assert np.allclose(image.affine, axis.affine), "not the grid of the data"
    names = tian_names(folder)
    labels = np.asarray(image.get_fdata(), dtype=int)
    i, j, k = axis.voxel[N_CORTEX:].T
    parcels = labels[i, j, k]
    assert (np.bincount(parcels, minlength=len(names) + 1)[1:] > 0).all()
    return parcels, names


def build(work: Path) -> None:
    folder = work / "atlases"
    folder.mkdir(parents=True, exist_ok=True)
    vertices = cortex_vertices(load(fetch(VERTICES, folder))
                               .header.get_axis(1))
    assert sum(map(len, vertices.values())) == N_CORTEX, vertices
    atlases = {}
    for k in (7, 17):
        for n in range(100, 1001, 100):
            parcels, names = dlabel(fetch(CBIG.format(n=n, k=k), folder),
                                    vertices)
            atlases[f"schaefer{n}_{k}"] = (
                parcels, [engine_name(name) for name in names])
    parcels, names = dlabel(fetch(GLASSER, folder), vertices)
    # in the order of the authors' Glasser matrices
    order = json.loads(fetch(GLASSER_ORDER, folder).read_text())
    order = [order[str(i)] for i in range(len(order))]
    position = {name: i + 1 for i, name in enumerate(order)}
    renumber = np.array([0] + [position[name] for name in names])
    atlases["glasser"] = (renumber[parcels], order)
    atlases["destrieux"] = destrieux(folder, vertices)
    arrays = {}
    for name, (parcels, names) in atlases.items():
        assert parcels.shape == (N_CORTEX,) and parcels.max() == len(names)
        arrays[name] = parcels.astype(np.int16)
        arrays[f"{name}:names"] = np.array(names)
        print(f"{name}: {len(names)} parcels, "
              f"{(parcels == 0).sum()} vertices in none")
    np.savez_compressed(work / "atlases.npz", **arrays)


def cortical(work: Path) -> dict:
    """The atlases of ``WORK/atlases.npz``: for each, the parcel of each
    cortical vertex and the names of the parcels."""
    arrays = np.load(work / "atlases.npz")
    return {a: (arrays[a].astype(int), [str(n) for n in arrays[f"{a}:names"]])
            for a in arrays.files if not a.endswith(":names")}
