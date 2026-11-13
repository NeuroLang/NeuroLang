"""
The cortical atlases on the vertices of the HCP high-resolution connectomes.

The connectomes of Mansour et al. (2021) are matrices over the 59,412
vertices of the cortex of the fs_LR 32k mesh (the left hemisphere's, then
the right's, without the medial wall: those of a CIFTI dense file). This
script gives each of these vertices its parcel in each atlas, and writes
``atlases.npz``: for an atlas ``a``, ``a`` (the parcel of each vertex, 0 for
none) and ``a:names`` (the name of parcel i at i - 1), in the names of the
NeuroLang engines (``17Networks LH VisCent ExStr 1``, ``L G and S
frontomargin``).

- Schaefer 2018, 100 to 1000 parcels, 7 and 17 networks: its fs_LR 32k
  label files (CBIG).
- Glasser 2016 (HCP-MMP1), to check the parcellation against the authors'
  own Glasser matrices.
- Destrieux 2009: nilearn's labels of fsaverage5, resampled to fs_LR 32k by
  the nearest vertex on the sphere, through the HCP's registration of
  fs_LR to fsaverage (HCPpipelines' resample_fsaverage).

Run it where the internet is reachable (the login node of a cluster)::

    python atlases.py WORK_DIR
"""

import json
import sys
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

N_VERTICES = 59412
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
# A dense file of the authors: its brain models are the connectomes' rows.
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
HEMISPHERES = {"L": "CIFTI_STRUCTURE_CORTEX_LEFT",
               "R": "CIFTI_STRUCTURE_CORTEX_RIGHT"}


def fetch(url: str, folder: Path) -> Path:
    """The file of *url* in *folder*, downloaded once."""
    path = folder / urllib.request.unquote(url.rsplit("/", 1)[1])
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)
    return path


def cortex_vertices(path: Path) -> dict:
    """The mesh vertices of each hemisphere in the rows of the CIFTI file
    *path*, in their order."""
    axis = nib.load(path).header.get_axis(1)
    vertices = {}
    for name, _, model in axis.iter_structures():
        for hemisphere, structure in HEMISPHERES.items():
            if name == structure:
                vertices[hemisphere] = np.asarray(model.vertex)
    return vertices


def dlabel(path: Path, vertices: dict):
    """The parcel of each connectome vertex in the label file *path*, and
    the names of the parcels (keys of its label table, renumbered from 1)."""
    image = nib.load(path)
    table = image.header.get_axis(0).label[0]
    data = np.asarray(image.get_fdata()[0], dtype=int)
    keys = sorted(k for k in table if k != 0)
    renumber = np.zeros(max(table) + 1, dtype=int)
    renumber[keys] = np.arange(1, len(keys) + 1)
    names = [table[k][0] for k in keys]
    parcels = []
    for hemisphere, structure in HEMISPHERES.items():
        for name, rows, model in image.header.get_axis(1).iter_structures():
            if name == structure:
                mesh = np.zeros(32492, dtype=int)
                mesh[model.vertex] = data[rows]
                parcels.append(renumber[mesh[vertices[hemisphere]]])
    return np.concatenate(parcels), names


def destrieux(folder: Path, vertices: dict):
    """The Destrieux parcel of each connectome vertex: nilearn's labels of
    fsaverage5 at the nearest vertex, on the sphere of fsaverage, of each
    vertex of fs_LR 32k."""
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(folder))
    labels = [x.decode() if isinstance(x, bytes) else str(x)
              for x in atlas["labels"]]
    kept = [i for i, name in enumerate(labels)
            if name not in ("Unknown", "Medial_wall")]
    names = [f"{h} {labels[i].replace('_', ' ')}" for h in HEMISPHERES
             for i in kept]
    parcels = []
    for offset, (hemisphere, side) in enumerate(
            (("L", "left"), ("R", "right"))):
        fsaverage5 = nib.load(fetch(
            f"{SPHERES}fsaverage5_std_sphere.{hemisphere}.10k_fsavg_"
            f"{hemisphere}.surf.gii", folder)).darrays[0].data
        fs_lr = nib.load(fetch(
            f"{SPHERES}fs_LR-deformed_to-fsaverage.{hemisphere}.sphere."
            "32k_fs_LR.surf.gii", folder)).darrays[0].data
        unit = (lambda xyz: xyz / np.linalg.norm(xyz, axis=1)[:, None])
        _, nearest = cKDTree(unit(fsaverage5)).query(unit(fs_lr))
        label = np.asarray(atlas[f"map_{side}"])[nearest]
        number = np.zeros(len(labels), dtype=int)
        number[kept] = np.arange(1, len(kept) + 1) + offset * len(kept)
        parcels.append(number[label[vertices[hemisphere]]])
    return np.concatenate(parcels), names


def main(work: Path) -> None:
    folder = work / "atlases"
    folder.mkdir(parents=True, exist_ok=True)
    vertices = cortex_vertices(fetch(VERTICES, folder))
    assert sum(map(len, vertices.values())) == N_VERTICES, vertices
    atlases = {}
    for k in (7, 17):
        for n in range(100, 1001, 100):
            parcels, names = dlabel(fetch(CBIG.format(n=n, k=k), folder),
                                    vertices)
            atlases[f"schaefer{n}_{k}"] = (
                parcels, [name.replace("_", " ") for name in names])
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
        assert parcels.shape == (N_VERTICES,) and parcels.max() == len(names)
        arrays[name] = parcels.astype(np.int16)
        arrays[f"{name}:names"] = np.array(names)
        print(f"{name}: {len(names)} parcels, "
              f"{(parcels == 0).sum()} vertices in none")
    np.savez_compressed(work / "atlases.npz", **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
