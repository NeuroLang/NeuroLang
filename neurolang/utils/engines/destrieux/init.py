"""Destrieux atlas engine initialization.

Registers the following predicates:

* ``destrieux(name, region)`` — cortical region name and
  :class:`~neurolang.regions.ExplicitVBR` geometry.
"""

from pathlib import Path
from typing import Callable, Iterable

import nibabel as nib
from nilearn import datasets

from neurolang.frontend import NeurolangPDL
from neurolang.regions import ExplicitVBR, region_union
from neurolang.utils.engines.base import init_base_engine


def init_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image, data_dir: Path
) -> None:
    """Populate a fresh engine with Destrieux atlas data.

    Parameters
    ----------
    nl :
        Fresh :class:`~neurolang.frontend.NeurolangPDL` instance.
    mask :
        MNI template image (used for base symbol registration).
    data_dir :
        Directory under which downloaded data is cached.
    """
    init_base_engine(nl, mask)

    nl.add_symbol(
        region_union,
        name="region_union",
        type_=Callable[[Iterable[ExplicitVBR]], ExplicitVBR],
    )

    destrieux_atlas = datasets.fetch_atlas_destrieux_2009(
        data_dir=str(data_dir / "destrieux")
    )
    nl.new_symbol(name="destrieux")
    destrieux_atlas_image = nib.load(destrieux_atlas["maps"])
    raw_labels = destrieux_atlas["labels"]

    # nilearn >= 0.13 returns a plain list (index = label value);
    # earlier versions returned a numpy array of (index, name) pairs.
    if isinstance(raw_labels, dict):
        label_items = raw_labels.items()
    elif isinstance(raw_labels, list):
        label_items = enumerate(raw_labels)
    else:
        # numpy structured array with two columns
        label_items = raw_labels

    destrieux_set = set()
    for k, v in label_items:
        if k == 0:
            continue
        if isinstance(v, bytes):
            v = v.decode("utf8")
        destrieux_set.add(
            (
                v.replace("-", " ").replace("_", " "),
                ExplicitVBR.from_spatial_image_label(
                    destrieux_atlas_image, k
                ),
            )
        )
    nl.add_tuple_set(destrieux_set, name="destrieux")
