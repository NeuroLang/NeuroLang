"""Destrieux atlas engine initialization.

Registers the following symbols:

* ``region_union`` — aggregates an iterable of regions into a single
  :class:`~neurolang.regions.ExplicitVBR`.
* Base engine symbols (``agg_count``, ``agg_create_region``,
  ``agg_create_region_overlay``, ``principal_direction``).

The ``destrieux`` predicate itself is loaded from the YAML ``atlases:``
section.
"""

from pathlib import Path
from typing import Callable, Iterable

import nibabel as nib

from neurolang.frontend import NeurolangPDL
from neurolang.regions import ExplicitVBR, region_union
from neurolang.utils.engines.base import init_base_engine


def init_engine(
    nl: NeurolangPDL, mask: nib.Nifti1Image, data_dir: Path
) -> None:
    """Populate a fresh engine with Destrieux atlas symbols.

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
