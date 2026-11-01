"""Atlases: a label volume, its affine and a label table (DESIGN.md, 4)."""

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .spaces import Space

__all__ = [
    "Atlas",
    "HemisphereRule",
    "HEMISPHERE_RULES",
    "ProbabilisticAtlas",
    "infer_hemispheres",
    "load_label_table_bids_dseg",
    "load_label_table_freesurfer_lut",
    "load_label_table_tsv",
]

#: ``hemi`` values of the ``label`` relation.
HEMISPHERES = ("left", "right", "none")


@dataclass(frozen=True)
class HemisphereRule:
    """Infer the hemisphere of a raw label name, and its base name.

    Attributes
    ----------
    pattern : str
        A regular expression that must *fully* match the raw name and have
        a named group ``base``: the name without its hemisphere token.
    hemi : str
        ``"left"`` or ``"right"``.
    """

    pattern: str
    hemi: str


#: The default, ordered rules; the first that matches wins. Left-/Right-,
#: ctx-lh-/ctx-rh-, _L/_R, lh./rh., .left/.right (DESIGN.md, 4.3).
HEMISPHERE_RULES: Sequence[HemisphereRule] = ()


def infer_hemispheres(
    names: Iterable[str],
    rules: Sequence[HemisphereRule] = HEMISPHERE_RULES,
) -> pd.DataFrame:
    """Split raw label names into base name and hemisphere.

    Returns
    -------
    pandas.DataFrame
        One row per name, columns ``name``, ``base_name``, ``hemi``;
        ``hemi`` is ``"none"`` and ``base_name == name`` if no rule matches.
    """
    raise NotImplementedError


def load_label_table_freesurfer_lut(
    path: Union[str, os.PathLike]
) -> pd.DataFrame:
    """Read a FreeSurfer colour LUT (``FreeSurferColorLUT.txt``).

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` (int64) and ``name``; hemispheres are inferred later.
    """
    raise NotImplementedError


def load_label_table_bids_dseg(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Read a BIDS ``*_dseg.tsv`` (columns ``index`` and ``name``)."""
    raise NotImplementedError


def load_label_table_tsv(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Read a TSV with columns ``id``, ``name`` and optionally ``hemi``."""
    raise NotImplementedError


class Atlas:
    """A deterministic atlas: label volume + affine + label table + space.

    Parameters
    ----------
    name : str
        The atlas's identifier, the value of the ``a`` column of every
        label-dependent relation. Unique in an engine.
    labels : numpy.ndarray
        3D integer volume; ``background`` marks voxels outside any region.
    affine : numpy.ndarray
        4x4, voxel indices (ijk) to the space's world coordinates in mm
        (RAS+, as nibabel).
    table : pandas.DataFrame
        Columns ``id``, ``name`` and optionally ``hemi``; if ``hemi`` is
        missing it is inferred with ``hemisphere_rules``. Only ids that are
        in the table *and* in the volume produce rows of ``label``.
    space : Space
        The space tag.
    background : int
        The label that means "no region" (default 0, DESIGN.md, 7.3).
    hemisphere_rules : Sequence[HemisphereRule]
        Rules for the inference of ``hemi`` and of the base names.
    term_mapping : pandas.DataFrame or path, optional
        Columns ``base_name`` and ``term``; defaults to the file shipped
        in ``nl_tracts/mappings`` for ``name``, if there is one.
    """

    def __init__(
        self,
        name: str,
        labels: np.ndarray,
        affine: np.ndarray,
        table: pd.DataFrame,
        space: Space,
        background: int = 0,
        hemisphere_rules: Sequence[HemisphereRule] = HEMISPHERE_RULES,
        term_mapping: Optional[Union[pd.DataFrame, str, os.PathLike]] = None,
    ) -> None:
        raise NotImplementedError

    # Constructors for the loaders of DESIGN.md, 4.2.

    @classmethod
    def from_freesurfer(
        cls,
        name: str,
        image: Union[str, os.PathLike, Any],
        lut: Union[str, os.PathLike],
        space: Space,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image (path or nibabel image) and a FreeSurfer LUT."""
        raise NotImplementedError

    @classmethod
    def from_bids_dseg(
        cls,
        name: str,
        image: Union[str, os.PathLike, Any],
        dseg_tsv: Union[str, os.PathLike],
        space: Space,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image and its BIDS ``dseg.tsv``."""
        raise NotImplementedError

    @classmethod
    def from_tsv(
        cls,
        name: str,
        image: Union[str, os.PathLike, Any],
        tsv: Union[str, os.PathLike],
        space: Space,
        **kwargs: Any,
    ) -> "Atlas":
        """From a label image and a TSV (``id``, ``name``[, ``hemi``])."""
        raise NotImplementedError

    @classmethod
    def from_nilearn(
        cls, name: str, bunch: Any, space: Space, **kwargs: Any
    ) -> "Atlas":
        """From the object a nilearn atlas fetcher returns.

        Duck-typed, nilearn is not imported: ``bunch.maps`` is a path or an
        image; ``bunch.labels`` the names, the id of a name being its
        position, unless ``bunch.lut`` (a DataFrame with ``index`` and
        ``name``) is present.
        """
        raise NotImplementedError

    # The relations of DESIGN.md, 3.

    def label_relation(self) -> pd.DataFrame:
        """``label(a, l, name, hemi)``: columns ``a, l, name, hemi``."""
        raise NotImplementedError

    def label_term_relation(self) -> pd.DataFrame:
        """``label_term(a, l, term)``: columns ``a, l, term``."""
        raise NotImplementedError

    def label_extent_relation(self) -> pd.DataFrame:
        """``label_extent(a, l, axis, lo, hi)``.

        The extent of each label's voxel *centres*, in world mm, along each
        world axis (0, 1, 2 = x, y, z), computed from all the voxels of the
        label, not from two corners: right for oblique affines.
        """
        raise NotImplementedError

    def content_hash(self) -> str:
        """Hash of the volume, affine, table, background and space."""
        raise NotImplementedError


class ProbabilisticAtlas:
    """Extension point for 4D (probability map) atlases. Not implemented.

    A probabilistic atlas holds, per voxel, a probability for each label.
    It will produce *probabilistic* facts ``traverses_p(a, t, s, l)`` with
    the probability of the streamline traversing the label, instead of the
    boolean ``traverses``; it shares ``label``, ``label_term`` and the
    loaders of :class:`Atlas`, and the engine side is
    ``add_probabilistic_facts_from_tuples`` (DESIGN.md, 4.4).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "probabilistic atlases are future work (DESIGN.md, 4.4)"
        )
