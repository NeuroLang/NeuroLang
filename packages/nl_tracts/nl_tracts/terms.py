"""Term mappings: raw label names to the shared vocabulary (DESIGN.md, 3.2).

A mapping is a TSV with the columns ``atlas_label_name``, ``term`` and,
optionally, ``hemi``. Lines that start with ``#`` and blank lines are
ignored. A name may have several rows (several terms); ``hemi`` is
``left``, ``right``, ``none`` or empty, and the rows of one name agree on it.
The terms are hemisphere-free: the hemisphere is the ``hemi`` of ``label``.
"""

import io
import os
from importlib import resources
from typing import Union

import pandas as pd

__all__ = [
    "COLUMNS",
    "HEMISPHERES",
    "SHIPPED_MAPPINGS",
    "load_term_mapping",
    "shipped_mapping_path",
]

#: ``hemi`` values of the ``label`` relation.
HEMISPHERES = ("left", "right", "none")

COLUMNS = ["atlas_label_name", "term", "hemi"]

#: The mappings in ``nl_tracts/mappings``, by key (the file name without
#: ``.tsv``).
SHIPPED_MAPPINGS = (
    "freesurfer_aseg_aparc", "freesurfer_destrieux", "tian_subcortex_s2",
)


def shipped_mapping_path(key: str):
    """The file of the shipped mapping ``key`` (a ``Traversable``)."""
    if key not in SHIPPED_MAPPINGS:
        raise ValueError(
            f"no shipped mapping {key!r}; the keys are "
            f"{', '.join(SHIPPED_MAPPINGS)}"
        )
    return resources.files("nl_tracts").joinpath("mappings", f"{key}.tsv")


def load_term_mapping(
    source: Union[str, os.PathLike, pd.DataFrame]
) -> pd.DataFrame:
    """Read and validate a term mapping.

    Parameters
    ----------
    source : DataFrame, path or str
        A DataFrame with the columns of the format, the path of a TSV, or
        the key of a shipped mapping (:data:`SHIPPED_MAPPINGS`).

    Returns
    -------
    pandas.DataFrame
        Columns ``atlas_label_name``, ``term``, ``hemi`` (``""`` where
        missing), all strings.

    Raises
    ------
    ValueError
        On a missing column, an empty name or term, a ``hemi`` that is not
        ``left``, ``right``, ``none`` or empty, a duplicate (name, term), or
        two rows of one name with different ``hemi``.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        if isinstance(source, str) and not os.path.exists(source):
            text = shipped_mapping_path(source).read_text(encoding="utf-8")
        else:
            with open(source, encoding="utf-8") as f:
                text = f.read()
        frame = pd.read_csv(
            io.StringIO(text), sep="\t", comment="#", dtype=str,
            keep_default_na=False,
        )
    return _validated(frame)


def _validated(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS[:2] if c not in frame.columns]
    if missing:
        raise ValueError(
            f"a term mapping needs the columns atlas_label_name and term, "
            f"missing {missing}; has {list(frame.columns)}"
        )
    if "hemi" not in frame.columns:
        frame = frame.assign(hemi="")
    frame = frame[COLUMNS].fillna("").astype(str).reset_index(drop=True)
    for column in COLUMNS[:2]:
        if (frame[column].str.strip() == "").any():
            raise ValueError(f"empty {column} in a term mapping")
    bad = sorted(set(frame["hemi"]) - set(HEMISPHERES) - {""})
    if bad:
        raise ValueError(
            f"hemi must be left, right, none or empty, not {bad}"
        )
    duplicated = frame.duplicated(["atlas_label_name", "term"], keep=False)
    if duplicated.any():
        raise ValueError(
            "duplicate (atlas_label_name, term) rows: "
            f"{frame[duplicated].drop_duplicates().values.tolist()}"
        )
    hemis = frame[frame["hemi"] != ""].groupby("atlas_label_name")["hemi"]
    inconsistent = hemis.nunique()
    inconsistent = inconsistent[inconsistent > 1].index.tolist()
    if inconsistent:
        raise ValueError(
            f"rows of the same name with different hemi: {inconsistent}"
        )
    return frame
