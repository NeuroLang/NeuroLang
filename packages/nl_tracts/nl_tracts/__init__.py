"""Tractography for NeuroLang.

The core of the package (``spaces``, ``atlas``, ``tractogram``,
``indexing``, ``cache``, ``fetch``) turns an atlas and a tractogram into pandas
DataFrames, the extensional relations of DESIGN.md, and depends only on
numpy, scipy, nibabel and pandas. ``nl_tracts.frontend`` hands them to a
NeuroLang engine.
"""

__all__ = [
    "Atlas",
    "IndexingParams",
    "ProbabilisticAtlas",
    "Space",
    "SpaceMismatchError",
    "Tractogram",
    "build_relations",
    "fetch_tractogram",
]

from .atlas import Atlas, ProbabilisticAtlas
from .fetch import fetch_tractogram
from .indexing import IndexingParams, build_relations
from .spaces import Space, SpaceMismatchError
from .tractogram import Tractogram
