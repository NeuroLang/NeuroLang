"""Tractography for NeuroLang.

The core of the package (``atlas``, ``terms``, ``tractogram``,
``indexing``, ``cache``, ``fetch``) turns an atlas and a tractogram into pandas
DataFrames, the extensional relations of DESIGN.md, and depends only on
numpy, scipy, nibabel, pandas and the atlases of NeuroLang,
:mod:`neurolang.atlases`, which need nothing else of NeuroLang.
``nl_tracts.frontend`` hands them to a NeuroLang engine.
"""

__all__ = [
    "IndexingParams",
    "ProbabilisticAtlas",
    "Tractogram",
    "build_relations",
    "fetch_tractogram",
]

from .atlas import ProbabilisticAtlas
from .fetch import fetch_tractogram
from .indexing import IndexingParams, build_relations
from .tractogram import Tractogram
