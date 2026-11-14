"""Tractography for NeuroLang.

The core of the package (``atlas``, ``terms``, ``tractogram``,
``indexing``, ``cache``, ``fetch``) turns an atlas and a tractogram into pandas
DataFrames, the extensional relations of DESIGN.md, and depends only on
numpy, scipy, nibabel, pandas and the atlases of NeuroLang,
:mod:`neurolang.atlases`, which need nothing else of NeuroLang.
``frontend`` hands them to a NeuroLang engine, and
``vocabulary.dl`` is the vocabulary of rules over them.
"""

__all__ = [
    "IndexingParams",
    "ProbabilisticAtlas",
    "Tractogram",
    "build_relations",
    "fetch_tractogram",
    "vocabulary_predicates",
]

from .atlas import ProbabilisticAtlas
from .fetch import fetch_tractogram
from .indexing import IndexingParams, build_relations
from .tractogram import Tractogram


def vocabulary_predicates():
    """The predicates of the vocabulary, ``vocabulary.dl``, in its order."""
    import re
    from importlib.resources import files

    text = files(__package__).joinpath("vocabulary.dl").read_text()
    heads = re.findall(r"^(\w+)\(", text, flags=re.MULTILINE)
    return tuple(dict.fromkeys(heads))
