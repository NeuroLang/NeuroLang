"""Tractography for NeuroLang.

The core of the package (``spaces``, ``terms``, ``atlas``) turns an atlas
into pandas DataFrames, the extensional relations of DESIGN.md, and
depends only on numpy, scipy, nibabel and pandas.
"""

__all__ = [
    "Atlas",
    "ProbabilisticAtlas",
    "Space",
    "SpaceMismatchError",
]

from .atlas import Atlas, ProbabilisticAtlas
from .spaces import Space, SpaceMismatchError
