"""On-disk parquet cache of the atlas-dependent relations (DESIGN.md, 7.3)."""

import os
from typing import Optional, Union

import pandas as pd

from .atlas import Atlas
from .indexing import IndexingParams
from .tractogram import Tractogram

__all__ = ["cache_key", "read_cached", "write_cached"]

#: Bumped whenever the indexing semantics change, to invalidate the caches.
INDEXING_VERSION = 1


def cache_key(
    tractogram: Tractogram, atlas: Atlas, params: IndexingParams
) -> str:
    """Hex digest of ``tractogram.content_hash()``, ``atlas.content_hash()``,
    ``params`` and :data:`INDEXING_VERSION`."""
    raise NotImplementedError


def read_cached(
    cache_dir: Union[str, os.PathLike], key: str, relation: str
) -> Optional[pd.DataFrame]:
    """``<cache_dir>/<key>/<relation>.parquet`` if it exists, else None."""
    raise NotImplementedError


def write_cached(
    cache_dir: Union[str, os.PathLike],
    key: str,
    relation: str,
    frame: pd.DataFrame,
) -> None:
    """Write atomically (temp file, then rename), zstd, row groups of 1M."""
    raise NotImplementedError
