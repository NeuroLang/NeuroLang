"""On-disk parquet cache of the atlas-dependent relations (DESIGN.md, 7.3)."""

import hashlib
import os
from pathlib import Path
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
    """Hex digest (32 characters) of the content of the tractogram and of
    the atlas, of ``params`` and of :data:`INDEXING_VERSION`.

    The ids of the tractogram and the name of the atlas are not in it, so
    that the same file under another id shares the entry; the ``a`` and
    ``t`` columns are put back when it is read.
    """
    h = hashlib.sha256(b"nl_tracts.cache")
    for part in (
        str(INDEXING_VERSION), tractogram.content_hash(),
        atlas.content_hash(), repr(params),
    ):
        h.update(part.encode() + b"\n")
    return h.hexdigest()[:32]


def _path(cache_dir: Union[str, os.PathLike], key: str, relation: str):
    return Path(cache_dir).expanduser() / key / f"{relation}.parquet"


def read_cached(
    cache_dir: Union[str, os.PathLike], key: str, relation: str
) -> Optional[pd.DataFrame]:
    """``<cache_dir>/<key>/<relation>.parquet`` if it exists, else None."""
    path = _path(cache_dir, key, relation)
    return pd.read_parquet(path) if path.is_file() else None


def write_cached(
    cache_dir: Union[str, os.PathLike],
    key: str,
    relation: str,
    frame: pd.DataFrame,
) -> None:
    """Write atomically (to a temporary name, then renamed: an interrupted
    write is never read as a result), zstd, row groups of 10^6."""
    path = _path(cache_dir, key, relation)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(
            temporary, engine="pyarrow", compression="zstd",
            row_group_size=1_000_000, index=False,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
