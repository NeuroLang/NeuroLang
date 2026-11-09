import numpy as np
import pandas as pd
import pytest

import nl_tracts.cache as cache
from nl_tracts.atlas import Atlas
from nl_tracts.cache import cache_key, read_cached, write_cached
from nl_tracts.indexing import IndexingParams, RelationStore
from nl_tracts.tractogram import Tractogram

from .conftest import AFFINES, synthetic_volume

TABLE = pd.DataFrame({"id": [0, 1, 2, 3, 4], "name": list("abcde")})


def atlas(name="fs", **kw):
    return Atlas(name, synthetic_volume(), AFFINES["aligned"], TABLE,
                 kw.pop("space", "s"), **kw)


def tractogram(id="t1", seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    lines = [
        (rng.normal(size=(rng.integers(2, 20), 3)) * 8 + [-4, -14, -26]
         + shift).astype(np.float32)
        for _ in range(30)
    ]
    return Tractogram.from_streamlines(id, lines, "s")


def test_the_key_depends_on_the_content_and_the_parameters_only():
    base = cache_key(tractogram(), atlas(), IndexingParams())
    assert base == cache_key(tractogram(), atlas(), IndexingParams())
    # not on the names: the same file under another id, an atlas renamed
    assert base == cache_key(tractogram("other"), atlas("other"),
                             IndexingParams())
    for key in (
        cache_key(tractogram(shift=1.0), atlas(), IndexingParams()),
        cache_key(tractogram(), atlas(space="other"), IndexingParams()),
        cache_key(tractogram(), atlas(background=1), IndexingParams()),
        cache_key(tractogram(), atlas(),
                  IndexingParams(crossing_threshold=5.0)),
        cache_key(tractogram(), atlas(), IndexingParams(length_threshold=3.0)),
        cache_key(tractogram(), atlas(), IndexingParams(emit_background=True)),
    ):
        assert key != base
    assert len(base) == 32 and int(base, 16) >= 0


def test_the_key_changes_with_the_version_of_the_indexing(monkeypatch):
    base = cache_key(tractogram(), atlas(), IndexingParams())
    monkeypatch.setattr(cache, "INDEXING_VERSION", cache.INDEXING_VERSION + 1)
    assert cache_key(tractogram(), atlas(), IndexingParams()) != base


def test_round_trip_keeps_the_types(tmp_path):
    frame = pd.DataFrame(
        {"s": np.array([0, 5], np.int32), "e": np.array([0, 1], np.int8),
         "l": np.array([3, 9], np.int32)}
    )
    assert read_cached(tmp_path, "k", "endpoint") is None
    write_cached(tmp_path, "k", "endpoint", frame)
    assert (tmp_path / "k" / "endpoint.parquet").is_file()
    back = read_cached(tmp_path, "k", "endpoint")
    pd.testing.assert_frame_equal(back, frame)


def test_a_failed_write_leaves_nothing_that_is_read(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs):
        open(args[0], "wb").write(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError):
        write_cached(tmp_path, "k", "traverses",
                     pd.DataFrame({"s": np.array([1], np.int32)}))
    assert read_cached(tmp_path, "k", "traverses") is None
    assert not list((tmp_path / "k").glob("*")) or all(
        p.suffix != ".parquet" for p in (tmp_path / "k").glob("*")
    )


def test_add_index_writes_the_cache_and_a_second_store_reads_it(
    tmp_path, monkeypatch
):
    import nl_tracts.indexing as indexing

    params = IndexingParams(crossing_threshold=3.0, emit_background=True)
    first = RelationStore()
    first.add_tractogram(tractogram(), params)
    first.add_index(tractogram(), atlas(), params, cache_dir=tmp_path)
    key = cache_key(tractogram(), atlas(), params)
    assert (tmp_path / key / "traverses.parquet").is_file()
    assert (tmp_path / key / "endpoint.parquet").is_file()

    def no_indexing(*args, **kwargs):
        raise AssertionError("indexed again, not read from the cache")

    monkeypatch.setattr(indexing, "_index", no_indexing)
    second = RelationStore()
    # the same file under another id and another atlas name, as the second
    # tractogram of the store: its x are those of the first plus 2 ** 32
    second.add_tractogram(tractogram("before", seed=1), params)
    second.add_tractogram(tractogram("another"), params)
    second.add_index(
        tractogram("another"), atlas("other"), params, cache_dir=tmp_path
    )
    for relation in ("streamline_label", "endpoint_label"):
        got = second.relations[relation]
        want = first.relations[relation].copy()
        assert got["a"].unique().tolist() == ["other"]
        want["x"] += 2 ** 32
        pd.testing.assert_frame_equal(
            got.drop(columns=["a"]), want.drop(columns=["a"])
        )
        assert got.columns.tolist() == want.columns.tolist()
        assert got.dtypes.equals(want.dtypes)
