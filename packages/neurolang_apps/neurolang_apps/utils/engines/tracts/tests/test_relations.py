"""The atlas-independent relations of a tractogram, and the relation store."""

import numpy as np
import pandas as pd
import pytest

from neurolang.atlases import Atlas
from ..indexing import (
    IndexingParams,
    RelationStore,
    endpoint_coord_relation,
    endpoint_point_relation,
    endpoint_relation,
    streamline_extent_relation,
    streamline_length_relation,
    traverses_relation,
)
from ..tractogram import Tractogram

from .conftest import AFFINES

PARAMS = IndexingParams()


def lines(seed=0, n=15):
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(size=(rng.integers(2, 25), 3)) * 8).astype(np.float32)
        for _ in range(n)
    ]


def tractogram(id="t1", seed=0, **kw):
    return Tractogram.from_streamlines(id, lines(seed, **kw), "subject:1")


def test_streamline_extent_columns_dtypes_and_values():
    tg = tractogram()
    rel = streamline_extent_relation(tg, PARAMS)
    assert rel.columns.tolist() == ["t", "s", "axis", "lo", "hi"]
    assert pd.api.types.is_string_dtype(rel["t"])
    assert rel["t"].unique().tolist() == ["t1"]
    assert rel.dtypes["s"] == np.int32 and rel.dtypes["axis"] == np.int8
    assert rel.dtypes["lo"] == np.float32 and rel.dtypes["hi"] == np.float32
    assert len(rel) == 3 * tg.n_streamlines
    assert rel["axis"].tolist() == [0, 1, 2] * tg.n_streamlines
    for (s, axis), g in rel.groupby(["s", "axis"]):
        pts = lines()[s][:, axis]
        assert g["lo"].iloc[0] == pytest.approx(pts.min())
        assert g["hi"].iloc[0] == pytest.approx(pts.max())


def test_endpoint_coord_columns_dtypes_and_values():
    tg = tractogram()
    rel = endpoint_coord_relation(tg, PARAMS)
    assert rel.columns.tolist() == ["t", "s", "e", "axis", "v"]
    assert rel.dtypes["e"] == np.int8 and rel.dtypes["axis"] == np.int8
    assert rel.dtypes["v"] == np.float32
    assert len(rel) == 6 * tg.n_streamlines
    lookup = rel.set_index(["s", "e", "axis"])["v"]
    for s, line in enumerate(lines()):
        for axis in range(3):
            assert lookup[(s, 0, axis)] == line[0, axis]    # first point
            assert lookup[(s, 1, axis)] == line[-1, axis]   # last point


def test_endpoint_point_is_endpoint_coord_one_row_per_end():
    tg = tractogram()
    rel = endpoint_point_relation(tg, PARAMS)
    assert rel.columns.tolist() == ["t", "s", "e", "px", "py", "pz"]
    assert rel.dtypes["e"] == np.int8
    assert (rel.dtypes[["px", "py", "pz"]] == np.float32).all()
    assert len(rel) == 2 * tg.n_streamlines
    coord = endpoint_coord_relation(tg, PARAMS).pivot(
        index=["s", "e"], columns="axis", values="v")
    assert np.array_equal(
        rel.set_index(["s", "e"])[["px", "py", "pz"]].to_numpy(),
        coord.loc[list(zip(rel["s"], rel["e"]))].to_numpy())


def test_streamline_length_columns_dtypes_and_values():
    tg = tractogram()
    rel = streamline_length_relation(tg, PARAMS)
    assert rel.columns.tolist() == ["t", "s", "mm"]
    assert rel.dtypes["mm"] == np.float32
    expected = [
        np.linalg.norm(np.diff(x.astype(float), axis=0), axis=1).sum()
        for x in lines()
    ]
    np.testing.assert_allclose(rel["mm"], expected, rtol=1e-5)
    assert rel["s"].tolist() == list(range(15))


def test_the_length_threshold_leaves_streamlines_out_and_keeps_the_ids():
    tg = tractogram()
    lengths = tg.lengths_in_mm()
    threshold = float(np.median(lengths))
    params = IndexingParams(length_threshold=threshold)
    kept = np.flatnonzero(lengths >= threshold)
    assert 0 < len(kept) < tg.n_streamlines
    for relation, ids in (
        (streamline_extent_relation(tg, params), "s"),
        (endpoint_coord_relation(tg, params), "s"),
        (streamline_length_relation(tg, params), "s"),
    ):
        # not renumbered
        assert sorted(relation[ids].unique()) == kept.tolist()


def test_the_relations_are_vectorised_not_looped():
    """No Python call per streamline: 20000 streamlines in well under the
    time of a loop (a loose bound against a regression to one)."""
    import time

    rng = np.random.default_rng(0)
    big = Tractogram.from_streamlines(
        "t", [rng.normal(size=(30, 3)).astype(np.float32)
              for _ in range(20000)], "s",
    )
    start = time.time()
    streamline_extent_relation(big, PARAMS)
    endpoint_coord_relation(big, PARAMS)
    streamline_length_relation(big, PARAMS)
    assert time.time() - start < 5


# the store
def atlas(name="fs", space="subject:1"):
    volume = np.zeros((6, 6, 6), np.int32)
    volume[1:3, 1:3, 1:3] = 1
    volume[4, 4, 4] = 2
    table = pd.DataFrame({"id": [0, 1, 2], "name": ["bg", "Left-A", "B"]})
    return Atlas(name, volume, AFFINES["aligned"], table, space)


MAPPING = pd.DataFrame(
    {"atlas_label_name": ["Left-A"], "term": ["a"], "hemi": ["left"]}
)


def test_add_atlas_and_add_tractogram_fill_the_relations():
    store = RelationStore()
    store.add_atlas(atlas(), MAPPING)
    store.add_tractogram(tractogram(), PARAMS)
    assert set(store.relations) == {
        "atlas", "label", "label_term", "label_extent",
        "tractogram", "streamline", "streamline_extent", "endpoint_coord",
        "endpoint_point", "streamline_length",
    }
    assert store.relations["atlas"]["a"].tolist() == ["fs"]
    assert store.relations["tractogram"]["t"].tolist() == ["t1"]
    assert store.tractograms == ["t1"]
    streamline = store.relations["streamline"]
    assert streamline.columns.tolist() == ["x", "t", "s"]
    assert streamline["s"].tolist() == list(range(15))
    # the first tractogram: x is s
    assert streamline["x"].tolist() == list(range(15))
    assert streamline.dtypes["x"] == np.int64
    assert streamline.dtypes["s"] == np.int32
    assert pd.api.types.is_string_dtype(streamline["t"])
    for name, columns in (
        ("streamline_extent", ["x", "axis", "lo", "hi"]),
        ("endpoint_coord", ["x", "e", "axis", "v"]),
        ("endpoint_point", ["x", "e", "px", "py", "pz"]),
        ("streamline_length", ["x", "mm"]),
    ):
        frame = store.relations[name]
        assert frame.columns.tolist() == columns, name
        assert frame.dtypes["x"] == np.int64, name
    assert len(store.relations["label"]) == 3
    assert store.relations["label_term"][["l", "term"]].values.tolist() == [
        [1, "a"]]
    assert len(store.relations["label_extent"]) == 6


def test_several_atlases_and_tractograms_accumulate():
    store = RelationStore()
    store.add_atlas(atlas("one"))
    store.add_atlas(atlas("two", space="subject:2"))
    store.add_tractogram(tractogram("a", seed=0), PARAMS)
    store.add_tractogram(tractogram("b", seed=1, n=4), PARAMS)
    assert store.relations["atlas"]["a"].tolist() == ["one", "two"]
    assert store.relations["tractogram"]["t"].tolist() == ["a", "b"]
    assert store.tractograms == ["a", "b"]
    streamline = store.relations["streamline"]
    assert len(streamline) == 15 + 4
    # x = (index << 32) | s, the index being the order of addition
    assert streamline["x"].tolist() == list(range(15)) + [
        4294967296, 4294967297, 4294967298, 4294967299]
    assert streamline["t"].tolist() == ["a"] * 15 + ["b"] * 4
    assert streamline["s"].tolist() == list(range(15)) + list(range(4))
    extent = store.relations["streamline_extent"]
    assert set(extent["x"]) == set(streamline["x"])
    # the rows of b are its own: x decodes to the streamline they are of
    b = extent[extent["x"] == 4294967296 + 2].set_index("axis")
    second = lines(1, n=4)[2]
    for axis in range(3):
        assert b.loc[axis, "lo"] == second[:, axis].min()
        assert b.loc[axis, "hi"] == second[:, axis].max()
    assert set(store.relations["label"]["a"]) == {"one", "two"}
    assert store.relations["label_extent"]["a"].nunique() == 2


def test_an_atlas_or_a_tractogram_cannot_be_added_twice():
    store = RelationStore()
    store.add_atlas(atlas())
    store.add_tractogram(tractogram(), PARAMS)
    with pytest.raises(ValueError, match="fs"):
        store.add_atlas(atlas())
    with pytest.raises(ValueError, match="t1"):
        store.add_tractogram(tractogram(), PARAMS)


# the index
def near_the_label(id="t1", seed=0):
    """Streamlines around the first label of ``atlas()`` (x -8..-4, y
    -17..-11, z -30..-22 mm), so that they have labels."""
    rng = np.random.default_rng(seed)
    lines = [
        (rng.normal(size=(rng.integers(2, 20), 3)) * 3
         + [-6, -14, -26]).astype(np.float32)
        for _ in range(30)
    ]
    return Tractogram.from_streamlines(id, lines, "subject:1")


def add_and_index(store, tractogram, atlas, params=PARAMS, **kw):
    if tractogram.id not in store.tractograms:
        store.add_tractogram(tractogram, params)
    store.add_index(tractogram, atlas, params, **kw)


def test_add_index_adds_streamline_label_and_endpoint_label():
    store = RelationStore()
    store.add_tractogram(near_the_label("t0", seed=1), PARAMS)
    tg = near_the_label()
    store.add_tractogram(tg, PARAMS)
    before = set(store.relations)
    store.add_index(tg, atlas(), PARAMS)
    assert set(store.relations) - before == {
        "streamline_label", "endpoint_label"}
    label = store.relations["streamline_label"]
    ends = store.relations["endpoint_label"]
    assert label.columns.tolist() == ["a", "x", "l"]
    assert ends.columns.tolist() == ["a", "x", "e", "l"]
    assert set(label["a"]) == {"fs"}
    assert label.dtypes["x"] == np.int64 and ends.dtypes["x"] == np.int64
    assert label.dtypes["l"] == np.int32 and ends.dtypes["e"] == np.int8
    # the rows of traverses_relation and endpoint_relation, with x for
    # (t, s): the second tractogram, so x = 2 ** 32 + s
    want = traverses_relation(tg, atlas(), PARAMS)
    assert len(want) > 0
    assert sorted(zip(label["x"], label["l"])) == sorted(
        zip(want["s"].astype(np.int64) + 2 ** 32, want["l"]))
    want = endpoint_relation(tg, atlas(), PARAMS)
    assert sorted(zip(ends["x"], ends["e"], ends["l"])) == sorted(
        zip(want["s"].astype(np.int64) + 2 ** 32, want["e"], want["l"]))


def test_add_index_needs_the_tractogram_added():
    """Its index in x is the order in which it was added."""
    store = RelationStore()
    with pytest.raises(ValueError, match="add_tractogram"):
        store.add_index(near_the_label(), atlas(), PARAMS)
    assert "streamline_label" not in store.relations


def test_add_index_accumulates_atlases_and_tractograms():
    store = RelationStore()
    for a in ("one", "two"):
        for t in ("a", "b"):
            add_and_index(store, near_the_label(t), atlas(a))
    frame = store.relations["streamline_label"].merge(
        store.relations["streamline"], on="x")
    assert set(zip(frame["a"], frame["t"])) == {
        ("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")}
    with pytest.raises(ValueError, match="one.*a|already"):
        store.add_index(near_the_label("a"), atlas("one"), PARAMS)


def test_add_index_checks_the_spaces():
    from neurolang.atlases import SpaceMismatchError

    store = RelationStore()
    store.add_tractogram(near_the_label(), PARAMS)
    with pytest.raises(SpaceMismatchError):
        store.add_index(near_the_label(), atlas(space="MNI"), PARAMS)
    assert "streamline_label" not in store.relations
    with pytest.warns(UserWarning, match="spaces differ"):
        store.add_index(
            near_the_label(), atlas(space="MNI"), PARAMS,
            allow_space_mismatch=True,
        )
    assert "streamline_label" in store.relations


def test_build_relations_is_the_whole_store_for_one_pair():
    from ..indexing import build_relations

    store = build_relations(near_the_label(), atlas(), PARAMS)
    assert set(store.relations) == {
        "atlas", "label", "label_term", "label_extent", "tractogram",
        "streamline", "streamline_extent", "endpoint_coord", "endpoint_point",
        "streamline_length", "streamline_label", "endpoint_label",
    }
