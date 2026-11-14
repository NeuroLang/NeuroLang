"""The indexing of a tractogram by an atlas: ``traverses`` and ``endpoint``.

The oracle is tract_querier's ``TractographySpatialIndexing`` on the same
volume, affine and streamlines (DESIGN.md, 6.1): the rounding of points to
voxels (half to even), the clipping with one warning, the fraction of the
points of a streamline in a label against the threshold, the denominator
that counts the background, the endpoints, the length threshold. The reference
numbers the streamlines that survive the length filter again; ours do not.
It also has a quirk at a threshold of 0 (it keeps the labels that no point
has, up to the largest one a streamline has), which is not reproduced, so
the thresholds here are above 0.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from neurolang.atlases import Atlas
from ..indexing import (
    IndexingParams,
    endpoint_relation,
    points_to_voxels,
    traverses_relation,
)
from ..tractogram import Tractogram

from .conftest import AFFINES
from .reference_tq import load_indexing

SHAPE = (15, 12, 9)
ATLAS_TABLE = pd.DataFrame(
    {"id": range(8), "name": [f"r{i}" for i in range(8)]}
)


def blocky_volume(seed=0):
    """Labels 0-7 in blocks of 3x3x3 voxels, so that streamlines cross
    several labels and sometimes stay in one."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 8, (5, 4, 3))
    return np.kron(blocks, np.ones((3, 3, 3), int)).astype(np.int32)


def atlas_of(affine_name, seed=0, **kw):
    kw.setdefault("space", "s")
    return Atlas(
        "fs", blocky_volume(seed), AFFINES[affine_name], ATLAS_TABLE, **kw
    )


def streamlines(affine, n=80, seed=0):
    """Random walks around the centre of the volume, some leaving it."""
    rng = np.random.default_rng(100 + seed)
    centre = (affine @ np.r_[np.array(SHAPE) / 2, 1])[:3]
    scale = np.abs(affine[:3, :3]).sum(axis=0).mean()
    lines = []
    for _ in range(n):
        k = rng.integers(2, 45)
        steps = rng.normal(size=(k, 3)) * scale * 0.7
        lines.append(
            (centre + np.cumsum(steps, axis=0)
             + rng.normal(size=3) * scale * 2).astype(np.float32)
        )
    return lines


def reference_sets(lines, atlas, params, reference):
    """The reference's crossing and ending sets, as {id: set} and
    {id: label}, in the ids of ``lines`` (it numbers them again after the
    length filter)."""
    index = reference.TractographySpatialIndexing(
        [x.astype(np.float64) for x in lines], atlas.labels, atlas.affine,
        params.length_threshold, params.crossing_threshold,
    )
    if params.length_threshold > 0:
        kept = [
            i for i, x in enumerate(lines)
            if ((((x[1:] - x[:-1]).astype(float) ** 2).sum(1)) ** .5).sum()
            >= params.length_threshold
        ]
    else:
        kept = list(range(len(lines)))
    crossing = {
        kept[j]: set(map(int, labels))
        for j, labels in index.crossing_tracts_labels.items()
    }
    ends = tuple(
        {kept[j]: int(label) for j, label in e.items()}
        for e in index.ending_tracts_labels
    )
    return crossing, ends


def rows(frame, *columns):
    return {tuple(r) for r in frame[list(columns)].itertuples(index=False)}


@pytest.fixture(scope="module")
def reference():
    return load_indexing()


@pytest.mark.parametrize("affine_name", sorted(AFFINES))
@pytest.mark.parametrize("threshold", [0.5, 2.0, 25.0, 50.0, 100.0])
def test_traverses_and_endpoint_against_the_reference(
    reference, affine_name, threshold
):
    atlas = atlas_of(affine_name)
    lines = streamlines(AFFINES[affine_name])
    tg = Tractogram.from_streamlines("t1", lines, "s")
    params = IndexingParams(
        crossing_threshold=threshold, emit_background=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # points outside, on both sides
        crossing, ends = reference_sets(lines, atlas, params, reference)
        traverses = traverses_relation(tg, atlas, params)
        endpoint = endpoint_relation(tg, atlas, params)

    assert rows(traverses, "s", "l") == {
        (s, label) for s, labels in crossing.items() for label in labels
    }
    # no endpoint of the background (6.3): the reference's, without label 0
    assert rows(endpoint, "s", "e", "l") == {
        (s, e, label) for e in (0, 1) for s, label in ends[e].items()
        if label != 0
    }


@pytest.mark.parametrize("affine_name", ["aligned", "oblique"])
def test_the_length_threshold_leaves_streamlines_out_and_keeps_the_ids(
    reference, affine_name
):
    atlas = atlas_of(affine_name)
    lines = streamlines(AFFINES[affine_name])
    tg = Tractogram.from_streamlines("t1", lines, "s")
    lengths = tg.lengths_in_mm()
    threshold = float(np.median(lengths))
    params = IndexingParams(
        crossing_threshold=10.0, length_threshold=threshold,
        emit_background=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crossing, ends = reference_sets(lines, atlas, params, reference)
        traverses = traverses_relation(tg, atlas, params)
        endpoint = endpoint_relation(tg, atlas, params)
    kept = set(np.flatnonzero(lengths >= threshold).tolist())
    assert 0 < len(kept) < len(lines)
    assert set(crossing) == kept
    assert set(traverses["s"]) <= kept  # never renumbered
    assert rows(traverses, "s", "l") == {
        (s, label) for s, labels in crossing.items() for label in labels
    }
    assert set(endpoint["s"]) <= kept


def test_the_background_is_in_the_denominator_but_not_in_the_rows_by_default():
    """A streamline with 3 of its 4 points in the background and 1 in
    label 2 (25 %): label 2 at a threshold of 25 or less, not above, and the
    background has no row unless asked for."""
    volume = np.zeros((6, 6, 6), np.int32)
    volume[3:, :, :] = 2
    table = pd.DataFrame({"id": [0, 2], "name": ["bg", "two"]})
    atlas = Atlas("a", volume, np.eye(4), table, "s")
    line = np.array([[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]], np.float32)
    tg = Tractogram.from_streamlines("t", [line], "s")
    for threshold, expected in ((25.0, {2}), (26.0, set())):
        params = IndexingParams(crossing_threshold=threshold)
        assert set(traverses_relation(tg, atlas, params)["l"]) == expected
    with_background = IndexingParams(
        crossing_threshold=25.0, emit_background=True
    )
    assert set(traverses_relation(tg, atlas, with_background)["l"]) == {0, 2}


def test_endpoints_in_the_background_have_no_row_whatever_the_flag():
    volume = np.zeros((6, 6, 6), np.int32)
    volume[3:, :, :] = 2
    table = pd.DataFrame({"id": [0, 2], "name": ["bg", "two"]})
    atlas = Atlas("a", volume, np.eye(4), table, "s")
    line = np.array([[0, 1, 1], [5, 1, 1]], np.float32)  # background, two
    tg = Tractogram.from_streamlines("t", [line], "s")
    for flag in (False, True):
        endpoint = endpoint_relation(
            tg, atlas, IndexingParams(emit_background=flag)
        )
        assert rows(endpoint, "s", "e", "l") == {(0, 1, 2)}


def test_columns_and_dtypes():
    atlas = atlas_of("aligned")
    tg = Tractogram.from_streamlines(
        "t1", streamlines(AFFINES["aligned"]), "s"
    )
    traverses = traverses_relation(tg, atlas, IndexingParams())
    endpoint = endpoint_relation(tg, atlas, IndexingParams())
    assert traverses.columns.tolist() == ["a", "t", "s", "l"]
    assert endpoint.columns.tolist() == ["a", "t", "s", "e", "l"]
    for frame in (traverses, endpoint):
        assert pd.api.types.is_string_dtype(frame["a"])
        assert frame["a"].unique().tolist() == ["fs"]
        assert frame["t"].unique().tolist() == ["t1"]
        assert frame.dtypes["s"] == np.int32 and frame.dtypes["l"] == np.int32
    assert endpoint.dtypes["e"] == np.int8


def test_chunks_give_the_same_relations(monkeypatch):
    import neurolang_apps.utils.engines.tracts.indexing as indexing

    atlas = atlas_of("oblique")
    tg = Tractogram.from_streamlines(
        "t1", streamlines(AFFINES["oblique"], n=60), "s"
    )
    params = IndexingParams(crossing_threshold=5.0, emit_background=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = traverses_relation(tg, atlas, params), \
            endpoint_relation(tg, atlas, params)
        monkeypatch.setattr(indexing, "CHUNK_POINTS", 50)
        chunked = traverses_relation(tg, atlas, params), \
            endpoint_relation(tg, atlas, params)
    for a, b in zip(whole, chunked):
        pd.testing.assert_frame_equal(a, b)


# the points to voxels
def test_half_voxel_coordinates_round_to_even():
    """np.round, as the reference: 0.5 and 2.5 to 0 and 2, 1.5 to 2."""
    pts = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 0.4999]])
    ijk = points_to_voxels(pts, np.eye(4), (10, 10, 10))
    np.testing.assert_array_equal(ijk, [[0, 2, 2], [4, 4, 0]])


def test_points_are_mapped_by_the_inverse_of_the_affine():
    affine = AFFINES["flipped"]
    voxels = np.array([[1, 2, 3], [4, 0, 7]])
    world = (affine @ np.c_[voxels, np.ones(2)].T).T[:, :3]
    np.testing.assert_array_equal(
        points_to_voxels(world, affine, (15, 12, 9)), voxels
    )


def test_points_outside_are_clipped_with_one_warning_that_counts_them():
    pts = np.array([[-3.0, 1, 1], [20.0, 1, 1], [2.0, 2, 2], [2, 2, -4]])
    with pytest.warns(UserWarning, match="3 of 4.*outside") as caught:
        ijk = points_to_voxels(pts, np.eye(4), (10, 10, 10))
    assert len(caught) == 1
    np.testing.assert_array_equal(
        ijk, [[0, 1, 1], [9, 1, 1], [2, 2, 2], [2, 2, 0]]
    )


def test_no_warning_when_every_point_is_inside():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        points_to_voxels(np.array([[1.0, 1, 1]]), np.eye(4), (4, 4, 4))


def test_a_traversal_run_warns_once_for_all_its_chunks(monkeypatch):
    import neurolang_apps.utils.engines.tracts.indexing as indexing

    atlas = atlas_of("aligned")
    tg = Tractogram.from_streamlines(
        "t", [np.array([[500.0, 500, 500], [501, 501, 501]])] * 30, "s"
    )
    monkeypatch.setattr(indexing, "CHUNK_POINTS", 10)
    with pytest.warns(UserWarning, match="outside") as caught:
        traverses_relation(tg, atlas, IndexingParams())
    assert len(caught) == 1


# parameters
@pytest.mark.parametrize(
    "kwargs", [{"crossing_threshold": -1.0}, {"crossing_threshold": 100.5},
               {"length_threshold": -1.0}],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        IndexingParams(**kwargs)
