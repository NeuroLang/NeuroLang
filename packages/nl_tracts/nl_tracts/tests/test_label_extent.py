"""``label_extent``: the world bounding box of each label, in mm.

Convention (DESIGN.md, 6.1): **voxel edges**. A label whose first and last
voxel are ``start`` and ``stop - 1`` along an axis spans, in voxel units,
``[start - 0.5, stop - 0.5]``: the physical extent of its voxels. The
reference, ``tract_querier.compute_label_bounding_boxes``, maps
``[start, stop]`` (centre of the first voxel to centre of the voxel *after*
the last), so it is shifted by half a voxel at both ends, and maps only two
corners, which gives the box only if the axes of the volume are those of the
space (each world axis depends on one voxel axis: aligned, flipped or
permuted; for an oblique affine it is not a box). nl_tracts maps the 8
corners of the box.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from nl_tracts.atlas import Atlas

from .conftest import (
    AFFINES,
    EXACT_AFFINES,
    box_corners_extent,
    brute_force_extent,
)

TABLE = pd.DataFrame(
    {"id": [0, 1, 2, 3, 4], "name": ["bg", "a", "b", "c", "d"]}
)


def atlas_of(volume, affine, **kw):
    return Atlas("x", volume, affine, TABLE, "s", **kw)


def extents(atlas):
    """{label: (lo (3,), hi (3,))} from the relation."""
    rel = atlas.label_extent_relation()
    out = {}
    for label, g in rel.groupby("l"):
        g = g.sort_values("axis")
        assert g["axis"].tolist() == [0, 1, 2]
        out[label] = (g["lo"].to_numpy(float), g["hi"].to_numpy(float))
    return out


def reference_extent(volume, affine, label):
    """``tract_querier.tract_label_indices.compute_label_bounding_boxes``,
    for one label, transcribed (the reference is not a dependency)."""
    sl = ndimage.find_objects(volume.astype(int))[label - 1]
    linear, translation = affine[:3, :3], affine[:-1, -1]
    pts = np.dot(linear, np.array([(s.start, s.stop) for s in sl])).T
    pts = pts + translation
    return pts.min(axis=0), pts.max(axis=0)  # BoundingBox: min and max


def test_relation_schema_and_dtypes(volume):
    rel = atlas_of(volume, AFFINES["aligned"]).label_extent_relation()
    assert rel.columns.tolist() == ["a", "l", "axis", "lo", "hi"]
    assert rel.dtypes["l"] == np.int32
    assert rel.dtypes["axis"] == np.int8
    assert rel.dtypes["lo"] == np.float32 and rel.dtypes["hi"] == np.float32
    assert rel["a"].tolist() == ["x"] * len(rel)
    assert len(rel) == 3 * 4  # three axes for each of labels 1-4


def test_axis_encoding_is_x_y_z_of_the_space(volume):
    rel = atlas_of(volume, AFFINES["aligned"]).label_extent_relation()
    # label 3 is the voxel (10, 9, 7): aligned affine diag(2,3,4)+(-10,-20,-30)
    one = rel[rel.l == 3].set_index("axis")
    assert one.loc[0, ["lo", "hi"]].tolist() == [9.0, 11.0]    # 2*10-10 +-1
    assert one.loc[1, ["lo", "hi"]].tolist() == [5.5, 8.5]     # 3*9-20 +-1.5
    assert one.loc[2, ["lo", "hi"]].tolist() == [-4.0, 0.0]    # 4*7-30 +-2


def test_no_row_for_the_background(volume):
    rel = atlas_of(volume, AFFINES["aligned"]).label_extent_relation()
    assert 0 not in rel["l"].tolist()


def test_a_label_of_the_table_absent_from_the_volume_has_no_row(volume):
    volume[volume == 2] = 0
    with pytest.warns(UserWarning):
        rel = atlas_of(volume, AFFINES["aligned"]).label_extent_relation()
    assert set(rel["l"]) == {1, 3, 4}


@pytest.mark.parametrize("name", EXACT_AFFINES)
def test_exact_for_axis_aligned_affines(volume, name):
    """Aligned, flipped (negative scales) and permuted axes: the box of
    voxel edges is the exact extent of the voxels."""
    affine = AFFINES[name]
    got = extents(atlas_of(volume, affine))
    assert set(got) == {1, 2, 3, 4}
    for label, (lo, hi) in got.items():
        true_lo, true_hi = brute_force_extent(volume, affine, label)
        np.testing.assert_allclose(lo, true_lo, atol=1e-4)
        np.testing.assert_allclose(hi, true_hi, atol=1e-4)
        assert (lo < hi).all()


def test_flipped_axes_have_lo_below_hi(volume):
    got = extents(atlas_of(volume, AFFINES["flipped"]))
    for lo, hi in got.values():
        assert (lo < hi).all()
    # label 3 = voxel (10, 9, 7), diag(-2, -3, 4) + (10, 20, -30)
    np.testing.assert_allclose(got[3][0], [-11.0, -8.5, -4.0], atol=1e-4)
    np.testing.assert_allclose(got[3][1], [-9.0, -5.5, 0.0], atol=1e-4)


def test_oblique_is_the_world_box_of_the_eight_corners(volume):
    affine = AFFINES["oblique"]
    got = extents(atlas_of(volume, affine))
    for label, (lo, hi) in got.items():
        exp_lo, exp_hi = box_corners_extent(volume, affine, label)
        np.testing.assert_allclose(lo, exp_lo, atol=1e-3)
        np.testing.assert_allclose(hi, exp_hi, atol=1e-3)


def test_oblique_contains_the_voxels_and_is_larger_for_a_diagonal_label(
    volume
):
    """The box of the volume box is a superset of the extent of the voxels
    under an oblique affine (DESIGN.md 6.1); the diagonal line shows it."""
    affine = AFFINES["oblique"]
    got = extents(atlas_of(volume, affine))
    larger = False
    for label, (lo, hi) in got.items():
        true_lo, true_hi = brute_force_extent(volume, affine, label)
        assert (lo <= true_lo + 1e-3).all() and (hi >= true_hi - 1e-3).all()
        larger |= bool(
            (lo < true_lo - 1e-3).any() or (hi > true_hi + 1e-3).any()
        )
    assert larger


def test_the_convention_is_voxel_edges_not_centres(volume):
    """A single voxel has the extent of one voxel, not zero."""
    got = extents(atlas_of(volume, AFFINES["aligned"]))
    lo, hi = got[3]
    np.testing.assert_allclose(hi - lo, [2.0, 3.0, 4.0], atol=1e-4)
    c_lo, c_hi = brute_force_extent(volume, AFFINES["aligned"], 3, 0)
    assert (c_hi - c_lo == 0).all()  # centres: a point


@pytest.mark.parametrize("name", EXACT_AFFINES)
def test_relation_to_the_reference_is_half_a_voxel(volume, name):
    """Reference [start, stop] vs ours [start - .5, stop - .5]: each end
    of each corner is shifted by ``linear @ (-.5, -.5, -.5)``."""
    affine = AFFINES[name]
    shift = affine[:3, :3] @ np.full(3, -0.5)
    got = extents(atlas_of(volume, affine))
    for label, (lo, hi) in got.items():
        ref_lo, ref_hi = reference_extent(volume, affine, label)
        np.testing.assert_allclose(lo, ref_lo + shift, atol=1e-4)
        np.testing.assert_allclose(hi, ref_hi + shift, atol=1e-4)


def test_sparse_large_ids(volume):
    """Destrieux's ids are in the 11000s: find_objects needs a list of
    that length, and the ids must come back as they are."""
    vol = np.zeros((6, 6, 6), np.int32)
    vol[1:3, 1:3, 1:3] = 11101
    vol[4, 4, 4] = 12175
    table = pd.DataFrame({"id": [11101, 12175], "name": ["a", "b"]})
    rel = Atlas("d", vol, np.eye(4), table, "s").label_extent_relation()
    assert set(rel["l"]) == {11101, 12175}
    one = rel[rel.l == 12175].sort_values("axis")
    assert one["lo"].tolist() == [3.5] * 3 and one["hi"].tolist() == [4.5] * 3


def test_one_pass_for_all_labels(volume, monkeypatch):
    calls = []
    original = ndimage.find_objects
    monkeypatch.setattr(
        ndimage, "find_objects",
        lambda *a, **k: calls.append(1) or original(*a, **k),
    )
    atlas_of(volume, AFFINES["aligned"]).label_extent_relation()
    assert calls == [1]


def test_the_relation_is_computed_once(volume):
    atlas = atlas_of(volume, AFFINES["aligned"])
    assert atlas.label_extent_relation() is not None
    first = atlas.label_extent_relation()
    again = atlas.label_extent_relation()
    pd.testing.assert_frame_equal(first, again)
