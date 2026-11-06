import nibabel as nib
import numpy as np
import pytest

from nl_tracts.tractogram import Tractogram


def streamlines(seed=0, n=12):
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(size=(rng.integers(2, 30), 3)) * 10).astype(np.float32)
        for _ in range(n)
    ]


def make(**kw):
    kw.setdefault("id", "100307")
    kw.setdefault("space", "subject:100307")
    return Tractogram.from_streamlines(streamlines=streamlines(), **kw)


def test_flat_arrays_and_offsets():
    lines = streamlines()
    tg = make()
    assert tg.id == "100307" and tg.space == "subject:100307"
    assert tg.n_streamlines == len(lines)
    assert tg.points.shape == (sum(map(len, lines)), 3)
    assert tg.points.dtype == np.float32
    assert tg.offsets.dtype == np.int64
    assert tg.offsets[0] == 0 and tg.offsets[-1] == len(tg.points)
    for s, line in enumerate(lines):
        np.testing.assert_array_equal(
            tg.points[tg.offsets[s]:tg.offsets[s + 1]], line
        )


def test_the_constructor_takes_the_flat_arrays():
    lines = streamlines()
    points = np.concatenate(lines)
    offsets = np.cumsum([0] + [len(x) for x in lines])
    tg = Tractogram("t", points, offsets, "s")
    assert tg.n_streamlines == len(lines)
    np.testing.assert_array_equal(tg.lengths_in_points(),
                                  [len(x) for x in lines])


def test_from_a_nibabel_array_sequence():
    lines = streamlines()
    seq = nib.streamlines.ArraySequence(lines)
    tg = Tractogram.from_streamlines("t", seq, "s")
    np.testing.assert_array_equal(tg.points, np.concatenate(lines))


def test_lengths_in_mm_against_a_loop():
    lines = streamlines()
    tg = make()
    expected = [
        np.linalg.norm(np.diff(x.astype(float), axis=0), axis=1).sum()
        for x in lines
    ]
    np.testing.assert_allclose(tg.lengths_in_mm(), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "points,offsets,match",
    [
        (np.zeros((4, 3)), [0, 4, 5], "offsets"),           # past the points
        (np.zeros((4, 3)), [1, 4], "offsets"),              # not from 0
        (np.zeros((4, 3)), [0, 3, 2, 4], "offsets"),        # not increasing
        (np.zeros((4, 2)), [0, 4], "points"),               # not 3D
        (np.zeros((5, 3)), [0, 3, 4, 5], "streamline 1"),   # one point
    ],
)
def test_invalid_tractograms_are_rejected(points, offsets, match):
    with pytest.raises(ValueError, match=match):
        Tractogram("t", points, np.asarray(offsets), "s")


def test_a_streamline_of_one_point_is_named():
    with pytest.raises(ValueError, match="streamline 2.*2 points"):
        Tractogram.from_streamlines(
            "t", [np.zeros((3, 3)), np.ones((4, 3)), np.zeros((1, 3))], "s"
        )


@pytest.mark.parametrize("suffix", [".trk", ".tck"])
def test_from_file_reads_trk_and_tck(tmp_path, suffix):
    lines = streamlines()
    tractogram = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    path = tmp_path / f"tracts{suffix}"
    header = None
    if suffix == ".trk":
        header = {
            nib.streamlines.Field.VOXEL_TO_RASMM: np.eye(4),
            nib.streamlines.Field.VOXEL_SIZES: (1, 1, 1),
            nib.streamlines.Field.DIMENSIONS: (200, 200, 200),
            nib.streamlines.Field.VOXEL_ORDER: "RAS",
        }
    nib.streamlines.save(tractogram, str(path), header=header)
    tg = Tractogram.from_file("t", path, "s")
    assert tg.n_streamlines == len(lines)
    np.testing.assert_allclose(tg.points, np.concatenate(lines), atol=1e-4)


def test_from_file_goes_to_rasmm(tmp_path):
    """A file whose points are in voxels is read in mm, in RAS+."""
    lines = [np.array([[0, 0, 0], [1, 2, 3.0]], np.float32)]
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    tractogram = nib.streamlines.Tractogram(lines, affine_to_rasmm=affine)
    path = tmp_path / "t.tck"
    nib.streamlines.save(tractogram, str(path))
    tg = Tractogram.from_file("t", path, "s")
    # the file is written in mm (rasmm), and read back as written
    np.testing.assert_allclose(tg.points, np.array(
        [[10, 20, 30], [12, 26, 42.0]]
    ), atol=1e-4)


def test_content_hash():
    base = make().content_hash()
    assert base == make().content_hash()
    # not the id; the points, the offsets, the space
    assert base == make(id="other").content_hash()
    assert base != make(space="subject:2").content_hash()
    lines = streamlines()
    lines[0] = lines[0] + 1
    assert base != Tractogram.from_streamlines(
        "t", lines, "subject:100307"
    ).content_hash()
    lines = streamlines()
    merged = [np.concatenate(lines[:2])] + lines[2:]
    assert base != Tractogram.from_streamlines(
        "t", merged, "subject:100307"
    ).content_hash()
