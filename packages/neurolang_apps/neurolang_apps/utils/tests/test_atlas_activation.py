"""Tests of the studies activating the regions of an atlas."""

import nibabel as nib
import numpy as np
import pytest

from neurolang_apps.utils.engine_registry import atlas_activation


def atlas(affine):
    """A 10 x 10 x 10 atlas: region 1 is the voxel (2, 2, 2), region 2 the
    voxels (7, 7, 7) and (7, 7, 8)."""
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2, 2, 2] = 1
    data[7, 7, 7] = data[7, 7, 8] = 2
    return nib.Nifti1Image(data, affine)


NAMES = {1: "left", 2: "right"}
GRID_2MM = np.diag([2.0, 2.0, 2.0, 1.0])


def centre(affine, ijk):
    return nib.affines.apply_affine(affine, np.array(ijk, dtype=float))


@pytest.mark.parametrize("affine", [
    GRID_2MM,
    # flipped x axis and an offset origin, as the Schaefer atlas has
    np.array([[-2.0, 0, 0, 18.0], [0, 2.0, 0, -10.0], [0, 0, 2.0, 4.0],
              [0, 0, 0, 1.0]]),
])
def test_peaks_within_the_radius_activate_the_region(affine):
    left = centre(affine, (2, 2, 2))
    right = centre(affine, (7, 7, 8))
    peaks = np.array([
        left + [3.0, 0, 0],      # exactly 3 mm: inside
        left + [0, 3.01, 0],     # just beyond 3 mm
        right + [1.0, 1.0, 1.0],  # sqrt(3) mm from (7, 7, 8)
        right + [0, 0, 40.0],    # far
    ])
    studies = np.array([10, 11, 12, 13])
    pairs = atlas_activation(atlas(affine), NAMES, peaks, studies, 3.0)
    assert pairs == [("left", 10), ("right", 12)]


def test_each_study_once_per_region_and_chunks():
    affine = GRID_2MM
    peaks = np.repeat([centre(affine, (7, 7, 7))], 5, axis=0)
    studies = np.array([1, 1, 2, 2, 3])
    pairs = atlas_activation(atlas(affine), NAMES, peaks, studies, 3.0,
                             chunk=2)
    assert pairs == [("right", 1), ("right", 2), ("right", 3)]


def test_peaks_outside_the_grid():
    peaks = np.array([[-50.0, -50.0, -50.0], [500.0, 0, 0]])
    assert atlas_activation(
        atlas(GRID_2MM), NAMES, peaks, np.array([1, 2]), 3.0
    ) == []


# The pairs kept in a file: what they depend on, and what leaves them be.
PEAKS = np.array([
    centre(GRID_2MM, (2, 2, 2)) + [1.0, 0, 0],
    centre(GRID_2MM, (7, 7, 8)),
    [500.0, 0, 0],
])
STUDIES = np.array([10, 11, 12])


# The same atlas with its origin 1 mm away.
MOVED_GRID = GRID_2MM + np.array(
    [[0, 0, 0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
)


@pytest.fixture
def finds(monkeypatch):
    """The calls of atlas_activation that cached_atlas_activation makes."""
    from neurolang_apps.utils import engine_registry

    calls = []
    find = engine_registry.atlas_activation

    def spy(*args, **kwargs):
        calls.append(args)
        return find(*args, **kwargs)

    monkeypatch.setattr(engine_registry, "atlas_activation", spy)
    return calls


def cached(directory, peaks=PEAKS, studies=STUDIES, names=NAMES, radius=3.0,
           img=None):
    from neurolang_apps.utils.engine_registry import cached_atlas_activation

    img = atlas(GRID_2MM) if img is None else img
    return cached_atlas_activation(directory, img, names, peaks, studies,
                                   radius)


def test_the_pairs_are_found_once_and_then_read(tmp_path, finds):
    expected = [("left", 10), ("right", 11)]
    assert cached(tmp_path) == expected
    assert len(finds) == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    assert cached(tmp_path) == expected
    assert len(finds) == 1  # read, not found again


def test_the_pairs_read_are_those_found(tmp_path):
    from neurolang_apps.utils.engine_registry import atlas_activation

    found = atlas_activation(atlas(GRID_2MM), NAMES, PEAKS, STUDIES, 3.0)
    cached(tmp_path)
    assert cached(tmp_path) == found
    assert all(
        type(name) is str and type(study) is int for name, study in
        cached(tmp_path)
    )


@pytest.mark.parametrize("change", [
    {"radius": 2.0},
    {"peaks": PEAKS + [0.5, 0, 0]},
    {"studies": np.array([10, 11, 13])},
    {"names": {1: "left", 2: "RIGHT"}},
    {"img": atlas(MOVED_GRID)},
])
def test_what_the_pairs_depend_on_names_another_file(tmp_path, finds, change):
    cached(tmp_path)
    cached(tmp_path, **change)
    assert len(finds) == 2 and len(list(tmp_path.glob("*.npz"))) == 2


def test_noise_below_a_micrometre_in_the_peaks_finds_the_same_file(
    tmp_path, finds
):
    """numpy versions differ in the last bit of a coordinate; the browser
    must find the pairs a build on another platform kept."""
    cached(tmp_path)
    cached(tmp_path, peaks=PEAKS + 1e-9)
    assert len(finds) == 1


def test_string_studies_come_back_as_strings(tmp_path):
    studies = np.array(["a1", "b2", "c3"])
    expected = [("left", "a1"), ("right", "b2")]
    assert cached(tmp_path, studies=studies) == expected
    assert cached(tmp_path, studies=studies) == expected


def test_no_pairs_are_kept_and_read_too(tmp_path, finds):
    outside = np.array([[500.0, 0, 0]])
    assert cached(tmp_path, peaks=outside, studies=np.array([1])) == []
    assert cached(tmp_path, peaks=outside, studies=np.array([1])) == []
    assert len(finds) == 1


def test_a_damaged_file_is_found_again_and_replaced(tmp_path, finds):
    cached(tmp_path)
    [path] = tmp_path.glob("*.npz")
    path.write_bytes(b"not an npz")
    assert cached(tmp_path) == [("left", 10), ("right", 11)]
    assert len(finds) == 2
    assert cached(tmp_path) == [("left", 10), ("right", 11)]
    assert len(finds) == 2


def test_a_directory_that_cannot_be_written_only_costs_the_search(
    tmp_path, finds
):
    blocked = tmp_path / "file"
    blocked.write_text("a file, not a directory")
    assert cached(blocked / "cache") == [("left", 10), ("right", 11)]
    assert cached(blocked / "cache") == [("left", 10), ("right", 11)]
    assert len(finds) == 2


def test_an_engine_registers_the_pairs_it_read(tmp_path, finds):
    from neurolang.frontend import NeurolangPDL

    from neurolang_apps.utils import engine_registry

    nl = NeurolangPDL()
    rows = [(*peak, int(study)) for peak, study in zip(PEAKS, STUDIES)]
    nl.add_tuple_set(rows, name="peak_mni")
    for _ in range(2):
        engine_registry._register_atlas_activation(
            nl, NAMES, atlas(GRID_2MM), {}, "tiny", cache_dir=tmp_path
        )
        assert sorted(nl.symbol_table["tiny_activated"].value.unwrap()) == [
            ("left", 10), ("right", 11)
        ]
    assert len(finds) == 1
