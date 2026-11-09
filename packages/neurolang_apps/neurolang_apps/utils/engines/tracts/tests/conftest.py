"""Fixtures of the the engine tracts tests (the marker ``network`` is the
repository's, ``pyproject.toml``)."""

import itertools

import numpy as np
import pytest


@pytest.fixture
def fetch_or_skip():
    """Call a fetcher; skip the test if it cannot get its data."""

    def fetch(fetcher, *args, **kwargs):
        try:
            return fetcher(*args, **kwargs)
        except Exception as e:  # offline, mirror down, cache unreadable
            pytest.skip(f"cannot fetch {fetcher.__name__}: {e!r}")

    return fetch


# Affines of the synthetic volume, voxel indices -> RAS mm. ``aligned``,
# ``flipped`` and ``permuted`` map the axes of the volume onto those of the
# space, so a box of voxels is a box of mm; ``oblique`` does not.
def _rotation(axis, degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    r = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def _affine(linear, translation):
    a = np.eye(4)
    a[:3, :3] = linear
    a[:3, 3] = translation
    return a


AFFINES = {
    "aligned": _affine(np.diag([2.0, 3.0, 4.0]), [-10.0, -20.0, -30.0]),
    "flipped": _affine(np.diag([-2.0, -3.0, 4.0]), [10.0, 20.0, -30.0]),
    "permuted": _affine(
        np.array([[0.0, 0.0, -4.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        [5.0, -7.0, 11.0],
    ),
    "oblique": _affine(
        _rotation(2, 25) @ _rotation(1, 15) @ np.diag([2.0, 3.0, 4.0]),
        [-10.0, -20.0, -30.0],
    ),
}
EXACT_AFFINES = ("aligned", "flipped", "permuted")


@pytest.fixture(params=sorted(AFFINES))
def affine_name(request):
    return request.param


def synthetic_volume():
    """A 12x10x8 volume with four labels of known voxels.

    1: a box, 2: a box, 3: a single voxel, 4: a diagonal line (whose box in
    the volume is much larger than the line).
    """
    vol = np.zeros((12, 10, 8), dtype=np.int32)
    vol[2:5, 1:4, 0:3] = 1
    vol[6:9, 5:8, 4:8] = 2
    vol[10, 9, 7] = 3
    for k in range(6):
        vol[k, k, k] = 4
    return vol


@pytest.fixture
def volume():
    return synthetic_volume()


def brute_force_extent(volume, affine, label, half_voxel=0.5):
    """World min and max over every corner of every voxel of ``label``,
    and, with ``half_voxel=0``, over their centres. Independent of the
    implementation: loops over voxels."""
    ijk = np.argwhere(volume == label)
    offsets = [-half_voxel, half_voxel] if half_voxel else [0.0]
    pts = [
        affine @ np.r_[v + np.array(d), 1.0]
        for v in ijk
        for d in itertools.product(offsets, repeat=3)
    ]
    pts = np.array(pts)[:, :3]
    return pts.min(axis=0), pts.max(axis=0)


def box_corners_extent(volume, affine, label):
    """World min and max over the 8 corners of the *box* of ``label``, from
    its first and last voxel, edges at half a voxel. Independent of
    ``scipy.ndimage.find_objects``."""
    ijk = np.argwhere(volume == label)
    lo, hi = ijk.min(axis=0) - 0.5, ijk.max(axis=0) + 0.5
    pts = np.array([
        affine @ np.r_[c, 1.0] for c in itertools.product(*zip(lo, hi))
    ])[:, :3]
    return pts.min(axis=0), pts.max(axis=0)
