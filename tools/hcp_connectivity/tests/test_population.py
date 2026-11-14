"""The assignment of streamline ends to regions, on a tiny tractogram."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hcp_connectivity.population import assign, count, endpoints, tractograms  # noqa: E402,E501

# a 6 x 3 x 3 grid of 2 mm voxels, its origin at (0, 0, 0): region 1 in
# voxels i = 0, region 2 in i = 1, region 3 in i = 5
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


def volume():
    labels = np.zeros((6, 3, 3), dtype=int)
    labels[0, 1, 1], labels[1, 1, 1], labels[5, 1, 1] = 1, 2, 3
    return labels


def test_assign_nearest_voxel_within_the_radius():
    points = np.array([[0.0, 2.0, 2.0],    # on region 1
                       [0.9, 2.0, 2.0],    # between 1 and 2, nearer 1
                       [3.0, 2.0, 2.0],    # nearer region 2
                       [6.0, 2.0, 2.0],    # 4 mm from 2, 4 from 3: none
                       [10.0, 3.5, 2.0]])  # 1.5 mm from region 3
    assert assign(points, volume(), AFFINE, 3.0).tolist() == [1, 1, 2, 0, 3]
    assert assign(points, volume(), AFFINE, 5.0).tolist() == [1, 1, 2, 2, 3]
    assert assign(points, volume(), AFFINE, 1.0)[-1] == 0


def test_assign_keeps_the_shape():
    points = np.zeros((4, 2, 3)) + [0.0, 2.0, 2.0]
    assert assign(points, volume(), AFFINE).shape == (4, 2)


def test_count_is_symmetric_with_the_within_region_on_the_diagonal():
    regions = np.array([[1, 2], [2, 1], [1, 2], [3, 3], [1, 0], [0, 0]])
    matrix = count(regions, 3)
    assert (matrix == matrix.T).all()
    assert matrix[0, 1] == matrix[1, 0] == 3
    assert matrix[2, 2] == 1 and matrix.sum() == 3 + 3 + 1
    assert matrix[0, 0] == 0  # the streamlines with an end in none are out


def test_endpoints_of_a_tractogram_file(tmp_path):
    lines = [np.array([[0, 2, 2], [2, 2, 2], [10, 2, 2]], dtype=np.float32),
             np.array([[2, 2, 2], [4, 2, 2]], dtype=np.float32)]
    tractogram = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    path = tmp_path / "bundle.trk"
    nib.streamlines.save(tractogram, str(path), header={
        "dimensions": (6, 3, 3), "voxel_sizes": (1.0, 1.0, 1.0),
        "voxel_to_rasmm": np.eye(4), "voxel_order": "RAS"})
    assert tractograms([tmp_path]) == [path]
    ends = endpoints(path)
    assert ends.shape == (2, 2, 3)
    assert np.allclose(ends[0], [[0, 2, 2], [10, 2, 2]])
    regions = assign(ends, volume(), AFFINE, 3.0)
    assert regions.tolist() == [[1, 3], [2, 2]]
    matrix = count(regions, 3)
    assert matrix[0, 2] == matrix[2, 0] == 1 and matrix[1, 1] == 1
