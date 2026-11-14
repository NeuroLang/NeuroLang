import numpy as np
import pytest

from hcp_connectivity import tractography as t
from hcp_connectivity.parcels import Part

def _cortex(side):
    return [f"17Networks {side} " +
            ("PFC" if i < 20 else "Temp" if i < 40 else "Net") + f" {i}"
            for i in range(200)]


NAMES = (_cortex("LH") + _cortex("RH") +
         ["aHIP rh", "pHIP rh", "lAMY rh", "mAMY rh", "THA DP rh",
          "THA VP rh", "THA VA rh", "THA DA rh", "NAc shell rh",
          "NAc core rh", "pGP rh", "aGP rh", "aPUT rh", "pPUT rh",
          "aCAU rh", "pCAU rh"] +
         ["aHIP lh", "pHIP lh", "lAMY lh", "mAMY lh", "THA DP lh",
          "THA VP lh", "THA VA lh", "THA DA lh", "NAc shell lh",
          "NAc core lh", "pGP lh", "aGP lh", "aPUT lh", "pPUT lh",
          "aCAU lh", "pCAU lh"])


def test_combine_labels_tian_wins_and_is_offset():
    cortex = np.array([0, 1, 400, 7])
    tian = np.array([0, 0, 0, 32])
    assert t.combine_labels(cortex, tian).tolist() == [0, 1, 400, 432]


def test_resample_nearest_upsamples():
    low = np.arange(8).reshape(2, 2, 2)
    affine = 2 * np.eye(4)
    affine[3, 3] = 1
    fine = t.resample_nearest(low, affine, (4, 4, 4), np.eye(4))
    assert fine.shape == (4, 4, 4)
    assert fine[0, 0, 0] == low[0, 0, 0] and fine[2, 2, 2] == low[1, 1, 1]


def test_warp_labels_identity_and_shift():
    shape = (8, 9, 10)
    labels = np.random.default_rng(0).integers(0, 5, shape)
    # FSL-style: x flipped affine
    affine = np.diag([-1.0, 1, 1, 1])
    affine[0, 3] = shape[0] - 1
    zero = np.zeros(shape + (3,), dtype=np.float32)
    same = t.warp_labels(labels, affine, zero, affine, shape, affine)
    assert (same == labels).all()
    # one mm along the voxel x axis (FSL: not flipped)
    moved = zero.copy()
    moved[..., 0] = 1
    shifted = t.warp_labels(labels, affine, moved, affine, shape, affine)
    assert (shifted[:-1] == labels[1:]).all()


def test_dice():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert t.dice(a, b) == 0.5 and t.dice(a, a) == 1.0


def test_tian_regions_by_structure_and_side():
    assert t.tian_region_numbers(NAMES, "thalamus", "l") == [
        421, 422, 423, 424]
    assert t.tian_region_numbers(NAMES, "hippocampus", "r") == [401, 402]
    assert len(t.tian_region_numbers(NAMES, "putamen", "l")) == 2


def test_matrix_check_of_a_planted_matrix():
    n = len(NAMES)
    matrix = np.zeros((n, n))
    left_thalamus = [i - 1 for i in t.tian_region_numbers(NAMES, "thalamus",
                                                          "l")]
    matrix[np.ix_(left_thalamus, [0, 1, 2])] = 1.0  # all to the left cortex
    matrix = matrix + matrix.T
    matrix[0, 1] = matrix[1, 0] = 1.0  # within the left cortex
    check = t.matrix_check(matrix, NAMES)
    assert check["thalamus_to_cortex"] == pytest.approx(1.0)
    assert check["thalamus_l_ipsilateral"] == pytest.approx(1.0)
    assert check["cortex_interhemispheric"] == 0.0


def test_assigned_fraction(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("1 2\n0 3\n4 4\n5 6\n")
    out = t.assigned_fraction(path, np.array([1.0, 1, 1, 1]))
    assert out["streamlines"] == 4
    assert out["assigned"] == 0.75 and out["assigned_distinct"] == 0.5


def test_read_matrix_and_part(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("\n".join(",".join(["1"] * 3) for _ in range(3)))
    assert t.read_matrix(path, 3).shape == (3, 3)
    part = Part(tmp_path / "part-0.npz", every=1)
    part.add("100206", {t.NAME: np.eye(2)}, total=1.0)
    assert Part(tmp_path / "part-0.npz").subjects == ["100206"]


def test_subjects_with_diffusion(tmp_path):
    for s, complete in (("2", True), ("1", True), ("3", False)):
        home = tmp_path / s
        for f in t.FILES:
            if complete or f != "bvals":
                (home / "T1w" / "Diffusion").mkdir(parents=True,
                                                   exist_ok=True)
                (home / "T1w" / "Diffusion" / f).write_text("")
        (home / "T1w" / "aparc+aseg.nii.gz").write_text("")
        (home / "MNINonLinear" / "xfms").mkdir(parents=True, exist_ok=True)
        (home / "MNINonLinear" / "xfms" /
         "standard2acpc_dc.nii.gz").write_text("")
    assert t.subjects_with_diffusion(tmp_path) == ["1", "2"]
    assert t.subjects_with_diffusion(tmp_path, 1) == ["1"]
