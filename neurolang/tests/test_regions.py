import numpy as np
import nibabel as nib
from numpy import random
from pytest import raises
from ..regions import *
from ..CD_relations import direction_matrix, cardinal_relation, is_in_direction
from ..exceptions import NeuroLangException
from ..brain_tree import AABB, Tree


def _generate_random_box(size_bounds, *args):
    N = len(args)
    lower_bound = np.array([np.random.uniform(*b) for b in tuple(args)])
    upper_bound = lower_bound + np.random.uniform(*size_bounds, size=N)
    return Region(lower_bound, upper_bound)


def test_region_eq():
    r1 = Region((0, 0, 0), (1, 1, 1))
    r2 = Region((0, 0, 0), (1, 1, 1))
    assert r1 == r2
    r3 = _generate_random_box((0, 10), (0, 0), (0, 0), (0, 0))
    r4 = _generate_random_box((50, 100), (50, 50), (100, 100), (200, 200))
    assert not r3 == r4


def test_invalid_regions_raise_exception():

    with raises(NeuroLangException):
        Region((0, 0, 0), (1, -1, 1))

    with raises(NeuroLangException):
        Region((0, 0, 0), (0, 10, 20))


def test_coordinates():
    r1 = Region((0, 0, 0), (1, 1, 1))
    assert np.array_equal(
        r1.bounding_box.limits,
        np.array([tuple([0, 1]), tuple([0, 1]), tuple([0, 1])])
    )
    r2 = Region((2, 0, 7), (4, 6, 8))
    assert np.array_equal(
        r2.bounding_box.limits,
        np.array([tuple([2, 4]), tuple([0, 6]), tuple([7, 8])])
    )


def _dir_matrix(region, other_region):
    return direction_matrix([region.bounding_box], [other_region.bounding_box])


def test_regions_dir_matrix():

    # 2d regions (R-L, P-A)
    r1 = Region((0, 0), (1, 1))
    r2 = Region((0, 5), (1, 6))
    assert is_in_direction(_dir_matrix(r1, r2), 'P')

    # r1 A:B:P:RA:R:RP r2
    r1 = Region((3, 3, 0), (8, 8, 1))
    r2 = Region((2, 4, 0), (5, 6, 1))
    dir_matrix = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
    assert np.array_equal(_dir_matrix(r1, r2)[1], dir_matrix)

    # r1 L:LA:A:B r2
    r1 = Region((1, 1, 0), (5, 5, 1))
    r2 = Region((3, 3, 0), (5, 7, 1))
    dir_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    dm = _dir_matrix(r1, r2)[1]
    assert np.array_equal(dm, dir_matrix)

    # r1 LP r2
    r1 = Region((6, 6, 0), (8, 8, 1))
    r2 = Region((8, 4, 0), (10, 6, 1))
    dir_matrix = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    dm = _dir_matrix(r1, r2)
    assert np.array_equal(dm[1], dir_matrix)

    # r1 B r2
    r1 = Region((5, 6, 0), (8, 8, 1))
    r2 = Region((5, 5, 0), (10, 10, 1))
    dir_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.array_equal(_dir_matrix(r1, r2)[1], dir_matrix)

    # r1 LA:A:RA:L:B:R:LP:P:RP r2
    r1 = Region((0, 0, 0), (10, 10, 1))
    r2 = Region((5, 5, 0), (6, 6, 1))
    dir_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert np.array_equal(_dir_matrix(r1, r2)[1], dir_matrix)

    r1 = Region((0, 0, 2), (10, 1, 9))
    r2 = Region((0, 0, 0), (10, 1, 1))
    # r1 S r2 - r2 I r1
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[2, 1, 1] = 1
    assert np.array_equal(_dir_matrix(r1, r2), dir_tensor)

    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[0, 1, 1] = 1
    assert np.array_equal(_dir_matrix(r2, r1), dir_tensor)

    # r1 SL r2
    r1 = Region((0, 0, 8), (10, 1, 9))
    r2 = Region((15, 0, 0), (17, 1, 1))
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[2, 1, 0] = 1
    assert np.array_equal(_dir_matrix(r1, r2), dir_tensor)

    # r1 RA r2
    r1 = Region((25, 0, 0), (30, 1, 1))
    r2 = Region((15, 5, 0), (20, 6, 1))
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[1, 0, 2] = 1
    assert np.array_equal(_dir_matrix(r1, r2), dir_tensor)

    # 4d regions overlapping at time intervals: r1 Before r2 - r2 After r1
    r1 = Region((0, 0, 0, 1), (1, 1, 1, 2))
    r2 = Region((0, 0, 0, 5), (1, 1, 1, 6))
    assert np.array_equal(
        _dir_matrix(r1, r2)[0, 1, :, :],
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    )
    assert np.array_equal(
        _dir_matrix(r1, r2)[1:],
        np.zeros(shape=(2, 3, 3, 3))
    )

    assert np.array_equal(
        _dir_matrix(r2, r1)[-1, 1, :, :],
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    )
    assert np.array_equal(
        _dir_matrix(r2, r1)[:-1],
        np.zeros(shape=(2, 3, 3, 3))
    )
    assert is_in_direction(_dir_matrix(r2, r1), 'F')


def test_basic_directionality():
    r1 = Region((0, 0), (1, 1))
    r2 = Region((0, -5), (1, -2))
    assert is_in_direction(_dir_matrix(r1, r2), 'A')
    assert is_in_direction(_dir_matrix(r2, r1), 'P')

    r1 = Region((0, 0), (1, 1))
    r2 = Region((4, 0), (6, 2))
    assert is_in_direction(_dir_matrix(r1, r2), 'L')
    assert is_in_direction(_dir_matrix(r2, r1), 'R')
    assert is_in_direction(_dir_matrix(r2, r1), 'A')
    assert is_in_direction(_dir_matrix(r2, r1), 'RA')

    r1 = Region((0, 0, 0), (1, 1, 1))
    r2 = Region((0, 0, 3), (1, 1, 4))
    assert is_in_direction(_dir_matrix(r1, r2), 'I')
    assert is_in_direction(_dir_matrix(r2, r1), 'S')

    r1 = Region((0, 0), (2, 2))
    r2 = Region((1, 0), (3, 2))
    assert is_in_direction(_dir_matrix(r1, r2), 'AO')
    assert is_in_direction(_dir_matrix(r2, r1), 'PO')

    r1 = Region((0, 0), (6, 6))
    r2 = Region((2, 3), (7, 4))
    assert is_in_direction(_dir_matrix(r1, r2), 'LAPO')
    assert is_in_direction(_dir_matrix(r2, r1), 'OR')

    r1 = Region((0, 0, 0), (1, 1, 1))
    r2 = Region((0, -5, -5), (1, 5, 5))
    assert is_in_direction(_dir_matrix(r1, r2), 'O')
    for rel in ['P', 'A', 'I', 'S', 'L', 'R']:
        assert not is_in_direction(_dir_matrix(r1, r2), rel)

    r1 = Region((0, 0, 0), (1, 3, 5))
    r2 = Region((0, 2, 1), (1, 7, 4))
    assert is_in_direction(_dir_matrix(r1, r2), 'O')

    r1 = Region((0, 0), (1, 1))
    r2 = Region((1, 0), (2, 1))
    assert is_in_direction(_dir_matrix(r1, r2), 'L')
    assert not is_in_direction(_dir_matrix(r1, r2), 'O')


def test_explicit_region():

    def randint(): return random.randint(0, 1000)

    voxels = [(randint(), randint(), randint()) for _ in range(50)]
    affine = np.eye(4)
    vbr = ExplicitVBR(voxels, affine)
    assert np.array_equal(vbr.to_ijk(affine), vbr.voxels)
    assert vbr.aabb_tree is not None
    assert np.all(vbr.bounding_box.lb >= 0)
    assert np.all(vbr.bounding_box.lb <= 1000)

    affine = np.eye(4)
    brain_stem = ExplicitVBR(voxels, affine)
    assert np.array_equal(brain_stem.voxels, brain_stem.to_ijk(affine))

    affine = np.eye(4) * 2
    affine[-1] = 1
    brain_stem = ExplicitVBR(voxels, affine)
    assert np.array_equal(brain_stem.voxels, brain_stem.to_ijk(affine))

    affine = np.eye(4)
    affine[:, -1] = np.array([1, 1, 1, 1])
    brain_stem = ExplicitVBR(voxels, affine)
    assert np.array_equal(brain_stem.voxels, brain_stem.to_ijk(affine))

    affine = np.array([
        [-0.69999999, 0., 0., 90.],
        [0., 0.69999999, 0., -126.],
        [0., 0., 0.69999999, -72.],
        [0., 0., 0., 1.]
    ]).round(2)
    brain_stem = ExplicitVBR(voxels, affine)
    assert np.array_equal(brain_stem.voxels, brain_stem.to_ijk(affine))


def test_build_tree_one_voxel_regions():

    region = ExplicitVBR(np.array([[2, 2, 2]]), np.eye(4))
    assert region.bounding_box == AABB((2, 2, 2), (3, 3, 3))
    assert region.aabb_tree.height == 0

    other_region = ExplicitVBR(np.array([[2, 2, 2]]), np.diag((10, 10, 10, 1)))
    assert other_region.bounding_box == AABB((20, 20, 20), (30, 30, 30))
    assert other_region.aabb_tree.height == 0
    assert is_in_direction(_dir_matrix(other_region, region), 'SA')


def test_tree_of_convex_regions():
    cube = ExplicitVBR(np.array([[0, 0, 0], [5, 5, 5]]), np.eye(4))
    assert cube.aabb_tree.height == 1
    triangle = ExplicitVBR(
        np.array([[0, 0, 0], [2, 0, 1], [5, 5, 5]]), np.eye(4)
    )
    assert triangle.aabb_tree.height == 2

    region = ExplicitVBR(
        np.array([[0, 0, 0], [2, 2, 1], [5, 5, 0], [8, 8, 0]]), np.eye(4)
    )
    assert region.aabb_tree.height == 2

    region = ExplicitVBR(
        np.array([[0, 0, 0], [2, 2, 1], [5, 5, 0], [10, 10, 0]]), np.eye(4)
    )
    assert region.aabb_tree.height == 3
    #rand length n of voxels takes you to log2(n) tree height only if equidist


def test_spherical_volumetric_region():

    def randint(): return random.randint(0, 1000)

    N = 500
    voxels = sorted([(randint(), randint(), randint()) for _ in range(N)])
    affine = np.eye(4)
    center = voxels[N//2]
    radius = 15
    sr = SphericalVolume(center, radius)
    vbr_voxels = sr.to_ijk(affine)
    rand_voxel = vbr_voxels[np.random.choice(len(vbr_voxels), 1)]
    coordinate = nib.affines.apply_affine(affine, np.array(rand_voxel))
    assert np.linalg.norm(np.array(coordinate) - np.array(center)) <= radius

    explicit_sr = sr.to_explicit_vbr(affine, None)
    assert np.all(
        np.array([
            np.linalg.norm(np.array(tuple([x, y, z])) - np.array(center))
            for [x, y, z] in explicit_sr.to_xyz()
        ]) <= 15
    )


def test_planar_region():
    center = (1, 5, 6)
    vector = (1, 0, 0)
    pr = PlanarVolume(center, vector, limit=10)
    assert pr.point_in_plane(center)
    assert not pr.point_in_plane((2, 8, 7))
    p = tuple(random.randint(1, 250, size=3))
    p_proj = pr.project_point_to_plane(p)
    assert not pr.point_in_plane(p_proj)
    assert np.array_equal([0, -10, -10], pr.bounding_box.lb)
    assert np.array_equal([10, 10, 10], pr.bounding_box.ub)


def test_regions_with_multiple_bb_directionality():
    r1 = Region((0, 0, 0), (6, 6, 1))
    r2 = Region((6, 0, 0), (12, 6, 1))
    assert is_in_direction(_dir_matrix(r1, r2), 'L')
    r2 = Region((2, -3, 0), (5, 3, 1))
    assert is_in_direction(_dir_matrix(r1, r2), 'LR')

    region = ExplicitVBR(np.array([[0, 0, 0], [5, 5, 0]]), np.eye(4))
    other_region = ExplicitVBR(np.array([[3, 0, 0]]), np.eye(4))
    assert is_in_direction(_dir_matrix(other_region, region), 'O')
    for r in ['L', 'R', 'P', 'A', 'I', 'S']:
        assert not is_in_direction(_dir_matrix(other_region, region), r)

    tree = Tree()
    tree.add(region.bounding_box)
    tree.add(AABB((0, 0, 0), (2.5, 5, 1)))
    tree.add(AABB((2.5, 0, 0), (5, 5, 1)))
    region_bbs = [tree.root.left.box, tree.root.right.box]
    assert is_in_direction(
        direction_matrix([other_region.bounding_box], region_bbs), 'O'
    )

    region_bbs = [
      AABB((0, 0, 0), (2.5, 2.5, 1)),
      AABB((0, 2.5, 0), (2.5, 5, 1)),
      AABB((2.5, 2.5, 0), (5, 5, 1)),
    ]

    for region in region_bbs:
      tree.add(region)

    assert is_in_direction(
        direction_matrix([other_region.bounding_box], region_bbs), 'P'
    )
    assert is_in_direction(
        direction_matrix([other_region.bounding_box], region_bbs), 'R'
    )

    for r in ['L', 'A', 'I', 'S', 'O']:
        assert not is_in_direction(
            direction_matrix([other_region.bounding_box], region_bbs), r
        )


def test_refinement_of_not_overlapping():

    triangle = ExplicitVBR(
        np.array([[0, 0, 0], [6, 0, 0], [6, 6, 1]]), np.eye(4)
    )
    other_region = ExplicitVBR(np.array([[0, 6, 0]]), np.eye(4))
    assert cardinal_relation(
        other_region, triangle, 'O', refine_overlapping=False
    )
    with raises(ValueError):
        cardinal_relation(
            other_region, triangle, 'O', refine_overlapping=True, stop_at=0
        )
    assert not cardinal_relation(
        other_region, triangle, 'O', refine_overlapping=True
    )
    for r in ['L', 'A']:
        assert cardinal_relation(
            other_region, triangle, r, refine_overlapping=True
        )
    for r in ['R', 'P', 'I', 'S', 'O']:
        assert not cardinal_relation(
            other_region, triangle, r, refine_overlapping=True
        )

    outer = ExplicitVBR(np.array([[0, 0, 0], [10, 10, 0]]), np.eye(4))
    inner = ExplicitVBR(np.array([[8, 0, 0]]), np.eye(4))
    assert cardinal_relation(inner, outer, 'O', refine_overlapping=False)
    assert not cardinal_relation(inner, outer, 'O', refine_overlapping=True)

    for r in ['L', 'R', 'A', 'P', 'I', 'S']:
        assert not cardinal_relation(inner, outer, r, refine_overlapping=False)

    for r in ['L', 'R', 'P']:
        assert cardinal_relation(inner, outer, r, refine_overlapping=True)
    for r in ['A', 'I', 'S', 'O']:
        assert not cardinal_relation(inner, outer, r, refine_overlapping=True)

    region = ExplicitVBR(
        np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]), np.eye(4)
    )
    other_region = ExplicitVBR(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.eye(4)
    )
    assert cardinal_relation(
        region, other_region, 'O', refine_overlapping=False
    )
    assert cardinal_relation(
        region, other_region, 'O', refine_overlapping=True
    )


def test_regions_union_intersection():

    def randint(): return random.randint(70, 100)

    voxels = [(randint(), randint(), randint()) for _ in range(50)]
    affine = np.array([
        [-0.69999999, 0., 0., 90.],
        [0., 0.69999999, 0., -126.],
        [0., 0., 0.69999999, -72.],
        [0., 0., 0., 1.]
    ]).round(2)
    region = ExplicitVBR(voxels, affine)
    union = region_union([region], affine)
    assert union.bounding_box == region.bounding_box
    #
    center = region.bounding_box.ub
    radius = 30
    sphere = SphericalVolume(center, radius)
    assert sphere.bounding_box.overlaps(region.bounding_box)
    intersect = region_intersection([region, sphere], affine)
    assert intersect is not None


def test_intersection_difference():

    def randint(): return random.randint(1, 5)
    affine = np.array([
        [-0.69999999, 0., 0., 90.],
        [0., 0.69999999, 0., -126.],
        [0., 0., 0.69999999, -72.],
        [0., 0., 0., 1.]
    ]).round(2)

    center = (randint(), randint(), randint())
    center2 = (randint(), randint(), randint())
    radius = 10
    sphere = SphericalVolume(center, radius)
    other_sphere = SphericalVolume(center2, radius)

    intersect = region_intersection([sphere, other_sphere], affine)
    d1 = region_difference([sphere, other_sphere], affine)
    d2 = region_difference([other_sphere, sphere], affine)
    union = region_union([sphere, other_sphere], affine)
    intersect2 = region_difference([union, d1, d2], affine)
    assert intersect.bounding_box == intersect2.bounding_box
