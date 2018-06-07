import numpy as np
import os
from numpy import random
from pytest import raises
from ..regions import *
from ..CD_relations import *
from ..exceptions import NeuroLangException
from ..utils.data_manipulation import *


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


def test_coordinates():
    r1 = Region((0, 0, 0), (1, 1, 1))
    assert np.array_equal(r1.bounding_box, np.array([[tuple([0, 1]), tuple([0, 1]), tuple([0, 1])]]))
    r2 = Region((2, 0, 7), (4, 6, 8))
    assert np.array_equal(r2.bounding_box, np.array([[tuple([2, 4]), tuple([0, 6]), tuple([7, 8])]]))


def test_get_interval_relations_of_regions():
    r1 = Region((1, 1, 1), (2, 2, 2))
    r2 = Region((5, 5, 5), (8, 8, 8))
    assert intervals_relations_from_regions(r1, r2) == tuple(['b', 'b', 'b'])
    r1 = Region((1, 1, 1), (10, 10, 10))
    assert intervals_relations_from_regions(r1, r2) == tuple(['di', 'di', 'di'])
    r1 = Region((1, 1, 1), (6, 6, 6))
    assert intervals_relations_from_regions(r1, r2) == tuple(['o', 'o', 'o'])
    r2 = Region((1, 1, 1), (2, 2, 2))
    assert intervals_relations_from_regions(r1, r2) == tuple(['si', 'si', 'si'])
    assert intervals_relations_from_regions(r1, Region((1, 1, 1), (6, 6, 6))) == tuple(['e', 'e', 'e'])

    r1 = Region((5, 5, 5), (8, 8, 8))
    r2 = Region((8, 7, 12), (10, 8, 14))
    assert intervals_relations_from_regions(r1, r2) == tuple(['m', 'fi', 'b'])
    assert intervals_relations_from_regions(r2, r1) == tuple(['mi', 'f', 'bi'])

    r1 = Region((5, 5, 5), (8, 8, 8))
    r2 = Region((3, 3, 7), (6, 6, 9))
    assert intervals_relations_from_regions(r1, r2) == tuple(['oi', 'oi', 'o'])
    assert intervals_relations_from_regions(r2, r1) == tuple(['o', 'o', 'oi'])


def test_regions_dir_matrix():

    # r1 A:B:P:RA:R:RP r2
    r1 = Region((3, 3, 0), (8, 8, 1))
    r2 = Region((2, 4, 0), (5, 6, 1))
    dir_matrix = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
    assert np.array_equal(direction_matrix(r1, r2)[1], dir_matrix)

    # r1 L:LA:A:B r2
    r1 = Region((1, 1, 0), (5, 5, 1))
    r2 = Region((3, 3, 0), (5, 7, 1))
    dir_matrix = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    dm = direction_matrix(r1, r2)[1]
    assert np.array_equal(dm, dir_matrix)

    # r1 LP r2
    r1 = Region((6, 6, 0), (8, 8, 1))
    r2 = Region((8, 4, 0), (10, 6, 1))
    dir_matrix = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    dm = direction_matrix(r1, r2)
    assert np.array_equal(dm[1], dir_matrix)

    # r1 B r2
    r1 = Region((5, 6, 0), (8, 8, 1))
    r2 = Region((5, 5, 0), (10, 10, 1))
    dir_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.array_equal(direction_matrix(r1, r2)[1], dir_matrix)

    # r1 LA:A:RA:L:B:R:LP:P:RP r2
    r1 = Region((0, 0, 0), (10, 10, 1))
    r2 = Region((5, 5, 0), (6, 6, 1))
    dir_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert np.array_equal(direction_matrix(r1, r2)[1], dir_matrix)

    r1 = Region((0, 0, 2), (10, 1, 9))
    r2 = Region((0, 0, 0), (10, 1, 1))
    # r1 S r2 - r2 I r1
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[2, 1, 1] = 1
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[0, 1, 1] = 1
    assert np.array_equal(direction_matrix(r2, r1), dir_tensor)

    # r1 SL r2
    r1 = Region((0, 0, 8), (10, 1, 9))
    r2 = Region((15, 0, 0), (17, 1, 1))
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[2, 1, 0] = 1
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    # r1 RA r2
    r1 = Region((25, 0, 0), (30, 1, 1))
    r2 = Region((15, 5, 0), (20, 6, 1))
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[1, 0, 2] = 1
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    # 4d regions overlapping at time intervals: r1 Before r2 - r2 After r1
    r1 = Region((0, 0, 0, 1), (1, 1, 1, 2))
    r2 = Region((0, 0, 0, 5), (1, 1, 1, 6))
    assert np.all(direction_matrix(r1, r2)[0, 1, :, :] == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    assert np.all(direction_matrix(r1, r2)[1:] == np.zeros(shape=(2, 3, 3, 3)))

    assert np.all(direction_matrix(r2, r1)[-1, 1, :, :] == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
    assert np.all(direction_matrix(r2, r1)[:-1] == np.zeros(shape=(2, 3, 3, 3)))
    assert is_in_direction(direction_matrix(r2, r1), 'F')

    # 2d regions (R-L, P-A)
    r1 = Region((0, 0), (1, 1))
    r2 = Region((0, 5), (1, 6))
    assert is_in_direction(direction_matrix(r1, r2), 'P')


def test_basic_overlap():
    r1 = Region((0, 0, 0), (1, 1, 1))
    r2 = Region((0, -5, -5), (1, 5, 5))
    assert is_in_direction(direction_matrix(r1, r2), 'O')
    for rel in ['P', 'A', 'I', 'S', 'L', 'R']:
        assert not is_in_direction(direction_matrix(r1, r2), rel)

    r1 = Region((0, 0, 0), (1, 3, 5))
    r2 = Region((0, 2, 1), (1, 7, 4))
    assert is_in_direction(direction_matrix(r1, r2), 'O')

    r1 = Region((0, 0), (1, 1))
    r2 = Region((1, 0), (2, 1))
    assert is_in_direction(direction_matrix(r1, r2), 'L')
    assert not is_in_direction(direction_matrix(r1, r2), 'O')


def test_invalid_regions_raise_exception():

    with raises(NeuroLangException):
        Region((0, 0, 0), (1, -1, 1))

    with raises(NeuroLangException):
        Region((0, 0, 0), (0, 10, 20))


def test_region_from_data():
    subject = '100206'
    region = 'CC_POSTERIOR'
    r1 = create_region_from_subject_data(subject, region)
    region = 'CC_ANTERIOR'
    r2 = create_region_from_subject_data(subject, region)
    region = 'CC_CENTRAL'
    r3 = create_region_from_subject_data(subject, region)
    region = 'CC_MID_ANTERIOR'
    r4 = create_region_from_subject_data(subject, region)
    region = 'CC_MID_POSTERIOR'
    r5 = create_region_from_subject_data(subject, region)

    for region in [r2, r3, r4, r5]:
        is_in_direction(direction_matrix(r1, region), 'P')

    for region in [r1, r3, r4, r5]:
        is_in_direction(direction_matrix(r2, region), 'A')


def create_region_from_subject_data(subject, region):
    #todo: generate the data to remove file dependency in tests
    path = 'data/%s/T1w/aparc.a2009s+aseg.nii.gz' % subject
    if os.path.isfile(path):
        parc_data = nib.load(path)
        label_region_key = parse_region_label_map(parc_data)
        region_data = transform_to_ras_coordinate_system(parc_data, label_region_key[region])
        (lb, ub) = region_data_limits(region_data)
        return Region(lb, ub)


def test_sphere_volumetric_region():
    subject = '100206'
    path = 'data/%s/T1w/aparc.a2009s+aseg.nii.gz' % subject
    if os.path.isfile(path):
        parc_data = nib.load(path)
        center = (1.10000151, -28.00000167, -43.30000049)
        radius = 15
        sr = SphericalVolume(center, radius)
        vbr_voxels = sr.to_ijk(parc_data.affine)
        rand_voxel = vbr_voxels[np.random.choice(len(vbr_voxels), 1)]
        coordinate = nib.affines.apply_affine(parc_data.affine, np.array(rand_voxel))
        assert np.linalg.norm(np.array(coordinate) - np.array(center)) <= radius

        esr = sr.to_explicit_region(parc_data.affine)
        assert np.all(np.array([np.linalg.norm(np.array(tuple([x, y, z])) - np.array(center)) for [x, y, z] in esr.to_xyz()]) <= 15)


def test_explicit_region():
    subject = '100206'
    path = 'data/%s/T1w/aparc.a2009s+aseg.nii.gz' % subject
    if os.path.isfile(path):
        parc_data = nib.load(path)
        region = np.array(list(zip(*parc_data.get_data()[:100, :100, :100].nonzero())))

        vbr = ExplicitVBR(region, parc_data.affine)
        assert np.array_equal(vbr.to_ijk(parc_data.affine), vbr._voxels)

        [i, j, k] = vbr._voxels[np.random.choice(len(vbr._voxels))]
        assert not parc_data.get_data()[i, j, k] == 0


def test_planar_region():
    center = (1, 5, 6)
    vector = (1, 0, 0)
    pr = PlanarVolume(center, vector, limit=10)
    assert pr.point_in_plane(center)
    assert not pr.point_in_plane((2, 8, 7))
    p = tuple(random.randint(1, 250, size=3))
    p_proj = pr.project_point_to_plane(p)
    assert not pr.point_in_plane(p_proj)
    assert np.all([0, -10, -10] == pr._lb)
    assert np.all([10, 10, 10] == pr._ub)

    r2 = Region((0, -5, -5), (1, 5, 5))
    print(r2._lb)