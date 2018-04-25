import numpy as np
import pytest
from ..regions import *
from ..RCD_relations import *


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


def test_axis_intervals():
    r1 = Region((0, 0, 0), (1, 1, 1))
    assert np.array_equal(r1.axis_intervals(), np.array([tuple([0, 1]), tuple([0, 1]), tuple([0, 1])]))
    r2 = Region((2, 0, 7), (4, 6, 8))
    assert np.array_equal(r2.axis_intervals(), np.array([tuple([2, 4]), tuple([0, 6]), tuple([7, 8])]))


def test_get_interval_relations_of_regions():
    r1 = Region((1, 1, 1), (2, 2, 2))
    r2 = Region((5, 5, 5), (8, 8, 8))
    assert get_interval_relation_to(r1, r2) == tuple(['b', 'b', 'b'])
    r1 = Region((1, 1, 1), (10, 10, 10))
    assert get_interval_relation_to(r1, r2) == tuple(['di', 'di', 'di'])
    r1 = Region((1, 1, 1), (6, 6, 6))
    assert get_interval_relation_to(r1, r2) == tuple(['o', 'o', 'o'])
    r2 = Region((1, 1, 1), (2, 2, 2))
    assert get_interval_relation_to(r1, r2) == tuple(['si', 'si', 'si'])
    assert get_interval_relation_to(r1, Region((1, 1, 1), (6, 6, 6))) == tuple(['e', 'e', 'e'])

    r1 = Region((5, 5, 5), (8, 8, 8))
    r2 = Region((8, 7, 12), (10, 8, 14))
    assert get_interval_relation_to(r1, r2) == tuple(['m', 'fi', 'b'])
    assert get_interval_relation_to(r2, r1) == tuple(['mi', 'f', 'bi'])

    r1 = Region((5, 5, 5), (8, 8, 8))
    r2 = Region((3, 3, 7), (6, 6, 9))
    assert get_interval_relation_to(r1, r2) == tuple(['oi', 'oi', 'o'])
    assert get_interval_relation_to(r2, r1) == tuple(['o', 'o', 'oi'])


def test_regions_dir_matrix():

    dir_tensor = np.zeros(shape=(3, 3, 3))
    # r1 B:I:S:SA:A:IA r2
    r1 = Region((3, 3), (8, 8))
    r2 = Region((4, 2), (6, 5))
    dir_tensor[1] = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    # r1 B:I:IP:P r2
    r1 = Region((1, 1), (5, 5))
    r2 = Region((3, 3), (5, 7))
    dir_tensor[1] = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])

    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    # r1 SP r2
    r1 = Region((6, 6), (8, 8))
    r2 = Region((4, 8), (6, 10))
    dir_tensor[1] = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    # r1 B r2
    r1 = Region((5, 6), (8, 8))
    r2 = Region((5, 5), (10, 10))
    dir_tensor[1] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    # r1 B:I:IP:P:SP:S:SA:A:IA r2
    r1 = Region((0, 0), (10, 10))
    r2 = Region((5, 5), (6, 6))
    dir_tensor[1] = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert np.array_equal(direction_matrix(r1, r2), dir_tensor)

    #Hyper-Rectangle Regions
    r1 = Region((0, 8, 0), (10, 9, 1))
    r2 = Region((0, 0, 0), (10, 1, 1))
    # r1 SC r2 - r2 IC r1
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[1, 0, 1] = 1
    obtained = direction_matrix(r1, r2)
    assert np.array_equal(obtained, dir_tensor)
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[1, 2, 1] = 1
    obtained = direction_matrix(r2, r1)
    assert np.array_equal(obtained, dir_tensor)

    r1 = Region((0, 8, 0), (10, 9, 1))
    r2 = Region((15, 0, 0), (17, 1, 1))
    # r1 SL r2
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[0, 0, 1] = 1
    obtained = direction_matrix(r1, r2)
    assert np.array_equal(obtained, dir_tensor)

    r1 = Region((25, 0, 0), (30, 1, 1))
    r2 = Region((15, 0, 5), (20, 1, 6))
    # r1 PR r2
    dir_tensor = np.array(np.zeros(shape=(3, 3, 3)))
    dir_tensor[2, 1, 0] = 1
    obtained = direction_matrix(r1, r2)
    assert np.array_equal(obtained, dir_tensor)


def test_invalid_regions_raise_exception():
    exception_msg = 'Lower bounds must lower (and not equal) than upper bounds when creating rectangular regions'
    with pytest.raises(Exception) as excinfo:
        Region((0, 0, 0), (1, -1, 1))
    assert str(excinfo.value) == exception_msg

    with pytest.raises(Exception) as excinfo:
        Region((0, 0, 0), (0, 10, 20))
    assert str(excinfo.value) == exception_msg
