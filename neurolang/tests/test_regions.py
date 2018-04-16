import numpy as np
from ..regions import *
from ..RCD_relations import *


def _generate_random_box(x_bounds, y_bounds, size_bounds):
    lower_bound = np.asanyarray([np.random.uniform(*b) for b in (x_bounds, y_bounds)])
    upper_bound = lower_bound + np.random.uniform(size_bounds, size=2)
    return Region(lower_bound, upper_bound)


def test_region_eq():
    r1 = Region((0, 0), (1, 1))
    r2 = Region((0, 0), (1, 1))
    assert r1 == r2
    r3 = _generate_random_box((0, 0), (0, 0), (0, 10))
    r4 = _generate_random_box((50, 50), (100, 100), (50, 100))
    assert not r3 == r4


def test_axis_intervals():
    r1 = Region((0, 0), (1, 1))
    assert np.array_equal(r1.axis_intervals(), np.array([tuple([0, 1]), tuple([0, 1])]))
    r2 = Region((7, 0), (8, 6))
    assert np.array_equal(r2.axis_intervals(), np.array([tuple([7, 8]), tuple([0, 6])]))


def test_get_interval_relations_of_regions():
    r1 = Region((1, 1), (2, 2))
    r2 = Region((5, 5), (8, 8))
    assert get_interval_relation_to(r1, r2) == tuple(['b', 'b'])
    r1 = Region((1, 1), (10, 10))
    assert get_interval_relation_to(r1, r2) == tuple(['di', 'di'])
    r1 = Region((1, 1), (6, 6))
    assert get_interval_relation_to(r1, r2) == tuple(['o', 'o'])
    r2 = Region((1, 1), (2, 2))
    assert get_interval_relation_to(r1, r2) == tuple(['si', 'si'])
    assert get_interval_relation_to(r1, Region((1, 1), (6, 6))) == tuple(['e', 'e'])

    # r1 W r2
    r1 = Region((5, 5), (8, 8))
    r2 = Region((10, 6), (12, 7))
    assert get_interval_relation_to(r1, r2) == tuple(['b', 'di'])
    assert get_interval_relation_to(r2, r1) == tuple(['bi', 'd'])

    # r1 B:W:NW:W r2
    r1 = Region((5, 5), (8, 8))
    r2 = Region((7, 3), (9, 6))
    assert get_interval_relation_to(r1, r2) == tuple(['o', 'oi'])
    assert get_interval_relation_to(r2, r1) == tuple(['oi', 'o'])


def test_regions_dir_matrix():
    r1 = Region((0, 8), (1, 9))
    r2 = Region((0, 0), (1, 1))
    # r1 N r2 - r2 S r1
    result = np.array(np.zeros(shape=(3, 3)))
    result[0, 1] = 1
    assert np.array_equal(direction_matrix(r1, r2), result)
    result = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    assert np.array_equal(direction_matrix(r2, r1), result)

    # r1 B:S:N:NE:E:SE r2
    r1 = Region((3, 3), (8, 8))
    r2 = Region((2, 4), (5, 6))
    result = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
    assert np.array_equal(direction_matrix(r1, r2), result)

    # r1 B:S:SW:W r2
    r1 = Region((1, 1), (5, 5))
    r2 = Region((3, 3), (7, 5))
    result = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
    assert np.array_equal(direction_matrix(r1, r2), result)

    # r1 NW r2
    r1 = Region((6, 6), (8, 8))
    r2 = Region((8, 4), (10, 6))
    result = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.array_equal(direction_matrix(r1, r2), result)

    # r1 B r2
    r1 = Region((6, 5), (8, 8))
    r2 = Region((5, 5), (10, 10))
    result = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.array_equal(direction_matrix(r1, r2), result)

    # r1 B:S:SW:W:NW:N:NE:E:SE r2
    r1 = Region((0, 0), (10, 10))
    r2 = Region((5, 5), (6, 6))
    result = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert np.array_equal(direction_matrix(r1, r2), result)
