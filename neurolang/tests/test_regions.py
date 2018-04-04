import numpy as np
from ..regions import *


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


def test_ia_relations_functions():
    intervals = [tuple([1, 2]), tuple([5, 7]), tuple([1, 5]), tuple([4, 6]), tuple([2, 4]), tuple([6, 7]), tuple([2, 4])]

    assert before(intervals[0], intervals[1])
    assert meets(intervals[0], intervals[4])
    assert starts(intervals[0], intervals[2])
    assert during(intervals[4], intervals[2])
    assert overlaps(intervals[3], intervals[1])
    assert finishes(intervals[5], intervals[1])
    assert equals(intervals[4], intervals[6])

    assert not equals(intervals[1], intervals[0])
    assert not during(intervals[1], intervals[2])
    assert not overlaps(intervals[0], intervals[2])
    assert not starts(intervals[3], intervals[4])


def test_get_interval_relations_of_regions():
    r1 = Region((1, 1), (2, 2))
    r2 = Region((5, 5), (8, 8))
    assert r1.get_interval_relation_to(r2) == tuple(['b', 'b'])
    r1 = Region((1, 1), (10, 10))
    assert r1.get_interval_relation_to(r2) == tuple(['di', 'di'])
    r1 = Region((1, 1), (6, 6))
    assert r1.get_interval_relation_to(r2) == tuple(['o', 'o'])
    r2 = Region((1, 1), (2, 2))
    assert r1.get_interval_relation_to(r2) == tuple(['si', 'si'])
    assert r1.get_interval_relation_to(Region((1, 1), (6, 6))) == tuple(['e', 'e'])

    #r1 W r2
    r1 = Region((5, 5), (8, 8))
    r2 = Region((10, 6), (12, 7))
    assert r1.get_interval_relation_to(r2) == tuple(['b', 'di'])
    assert r2.get_interval_relation_to(r1) == tuple(['bi', 'd'])

    #r1 B:W:NW:W r2
    r1 = Region((5, 5), (8, 8))
    r2 = Region((7, 3), (9, 6))
    assert r1.get_interval_relation_to(r2) == tuple(['o', 'oi'])
    assert r2.get_interval_relation_to(r1) == tuple(['oi', 'o'])