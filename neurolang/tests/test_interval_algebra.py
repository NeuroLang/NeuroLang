from ..interval_algebra import *


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