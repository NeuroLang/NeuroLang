"""Rows of a relational algebra set read without dropping duplicates."""
from collections import Counter

from ..pandas import NamedRelationalAlgebraFrozenSet


def test_sample_rows():
    """The first *n* rows if contiguous, otherwise *n* rows at distinct
    positions chosen at random, the same for one set; tuples in the order
    of the set's columns, duplicates kept."""
    rows = [(i // 2, i) for i in range(20)] + [(0, 0)]
    relation = NamedRelationalAlgebraFrozenSet(("x", "y"), rows)
    projected = relation.projection("y", "x")

    assert list(relation.sample_rows(3, contiguous=True)) == rows[:3]
    assert list(projected.sample_rows(3, contiguous=True)) == [
        (y, x) for x, y in rows[:3]
    ]
    assert list(relation.sample_rows(30, contiguous=True)) == rows

    sample = list(relation.sample_rows(5))
    # no row read more often than it is in the set: distinct positions
    assert len(sample) == 5 and not Counter(sample) - Counter(rows)
    assert sample == list(relation.sample_rows(5))
    assert sample != rows[:5]
    assert sorted(relation.sample_rows(30)) == sorted(rows)

    empty = NamedRelationalAlgebraFrozenSet(("x", "y"), [])
    assert list(empty.sample_rows(3, contiguous=True)) == []
    assert list(empty.sample_rows(3)) == []
