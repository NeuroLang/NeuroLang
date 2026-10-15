import itertools
import math

import numpy
import pandas
import pytest
import scipy.spatial

from ..euclidean_join import (
    euclidean_distance,
    euclidean_join,
    euclidean_pairs,
)


def _brute_force(first, second, radius, strict):
    result = set()
    for p, q in itertools.product(map(tuple, first), map(tuple, second)):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
        if dist < radius or (not strict and dist == radius):
            result.add(p + q)
    return result


def _as_set(first, second, pairs):
    first_index, second_index, distances = pairs
    assert len(first_index) == len(second_index) == len(distances)
    result = set()
    for i, j, dist in zip(first_index, second_index, distances):
        assert dist == euclidean_distance(*first[i], *second[j])
        result.add(tuple(first[i]) + tuple(second[j]))
    assert len(result) == len(distances)
    return result


def _found(first, second, radius, strict):
    return _as_set(
        first, second, euclidean_pairs(first, second, radius, strict)
    )


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("radius", [0, 1, 1.5, 2, 3])
@pytest.mark.parametrize("dims", [1, 2, 3, 4])
def test_integer_pairs_match_brute_force(strict, radius, dims):
    rng = numpy.random.default_rng(dims)
    first = rng.integers(-4, 5, size=(40, dims))
    second = rng.integers(-4, 5, size=(30, dims))
    first = numpy.unique(first, axis=0)
    second = numpy.unique(second, axis=0)
    expected = _brute_force(first, second, radius, strict)
    assert _found(first, second, radius, strict) == expected
    assert _found(second, first, radius, strict) == {
        p[dims:] + p[:dims] for p in expected
    }


@pytest.mark.parametrize("strict", [True, False])
def test_float_pairs_match_brute_force(strict):
    rng = numpy.random.default_rng(0)
    first = numpy.round(rng.uniform(-3, 3, size=(50, 3)), 1)
    second = numpy.round(rng.uniform(-3, 3, size=(20, 3)), 1)
    second[0] = first[0]
    second[1] = first[1] + numpy.array([2.0, 0.0, 0.0])
    first = numpy.unique(first, axis=0)
    second = numpy.unique(second, axis=0)
    expected = _brute_force(first, second, 2, strict)
    assert _found(first, second, 2, strict) == expected


def test_mixed_integer_and_float_coordinates():
    first = numpy.array([[0, 0, 0], [3, 0, 0]])
    second = numpy.array([[1.5, 0.0, 0.0]])
    assert _found(first, second, 1.5, True) == set()
    assert _found(first, second, 1.5, False) == {
        (0, 0, 0, 1.5, 0.0, 0.0), (3, 0, 0, 1.5, 0.0, 0.0)
    }


def test_sparse_integer_sets_use_exact_distances():
    first = numpy.array([[0, 0, 0], [1000, 0, 0]])
    second = numpy.array([[0, 0, 30], [1000, 40, 0], [500, 500, 500]])
    expected = _brute_force(first, second, 40, False)
    assert _found(first, second, 40, False) == expected
    assert _found(first, second, 40, True) == {
        (0, 0, 0, 0, 0, 30)
    }


@pytest.mark.parametrize(
    "first,second",
    [
        (numpy.empty((0, 3), dtype=int), numpy.array([[0, 0, 0]])),
        (numpy.array([[0, 0, 0]]), numpy.empty((0, 3), dtype=int)),
    ],
)
def test_empty_inputs(first, second):
    result = euclidean_pairs(first, second, 2, True)
    assert all(len(r) == 0 for r in result)


def test_negative_radius_gives_no_pairs():
    first = numpy.array([[0, 0, 0]])
    assert _found(first, first, -1, False) == set()


def test_euclidean_distance_is_vectorised():
    assert euclidean_distance(0, 0, 0, 1, 2, 2) == 3.0
    assert isinstance(euclidean_distance(0, 0, 0, 1, 2, 2), float)
    result = euclidean_distance(
        numpy.array([0, 1]), numpy.array([0, 1]), numpy.array([3, 4]),
        numpy.array([4, 1])
    )
    numpy.testing.assert_allclose(result, [5.0, 3.0])
    with pytest.raises(ValueError):
        euclidean_distance(1, 2, 3)


GRID = pandas.DataFrame(
    list(itertools.product(range(6), repeat=3)), columns=["i", "j", "k"]
)
PEAKS = pandas.DataFrame(
    [
        (2, 2, 2, "s1"), (2, 2, 2, "s2"), (3, 2, 2, "s1"), (4, 3, 2, "s1"),
        (0, 0, 5, "s3"), (9, 9, 9, "s4"),
    ],
    columns=["i1", "j1", "k1", "s"],
)
LABELLED = GRID.assign(l=[f"l{(i + j) % 3}" for i, j in zip(GRID.i, GRID.j)])
POINT, PEAK_POINT = ["i", "j", "k"], ["i1", "j1", "k1"]


def _brute_force_join(first, second, radius, strict, columns, distance):
    (first, first_point), (second, second_point) = first, second
    cross = first.merge(second, how="cross")
    cross[distance] = [
        math.dist(p, q) for p, q in zip(
            cross[first_point].itertuples(index=False),
            cross[second_point].itertuples(index=False),
        )
    ]
    within = cross[distance] < radius
    if not strict:
        within |= cross[distance] == radius
    return set(cross[within][columns].itertuples(index=False, name=None))


def _check_join(first, second, radius, strict, columns):
    result = euclidean_join(first, second, radius, strict, columns, "d")
    assert list(result.columns) == columns
    rows = list(result.itertuples(index=False, name=None))
    assert len(rows) == len(set(rows))
    expected = _brute_force_join(first, second, radius, strict, columns, "d")
    assert set(rows) == expected
    return result


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("radius", [1, 2, 2.5])
@pytest.mark.parametrize(
    "columns",
    [
        ["i", "j", "k", "s"],
        ["i", "j", "k", "i1", "j1", "k1", "s", "d"],
        ["s"],
        ["s", "d"],
        ["i1", "j1", "k1"],
        ["d", "k"],
    ],
)
def test_join_of_points_with_rows_of_other_columns(strict, radius, columns):
    _check_join((GRID, POINT), (PEAKS, PEAK_POINT), radius, strict, columns)
    _check_join((PEAKS, PEAK_POINT), (GRID, POINT), radius, strict, columns)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "columns",
    [
        ["l", "s"],
        ["l"],
        ["i", "j", "k", "l", "i1", "j1", "k1", "s"],
        ["i", "l", "d"],
    ],
)
def test_join_of_two_tables_with_other_columns(strict, columns):
    _check_join(
        (LABELLED, POINT), (PEAKS, PEAK_POINT), 2, strict, columns
    )


def test_join_of_float_points():
    rng = numpy.random.default_rng(0)
    first = pandas.DataFrame(
        numpy.round(rng.uniform(0, 4, size=(40, 2)), 1), columns=["x", "y"]
    )
    second = pandas.DataFrame({
        "u": numpy.round(rng.uniform(0, 4, size=15), 1),
        "v": numpy.round(rng.uniform(0, 4, size=15), 1),
        "name": [f"n{n % 4}" for n in range(15)],
    })
    for strict in (True, False):
        _check_join(
            (first, ["x", "y"]), (second, ["u", "v"]), 1.5, strict,
            ["x", "name"],
        )


def test_join_of_empty_tables():
    result = euclidean_join(
        (GRID.iloc[:0], POINT), (PEAKS, PEAK_POINT), 2, True,
        ["i", "s", "d"], "d",
    )
    assert list(result.columns) == ["i", "s", "d"] and len(result) == 0


def test_join_columns_default_to_all():
    result = euclidean_join(
        (GRID, POINT), (PEAKS, PEAK_POINT), 1, False, distance="d"
    )
    assert list(result.columns) == POINT + list(PEAKS.columns) + ["d"]


def test_integer_join_does_not_build_kd_trees(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a k-d tree was built")

    monkeypatch.setattr(scipy.spatial, "cKDTree", fail)
    _check_join(
        (GRID, POINT), (PEAKS, PEAK_POINT), 2, True, ["i", "j", "k", "s"]
    )
