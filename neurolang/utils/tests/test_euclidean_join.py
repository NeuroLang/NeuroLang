import itertools
import math

import numpy
import pytest

from ..euclidean_join import euclidean_distance, euclidean_pairs


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
