import pytest

from typing import Union, Set, Callable, AbstractSet, Generic, T
from .. import Unknown, is_leq_informative


def test_is_leq_informative_type():
    assert is_leq_informative(Unknown, int)
    assert ~is_leq_informative(int, Unknown)

    assert is_leq_informative(int, Union[int, float])
    assert is_leq_informative(float, Union[int, float])
    assert is_leq_informative(Union[float, int], Union[int, float, complex])
    assert ~is_leq_informative(str, Union[int, float])
    assert ~is_leq_informative(Union[int, float], int)

    assert ~is_leq_informative(Set[int], Set)
    assert is_leq_informative(Set, Set[int])
    assert is_leq_informative(Set[Unknown], Set[int])
    assert ~is_leq_informative(Set[int], Set[Unknown])
    assert is_leq_informative(Set[int], Set[int])


def test_is_leq_informative_base_types():
    assert is_leq_informative(int, int)
    assert is_leq_informative(int, float)
    assert is_leq_informative(str, str)
    assert not is_leq_informative(str, int)
    assert is_leq_informative(int, Union[int, str])
    assert is_leq_informative(str, Union[int, str])
    assert not is_leq_informative(Set, Union[int, str])

    assert is_leq_informative(
        Callable[[int], int], Callable[[int], int]
    )
    assert is_leq_informative(
        Callable[[int], int], Callable[[int], float]
    )
    assert not is_leq_informative(Set, Callable)

    assert is_leq_informative(
        AbstractSet[int], AbstractSet[int]
    )
    assert is_leq_informative(
        AbstractSet[int], AbstractSet[float]
    )

    with pytest.raises(ValueError, message="Generic not supported"):
        assert is_leq_informative(
            Set[int], Generic[T]
        )
