from typing import Union, Set

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
