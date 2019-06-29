import pytest

from typing import Union, Set, Callable, AbstractSet, Generic, Tuple, T
from .. import (
    Unknown, is_leq_informative,
    typing_callable_from_annotated_function, get_args,
    replace_type_variable
)


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


def test_typing_callable_from_annotated_function():
    def fun(a: int, b: str) -> float:
        pass

    t = typing_callable_from_annotated_function(fun)

    assert t.__origin__ is Callable
    assert t.__args__[0] is int and t.__args__[1] is str
    assert t.__args__[2] is float


def test_get_type_args():
    args = get_args(AbstractSet)
    assert args == tuple()

    args = get_args(AbstractSet[int])
    assert args == (int, )


def test_replace_subtype():
    assert (
        Set is replace_type_variable(
            int, Set, T
        )
    )
    assert str is replace_type_variable(int, str, T)

    assert (
        Set[float] == replace_type_variable(
            int, Set[float], T
        )
    )

    assert (
        Set[float] == replace_type_variable(
            float, Set[T], T
        )
    )

    assert (
        Tuple[float, int] is replace_type_variable(
            float, Tuple[T, int], T
        )
    )

    assert (
        Set[str] != replace_type_variable(float, Set[T], T)
    )
