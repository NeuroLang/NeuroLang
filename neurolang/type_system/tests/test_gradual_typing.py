import operator as op
from typing import (AbstractSet, Any, Callable, Generic, Mapping, Set,
                    SupportsInt, T, Tuple, Union)

import pytest

from .. import (NeuroLangTypeException, Unknown, get_args, get_origin,
                infer_type, is_leq_informative, is_parameterized,
                is_parametrical, replace_type_variable,
                typing_callable_from_annotated_function)


def test_parametrical():
    assert is_parametrical(Set)
    assert is_parametrical(AbstractSet)
    assert is_parametrical(Union)
    assert is_parametrical(Callable)
    assert not is_parametrical(Set[int])
    assert not is_parametrical(AbstractSet[int])
    assert not is_parametrical(Union[int, float])
    assert not is_parametrical(Callable[[int], float])
    assert not is_parametrical(int)
    assert not is_parametrical(T)
    assert not is_parametrical(SupportsInt)


def test_parameterized():
    assert not is_parameterized(Set)
    assert not is_parameterized(AbstractSet)
    assert not is_parameterized(Union)
    assert not is_parameterized(Callable)
    assert is_parameterized(Set[int])
    assert is_parameterized(AbstractSet[int])
    assert is_parameterized(Union[int, float])
    assert is_parameterized(Callable[[int], float])
    assert not is_parameterized(int)
    assert not is_parameterized(T)
    assert not is_parameterized(SupportsInt)


def test_get_type_args():
    args = get_args(AbstractSet)
    assert args == tuple()

    args = get_args(AbstractSet[int])
    assert args == (int, )


def test_is_leq_informative_type():
    assert is_leq_informative(Unknown, int)
    assert not is_leq_informative(int, Unknown)
    assert not is_leq_informative(Any, int)

    assert is_leq_informative(int, Union[int, float])
    assert is_leq_informative(float, Union[int, float])
    assert is_leq_informative(Union[float, int], Union[int, float, complex])
    assert not is_leq_informative(str, Union[int, float])
    assert not is_leq_informative(Union[int, float], int)

    assert is_leq_informative(Set[int], Set)
    assert not is_leq_informative(Set, Set[int])
    assert is_leq_informative(Set[Unknown], Set[int])
    assert not is_leq_informative(Set[int], Set[Unknown])
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

    with pytest.raises(ValueError, match="Generic not supported"):
        assert is_leq_informative(
            Set[int], Generic[T]
        )


def test_typing_callable_from_annotated_function():
    def fun(a: int, b: str) -> float:
        pass

    t = typing_callable_from_annotated_function(fun)

    origin = get_origin(Callable)
    args = get_args(t)
    assert issubclass(origin, Callable)
    assert args[0] is int and args[1] is str
    assert args[2] is float


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
        Tuple[float, int] == replace_type_variable(
            float, Tuple[T, int], T
        )
    )

    assert (
        Set[str] != replace_type_variable(float, Set[T], T)
    )


def test_infer_type():

    def a(x: int, y: str) -> bool:
        return False

    assert infer_type(1) is int
    assert infer_type(tuple([1, 'a'])) is Tuple[int, str]
    assert infer_type(lambda x: x) is Callable[[Unknown], Unknown]
    assert infer_type(a) is Callable[[int, str], bool]

    assert infer_type(frozenset()) is AbstractSet[Unknown]
    assert infer_type(frozenset([1])) is AbstractSet[int]
    assert infer_type(frozenset([1, 3.]), deep=True) is AbstractSet[float]
    assert infer_type(frozenset([1, 3]), deep=True) is AbstractSet[int]
    with pytest.raises(NeuroLangTypeException):
        infer_type(frozenset([1, 'a']), deep=True)

    assert infer_type(dict()) is Mapping[Unknown, Unknown]
    assert infer_type(dict(a=2)) is Mapping[str, int]

    assert infer_type(op.and_) is Callable[[bool, bool], bool]
    assert infer_type(op.neg) is Callable[[Unknown], Unknown]
    assert infer_type(op.invert) is Callable[[bool], bool]
    assert infer_type(op.eq) is Callable[[Unknown, Unknown], bool]
    assert infer_type(op.add) is Callable[[Unknown, Unknown], Unknown]
