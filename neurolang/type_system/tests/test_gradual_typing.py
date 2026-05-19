import operator as op
import unittest
from typing import (AbstractSet, Any, Callable, Generic, Mapping, Set,
                    SupportsInt, T, Tuple, Union)

import pytest

from .. import (NeuroLangTypeException, Unknown, get_args, get_origin,
                infer_type, is_leq_informative, is_parameterized,
                is_parametrical, replace_type_variable,
                typing_callable_from_annotated_function)


class TestGradualTyping(unittest.TestCase):
    """Test gradual typing functionality."""

    def test_parametrical(self):
        self.assertTrue(is_parametrical(Set))
        self.assertTrue(is_parametrical(AbstractSet))
        self.assertTrue(is_parametrical(Union))
        self.assertTrue(is_parametrical(Callable))
        self.assertFalse(is_parametrical(Set[int]))
        self.assertFalse(is_parametrical(AbstractSet[int]))
        self.assertFalse(is_parametrical(Union[int, float]))
        self.assertFalse(is_parametrical(Callable[[int], float]))
        self.assertFalse(is_parametrical(int))
        self.assertFalse(is_parametrical(T))
        self.assertFalse(is_parametrical(SupportsInt))


    def test_parameterized(self):
        self.assertFalse(is_parameterized(Set))
        self.assertFalse(is_parameterized(AbstractSet))
        self.assertFalse(is_parameterized(Union))
        self.assertFalse(is_parameterized(Callable))
        self.assertTrue(is_parameterized(Set[int]))
        self.assertTrue(is_parameterized(AbstractSet[int]))
        self.assertTrue(is_parameterized(Union[int, float]))
        self.assertTrue(is_parameterized(Callable[[int], float]))
        self.assertFalse(is_parameterized(int))
        self.assertFalse(is_parameterized(T))
        self.assertFalse(is_parameterized(SupportsInt))


def test_get_type_args():
    """Test getting type arguments from parametric types."""
    args = get_args(AbstractSet)
    assert args == tuple()

    args = get_args(AbstractSet[int])
    assert args == (int, )


def test_is_leq_informative_type():
    """Test if types are less informative than others."""
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
    """Test if base types are less informative than others."""
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
    """Test creating callable types from annotated functions."""
    def fun(a: int, b: str) -> float:
        pass

    t = typing_callable_from_annotated_function(fun)

    origin = get_origin(Callable)
    arg_types, return_type = get_args(t)
    assert issubclass(origin, Callable)
    assert arg_types[0] is int and arg_types[1] is str
    assert return_type is float


def test_replace_subtype():
    """Test replacing type variables in types."""
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
    """Test type inference for various expressions."""

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


def test_is_leq_informative_callable_ellipsis():
    """
    Regression test: `Callable[..., T]` uses `...` (Ellipsis) as a type
    parameter meaning "any/unknown arguments". `is_leq_informative` must not
    raise ValueError and must treat `...` as the least-informative arg-spec,
    analogous to `Unknown` for ordinary types.

    `Callable[..., T] ≤ Callable[anything, T]` regardless of arity, because
    unknown args are less informative than any concrete argument spec.
    """
    # Callable[..., T] is less informative than any same-return-type Callable
    assert is_leq_informative(Callable[..., bool], Callable[[int], bool])
    assert is_leq_informative(Callable[..., bool], Callable[[int, str], bool])
    assert is_leq_informative(Callable[..., bool], Callable[..., bool])

    # Unknown return type flows through correctly
    assert is_leq_informative(Callable[..., Unknown], Callable[[int], bool])

    # A concrete Callable is NOT less informative than Callable[..., T]
    assert not is_leq_informative(Callable[[int], bool], Callable[..., bool])
    assert not is_leq_informative(Callable[[int, str], bool], Callable[..., bool])
