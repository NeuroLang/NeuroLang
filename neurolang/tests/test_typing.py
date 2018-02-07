import pytest

import typing

from .. import symbols_and_types


def test_typing_callable_from_annotated_function():
    def fun(a: int, b: str)->float:
        pass

    t = symbols_and_types.typing_callable_from_annotated_function(fun)

    assert t.__origin__ == typing.Callable
    assert (t.__args__[0] == int) and (t.__args__[1] == str)
    assert t.__args__[2] == float


def test_get_type_args():
    args = symbols_and_types.get_type_args(typing.Set)
    assert args == tuple()

    args = symbols_and_types.get_type_args(typing.Set[int])
    assert args == (int,)


def test_is_subtype_base_types():
    assert symbols_and_types.is_subtype(int, int)
    assert symbols_and_types.is_subtype(int, float)
    assert symbols_and_types.is_subtype(str, str)
    assert not symbols_and_types.is_subtype(str, int)
    assert symbols_and_types.is_subtype(int, typing.Any)
    assert symbols_and_types.is_subtype(int, typing.Union[int, str])
    assert symbols_and_types.is_subtype(str, typing.Union[int, str])
    assert not symbols_and_types.is_subtype(typing.Set, typing.Union[int, str])

    assert symbols_and_types.is_subtype(
        typing.Callable[[int], int],
        typing.Callable[[int], int]
    )
    assert symbols_and_types.is_subtype(
        typing.Callable[[int], int],
        typing.Callable[[int], float]
    )
    assert not symbols_and_types.is_subtype(
        typing.Set,
        typing.Callable
    )

    assert symbols_and_types.is_subtype(
        typing.AbstractSet[int], typing.AbstractSet[int]
    )
    assert symbols_and_types.is_subtype(
        typing.AbstractSet[int], typing.AbstractSet[float]
    )

    with pytest.raises(ValueError, message="typing Generic not supported"):
        assert symbols_and_types.is_subtype(
            typing.Set[int],
            typing.Generic[typing.T]
        )


def test_replace_subtype():
    assert (
        typing.Set ==
        symbols_and_types.replace_type_variable(int, typing.Set, typing.T)
    )
    assert (
        str ==
        symbols_and_types.replace_type_variable(int, str, typing.T)
    )

    assert (
        typing.Set[float] ==
        symbols_and_types.replace_type_variable(
            int, typing.Set[float], typing.T
        )
    )

    assert (
        typing.Set[float] ==
        symbols_and_types.replace_type_variable(
            float, typing.Set[typing.T], typing.T
        )
    )

    assert (
        typing.Tuple[float, int] ==
        symbols_and_types.replace_type_variable(
            float, typing.Tuple[typing.T, int], typing.T
        )
    )

    assert (
        typing.Set[str] !=
        symbols_and_types.replace_type_variable(
            float, typing.Set[typing.T], typing.T
        )
    )


def test_get_type_and_value():
    type_, value = symbols_and_types.get_type_and_value(3)
    assert type_ == int
    assert value == 3

    type_, value = symbols_and_types.get_type_and_value(
        symbols_and_types.Symbol(int, 3)
    )
    assert type_ == int
    assert value == 3

    type_, value = symbols_and_types.get_type_and_value(
        symbols_and_types.Identifier('a'),
        {symbols_and_types.Identifier('a'): symbols_and_types.Symbol(int, 3)}
    )
    assert type_ == int
    assert value == 3

    def f(a: int)->int:
        return 0

    type_, value = symbols_and_types.get_type_and_value(f)

    assert type_ == typing.Callable[[int], int]
    assert value == f


def test_type_validation_value():
    def f(a: int)->int:
        return 0

    symbol_table = {
        symbols_and_types.Identifier('r'): symbols_and_types.Symbol(
             typing.AbstractSet[str],
             {'a'}
        )
    }

    values = (
        3, {3, 8}, 'try', f, (3, 'a'),
        symbols_and_types.Symbol(typing.Tuple[str, float], ('a', 3.)),
        symbols_and_types.Identifier('r')
    )
    types_ = (
        int, typing.AbstractSet[int],
        typing.Text, typing.Callable[[int], int],
        typing.Tuple[int, str], typing.Tuple[str, float],
        symbol_table[symbols_and_types.Identifier('r')].type
    )

    for i, v in enumerate(values):
        assert symbols_and_types.type_validation_value(
            v, typing.Any,
            symbol_table=symbol_table
        )

        for j, t in enumerate(types_):
            if i == j:
                assert symbols_and_types.type_validation_value(
                    v, t, symbol_table=symbol_table
                )
                assert symbols_and_types.type_validation_value(
                    v, typing.Union[t, types_[(i + 1) % len(types_)]],
                    symbol_table=symbol_table
                )
            else:
                assert not symbols_and_types.type_validation_value(
                    v, t, symbol_table=symbol_table
                )
                assert not symbols_and_types.type_validation_value(
                    v, typing.Union[t, types_[(i + 1) % len(types_)]],
                    symbol_table=symbol_table
                )

    with pytest.raises(ValueError, message="typing Generic not supported"):
        assert symbols_and_types.type_validation_value(
            None,
            typing.Generic[typing.T]
        )
