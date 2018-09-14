import pytest

import typing

from .. import symbols_and_types
from .. import expressions

C_ = expressions.Constant
S_ = expressions.Symbol


def test_get_type_args():
    args = symbols_and_types.get_type_args(typing.Set)
    assert args == tuple()

    args = symbols_and_types.get_type_args(typing.Set[int])
    assert args == (int, )


def test_subclass():
    assert not issubclass(
        expressions.Constant[int], expressions.Constant[typing.AbstractSet]
    )


def test_replace_subtype():
    assert (
        typing.Set is symbols_and_types.replace_type_variable(
            int, typing.Set, typing.T
        )
    )
    assert (str is symbols_and_types.replace_type_variable(int, str, typing.T))

    assert (
        typing.Set[float] == symbols_and_types.replace_type_variable(
            int, typing.Set[float], typing.T
        )
    )

    assert (
        typing.Set[float] == symbols_and_types.replace_type_variable(
            float, typing.Set[typing.T], typing.T
        )
    )

    assert (
        typing.Tuple[float, int] is symbols_and_types.replace_type_variable(
            float, typing.Tuple[typing.T, int], typing.T
        )
    )

    assert (
        typing.Set[str] != symbols_and_types.
        replace_type_variable(float, typing.Set[typing.T], typing.T)
    )


def test_get_type_and_value():
    type_, value = symbols_and_types.get_type_and_value(3)
    assert type_ is int
    assert value == 3

    type_, value = symbols_and_types.get_type_and_value(C_[int](3))
    assert type_ is int
    assert value == 3

    type_, value = symbols_and_types.get_type_and_value(
        S_('a'), {S_('a'): C_[int](3)}
    )
    assert type_ is int
    assert value == 3

    def f(a: int) -> int:
        return 0

    type_, value = symbols_and_types.get_type_and_value(f)

    assert type_ is typing.Callable[[int], int]
    assert value == f


def test_type_validation_value():
    def f(a: int) -> int:
        return 0

    values = (
        3, {3, 8}, 'try', f, (3, 'a'),
        C_[typing.Tuple[str, float]](('a', 3.)), {'a': 3}
    )

    types_ = (
        int, typing.AbstractSet[int], typing.Text, typing.Callable[[int], int],
        typing.Tuple[int, str], typing.Tuple[str, float],
        typing.Mapping[str, int]
    )

    for i, v in enumerate(values):
        assert symbols_and_types.type_validation_value(
            v, typing.Any
        )

        for j, t in enumerate(types_):
            if i is j:
                assert symbols_and_types.type_validation_value(
                    v, t
                )
                assert symbols_and_types.type_validation_value(
                    v,
                    typing.Union[t, types_[(i + 1) % len(types_)]],
                )
            else:
                assert not symbols_and_types.type_validation_value(
                    v, t
                )
                assert not symbols_and_types.type_validation_value(
                    v,
                    typing.Union[t, types_[(i + 1) % len(types_)]]
                )

    with pytest.raises(ValueError, message="typing Generic not supported"):
        assert symbols_and_types.type_validation_value(
            None, typing.Generic[typing.T]
        )


def test_TypedSymbol():
    v = 3
    t = int
    s = C_[t](v)
    assert s.value == v
    assert s.type is t

    with pytest.raises(symbols_and_types.NeuroLangTypeException):
        s = C_[t]('a')


def test_TypedSymbolTable():
    st = symbols_and_types.TypedSymbolTable()
    s1 = C_[int](3)
    s2 = C_[int](4)
    s3 = C_[float](5.)
    s4 = C_[int](5)
    s6 = C_[str]('a')

    assert len(st) == 0

    st[S_('s1')] = s1
    assert len(st) == 1
    assert 's1' in st
    assert st['s1'] == s1
    assert st.symbols_by_type(s1.type) == {'s1': s1}

    st[S_('s2')] = s2
    assert len(st) == 2
    assert 's2' in st
    assert st['s2'] == s2
    assert st.symbols_by_type(s1.type) == {'s1': s1, 's2': s2}

    st[S_('s3')] = s3
    assert len(st) == 3
    assert 's3' in st
    assert st['s3'] == s3
    assert st.symbols_by_type(s1.type) == {'s1': s1, 's2': s2}
    assert st.symbols_by_type(s3.type, False) == {'s3': s3}
    assert st.symbols_by_type(s3.type, True) == {'s1': s1, 's2': s2, 's3': s3}

    del st['s1']
    assert len(st) == 2
    assert 's1' not in st
    assert 's1' not in st.symbols_by_type(s1.type)

    assert {int, float} == st.types()

    stb = st.create_scope()
    assert 's2' in stb
    assert 's3' in stb
    stb[S_('s4')] = s4
    assert 's4' in stb
    assert 's4' not in st

    stb[S_('s5')] = None
    assert 's5' in stb
    assert stb[S_('s5')] is None

    stc = stb.create_scope()
    stc[S_('s6')] = s6
    assert {int, float, str} == stc.types()
    assert stc.symbols_by_type(int) == {'s2': s2, 's4': s4}

    assert set(iter(stc)) == {'s2', 's3', 's4', 's5', 's6'}

    with pytest.raises(ValueError):
        stb[S_('s6')] = 5


def test_get_callable_arguments_and_return():
    c = typing.Callable[[int, str], float]
    args, ret = symbols_and_types.get_Callable_arguments_and_return(c)
    assert args == (int, str)
    assert ret is float


def test_free_variable_wrapping():
    def f(a: int) -> float:
        return 2. * int(a)

    fva = C_(f)
    x = S_[int]('x')
    fvb = fva(x)
    fva_type, fva_value = symbols_and_types.get_type_and_value(fva)
    assert fva_type is typing.Callable[[int], float]
    assert fva_value == f

    assert symbols_and_types.get_type_and_value(fvb) == (float, fvb)
    assert symbols_and_types.get_type_and_value(x) == (int, x)
