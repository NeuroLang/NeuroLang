import pytest

import typing

from .. import expressions

C_ = expressions.Constant
S_ = expressions.Symbol


def test_get_type_args():
    args = expressions.get_type_args(typing.Set)
    assert args == tuple()

    args = expressions.get_type_args(typing.Set[int])
    assert args == (int, )


def test_subclass():
    assert not issubclass(
        expressions.Constant[int], expressions.Constant[typing.AbstractSet]
    )


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
        assert expressions.type_validation_value(
            v, typing.Any
        )

        for j, t in enumerate(types_):
            if i is j:
                assert expressions.type_validation_value(
                    v, t
                )
                assert expressions.type_validation_value(
                    v,
                    typing.Union[t, types_[(i + 1) % len(types_)]],
                )
            else:
                assert not expressions.type_validation_value(
                    v, t
                )
                assert not expressions.type_validation_value(
                    v,
                    typing.Union[t, types_[(i + 1) % len(types_)]]
                )

    with pytest.raises(ValueError, match="typing Generic not supported"):
        assert expressions.type_validation_value(
            None, typing.Generic[typing.T]
        )


def test_TypedSymbol():
    v = 3
    t = int
    s = C_[t](v)
    assert s.value == v
    assert s.type is t

    with pytest.raises(expressions.NeuroLangTypeException):
        s = C_[t]('a')


def test_TypedSymbolTable():
    st = expressions.TypedSymbolTable()
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


def test_free_variable_wrapping():
    def f(a: int) -> float:
        return 2. * int(a)

    fva = C_(f)
    x = S_[int]('x')
    fvb = fva(x)
    fva_type = fva.type
    fva_value = fva.value
    assert fva_type is typing.Callable[[int], float]
    assert fva_value == f

    assert fvb.type is float
    assert x.type is int


def test_symbols_by_type_kept_until_a_table_changes(monkeypatch):
    """The symbols of a type are compared to the types of the table once,
    until the table or a scope enclosing it changes; each call gets its
    own dict."""
    from .. import typed_symbol_table

    comparisons = []
    leq = typed_symbol_table.is_leq_informative
    monkeypatch.setattr(
        typed_symbol_table, "is_leq_informative",
        lambda a, b: comparisons.append(1) or leq(a, b),
    )
    outer = expressions.TypedSymbolTable()
    outer[S_('a')] = C_[int](1)
    inner = outer.create_scope()
    inner[S_('b')] = C_[int](2)
    inner[S_('c')] = C_[str]('x')

    first = inner.symbols_by_type(int)
    assert set(first) == {S_('a'), S_('b')}
    comparisons.clear()
    second = inner.symbols_by_type(int)
    assert second is first and not comparisons
    with pytest.raises(TypeError):  # read-only: shared by the callers
        second[S_('z')] = C_[int](0)

    inner[S_('d')] = C_[int](3)
    assert set(inner.symbols_by_type(int)) == {S_('a'), S_('b'), S_('d')}
    outer[S_('e')] = C_[int](4)  # an enclosing scope changes
    assert S_('e') in inner.symbols_by_type(int)
    del inner[S_('b')]
    assert S_('b') not in inner.symbols_by_type(int)
