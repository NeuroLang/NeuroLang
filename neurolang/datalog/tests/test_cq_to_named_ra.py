from typing import AbstractSet, Tuple

from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import Difference, NaturalJoin
from ...utils import (
    NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet
)
from ..expressions import Conjunction, Negation
from ..translate_to_named_ra import TranslateToNamedRA

R1 = RelationalAlgebraFrozenSet([(i, i * 2) for i in range(10)])

R2 = RelationalAlgebraFrozenSet([(i * 2, i * 3) for i in range(10)])

C_ = Constant
S_ = Symbol
F_ = FunctionApplication

Symbol_Table = {
    'R1': C_[AbstractSet[Tuple[int, int]]](R1),
    'R2': C_[AbstractSet[Tuple[int, int]]](R2),
}


def test_translate_set():
    x = S_('x')
    y = S_('y')
    fa = S_('R1')(x, y)

    tr = TranslateToNamedRA(Symbol_Table)
    res = tr.walk(fa)
    assert isinstance(res, Constant[AbstractSet[Tuple[int, int]]])
    assert isinstance(res.value, NamedRelationalAlgebraFrozenSet)
    assert res.value.columns == ('x', 'y')
    assert res.value.to_unnamed() == R1

    fa = S_('R1')(C_(1), y)

    tr = TranslateToNamedRA(Symbol_Table)
    res = tr.walk(fa)
    assert isinstance(res, Constant[AbstractSet[Tuple[int]]])
    assert isinstance(res.value, NamedRelationalAlgebraFrozenSet)
    assert res.value.columns == ('y', )
    assert res.value.to_unnamed() == R1.selection({0: 1}).projection(1)


def test_joins():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    fa = S_('R1')(x, y)
    fb = S_('R1')(y, z)
    exp = Conjunction((fa, fb))

    tr = TranslateToNamedRA(Symbol_Table)
    res = tr.walk(exp)
    assert isinstance(res, NaturalJoin)
    assert isinstance(
        res.relation_left, Constant[AbstractSet[Tuple[int, int]]]
    )
    assert isinstance(
        res.relation_right, Constant[AbstractSet[Tuple[int, int]]]
    )
    assert isinstance(res.relation_left.value, NamedRelationalAlgebraFrozenSet)
    assert res.relation_left.value.columns == ('x', 'y')
    assert res.relation_left.value.to_unnamed() == R1
    assert isinstance(
        res.relation_right.value, NamedRelationalAlgebraFrozenSet
    )
    assert res.relation_right.value.columns == ('y', 'z')
    assert res.relation_right.value.to_unnamed() == R1

    fb = S_('R2')(x, y)
    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA(Symbol_Table)
    res = tr.walk(exp)
    assert isinstance(res, Difference)
    assert isinstance(
        res.relation_left, Constant[AbstractSet[Tuple[int, int]]]
    )
    assert isinstance(
        res.relation_right, Constant[AbstractSet[Tuple[int, int]]]
    )
    assert isinstance(res.relation_left.value, NamedRelationalAlgebraFrozenSet)
    assert res.relation_left.value.columns == ('x', 'y')
    assert res.relation_left.value.to_unnamed() == R1
    assert isinstance(
        res.relation_right.value, NamedRelationalAlgebraFrozenSet
    )
    assert res.relation_right.value.columns == ('x', 'y')
    assert res.relation_right.value.to_unnamed() == R2

    fa = S_('R1')(x, y)
    fb = S_('R2')(y, C_(0))
    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA(Symbol_Table)
    res = tr.walk(exp)

    assert isinstance(res, Difference)
    assert isinstance(
        res.relation_left, Constant[AbstractSet[Tuple[int, int]]]
    )

    rel_right = res.relation_right
    assert isinstance(rel_right, NaturalJoin)
    assert rel_right.relation_left == res.relation_left

    assert isinstance(
        rel_right.relation_right, Constant[AbstractSet[Tuple[int]]]
    )
    assert isinstance(
        rel_right.relation_right.value, NamedRelationalAlgebraFrozenSet
    )
    assert rel_right.relation_right.value.columns == ('y',)
    assert (
        rel_right.relation_right.value.to_unnamed() ==
        R2.selection({1: 0}).projection(0)
    )
