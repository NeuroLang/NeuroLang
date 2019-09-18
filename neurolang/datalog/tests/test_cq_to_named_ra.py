from operator import eq
from typing import AbstractSet, Tuple

from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import (ColumnInt, Difference, NameColumns,
                                   NaturalJoin, Projection, Selection)
from ...utils import NamedRelationalAlgebraFrozenSet
from ..expressions import Conjunction, Negation
from ..translate_to_named_ra import TranslateToNamedRA


C_ = Constant
S_ = Symbol
F_ = FunctionApplication


def test_translate_set():
    x = S_('x')
    y = S_('y')
    fa = S_('R1')(x, y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(S_('R1'), (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    fa = S_('R1')(C_(1), y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(
            Selection(S_('R1'), C_(eq)(C_(ColumnInt(0)), C_(1))),
            (C_(ColumnInt(1)),)
        ),
        (y,)
    )


def test_equality_constant_symbol():
    x = S_('x')
    a = C_('a')

    expected_result = \
        C_[AbstractSet[Tuple[str]]](
            NamedRelationalAlgebraFrozenSet(('x',), {'a'})
        )

    fa = C_(eq)(x, a)
    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == expected_result

    fa = C_(eq)(a, x)
    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == expected_result

    y = S_('y')
    fb = S_('R1')(x, y)

    exp = Conjunction((fb, fa))

    fb_trans = NameColumns(
        Projection(S_('R1'), (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    res = tr.walk(exp)
    assert res == NaturalJoin(fb_trans, expected_result)


def test_joins():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    fa = S_('R1')(x, y)
    fb = S_('R1')(y, z)
    exp = Conjunction((fa, fb))

    fa_trans = NameColumns(
        Projection(S_('R1'), (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    fb_trans = NameColumns(
        Projection(S_('R1'), (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (y, z)
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == NaturalJoin(fa_trans, fb_trans)

    fb = S_('R2')(x, y)
    fb_trans = NameColumns(
        Projection(S_('R2'), (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )
    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == Difference(fa_trans, fb_trans)

    fa = S_('R1')(x, y)
    fb = S_('R2')(y, C_(0))
    fb_trans = NameColumns(
        Projection(
            Selection(S_('R2'), C_(eq)(C_(ColumnInt(1)), C_(0))),
            (C_(ColumnInt(0)),)
        ),
        (y,)
    )

    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    assert res == Difference(fa_trans, NaturalJoin(fa_trans, fb_trans))
