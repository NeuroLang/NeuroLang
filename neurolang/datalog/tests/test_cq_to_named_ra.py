from operator import eq
from typing import AbstractSet, Tuple

import pytest

from ...exceptions import NeuroLangException
from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import (ColumnInt, ColumnStr, Difference,
                                   NameColumns, NaturalJoin, Projection,
                                   RenameColumn, Selection)
from ...utils import NamedRelationalAlgebraFrozenSet
from ..expressions import Conjunction, Negation
from ..translate_to_named_ra import TranslateToNamedRA

C_ = Constant
S_ = Symbol
F_ = FunctionApplication


def test_translate_set():
    x = S_('x')
    y = S_('y')
    R1 = S_('R1')
    fa = R1(x, y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    fa = R1(C_(1), y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(
            Selection(R1, C_(eq)(C_(ColumnInt(0)), C_(1))),
            (C_(ColumnInt(1)),)
        ),
        (y,)
    )


def test_equality_constant_symbol():
    x = S_('x')
    a = C_('a')
    R1 = S_('R1')

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
    fb = R1(x, y)

    exp = Conjunction((fb, fa))

    fb_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    res = tr.walk(exp)
    assert res == NaturalJoin(fb_trans, expected_result)


def test_equality_symbols():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    w = S_('w')
    R1 = S_('R1')

    y = S_('y')
    fb = R1(x, y)
    fb_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    exp = Conjunction((fb, C_(eq)(x, y)))

    expected_result = Selection(
        fb_trans,
        C_(eq)(C_(ColumnStr('x')), C_(ColumnStr('y')))
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(x, z)))

    expected_result = Selection(
        NaturalJoin(fb_trans, RenameColumn(fb_trans, x, z)),
        C_(eq)(C_(ColumnStr('x')), C_(ColumnStr('z')))
    )

    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(z, x)))

    expected_result = Selection(
        NaturalJoin(fb_trans, RenameColumn(fb_trans, x, z)),
        C_(eq)(C_(ColumnStr('z')), C_(ColumnStr('x')))
    )

    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(z, w)))
    with pytest.raises(NeuroLangException, match="At least.*"):
        res = tr.walk(exp)


def test_joins():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    R1 = S_('R1')
    fa = R1(x, y)
    fb = R1(y, z)
    exp = Conjunction((fa, fb))

    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )

    fb_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (y, z)
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == NaturalJoin(fa_trans, fb_trans)

    R2 = S_('R2')
    fb = R2(x, y)
    fb_trans = NameColumns(
        Projection(R2, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )
    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == Difference(fa_trans, fb_trans)

    fa = R1(x, y)
    fb = R2(y, C_(0))
    fb_trans = NameColumns(
        Projection(
            Selection(R2, C_(eq)(C_(ColumnInt(1)), C_(0))),
            (C_(ColumnInt(0)),)
        ),
        (y,)
    )

    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    assert res == Difference(fa_trans, NaturalJoin(fa_trans, fb_trans))
