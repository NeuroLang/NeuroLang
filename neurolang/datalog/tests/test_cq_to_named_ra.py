from operator import eq

from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import (Column, Difference, NameColumns,
                                   NaturalJoin, Projection, Selection)
from ...utils import RelationalAlgebraFrozenSet
from ..expressions import Conjunction, Negation
from ..translate_to_named_ra import TranslateToNamedRA

R1 = RelationalAlgebraFrozenSet([(i, i * 2) for i in range(10)])

R2 = RelationalAlgebraFrozenSet([(i * 2, i * 3) for i in range(10)])

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
        Projection(S_('R1'), (C_(Column(0)), C_(Column(1)))),
        (x, y)
    )

    fa = S_('R1')(C_(1), y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(
            Selection(S_('R1'), C_(eq)(C_(Column(0)), C_(1))),
            (C_(Column(1)),)
        ),
        (y,)
    )

def test_joins():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    fa = S_('R1')(x, y)
    fb = S_('R1')(y, z)
    exp = Conjunction((fa, fb))

    fa_trans = NameColumns(
        Projection(S_('R1'), (C_(Column(0)), C_(Column(1)))),
        (x, y)
    )

    fb_trans = NameColumns(
        Projection(S_('R1'), (C_(Column(0)), C_(Column(1)))),
        (y, z)
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == NaturalJoin(fa_trans, fb_trans)

    fb = S_('R2')(x, y)
    fb_trans = NameColumns(
        Projection(S_('R2'), (C_(Column(0)), C_(Column(1)))),
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
            Selection(S_('R2'), C_(eq)(C_(Column(1)), C_(0))),
            (C_(Column(0)),)
        ),
        (y,)
    )

    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    assert res == Difference(fa_trans, NaturalJoin(fa_trans, fb_trans))
