from operator import contains, eq, gt, mul, not_
from typing import AbstractSet, Tuple

import pytest

from ...exceptions import NeuroLangException
from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import (ColumnInt, ColumnStr, Destroy, Difference,
                                   ExtendedProjection,
                                   ExtendedProjectionListMember, NameColumns,
                                   NaturalJoin, Projection, RenameColumn,
                                   Selection)
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
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
    )

    fa = R1(C_(1), y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(
            Selection(R1, C_(eq)(C_(ColumnInt(0)), C_(1))),
            (C_(ColumnInt(1)),)
        ),
        (Constant(ColumnStr('y')),)
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
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
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
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
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
        NaturalJoin(fb_trans, RenameColumn(fb_trans, Constant(ColumnStr('x')),
                                           Constant(ColumnStr('z')))),
        C_(eq)(C_(ColumnStr('x')), C_(ColumnStr('z')))
    )

    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(z, x)))

    expected_result = Selection(
        NaturalJoin(fb_trans, RenameColumn(fb_trans, Constant(ColumnStr('x')),
                                           Constant(ColumnStr('z')))),
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
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
    )

    fb_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr('y')), Constant(ColumnStr('z')))
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == NaturalJoin(fa_trans, fb_trans)

    R2 = S_('R2')
    fb = R2(x, y)
    fb_trans = NameColumns(
        Projection(R2, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
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
        (Constant(ColumnStr('y')),)
    )

    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    assert res == Difference(fa_trans, NaturalJoin(fa_trans, fb_trans))


def test_selection():
    x = S_('x')
    y = S_('y')
    R1 = S_('R1')
    fa = R1(x, y)
    builtin_condition = C_(gt)(x, C_(3))
    exp = Conjunction((fa, builtin_condition))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
    )
    assert res == Selection(fa_trans, builtin_condition)


def test_extended_projection():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    R1 = S_('R1')
    fa = R1(x, y)
    builtin_condition = C_(eq)(C_(mul)(x, C_(3)), z)
    exp = Conjunction((fa, builtin_condition))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
    )
    exp_trans = ExtendedProjection(
        fa_trans, [
            ExtendedProjectionListMember(*builtin_condition.args),
            ExtendedProjectionListMember(x, x),
            ExtendedProjectionListMember(y, y)
        ]
    )
    assert res == exp_trans


def test_extended_projection_algebraic_expression():
    x = S_('x')
    y = S_('y')
    R1 = S_('R1')
    fa = R1(x, y)
    builtin_condition = C_(eq)(C_(mul)(C_(2), C_(3)), y)
    exp = Conjunction((fa, builtin_condition))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr('x')), Constant(ColumnStr('y')))
    )
    assert res.relation_left == fa_trans
    assert len(res.relation_right.value)
    assert ({'y': 6} in res.relation_right.value)


def test_set_destroy():
    r1 = S_('R1')
    x = S_('x')
    y = S_('y')
    exp = Conjunction((C_(contains)(x, y), r1(x)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    exp_result = Destroy(
        NameColumns(Projection(r1, (C_(0),)), (C_('x'),)),
        x, y
    )
    assert res == exp_result


def test_set_constant_contains():
    r1 = S_('R1')
    x = S_('x')
    exp = Conjunction((C_(contains)(x, C_(0)), r1(x)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    exp_result = Selection(
        NameColumns(Projection(r1, (C_(0),)), (C_('x'),)),
        C_(contains)(x, C_(0))
    )
    assert res == exp_result


def test_only_equality():
    exp = Conjunction((C_(eq)(S_('x'), C_(3)),))

    res = TranslateToNamedRA().walk(exp)

    exp_result = NamedRelationalAlgebraFrozenSet(('x',), (3,))

    assert isinstance(res, Constant)
    assert res.value == exp_result


def test_border_cases():
    R1 = S_('R1')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    tr = TranslateToNamedRA()

    exp = Conjunction((
        R1(x),
        C_(eq)(x, x)
    ))
    res = tr.walk(exp)

    assert res == NameColumns(
        Projection(R1, (C_(0),)),
        (Constant(ColumnStr('x')),)
    )

    exp = Conjunction((
        Negation(Negation(R1(x))),
    ))
    res = tr.walk(exp)

    assert res == NameColumns(
        Projection(R1, (C_(0),)),
        (Constant(ColumnStr('x')),)
    )

    tr = TranslateToNamedRA()

    exp = Conjunction((
        R1(x),
        C_(eq)(x, y),
        C_(eq)(z, C_(2) * y)
    ))
    res = tr.walk(exp)
    expected_res = (
        ExtendedProjection(
            Selection(
                NaturalJoin(
                    NameColumns(
                        Projection(R1, (C_(0),)),
                        (C_('x'),)
                    ),
                    RenameColumn(
                        NameColumns(
                            Projection(R1, (C_(0),)),
                            (C_('x'),)
                        ),
                        C_('x'),
                        C_('y')
                    )
                ),
                C_(eq)(C_('x'), C_('y'))
            ),
            (
                ExtendedProjectionListMember(C_('x'), C_('x')),
                ExtendedProjectionListMember(C_('y'), C_('y')),
                ExtendedProjectionListMember(C_(2) * C_('y'), C_('z')),
            )
        )
    )
    assert res == expected_res


def test_border_case_2():
    T = Symbol[AbstractSet[int]]('T')
    x = Symbol[int]('x')

    def gtz_f(x):
        return x > 0

    gtz = Constant(gtz_f)

    exp = Conjunction((
        T(x),
        Negation(
            gtz(x)
        )
    ))

    expected_res = Selection(
        NameColumns(
            Projection(
                T,
                (C_(0),)
            ),
            (C_('x'),)
        ),
        C_(not_)(
            gtz(C_('x'))
        )
    )

    res = TranslateToNamedRA().walk(exp)
    assert res == expected_res
