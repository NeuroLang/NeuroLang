from operator import contains, eq, gt, mul, not_
from typing import AbstractSet, Tuple

import pytest

from ...exceptions import NeuroLangException
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Disjunction
from ...relational_algebra import (
    ColumnInt,
    ColumnStr,
    Destroy,
    Difference,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NameColumns,
    NaturalJoin,
    Projection,
    Selection,
    Union,
    str2columnstr_constant,
)
from ...utils import NamedRelationalAlgebraFrozenSet
from ..expressions import Conjunction, Negation
from ..translate_to_named_ra import TranslateToNamedRA

C_ = Constant
S_ = Symbol
F_ = FunctionApplication

EQ = Constant(eq)


def test_translate_set():
    x = S_("x")
    y = S_("y")
    R1 = S_("R1")
    fa = R1(x, y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )

    fa = R1(C_(1), y)

    tr = TranslateToNamedRA()
    res = tr.walk(fa)
    assert res == NameColumns(
        Projection(
            Selection(R1, C_(eq)(C_(ColumnInt(0)), C_(1))), (C_(ColumnInt(1)),)
        ),
        (Constant(ColumnStr("y")),),
    )


def test_equality_constant_symbol():
    x = S_("x")
    a = C_("a")
    R1 = S_("R1")

    expected_result = C_[AbstractSet[Tuple[str]]](
        NamedRelationalAlgebraFrozenSet(("x",), {"a"})
    )

    fa = C_(eq)(x, a)
    tr = TranslateToNamedRA()
    res = tr.walk(Conjunction((fa,)))
    assert res == expected_result

    fa = C_(eq)(a, x)
    tr = TranslateToNamedRA()
    res = tr.walk(Conjunction((fa,)))
    assert res == expected_result

    y = S_("y")
    fb = R1(x, y)

    exp = Conjunction((fb, fa))

    expected_result = Selection(
        NameColumns(
            Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
            (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
        ),
        C_(eq)(C_(ColumnStr("x")), a),
    )

    res = tr.walk(exp)
    assert res == expected_result


def test_equality_symbols():
    x = S_("x")
    y = S_("y")
    z = S_("z")
    w = S_("w")
    R1 = S_("R1")

    y = S_("y")
    fb = R1(x, y)
    fb_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )

    exp = Conjunction((fb, C_(eq)(x, y)))

    expected_result = Selection(
        fb_trans, C_(eq)(C_(ColumnStr("x")), C_(ColumnStr("y")))
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(x, z)))

    expected_result = ExtendedProjection(
        fb_trans,
        (
            ExtendedProjectionListMember(
                Constant(ColumnStr("x")), Constant(ColumnStr("x"))
            ),
            ExtendedProjectionListMember(
                Constant(ColumnStr("y")), Constant(ColumnStr("y"))
            ),
            ExtendedProjectionListMember(
                Constant(ColumnStr("x")), Constant(ColumnStr("z"))
            ),
        ),
    )

    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(z, x)))

    expected_result = ExtendedProjection(
        fb_trans,
        (
            ExtendedProjectionListMember(
                Constant(ColumnStr("x")), Constant(ColumnStr("x"))
            ),
            ExtendedProjectionListMember(
                Constant(ColumnStr("y")), Constant(ColumnStr("y"))
            ),
            ExtendedProjectionListMember(
                Constant(ColumnStr("x")), Constant(ColumnStr("z"))
            ),
        ),
    )

    res = tr.walk(exp)
    assert res == expected_result

    exp = Conjunction((fb, C_(eq)(z, w)))
    with pytest.raises(NeuroLangException, match="Could not resolve*"):
        res = tr.walk(exp)


def test_joins():
    x = S_("x")
    y = S_("y")
    z = S_("z")
    R1 = S_("R1")
    fa = R1(x, y)
    fb = R1(y, z)
    exp = Conjunction((fa, fb))

    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )

    fb_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("y")), Constant(ColumnStr("z"))),
    )

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == NaturalJoin(fa_trans, fb_trans)

    R2 = S_("R2")
    fb = R2(x, y)
    fb_trans = NameColumns(
        Projection(R2, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )
    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    assert res == Difference(fa_trans, fb_trans)

    fa = R1(x, y)
    fb = R2(y, C_(0))
    fb_trans = NameColumns(
        Projection(
            Selection(R2, C_(eq)(C_(ColumnInt(1)), C_(0))), (C_(ColumnInt(0)),)
        ),
        (Constant(ColumnStr("y")),),
    )

    exp = Conjunction((fa, Negation(fb)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    assert res == Difference(fa_trans, NaturalJoin(fa_trans, fb_trans))


def test_selection():
    x = S_("x")
    y = S_("y")
    R1 = S_("R1")
    fa = R1(x, y)
    builtin_condition = C_(gt)(x, C_(3))
    exp = Conjunction((fa, builtin_condition))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )
    assert res == Selection(fa_trans, builtin_condition)


def test_extended_projection():
    x = S_("x")
    y = S_("y")
    z = S_("z")
    R1 = S_("R1")
    fa = R1(x, y)
    builtin_condition = C_(eq)(C_(mul)(x, C_(3)), z)
    exp = Conjunction((fa, builtin_condition))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )
    exp_trans = ExtendedProjection(
        fa_trans,
        [
            ExtendedProjectionListMember(*builtin_condition.args),
            ExtendedProjectionListMember(x, x),
            ExtendedProjectionListMember(y, y),
        ],
    )
    assert res == exp_trans


def test_extended_projection_2():
    u = S_("u")
    v = S_("v")
    w = S_("w")
    x = S_("x")
    y = S_("y")
    z = S_("z")
    R1 = S_("R1")
    fa = R1(x, y, v, u)
    builtin_condition_1 = C_(eq)(C_(mul)(x, C_(3)), z)
    builtin_condition_2 = C_(eq)(C_(mul)(v, C_(2)), w)
    exp = Conjunction((fa, builtin_condition_1, builtin_condition_2))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(
            R1,
            (
                C_(ColumnInt(0)),
                C_(ColumnInt(1)),
                C_(ColumnInt(2)),
                C_(ColumnInt(3)),
            ),
        ),
        (
            Constant(ColumnStr("x")),
            Constant(ColumnStr("y")),
            Constant(ColumnStr("v")),
            Constant(ColumnStr("u")),
        ),
    )
    exp_trans = ExtendedProjection(
        fa_trans,
        [
            ExtendedProjectionListMember(*builtin_condition_1.args),
            ExtendedProjectionListMember(*builtin_condition_2.args),
            ExtendedProjectionListMember(x, x),
            ExtendedProjectionListMember(y, y),
            ExtendedProjectionListMember(u, u),
            ExtendedProjectionListMember(v, v),
        ],
    )
    assert res == exp_trans


def test_extended_projection_algebraic_expression():
    x = S_("x")
    y = S_("y")
    R1 = S_("R1")
    fa = R1(x, y)
    builtin_condition = C_(eq)(C_(mul)(C_(2), C_(3)), y)
    exp = Conjunction((fa, builtin_condition))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
    )
    assert res == Selection(
        fa_trans, Constant(eq)(Constant(ColumnStr("y")), Constant(6))
    )


def test_set_destroy():
    r1 = S_("R1")
    x = S_("x")
    y = S_("y")

    tr = TranslateToNamedRA()

    exp = Conjunction((C_(contains)(x, y), r1(x)))
    res = tr.walk(exp)

    exp_result = Destroy(
        NameColumns(Projection(r1, (C_(0),)), (C_("x"),)), x, y
    )
    assert res == exp_result


def test_set_destroy_multicolumn():
    r1 = S_("R1")
    x = S_("x")
    y = S_("y")
    z = S_("z")

    tr = TranslateToNamedRA()
    exp = Conjunction((C_(contains)(x, C_((y, z))), r1(x)))
    res = tr.walk(exp)

    exp_result = Destroy(
        NameColumns(Projection(r1, (C_(0),)), (C_("x"),)),
        x,
        C_[Tuple[ColumnStr, ColumnStr]]((ColumnStr("y"), ColumnStr("z"))),
    )
    assert res == exp_result


def test_set_constant_contains():
    r1 = S_("R1")
    x = S_("x")
    exp = Conjunction((C_(contains)(x, C_(0)), r1(x)))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)

    exp_result = Selection(
        NameColumns(Projection(r1, (C_(0),)), (C_("x"),)),
        C_(contains)(x, C_(0)),
    )
    assert res == exp_result


def test_only_equality():
    exp = Conjunction((C_(eq)(S_("x"), C_(3)),))

    res = TranslateToNamedRA().walk(exp)

    exp_result = NamedRelationalAlgebraFrozenSet(("x",), (3,))

    assert isinstance(res, Constant)
    assert res.value == exp_result


def test_disjunction():
    p1 = S_("T")(S_("x"))
    p2 = S_("U")(S_("x"))
    p3 = S_("V")(S_("x"))
    r1 = TranslateToNamedRA().walk(p1)
    r2 = TranslateToNamedRA().walk(p2)
    r3 = TranslateToNamedRA().walk(p3)

    exp = Disjunction((p1,))
    res = TranslateToNamedRA().walk(exp)
    assert res == r1

    exp = Disjunction((p1, p2))
    res = TranslateToNamedRA().walk(exp)
    assert res == Union(r1, r2)

    exp = Disjunction((p1, p2, p3))
    res = TranslateToNamedRA().walk(exp)
    assert res == Union(r1, Union(r2, r3))

    exp = Disjunction((p1, Conjunction((p2, p3))))
    res = TranslateToNamedRA().walk(exp)
    assert res == Union(r1, TranslateToNamedRA().walk(Conjunction((p2, p3))))

    exp = Conjunction((p1, Disjunction((p2, p3))))
    res = TranslateToNamedRA().walk(exp)
    assert res == NaturalJoin(
        r1, TranslateToNamedRA().walk(Disjunction((p2, p3)))
    )


def test_border_cases():
    R1 = S_("R1")
    x = S_("x")
    y = S_("y")
    z = S_("z")

    tr = TranslateToNamedRA()

    exp = Conjunction((R1(x), C_(eq)(x, x)))
    res = tr.walk(exp)

    assert res == NameColumns(
        Projection(R1, (C_(0),)), (Constant(ColumnStr("x")),)
    )

    exp = Conjunction((Negation(Negation(R1(x))),))
    res = tr.walk(exp)

    assert res == NameColumns(
        Projection(R1, (C_(0),)), (Constant(ColumnStr("x")),)
    )

    tr = TranslateToNamedRA()

    exp = Conjunction((R1(x), C_(eq)(x, y), C_(eq)(z, C_(2) * y)))
    res = tr.walk(exp)
    expected_res = ExtendedProjection(
        ExtendedProjection(
            NameColumns(Projection(R1, (C_(0),)), (C_("x"),)),
            (
                ExtendedProjectionListMember(
                    C_(ColumnStr("x")),
                    C_(ColumnStr("x")),
                ),
                ExtendedProjectionListMember(
                    C_(ColumnStr("x")),
                    C_(ColumnStr("y")),
                ),
            ),
        ),
        (
            ExtendedProjectionListMember(C_("x"), C_("x")),
            ExtendedProjectionListMember(C_("y"), C_("y")),
            ExtendedProjectionListMember(C_(2) * C_("y"), C_("z")),
        ),
    )
    assert res == expected_res


def test_border_case_2():
    T = Symbol[AbstractSet[int]]("T")
    x = Symbol[int]("x")
    y = Symbol[int]("y")

    def gtz_f(x):
        return x > 0

    gtz = Constant(gtz_f)

    exp = Conjunction((T(x), Negation(gtz(x))))

    expected_res = Selection(
        NameColumns(Projection(T, (C_(0),)), (C_(ColumnStr("x")),)),
        C_(not_)(gtz(C_(ColumnStr("x")))),
    )

    res = TranslateToNamedRA().walk(exp)
    assert res == expected_res

    exp = Conjunction((T(x), Negation(C_(eq)(x, C_(3)))))

    res = TranslateToNamedRA().walk(exp)

    expected_res = Selection(
        NameColumns(Projection(T, (C_(0),)), (C_(ColumnStr("x")),)),
        C_(not_)(C_(eq)(C_(ColumnStr("x")), C_(3))),
    )
    assert res == expected_res

    exp = Conjunction((T(x, y), Negation(C_(eq)(x, y))))

    res = TranslateToNamedRA().walk(exp)

    expected_res = Selection(
        NameColumns(Projection(T, (C_(0), C_(1))), (C_("x"), C_("y"))),
        C_(not_)(C_(eq)(C_(ColumnStr("x")), C_(ColumnStr("y")))),
    )
    assert res == expected_res


def test_extended_projection_variable_equality():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    conjunction = Conjunction((Q(x), EQ(y, x)))
    result = TranslateToNamedRA().walk(conjunction)
    assert result == ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("y"),
            ),
        ),
    )


def test_extended_projection_variable_equality_constant():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    a = Constant("a")
    conjunction = Conjunction((Q(x), EQ(y, a)))
    result = TranslateToNamedRA().walk(conjunction)
    assert result == ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                Constant("a"),
                str2columnstr_constant("y"),
            ),
        ),
    )


def test_extended_projection_variable_equality_not_named():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    conjunction = Conjunction((Q(x), EQ(y, x), EQ(z, y)))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("y"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_extended_projection_variable_equality_twice_same_lhs():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    conjunction = Conjunction((Q(x), EQ(x, y), EQ(x, z)))
    result = TranslateToNamedRA().walk(conjunction)
    assert result == ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("y"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("z"),
            ),
        ),
    )


def test_double_equality():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    conjunction = Conjunction((Q(x), EQ(x, y), EQ(x, z)))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("z"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("y"),
            ),
        ),
    )
    assert result == expected


def test_three_way_equality():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    w = Symbol("w")
    z = Symbol("z")
    conjunction = Conjunction((Q(x), EQ(z, y), EQ(y, w), EQ(w, x)))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("w"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("w"),
                str2columnstr_constant("y"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_not_ordered_equalites():
    Q = Symbol("Q")
    x = Symbol("x")
    y = Symbol("y")
    w = Symbol("w")
    h = Symbol("h")
    z = Symbol("z")
    conjunction = Conjunction((Q(x), EQ(w, h), EQ(h, x), EQ(y, w), EQ(z, y)))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(Q, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("h"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("h"),
                str2columnstr_constant("w"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("w"),
                str2columnstr_constant("y"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_equality_builtin_application():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    P = Symbol("P")
    f = Constant(lambda x: x ** 2)
    conjunction = Conjunction((P(x), P(y), EQ(z, f(x, y))))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NaturalJoin(
            NameColumns(
                Projection(P, (C_(ColumnInt(0)),)),
                (C_(ColumnStr("x")),),
            ),
            NameColumns(
                Projection(P, (C_(ColumnInt(0)),)),
                (C_(ColumnStr("y")),),
            ),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("y"),
            ),
            ExtendedProjectionListMember(
                f(C_(ColumnStr("x")), C_(ColumnStr("y"))),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_equality_nested_builtin_application():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    P = Symbol("P")
    f = Constant(lambda x: x // 2)
    g = Constant(lambda x: x ** 2)
    conjunction = Conjunction((P(x), P(y), EQ(z, f(g(x), g(y)))))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NaturalJoin(
            NameColumns(
                Projection(P, (C_(ColumnInt(0)),)),
                (C_(ColumnStr("x")),),
            ),
            NameColumns(
                Projection(P, (C_(ColumnInt(0)),)),
                (C_(ColumnStr("y")),),
            ),
        ),
        (
            ExtendedProjectionListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            ExtendedProjectionListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("y"),
            ),
            ExtendedProjectionListMember(
                f(g(C_(ColumnStr("x"))), g(C_(ColumnStr("y")))),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected
