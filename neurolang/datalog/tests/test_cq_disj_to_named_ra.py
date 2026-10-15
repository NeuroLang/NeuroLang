from operator import add, contains, eq, gt, mul, not_
from typing import AbstractSet, Callable, Tuple

import numpy
import pytest

from ... import exceptions
from ...exceptions import NeuroLangException
from ...expression_walker import expression_iterator
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Disjunction, ExistentialPredicate
from ...relational_algebra import (
    ColumnInt,
    ColumnStr,
    Destroy,
    Difference,
    ExtendedProjection,
    FunctionApplicationListMember,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraOptimiser,
    RelationalAlgebraSolver,
    Selection,
    Union,
    str2columnstr_constant,
)
from ..chase.relational_algebra import NamedRelationalAlgebraOptimiser
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
            FunctionApplicationListMember(
                Constant(ColumnStr("x")), Constant(ColumnStr("x"))
            ),
            FunctionApplicationListMember(
                Constant(ColumnStr("y")), Constant(ColumnStr("y"))
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                Constant(ColumnStr("x")), Constant(ColumnStr("x"))
            ),
            FunctionApplicationListMember(
                Constant(ColumnStr("y")), Constant(ColumnStr("y"))
            ),
            FunctionApplicationListMember(
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
    expected_builtin_condition = C_(gt)(C_(ColumnStr("x")), C_(3))
    assert res == Selection(fa_trans, expected_builtin_condition)


def test_selection_over_unbound_variable():
    x = S_("x")
    y = S_("y")
    R1 = S_("R1")
    exp = Conjunction((R1(x), C_(gt)(x, y)))

    with pytest.raises(exceptions.UnboundVariablesError) as exc_info:
        TranslateToNamedRA().walk(exp)
    assert exc_info.value.variables == ("y",)
    assert exc_info.value.rule is None
    assert str(exc_info.value).startswith(
        "In a rule body: the variable y appears only in `x > y`"
    )
    assert "normal form" not in exc_info.value.suggestion


def test_extended_projection():
    x = S_("x")
    y = S_("y")
    z = S_("z")
    R1 = S_("R1")
    fa = R1(x, y)
    builtin_condition = C_(eq)(C_(mul)(x, C_(3)), z)
    exp = Conjunction((fa, builtin_condition))
    cs_x = C_(ColumnStr("x"))
    cs_y = C_(ColumnStr("y"))
    cs_z = C_(ColumnStr("z"))

    tr = TranslateToNamedRA()
    res = tr.walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (cs_x, cs_y),
    )
    exp_trans = ExtendedProjection(
        fa_trans,
        [
            FunctionApplicationListMember(C_(mul)(cs_x, C_(3)), cs_z),
            FunctionApplicationListMember(cs_x, cs_x),
            FunctionApplicationListMember(cs_y, cs_y),
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
    cs_x = C_(ColumnStr("x"))
    cs_y = C_(ColumnStr("y"))
    cs_z = C_(ColumnStr("z"))
    cs_u = C_(ColumnStr("u"))
    cs_v = C_(ColumnStr("v"))
    cs_w = C_(ColumnStr("w"))

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
        (cs_x, cs_y, cs_v, cs_u),
    )
    exp_trans = ExtendedProjection(
        fa_trans,
        [
            FunctionApplicationListMember(C_(mul)(cs_x, C_(3)), cs_z),
            FunctionApplicationListMember(C_(mul)(cs_v, C_(2)), cs_w),
            FunctionApplicationListMember(cs_x, cs_x),
            FunctionApplicationListMember(cs_y, cs_y),
            FunctionApplicationListMember(cs_u, cs_u),
            FunctionApplicationListMember(cs_v, cs_v),
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


@pytest.mark.parametrize("swap", [False, True])
def test_equality_expression_with_bound_variable_is_selection(swap):
    x = S_("x")
    y = S_("y")
    R1 = S_("R1")
    condition = (C_(mul)(x, C_(3)), y)
    if not swap:
        condition = condition[::-1]
    exp = Conjunction((R1(x, y), C_(eq)(*condition)))
    cs_x = C_(ColumnStr("x"))
    cs_y = C_(ColumnStr("y"))

    res = TranslateToNamedRA().walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (cs_x, cs_y),
    )
    assert res == Selection(fa_trans, C_(eq)(cs_y, C_(mul)(cs_x, C_(3))))


def test_equality_expression_with_variable_bound_by_other_atom():
    i = S_("i")
    i1 = S_("i1")
    di = S_("di")
    q = S_("q")
    v = S_("v")
    exp = Conjunction((q(i1, di), v(i), C_(eq)(i, C_(add)(i1, di))))
    cs_i = C_(ColumnStr("i"))
    cs_i1 = C_(ColumnStr("i1"))
    cs_di = C_(ColumnStr("di"))

    res = TranslateToNamedRA().walk(exp)
    q_trans = NameColumns(
        Projection(q, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (cs_i1, cs_di),
    )
    v_trans = NameColumns(Projection(v, (C_(ColumnInt(0)),)), (cs_i,))
    assert res == NaturalJoin(
        ExtendedProjection(
            q_trans,
            (
                FunctionApplicationListMember(C_(add)(cs_i1, cs_di), cs_i),
                FunctionApplicationListMember(cs_i1, cs_i1),
                FunctionApplicationListMember(cs_di, cs_di),
            ),
        ),
        v_trans,
    )


def test_second_equality_on_computed_join_column_is_selection():
    i = S_("i")
    i1 = S_("i1")
    di = S_("di")
    q = S_("q")
    v = S_("v")
    exp = Conjunction(
        (
            q(i1, di),
            v(i),
            C_(eq)(i, C_(add)(i1, di)),
            C_(eq)(i, C_(mul)(di, C_(2))),
        )
    )
    cs_i = C_(ColumnStr("i"))
    cs_i1 = C_(ColumnStr("i1"))
    cs_di = C_(ColumnStr("di"))

    res = TranslateToNamedRA().walk(exp)
    q_trans = NameColumns(
        Projection(q, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (cs_i1, cs_di),
    )
    v_trans = NameColumns(Projection(v, (C_(ColumnInt(0)),)), (cs_i,))
    assert res == Selection(
        NaturalJoin(
            ExtendedProjection(
                q_trans,
                (
                    FunctionApplicationListMember(
                        C_(add)(cs_i1, cs_di), cs_i
                    ),
                    FunctionApplicationListMember(cs_i1, cs_i1),
                    FunctionApplicationListMember(cs_di, cs_di),
                ),
            ),
            v_trans,
        ),
        C_(eq)(cs_i, C_(mul)(cs_di, C_(2))),
    )


def neighbourhood_conjunction():
    """
    Return the conjunction of the voxels close to a reported peak.

    It is the body of ``reports_close(i, j, k, s) :-
    peak_reported(i1, j1, k1, s), offset(di, dj, dk), (i == i1 + di),
    (j == j1 + dj), (k == k1 + dk), voxel(i, j, k)``.
    """
    i, j, k, s = S_("i"), S_("j"), S_("k"), S_("s")
    i1, j1, k1 = S_("i1"), S_("j1"), S_("k1")
    di, dj, dk = S_("di"), S_("dj"), S_("dk")
    return Conjunction(
        (
            S_("peak_reported")(i1, j1, k1, s),
            S_("offset")(di, dj, dk),
            C_(eq)(i, C_(add)(i1, di)),
            C_(eq)(j, C_(add)(j1, dj)),
            C_(eq)(k, C_(add)(k1, dk)),
            S_("voxel")(i, j, k),
        )
    )


def test_equalities_on_bound_variables_join_on_computed_columns():
    cs = {
        name: C_(ColumnStr(name))
        for name in ("i", "j", "k", "s", "i1", "j1", "k1", "di", "dj", "dk")
    }

    def named(symbol, *columns):
        return NameColumns(
            Projection(
                S_(symbol),
                tuple(C_(ColumnInt(n)) for n in range(len(columns))),
            ),
            tuple(cs[c] for c in columns),
        )

    res = TranslateToNamedRA().walk(neighbourhood_conjunction())
    sources = NaturalJoin(
        named("peak_reported", "i1", "j1", "k1", "s"),
        named("offset", "di", "dj", "dk"),
    )
    computed = tuple(
        FunctionApplicationListMember(
            C_(add)(cs[f"{c}1"], cs[f"d{c}"]), cs[c]
        )
        for c in "ijk"
    ) + tuple(
        FunctionApplicationListMember(cs[c], cs[c])
        for c in ("i1", "j1", "k1", "s", "di", "dj", "dk")
    )
    assert res == NaturalJoin(
        ExtendedProjection(sources, computed),
        named("voxel", "i", "j", "k"),
    )


def test_optimised_neighbourhood_plan_has_no_product_with_voxel():
    res = TranslateToNamedRA().walk(neighbourhood_conjunction())
    res = NamedRelationalAlgebraOptimiser().walk(res)
    res = RelationalAlgebraOptimiser().walk(res)
    voxel = S_("voxel")
    nodes = [node for _, node in expression_iterator(res)]
    assert not any(isinstance(node, Selection) for node in nodes)
    assert any(isinstance(node, ExtendedProjection) for node in nodes)
    for node in nodes:
        if isinstance(node, Product):
            assert all(
                voxel not in (n for _, n in expression_iterator(relation))
                for relation in node.relations
            )


def test_two_equality_expressions_on_same_unbound_variable():
    x = S_("x")
    y = S_("y")
    z = S_("z")
    R1 = S_("R1")
    exp = Conjunction(
        (
            R1(x, y),
            C_(eq)(z, C_(mul)(x, C_(3))),
            C_(eq)(z, C_(mul)(y, C_(2))),
        )
    )
    cs_x = C_(ColumnStr("x"))
    cs_y = C_(ColumnStr("y"))
    cs_z = C_(ColumnStr("z"))

    res = TranslateToNamedRA().walk(exp)
    fa_trans = NameColumns(
        Projection(R1, (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (cs_x, cs_y),
    )
    assert isinstance(res, Selection)
    assert res.formula == C_(eq)(cs_z, C_(mul)(cs_y, C_(2)))
    assert isinstance(res.relation, ExtendedProjection)
    assert res.relation.relation == fa_trans
    new_column, *kept_columns = res.relation.projection_list
    assert new_column == FunctionApplicationListMember(
        C_(mul)(cs_x, C_(3)), cs_z
    )
    assert all(m.fun_exp == m.dst_column for m in kept_columns)
    assert {m.dst_column for m in kept_columns} == {cs_x, cs_y}


def test_set_destroy():
    r1 = S_("R1")
    x = S_("x")
    y = S_("y")

    tr = TranslateToNamedRA()

    exp = Conjunction((C_(contains)(x, y), r1(x)))
    res = tr.walk(exp)

    exp_result = Destroy(
        NameColumns(Projection(r1, (C_(0),)), (C_("x"),)),
        C_(ColumnStr("x")), C_(ColumnStr("y"))
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
        NameColumns(Projection(r1, (C_(0),)), (C_(ColumnStr("x")),)),
        C_(ColumnStr("x")),
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
        C_(contains)(C_(ColumnStr("x")), C_(0)),
    )
    assert res == exp_result


def test_only_equality():
    exp = Conjunction((C_(eq)(S_("x"), C_(3)),))

    res = TranslateToNamedRA().walk(exp)

    exp_result = NamedRelationalAlgebraFrozenSet(("x",), (3,))

    assert isinstance(res, Constant)
    assert res.value == exp_result


def test_only_equalities_to_constants_keep_all_columns():
    exp = Conjunction((
        C_(eq)(S_("x"), C_(1)),
        C_(eq)(S_("y"), C_(2)),
        C_(eq)(S_("z"), C_(3)),
    ))

    res = RelationalAlgebraSolver().walk(TranslateToNamedRA().walk(exp))

    assert res.value == NamedRelationalAlgebraFrozenSet(
        ("x", "y", "z"), [(1, 2, 3)]
    )


def test_only_equalities_to_constants():
    """Translate ``x = 3 & y = 'a'``, as in ``magic_p(3, 'a') :- ()``."""
    exp = Conjunction((C_(eq)(S_("x"), C_(3)), C_(eq)(S_("y"), C_("a"))))

    res = RelationalAlgebraSolver().walk(TranslateToNamedRA().walk(exp))

    assert res.value == NamedRelationalAlgebraFrozenSet(
        ("x", "y"), [(3, "a")]
    )


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
                FunctionApplicationListMember(
                    C_(ColumnStr("x")),
                    C_(ColumnStr("x")),
                ),
                FunctionApplicationListMember(
                    C_(ColumnStr("x")),
                    C_(ColumnStr("y")),
                ),
            ),
        ),
        (
            FunctionApplicationListMember(C_("x"), C_("x")),
            FunctionApplicationListMember(C_("y"), C_("y")),
            FunctionApplicationListMember(C_(2) * C_("y"), C_("z")),
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("y"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("y"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("z"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("w"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("w"),
                str2columnstr_constant("y"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("h"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("h"),
                str2columnstr_constant("w"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("w"),
                str2columnstr_constant("y"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("y"),
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("y"),
                str2columnstr_constant("y"),
            ),
            FunctionApplicationListMember(
                f(g(C_(ColumnStr("x"))), g(C_(ColumnStr("y")))),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_arithmetic_extended_projection():
    x = Symbol("x")
    z = Symbol("z")
    P = Symbol("P")
    conjunction = Conjunction(
        (P(x), EQ(z, Constant(1) / (Constant(1) + (Constant(3) * x))))
    )
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(P, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                Constant(1)
                / (Constant(1) + (Constant(3) * C_(ColumnStr("x")))),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_extended_projection_constant_callable():
    x = Symbol("x")
    z = Symbol("z")
    P = Symbol("P")
    EXP = Constant[Callable[[float], float]](
        numpy.exp, verify_type=False, auto_infer_type=False
    )
    conjunction = Conjunction((P(x), EQ(z, EXP(x))))
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(P, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
            FunctionApplicationListMember(
                EXP(Constant(ColumnStr("x"))),
                str2columnstr_constant("z"),
            ),
        ),
    )
    assert result == expected


def test_sigmoid():
    x = Symbol("x")
    z = Symbol("z")
    P = Symbol("P")
    EXP = Constant[Callable[[float], float]](
        numpy.exp, verify_type=False, auto_infer_type=False
    )
    conjunction = Conjunction(
        (P(x), EQ(z, Constant(1) / (Constant(1) + EXP(-x))))
    )
    result = TranslateToNamedRA().walk(conjunction)
    expected = ExtendedProjection(
        NameColumns(
            Projection(P, (C_(ColumnInt(0)),)),
            (C_(ColumnStr("x")),),
        ),
        (
            FunctionApplicationListMember(
                Constant(1) / (Constant(1) + EXP(-C_(ColumnStr("x")))),
                str2columnstr_constant("z"),
            ),
            FunctionApplicationListMember(
                str2columnstr_constant("x"),
                str2columnstr_constant("x"),
            ),
        ),
    )
    assert result == expected


def test_existential():
    x = Symbol("x")
    y = Symbol("y")
    P = Symbol("P")

    conjunction = ExistentialPredicate(x, P(x, y))
    result = TranslateToNamedRA().walk(conjunction)
    p_translated = TranslateToNamedRA().walk(P(x, y))
    expected = Projection(p_translated, (C_(ColumnStr("y")),))
    assert result == expected

    conjunction = ExistentialPredicate(y, P(x, y))
    result = TranslateToNamedRA().walk(conjunction)
    expected = Projection(p_translated, (C_(ColumnStr("x")),))
    assert result == expected
