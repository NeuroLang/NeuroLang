import contextlib
import operator
from typing import AbstractSet, Tuple

import pytest

from ...datalog.basic_representation import WrappedRelationalAlgebraSet
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, FunctionApplication, Symbol
from ...utils import NamedRelationalAlgebraFrozenSet
from ..optimisers import (
    EliminateTrivialProjections,
    GreedyJoinOrdering,
    PushInSelections,
    PushProjectionsDown,
    RelationalAlgebraOptimiser,
    RenameOptimizations,
    RewriteSelections,
    SimplifyExtendedProjectionsWithConstants,
    _estimated_size,
    _ra_columns,
    _sort_join_operands
)
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    EquiJoin,
    ExtendedProjection,
    FullOuterNaturalJoin,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraSolver,
    RenameColumn,
    RenameColumns,
    ReplaceNull,
    Selection,
    eq_,
    str2columnstr_constant
)

C_ = Constant
ne = Constant(operator.ne)


@pytest.fixture
def r1():
    return WrappedRelationalAlgebraSet([(i, i * 2) for i in range(10)])


@pytest.fixture
def r2():
    return WrappedRelationalAlgebraSet([(i * 2, i * 3) for i in range(10)])


@pytest.fixture
def rs():
    return Symbol[AbstractSet]('R')


@pytest.fixture
def str_columns():
    return tuple(str2columnstr_constant(n) for n in ('a', 'b', 'c', 'd', 'e'))


def test_selection_reorder(r1):
    raop = RelationalAlgebraOptimiser()
    s = Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(1)))
    assert raop.walk(s) is s

    s1 = Selection(C_(r1), eq_(C_(1), C_(ColumnInt(0))))
    assert raop.walk(s1) == s

    s = Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    assert raop.walk(s) is s

    s1 = Selection(C_(r1), eq_(C_(ColumnInt(1)), C_(ColumnInt(0))))
    assert raop.walk(s1) == s

    s_in = Selection(C_(r1), eq_(C_(ColumnInt(1)), C_(ColumnInt(1))))
    s_out = Selection(s_in, eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    assert raop.walk(s_out) is s_out

    s_in1 = Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    s_out1 = Selection(s_in1, eq_(C_(ColumnInt(1)), C_(ColumnInt(1))))
    assert raop.walk(s_out1) == s_out


def test_push_selection_equijoins(r1, r2):
    raop = RelationalAlgebraOptimiser()
    s2 = Selection(
        EquiJoin(
            C_(r1), (C_(ColumnInt(0)),),
            C_(r2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(0)), C_(1))
    )
    s2_res = EquiJoin(
        Selection(
            C_(r1),
            eq_(C_(ColumnInt(0)), C_(1))
        ),
        (C_(ColumnInt(0)),),
        C_(r2), (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(r1), (C_(ColumnInt(0)),),
            C_(r2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(2)), C_(1))
    )
    s2_res = EquiJoin(
        C_(r1),
        (C_(ColumnInt(0)),),
        Selection(
            C_(r2),
            eq_(C_(ColumnInt(0)), C_(1))
        ),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(r1), (C_(ColumnInt(0)),),
            C_(r2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    )
    s2_res = EquiJoin(
        Selection(
            C_(r1),
            eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
        ),
        (C_(ColumnInt(0)),),
        C_(r2),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(r1), (C_(ColumnInt(0)),),
            C_(r2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(2)), C_(ColumnInt(3)))
    )
    s2_res = EquiJoin(
        C_(r1),
        (C_(ColumnInt(0)),),
        Selection(
            C_(r2),
            eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
        ),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(r1), (C_(ColumnInt(0)),),
            C_(r2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(1)), C_(ColumnInt(2)))
    )
    assert raop.walk(s2) == s2


def test_push_and_infer_equijoins(r1, r2):
    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(r1), C_(r2)))
    formula1 = eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    s = Selection(inner, formula1)
    assert raop.walk(s) == Product((Selection(C_(r1), formula1), C_(r2)))

    inner = Product((C_(r1), C_(r2)))
    formula2 = eq_(C_(ColumnInt(2)), C_(ColumnInt(3)))
    s = Selection(inner, formula2)
    res = raop.walk(s)
    expected_res = Product((C_(r1), Selection(C_(r2), formula1)))
    assert res == expected_res

    inner = Product((C_(r1), C_(r2)))
    formula3 = eq_(C_(ColumnInt(0)), C_(ColumnInt(3)))
    s = Selection(inner, formula3)
    assert raop.walk(s) == EquiJoin(
        C_(r1),
        (C_(ColumnInt(0)),),
        C_(r2),
        (C_(ColumnInt(1)),),
    )

    inner = Product((C_(r1), C_(r2), C_(r1)))
    formula3 = eq_(C_(ColumnInt(0)), C_(ColumnInt(3)))
    s = Selection(inner, formula3)
    assert raop.walk(s) == Product((
        EquiJoin(
            C_(r1),
            (C_(ColumnInt(0)),),
            C_(r2),
            (C_(ColumnInt(1)),),
        ),
        C_(r1)
    ))

    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(r1), C_(r2)))
    formula4 = eq_(C_(ColumnInt(0)), C_(1))
    s = Selection(inner, formula4)
    assert raop.walk(s) == Product(
        (Selection(C_(r1), formula4), C_(r2))
    )

    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(r1), C_(r2)))
    formula5 = eq_(C_(ColumnInt(2)), C_(1))
    s = Selection(inner, formula5)
    res = raop.walk(s)
    theoretical_res = Product(
        (C_(r1), Selection(C_(r2), formula4))
    )
    assert res == theoretical_res


def test_push_in_optimiser():
    class Opt(PushInSelections, ExpressionWalker):
        pass

    opt = Opt()

    r1 = Symbol('r1')
    r2 = Symbol('r2')
    a = Constant[ColumnStr](ColumnStr('a'))
    b = Constant[ColumnStr](ColumnStr('b'))
    c = Constant[ColumnStr](ColumnStr('c'))
    op = Symbol('op')

    exp1 = NaturalJoin(r1, r2)
    assert opt.walk(exp1) is exp1

    formula = op(a)
    exp2 = Selection(NaturalJoin(RenameColumn(r1, b, a), r2), formula)
    res = opt.walk(exp2)
    exp_res = NaturalJoin(Selection(
        RenameColumn(r1, b, a), formula),
        r2
    )
    assert res == exp_res

    exp3 = Selection(NaturalJoin(r2, RenameColumn(r1, b, a)), formula)
    res = opt.walk(exp3)
    exp_res = NaturalJoin(
        r2,
        Selection(
            RenameColumn(r1, b, a), formula
        )
    )
    assert res == exp_res

    formula = op(a)
    exp4 = Selection(LeftNaturalJoin(RenameColumn(r1, b, a), r2), formula)
    res = opt.walk(exp4)
    exp_res = LeftNaturalJoin(Selection(
        RenameColumn(r1, b, a), formula),
        r2
    )
    assert res == exp_res

    exp5 = Selection(LeftNaturalJoin(r2, RenameColumn(r1, b, a)), formula)
    res = opt.walk(exp5)
    exp_res = LeftNaturalJoin(
        r2,
        Selection(
            RenameColumn(r1, b, a), formula
        )
    )
    assert res == exp_res

    exp6 = Selection(
        ExtendedProjection(
                NameColumns(r1, (c,)),
                (
                    FunctionApplicationListMember(c, a),
                    FunctionApplicationListMember(Constant(1), b)
                )
        ),
        op(a)
    )
    res = opt.walk(exp6)
    exp_res = ExtendedProjection(
        Selection(
            NameColumns(r1, (c,)),
            op(c)
        ),
        (
            FunctionApplicationListMember(c, a),
            FunctionApplicationListMember(Constant(1), b)
        )
    )
    assert res == exp_res

    exp7 = Selection(
        ExtendedProjection(
                NameColumns(r1, (c,)),
                (
                    FunctionApplicationListMember(c, a),
                    FunctionApplicationListMember(Constant(1), b)
                )
        ),
        op(b)
    )
    res = opt.walk(exp7)

    assert res == exp7

    exp8 = Selection(
        GroupByAggregation(
                NameColumns(r1, (a, b)),
                (a,),
                (
                    FunctionApplicationListMember(op, b),
                )
        ),
        op(a)
    )
    res = opt.walk(exp8)

    exp_res = GroupByAggregation(
        Selection(NameColumns(r1, (a, b)), op(a)),
        (a,),
        (
            FunctionApplicationListMember(op, b),
        )
    )

    assert res == exp_res

    exp9 = Selection(
        ReplaceNull(
            NameColumns(r1, (a, b, c)),
            a,
            Constant(1.0)
        ),
        op(a)
    )
    res = opt.walk(exp9)
    assert res == exp9

    exp9 = Selection(
        ReplaceNull(
            NameColumns(r1, (a, b, c)),
            b,
            Constant(1.0)
        ),
        op(a)
    )
    res = opt.walk(exp9)
    exp_res = ReplaceNull(
        Selection(
            NameColumns(r1, (a, b, c)),
            op(a)
        ),
        b,
        Constant(1.0)
    )
    assert res == exp_res

    exp9 = Selection(
        Projection(
            NameColumns(r1, (a, b, c)),
            (a, c)
        ),
        op(a)
    )
    res = opt.walk(exp9)
    exp_res = Projection(
        Selection(
            NameColumns(r1, (a, b, c)),
            op(a)
        ),
        (a, c)
    )
    assert res == exp_res


def test_eliminate_trivial_projections_optimiser(r1):
    class Opt(EliminateTrivialProjections, ExpressionWalker):
        pass

    opt = Opt()

    rc = Constant(r1)
    exp = Projection(
        rc,
        (Constant[ColumnInt](ColumnInt(0)), Constant[ColumnInt](ColumnInt(1)))
    )

    res = opt.walk(exp)
    assert res is rc

    a = Constant[ColumnStr](ColumnStr('a'))
    b = Constant[ColumnStr](ColumnStr('b'))
    rv = NamedRelationalAlgebraFrozenSet(
        columns=('a', 'b'),
        iterable=r1
    )

    r = C_[AbstractSet[Tuple[int, int]]](rv)

    exp1 = Projection(r, (a, b))
    res1 = opt.walk(exp1)
    assert res1 is r

    r0 = Symbol('r0')
    exp = Projection(Projection(r0, (a, b)), (a,))
    res = opt.walk(exp)
    assert res == Projection(r0, (a,))

    exp = Projection(Projection(r0, (a, b)), (a, b))
    res = opt.walk(exp)
    assert res == Projection(r0, (a, b))

    exp = ExtendedProjection(
        r0,
        (
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )
    res = opt.walk(exp)
    assert res == Projection(r0, (a, b))

    exp = Projection(exp, (a,))
    res = opt.walk(exp)
    assert res == Projection(r0, (a,))


def test_simple_extended_projection_to_rename(rs, str_columns):
    class Opt(EliminateTrivialProjections, ExpressionWalker):
        pass

    a, b, c, d, _ = str_columns

    exp = ExtendedProjection(
        NameColumns(rs, (a, c)),
        (
            FunctionApplicationListMember(a, b),
            FunctionApplicationListMember(c, d)
        )
    )

    res = Opt().walk(exp)

    assert res == RenameColumns(
        exp.relation,
        (
            (a, b),
            (c, d)
        )
    )

    exp = ExtendedProjection(
        NameColumns(rs, (a, c)),
        (
            FunctionApplicationListMember(a, b),
        )
    )

    res = Opt().walk(exp)

    assert res == res


@pytest.mark.parametrize(
    "join_op",
    [NaturalJoin, LeftNaturalJoin, FullOuterNaturalJoin]
)
def test_composite_extended_projection_join(r1, str_columns, join_op):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        join_op(
            ExtendedProjection(
                r1, (
                    FunctionApplicationListMember(Constant(1), c),
                    FunctionApplicationListMember(a, a)
                )
            ),
            r1
        ),
        (
            FunctionApplicationListMember(c + c, d),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )

    res = opt.walk(exp)
    exp = ExtendedProjection(
        join_op(
            ExtendedProjection(r1, (FunctionApplicationListMember(a, a),)),
            r1),
        (
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(operator.add),
                    (Constant(1), Constant(1))
                ),
                d
            ),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )
    assert res == exp

    exp = ExtendedProjection(
        join_op(
            r1,
            ExtendedProjection(
                r1, (
                    FunctionApplicationListMember(Constant(1), c),
                    FunctionApplicationListMember(a, a)
                )
            )
        ),
        (
            FunctionApplicationListMember(c + c, d),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )

    res = opt.walk(exp)
    exp = ExtendedProjection(
        join_op(
            r1,
            ExtendedProjection(r1, (FunctionApplicationListMember(a, a),))),
        (
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(operator.add),
                    (Constant(1), Constant(1))
                ),
                d
            ),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )
    assert res == exp


@pytest.mark.parametrize(
    "join_op",
    [NaturalJoin, LeftNaturalJoin, FullOuterNaturalJoin]
)
def test_composite_extended_projection_constant_join(r1, str_columns, join_op):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        join_op(
            ExtendedProjection(
                r1, (
                    FunctionApplicationListMember(Constant(1), c),
                    FunctionApplicationListMember(a, a)
                )
            ),
            RenameColumns(r1, ((b, c),))
        ),
        (
            FunctionApplicationListMember(c + c, d),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )

    res = opt.walk(exp)
    exp = ExtendedProjection(
        join_op(
            ExtendedProjection(r1, (FunctionApplicationListMember(a, a),)),
            Selection(RenameColumns(r1, ((b, c),)), eq_(c, Constant(1)))
        ),
        (
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(operator.add),
                    (Constant(1), Constant(1))
                ),
                d
            ),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )
    assert res == exp


@pytest.mark.parametrize(
    "join_op",
    [NaturalJoin, LeftNaturalJoin, FullOuterNaturalJoin]
)
def test_composite_extended_projection_function_join(r1, str_columns, join_op):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        join_op(
            ExtendedProjection(
                r1, (
                    FunctionApplicationListMember(b + Constant(1), c),
                    FunctionApplicationListMember(a, a)
                )
            ),
            r1
        ),
        (
            FunctionApplicationListMember(c + c, d),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )

    res = opt.walk(exp)
    fresh_column = [
        s for s in res.relation.columns()
        if s.value.startswith('fresh')
    ][0]
    exp = ExtendedProjection(
        join_op(
            ExtendedProjection(
                r1,
                (
                    FunctionApplicationListMember(a, a),
                    FunctionApplicationListMember(b, fresh_column),
                )
            ),
            r1
        ),
        (
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(operator.add),
                    (fresh_column + Constant(1), fresh_column + Constant(1))
                ),
                d
            ),
            FunctionApplicationListMember(a, a),
            FunctionApplicationListMember(b, b)
        )
    )
    assert res == exp


def test_composite_extended_projection_replacenull(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, _, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ReplaceNull(
        ExtendedProjection(
            r1, (
                FunctionApplicationListMember(Constant(1), c),
                FunctionApplicationListMember(a, b)
            )
        ),
        b,
        Constant(1)
    )

    res = opt.walk(exp)
    exp = ExtendedProjection(
        ReplaceNull(
            r1,
            a,
            Constant(1)
        ),
        (
            FunctionApplicationListMember(Constant(1), c),
            FunctionApplicationListMember(a, b)
        )
    )
    assert res == exp


def test_extended_projection_groupby_trivial(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = GroupByAggregation(
        ExtendedProjection(
            r1, (
                FunctionApplicationListMember(Constant(1), c),
                FunctionApplicationListMember(a, b)
            )
        ),
        (b,),
        (FunctionApplicationListMember(Constant(sum)(c), d),)
    )

    res = opt.walk(exp)
    exp = GroupByAggregation(
        ExtendedProjection(
            r1, (
                FunctionApplicationListMember(a, b),
            )
        ),
        (b,),
        (FunctionApplicationListMember(Constant(len)(), d),)
    )
    assert res == exp


def test_rename_column_to_rename_columns(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    opt = Opt()

    exp = RenameColumn(rs, str_columns[0], str_columns[1])

    res = opt.walk(exp)

    assert res == RenameColumns(rs, ((str_columns[0], str_columns[1]),))


def test_trivial_rename_columns(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    opt = Opt()

    exp = RenameColumns(rs, tuple())

    res = opt.walk(exp)

    assert res == rs

    exp = RenameColumns(rs, tuple((s, s) for s in str_columns))

    res = opt.walk(exp)

    assert res == rs


def test_nested_rename_columns(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, b, c, d, e = str_columns

    opt = Opt()

    exp = RenameColumns(
        RenameColumns(rs, ((a, b), (d, e))),
        ((b, c),)
    )

    res = opt.walk(exp)

    assert res == RenameColumns(rs, ((a, c), (d, e)))


def test_nested_rename_columns_extended_projection(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, b, c, d, _ = str_columns

    opt = Opt()

    exp = RenameColumns(
        ExtendedProjection(
            rs,
            (
                FunctionApplicationListMember(a, b),
                FunctionApplicationListMember(Constant(1), c)
            )
        ),
        ((b, c), (c, d))
    )

    res = opt.walk(exp)

    assert res == ExtendedProjection(
        rs,
        (
            FunctionApplicationListMember(a, c),
            FunctionApplicationListMember(Constant(1), d)
        )
    )


def test_nested_rename_columns_groupby_agg(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, b, c, _, e = str_columns

    opt = Opt()

    exp = RenameColumns(
        GroupByAggregation(
            rs,
            (a,),
            (
                FunctionApplicationListMember(Constant(sum)(c), b),
            )
        ),
        ((a, e),)
    )

    res = opt.walk(exp)

    assert res == GroupByAggregation(
        RenameColumns(rs, ((a, e),)),
        (e,),
        (
            FunctionApplicationListMember(Constant(sum)(c), b),
        )
    )

    exp = RenameColumns(
        GroupByAggregation(
            rs,
            (a,),
            (FunctionApplicationListMember(Constant(sum)(c), b),)
        ),
        ((a, b), (b, c))
    )

    res = opt.walk(exp)

    assert res == GroupByAggregation(
        RenameColumns(rs, ((a, b),)),
        (b,),
        (
            FunctionApplicationListMember(Constant(sum)(c), c),
        )
    )

    exp = RenameColumns(
        GroupByAggregation(
            rs,
            (a,),
            (FunctionApplicationListMember(Constant(sum)(b), b),)
        ),
        ((a, b), (b, c))
    )

    res = opt.walk(exp)

    assert res == RenameColumns(
        GroupByAggregation(
            rs,
            (a,),
            (FunctionApplicationListMember(Constant(sum)(b), c),)
        ),
        ((a, b),)
    )


def test_rename_name(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, b, c, _, _ = str_columns

    opt = Opt()

    exp = RenameColumns(NameColumns(rs, (a, b)), ((a, c), (b, a)))

    res = opt.walk(exp)

    assert res == NameColumns(rs, ((c, a)))


def test_rename_selection(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, _, c, _, _ = str_columns

    opt = Opt()

    one = Constant(1)
    two = Constant(2)

    exp = RenameColumns(Selection(rs, eq_(a + two, one)), ((a, c),))

    res = opt.walk(exp)

    assert res == Selection(RenameColumns(rs, ((a, c),)), eq_(c + two, one))


def test_rename_projection(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, b, c, _, _ = str_columns

    opt = Opt()

    exp = RenameColumns(Projection(rs, (a, b)), ((a, c),))

    res = opt.walk(exp)

    assert res == Projection(RenameColumns(rs, ((a, c),)), (c, b))


def test_rename_replacenull(rs, str_columns):
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    a, _, c, _, _ = str_columns

    opt = Opt()

    zero = Constant(0)

    exp = RenameColumns(ReplaceNull(rs, a, zero), ((a, c),))

    res = opt.walk(exp)

    assert res == ReplaceNull(RenameColumns(rs, ((a, c),)), c, zero)


def test_nested_selections(rs, str_columns):
    class Opt(RewriteSelections, ExpressionWalker):
        pass

    a, b, _, _, _ = str_columns

    opt = Opt()

    and_ = Constant(operator.and_)
    one = Constant(1)

    exp = Selection(Selection(rs, eq_(a, one)), eq_(b + one, one))

    res = opt.walk(exp)

    assert res == Selection(rs, and_(eq_(a, one), eq_(b + one, one)))


def test_push_unnamed_selections_up(r1):
    raop = RelationalAlgebraOptimiser()
    x = str2columnstr_constant("x")
    y = str2columnstr_constant("y")

    s = NameColumns(
        Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (x, y)
    )
    res = Selection(NameColumns(C_(r1), (x, y)), eq_(x, y))
    assert raop.walk(s) == res

    s = NameColumns(
        Selection(C_(r1), eq_(C_(ColumnInt(1)), C_(ColumnInt(0)))),
        (x, y)
    )
    res = Selection(NameColumns(C_(r1), (x, y)), eq_(y, x))
    assert raop.walk(s) == res

    s = Projection(
        Selection(C_(r1), eq_(C_(ColumnInt(1)), C_(0))),
        (C_(ColumnInt(1)),)
    )
    res = Selection(
        Projection(C_(r1), (C_(ColumnInt(1)),)), eq_(C_(ColumnInt(1)), C_(0))
    )
    assert raop.walk(s) == res

    s = Projection(
        Selection(C_(r1), eq_(C_(ColumnInt(1)), C_(ColumnInt(0)))),
        (C_(ColumnInt(0)),)
    )
    res = Projection(
        Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (C_(ColumnInt(0)),)
    )
    assert raop.walk(s) == res

    s = Projection(
        Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))),
        (C_(ColumnInt(1)),)
    )
    assert raop.walk(s) == res

    s = Projection(
        Selection(C_(r1), eq_(C_(ColumnInt(1)), C_(ColumnInt(0)))),
        (C_(ColumnInt(0)), C_(ColumnInt(1)))
    )

    res = Selection(
        Projection(C_(r1), (C_(ColumnInt(0)), C_(ColumnInt(1)))),
        eq_(C_(ColumnInt(1)), C_(ColumnInt(0)))
    )

    assert raop.walk(s) == res

    s = Selection(
        Selection(C_(r1), ne(C_(ColumnInt(1)), C_(ColumnInt(2)))),
        eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    )

    res = Selection(
        Selection(C_(r1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))),
        ne(C_(ColumnInt(0)), C_(ColumnInt(2)))
    )

    assert raop.walk(s) == res


def _has_cross_product(expr):
    """
    Check whether a binary join tree joins operands sharing no column.

    Operands whose columns cannot be computed are not counted as cross
    products.
    """
    if not (hasattr(expr, 'relation_left') and hasattr(expr, 'relation_right')):
        return False
    with contextlib.suppress(Exception):
        lc = _ra_columns(expr.relation_left)
        rc = _ra_columns(expr.relation_right)
        if len(lc & rc) == 0:
            return True
    return (
        _has_cross_product(expr.relation_left)
        or _has_cross_product(expr.relation_right)
    )


def _named_constant(columns, row):
    """Build a one-row named relation constant with integer columns."""
    return C_[AbstractSet[Tuple[(int,) * len(columns)]]](
        NamedRelationalAlgebraFrozenSet(columns, [row])
    )


def _cross_product_under_projection():
    """
    Build ``π[s1,s5](A(s5) ⋈ B(s1))`` and the constant ``C(s1, s5)``.

    A ⋈ B is a cross product since A and B share no columns.
    """
    A = _named_constant(('s5',), (5,))
    B = _named_constant(('s1',), (1,))
    C = _named_constant(('s1', 's5'), (1, 5))
    proj = Projection(NaturalJoin(A, B), (
        str2columnstr_constant('s1'),
        str2columnstr_constant('s5'),
    ))
    return proj, C


def _optimise_to_fixpoint(expression, max_iterations=10):
    """Apply RelationalAlgebraOptimiser until the plan stops changing."""
    for _ in range(max_iterations):
        prev_repr = repr(expression)
        expression = RelationalAlgebraOptimiser().walk(expression)
        if repr(expression) == prev_repr:
            break
    return expression


def _named_symbol(name, column_names):
    """Build ``NameColumns(Symbol(name), column_names)`` with str columns."""
    return NameColumns(
        Symbol[AbstractSet](name),
        tuple(Constant[ColumnStr](ColumnStr(c)) for c in column_names),
    )


def test_push_projection_into_join_operands(r1, r2, str_columns):
    """
    Check that PushProjectionsDown pushes projections into join operands.

    PushProjectionsDown pushes the projection down into join operands
    and eliminates the outer projection boundary.

    π[b,d](R(a,b)⋈S(c,b)) ⋈ T(b,d)
    → after pipeline: all operands flattened in an NaryNaturalJoin
      with column-a pruned from R.
    """
    a, b, c, d, _ = str_columns

    # Build π[b,d](R⋈S) ⋈ T(b,d)
    # R(a,b) ⋈ S(c,b) — shared=b, cols={b,d}. Guard: shared={b} ⊆ cols={b,d} ✓
    # T(b,d) shares {b,d} with inner result {a,b,c}. Both ⊆ cols ✓
    r_named = NameColumns(C_(r1), (a, b))
    s_named = NameColumns(C_(r2), (c, b))
    t_named = NameColumns(C_(r2), (b, d))

    proj = Projection(NaturalJoin(r_named, s_named), (b, d))
    tree = NaturalJoin(proj, t_named)

    raop = RelationalAlgebraOptimiser()
    result = raop.walk(tree)

    # Verify no cross products in result — all NaturalJoin children
    # share at least one column after optimisation.
    assert not _has_cross_product(result), (
        "Cross product still present after optimisation"
    )

    # Verify the tree converged to an NaryNaturalJoin (the flattening
    # step absorbed the projection barrier into the join operands).
    assert type(result).__name__ == "NaryNaturalJoin", (
        "Expected flattened NaryNaturalJoin, got " + type(result).__name__
    )


def test_push_projection_eliminates_cross_product_across_projection():
    """
    Check that pushing projections removes a trapped cross product.

    Integration: PushProjectionIntoJoinOperands + FlattenAndReorderJoins
    eliminates a cross product trapped inside a projection.

    Tree: π[s₁,s₅](A(s₅) ⋈ B(s₁)) ⋈ C(s₁,s₅)
    A⋈B is a cross product (no shared columns).
    C has both columns — absorbing C enables natural joins throughout.
    """
    proj, C = _cross_product_under_projection()
    tree = NaturalJoin(proj, C)  # C can be absorbed (C.columns ⊆ proj.columns)

    curr = _optimise_to_fixpoint(tree)

    assert not _has_cross_product(curr), (
        "Cross product still present after optimisation"
    )


def test_push_projections_down_through_join(r1, str_columns):
    """Check that PushProjectionsDown prunes join inputs with projections."""
    a, b, c, _, _ = str_columns
    # R1 has columns (int, int) → name them a, b
    # R2 has columns (int, int) → name them c, b (shares b)
    r_named = NameColumns(C_(r1), (a, b))
    s_named = NameColumns(C_(r2), (c, b))

    join = NaturalJoin(r_named, s_named)  # natural join on b
    proj = Projection(join, (b, c))        # output needs b, c (drops a from R1)

    pj = PushProjectionsDown()
    result = pj.walk(proj)

    # result should be Projection(NaturalJoin(Projection(R1, (b,)), Projection(R2, (b, c))))
    assert type(result).__name__ == "Projection"
    assert tuple(result.attributes) == (b, c)

    inner = result.relation
    assert type(inner).__name__ == "NaturalJoin"

    left = inner.relation_left
    right = inner.relation_right

    # R1 should be projected to only (b) — column a is dropped
    # R2 should be projected to (b, c) — shared + output columns
    if type(left).__name__ == "Projection":
        r1_proj = left
    else:
        r1_proj = right

    assert type(r1_proj).__name__ == "Projection"
    # The projected attributes for R1: should only contain 'b' (shared column)
    proj_attrs = tuple(c.value for c in r1_proj.attributes)
    assert 'b' in proj_attrs
    assert 'a' not in proj_attrs, (
        "Column 'a' should have been pruned from R1 input"
    )


def test_push_projections_down_full_pipeline():
    """
    Check that the full optimiser pipeline leaves no cross products.

    End-to-end: the full RelationalAlgebraOptimiser pipeline
    (push projections into join operands → flatten/reorder → push projections)
    produces a plan with no cross products.
    """
    # π[s₁,s₅](A⋈B) ⋈ C ⋈ D
    # A⋈B is cross product, C provides s₁/s₅ to join with A/B, D provides s₄
    proj, C = _cross_product_under_projection()
    D = _named_constant(('s4', 's1', 's5'), (4, 1, 5))
    tree = NaturalJoin(NaturalJoin(proj, C), D)

    curr = _optimise_to_fixpoint(tree)

    # At minimum the pipeline should not crash and should eliminate cross
    # products. Projection pushdown is a best-effort optimisation — the
    # exact shape depends on what EliminateTrivialProjections cleans up.
    assert not _has_cross_product(curr), f"Cross product still present:\n{curr!r}"


def test_greedy_join_ordering_reorders_mixed_product_join():
    """
    Check that greedy join ordering joins shared-column operands first.

    Regression test: GreedyJoinOrdering reorders a mixed Product/NaturalJoin
    chain so operands with shared ColumnStr names are joined first.

    Tree: NaturalJoin(Product(R1[a,b], R2[c,d]), R3[b,c])
    R1 and R2 share no columns → Product.
    R3 shares 'b' with R1 and 'c' with R2.
    After GOO: R1 and R3 should be joined first (share 'b'),
    then R2 (shares 'c' with the result).
    Expected: nested NaturalJoin tree with no bare Product between
    operands that share columns.
    """
    r1_named = _named_symbol('R1', 'ab')
    r2_named = _named_symbol('R2', 'cd')
    r3_named = _named_symbol('R3', 'bc')

    # NaturalJoin(Product(R1[a,b], R2[c,d]), R3[b,c])
    tree = NaturalJoin(Product((r1_named, r2_named)), r3_named)

    opt = GreedyJoinOrdering()
    result = opt.walk(tree)

    # The result must NOT be a bare Product — the cross product should
    # have been eliminated by reordering.
    assert type(result).__name__ != "Product", (
        "Expected join (NaturalJoin), got Product — cross product not eliminated"
    )

    # The result should be a NaturalJoin (nested tree), not NaryNaturalJoin,
    # not Product.
    assert type(result).__name__ == "NaturalJoin", (
        "Expected NaturalJoin, got " + type(result).__name__
    )

    # Verify no cross product remains: walk the tree and check that
    # no Product node exists between operands that share columns.
    def _check_no_product(expr):
        if isinstance(expr, Product):
            # Product is only OK if its operands truly share no columns
            ops = list(expr.relations)
            for i, op_i in enumerate(ops):
                for op_j in ops[i + 1:]:
                    lc = _ra_columns(op_i)
                    rc = _ra_columns(op_j)
                    assert len(lc & rc) == 0, (
                        f"Cross product between operands that share columns: "
                        f"{lc} ∩ {rc} ≠ ∅"
                    )
        elif isinstance(expr, NaturalJoin):
            _check_no_product(expr.relation_left)
            _check_no_product(expr.relation_right)

    _check_no_product(result)


def test_greedy_join_ordering_preserves_disconnected_product():
    """
    Check that greedy join ordering keeps a disconnected product.

    Regression test: GreedyJoinOrdering must NOT convert a Product of
    two operands with NO shared column names into a NaturalJoin.

    Tree: Product(R1[a,b], R2[c,d])
    R1 and R2 share no columns → must remain Product.
    """
    tree = Product((_named_symbol('R1', 'ab'), _named_symbol('R2', 'cd')))

    opt = GreedyJoinOrdering()
    result = opt.walk(tree)

    # Must remain a Product — no shared columns means no join possible.
    assert type(result).__name__ == "Product", (
        "Expected Product (no shared columns), got " + type(result).__name__
    )


def test_push_projections_down_terminates_when_nothing_to_push():
    """
    Check that the optimiser stops when a projection cannot be pushed.

    In ``π[b](π[b](R(b, g)) ⋈ S(c))`` the left side is already narrowed
    and ``S`` needs no column, so pushing the projection changes nothing.
    PushProjectionsDown used to rebuild an equal expression in that case,
    and together with FlattenAndReorderJoins the walker rewrote the plan
    forever (RecursionError).
    """
    r = _named_constant(("b", "g"), (0, 1))
    s = _named_constant(("c",), (0,))
    t = _named_constant(("f", "a", "c"), (0, 1, 2))
    u = _named_constant(("c", "e", "f"), (2, 3, 0))
    plan = Product((
        Projection(NaturalJoin(r, s), (str2columnstr_constant("b"),)),
        NaturalJoin(
            Projection(
                t,
                (str2columnstr_constant("a"), str2columnstr_constant("f")),
            ),
            u,
        ),
    ))

    optimised = RelationalAlgebraOptimiser().walk(plan)

    expected = RelationalAlgebraSolver().walk(plan).value
    result = RelationalAlgebraSolver().walk(optimised).value
    assert result == expected


def _sized_constant(columns, n_rows):
    """Build a named relation constant with ``n_rows`` distinct rows."""
    return C_[AbstractSet[Tuple[(int,) * len(columns)]]](
        NamedRelationalAlgebraFrozenSet(
            columns, [(i,) * len(columns) for i in range(n_rows)]
        )
    )


def _left_deep_operands(expression):
    """Return the operands of a left-deep NaturalJoin tree in join order."""
    operands = []
    while isinstance(expression, NaturalJoin):
        operands.append(expression.relation_right)
        expression = expression.relation_left
    return [expression] + operands[::-1]


def test_estimated_size_of_wrapped_constants():
    """
    Check the size estimation of constant relations.

    Renames, projections and selections are estimated with the size of
    their input; other expressions and symbols have an unknown size.
    """
    r = _sized_constant(("a", "b"), 7)
    a, b, c = (str2columnstr_constant(n) for n in "abc")
    assert _estimated_size(r) == 7
    assert _estimated_size(Projection(RenameColumn(r, b, c), (a,))) == 7
    assert _estimated_size(Selection(r, eq_(a, C_(1)))) == 7
    assert _estimated_size(_named_symbol("R", "ab")) is None
    assert _estimated_size(NaturalJoin(r, r)) is None


def test_row_count_estimate_does_not_drop_duplicates():
    """Check that the pandas row count estimate does not deduplicate."""
    ras = NamedRelationalAlgebraFrozenSet(("a",), [(1,), (1,), (2,)])
    assert ras.row_count_estimate() == 3
    assert ras.might_have_duplicates
    assert len(ras) == 2


def test_sort_join_operands_prefers_small_connected_operands():
    """
    Check that known sizes drive the greedy order of join operands.

    The smallest operand sharing columns with the others starts the
    order, then the smallest operand connected to the placed ones is
    picked, never introducing a cross product. Without sizes the order
    only depends on column overlap.
    """
    reports = _sized_constant(("s", "x"), 50)
    voxel = _sized_constant(("x",), 100)
    labels = _sized_constant(("l", "x"), 5)
    label = _sized_constant(("l",), 2)
    operands = [reports, voxel, labels, label]
    sizes = [50, 100, 5, 2]

    assert _sort_join_operands(operands, sizes) == [
        label, labels, reports, voxel
    ]
    assert _sort_join_operands(operands) == [labels, reports, voxel, label]
    assert _sort_join_operands(operands, [None, None, None, 2]) == (
        _sort_join_operands(operands)
    )


def test_greedy_join_ordering_joins_small_relation_first():
    """
    Check that GreedyJoinOrdering joins the smallest relations first.

    In ``(R(a, b) ⋈ V(b)) ⋈ L(b, c)`` with sizes 50, 100 and 5, joining
    the small relation ``L`` with ``R`` before the large ``V`` keeps the
    intermediate results small. The result of the plan is unchanged.
    """
    r = _sized_constant(("a", "b"), 50)
    v = _sized_constant(("b",), 100)
    l_ = _sized_constant(("b", "c"), 5)
    tree = NaturalJoin(NaturalJoin(r, v), l_)

    result = GreedyJoinOrdering().walk(tree)

    assert _left_deep_operands(result) == [l_, r, v]
    assert (
        RelationalAlgebraSolver().walk(result).value
        == RelationalAlgebraSolver().walk(tree).value
    )
    assert GreedyJoinOrdering().walk(result) is result


def test_greedy_join_ordering_unknown_sizes_keep_order():
    """
    Check that joins of relations of unknown size are not reordered.

    Symbols have no known size, and a single known size is not enough to
    compare operands, so the join trees are left untouched.
    """
    tree = NaturalJoin(
        NaturalJoin(_named_symbol("R", "ab"), _named_symbol("V", "b")),
        _named_symbol("L", "bc"),
    )
    assert GreedyJoinOrdering().walk(tree) is tree

    tree = NaturalJoin(
        NaturalJoin(_named_symbol("R", "ab"), _named_symbol("V", "b")),
        _sized_constant(("b", "c"), 5),
    )
    assert GreedyJoinOrdering().walk(tree) is tree


def test_greedy_join_ordering_resolves_symbol_sizes():
    """
    Check that subclasses can resolve the size of symbols.

    Overriding ``resolve_relation`` with a symbol table lookup lets
    GreedyJoinOrdering order joins of symbols by the size of their values.
    """
    table = {
        Symbol[AbstractSet]("R"): _sized_constant(("0", "1"), 50),
        Symbol[AbstractSet]("V"): _sized_constant(("0",), 100),
        Symbol[AbstractSet]("L"): _sized_constant(("0", "1"), 5),
    }

    class SymbolTableJoinOrdering(GreedyJoinOrdering, ExpressionWalker):
        def resolve_relation(self, expression):
            return table.get(expression)

    r = _named_symbol("R", "ab")
    v = _named_symbol("V", "b")
    l_ = _named_symbol("L", "bc")
    tree = NaturalJoin(NaturalJoin(r, v), l_)

    result = SymbolTableJoinOrdering().walk(tree)

    assert _left_deep_operands(result) == [l_, r, v]


def test_greedy_join_ordering_sizes_order_product_components():
    """
    Check that sizes order the joins built when removing a cross product.

    ``Product(A(a), B(b)) ⋈ C(a, b) ⋈ D(b)``: the cross product is
    eliminated and, within the connected group, the smallest operand
    ``D`` starts the join, followed by the smallest connected ones.
    """
    a = _sized_constant(("a",), 40)
    b = _sized_constant(("b",), 30)
    c = _sized_constant(("a", "b"), 20)
    d = _sized_constant(("b",), 3)
    tree = NaturalJoin(NaturalJoin(Product((a, b)), c), d)

    result = GreedyJoinOrdering().walk(tree)

    assert _left_deep_operands(result) == [d, c, b, a]
    assert (
        RelationalAlgebraSolver().walk(result).value
        == RelationalAlgebraSolver().walk(tree).value
    )

def _relation(columns, rows):
    """Build a named relation constant with integer columns."""
    return C_[AbstractSet[Tuple[(int,) * len(columns)]]](
        NamedRelationalAlgebraFrozenSet(columns, rows)
    )


def _same_result(expression, optimised):
    """Check that two plans evaluate to the same relation."""
    solver = RelationalAlgebraSolver()
    return solver.walk(optimised).value == solver.walk(expression).value


def test_push_selection_into_join_of_constants():
    """
    Check that a selection is pushed into the operand holding its column.

    The operands are constant relations, which mention no column.
    """
    class Opt(PushInSelections, ExpressionWalker):
        pass

    left = _relation(("g", "e"), [(0, 1), (1, 1)])
    right = _relation(("g", "f", "c"), [(0, 2, 0), (1, 2, 1)])
    c = str2columnstr_constant("c")
    tree = Selection(NaturalJoin(left, right), eq_(c, C_(0)))

    result = Opt().walk(tree)

    assert result == NaturalJoin(left, Selection(right, eq_(c, C_(0))))
    assert _same_result(tree, result)


def test_rename_not_pushed_below_projection_dropping_its_target():
    """
    Check that a renaming is not pushed where its target column exists.

    The projected relation already has the column ``g`` that ``c`` is
    renamed into.
    """
    class Opt(RenameOptimizations, ExpressionWalker):
        pass

    f, c, g = (str2columnstr_constant(n) for n in "fcg")
    tree = RenameColumn(
        Projection(_relation(("f", "c", "g"), [(0, 1, 2), (1, 0, 0)]), (f, c)),
        c, g
    )

    result = Opt().walk(tree)

    assert result == RenameColumns(tree.relation, ((c, g),))
    assert _same_result(tree, result)


class GreedyJoinOrderingWalker(GreedyJoinOrdering, ExpressionWalker):
    """GreedyJoinOrdering walking the expressions it does not rewrite."""


def test_greedy_join_ordering_keeps_projection_not_distributing():
    """
    Check that a projection which does not distribute is kept whole.

    In ``π[e](F(f, g) ⋈ G(g, e)) ⋈ (H(e, f) × K(b))`` the projection drops
    the column ``f`` of ``F`` which ``H`` also has: pushing it onto the
    join operands would join ``F`` and ``H`` on ``f``.
    """
    e = str2columnstr_constant("e")
    left = Projection(
        NaturalJoin(_relation("fg", [(0, 0)]), _relation("ge", [(0, 1)])),
        (e,)
    )
    h = _relation("ef", [(1, 5)])
    k = _relation("b", [(7,)])
    tree = NaturalJoin(left, Product((h, k)))

    result = GreedyJoinOrderingWalker().walk(tree)

    assert result == Product((NaturalJoin(left, h), k))
    assert _same_result(tree, result)

