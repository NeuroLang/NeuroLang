import operator
from typing import AbstractSet, Tuple

import pytest

from ...datalog.basic_representation import WrappedRelationalAlgebraSet
from ...expression_walker import ExpressionWalker
from ...expressions import Constant, FunctionApplication, Symbol
from ...utils import NamedRelationalAlgebraFrozenSet
from ..optimisers import (
    EliminateTrivialProjections,
    PushInSelections,
    RelationalAlgebraOptimiser,
    RenameOptimizations,
    RewriteSelections,
    SimplifyExtendedProjectionsWithConstants
)
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    EquiJoin,
    ExtendedProjection,
    FunctionApplicationListMember,
    GroupByAggregation,
    LeftNaturalJoin,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
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


def test_composite_extended_projection_join(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        NaturalJoin(
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
        NaturalJoin(
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
        NaturalJoin(
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
        NaturalJoin(
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


def test_composite_extended_projection_leftjoin(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        LeftNaturalJoin(
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
        LeftNaturalJoin(
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
        LeftNaturalJoin(
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
        LeftNaturalJoin(
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


def test_composite_extended_projection_constant_join(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        LeftNaturalJoin(
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
        LeftNaturalJoin(
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


def test_composite_extended_projection_function_join(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        LeftNaturalJoin(
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
        LeftNaturalJoin(
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


def test_composite_extended_projection_function_join_flip(r1, str_columns):
    class Opt(SimplifyExtendedProjectionsWithConstants, ExpressionWalker):
        pass

    opt = Opt()

    a, b, c, d, _ = str_columns
    r1 = NameColumns(Constant(r1), (a, b))

    exp = ExtendedProjection(
        LeftNaturalJoin(
            r1,
            ExtendedProjection(
                r1, (
                    FunctionApplicationListMember(b + Constant(1), c),
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
    fresh_column = [
        s for s in res.relation.columns()
        if s.value.startswith('fresh')
    ][0]
    exp = ExtendedProjection(
        LeftNaturalJoin(
            r1,
            ExtendedProjection(
                r1,
                (
                    FunctionApplicationListMember(a, a),
                    FunctionApplicationListMember(b, fresh_column),
                )
            ),
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
