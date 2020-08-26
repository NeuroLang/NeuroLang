import operator
from typing import AbstractSet, Tuple

import pytest

from ..datalog.basic_representation import WrappedRelationalAlgebraSet
from ..exceptions import NeuroLangException
from ..expressions import Constant, Symbol
from ..expression_walker import ExpressionWalker
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    ConcatenateConstantColumn,
    Destroy,
    EquiJoin,
    ExtendedProjection,
    ExtendedProjectionListMember,
    Intersection,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraOptimiser,
    RelationalAlgebraSolver,
    EliminateTrivialProjections,
    RelationalAlgebraPushInSelections,
    RenameColumn,
    RenameColumns,
    Selection,
    str2columnstr_constant,
    Union,
    eq_,
    _const_relation_type_is_known,
    _sort_typed_const_named_relation_tuple_type_args,
    _infer_relation_type,
    _get_const_relation_type,
)
from ..utils import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
    RelationalAlgebraSet,
)
from ..utils.relational_algebra_set import RelationalAlgebraStringExpression


R1 = WrappedRelationalAlgebraSet([
    (i, i * 2)
    for i in range(10)
])

R2 = WrappedRelationalAlgebraSet([
    (i * 2, i * 3)
    for i in range(10)
])


C_ = Constant


def test_selection():
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(0)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection({0: 0})


def test_selection_columns():
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection_columns({0: 1})


def test_selection_general():
    gt_ = C_(operator.gt)
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    c = C_[AbstractSet[Tuple[int, int]]](r1_named)
    s = Selection(c, gt_(C_(ColumnStr('x')), C_(5)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == r1_named.selection(lambda t: t.x > 5)


def test_projections():
    s = Projection(C_(R1), (C_(ColumnInt(0)),))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.projection(0)


def test_equijoin():
    s = EquiJoin(
        C_(R1), (C_(ColumnInt(0)),),
        C_(R2), (C_(ColumnInt(0)),)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.equijoin(R2, [(0, 0)])


def test_naturaljoin():
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    r2_named = NamedRelationalAlgebraFrozenSet(('x', 'z'), R2)
    s = NaturalJoin(
        C_[AbstractSet[Tuple[int, int]]](r1_named),
        C_[AbstractSet[Tuple[int, int]]](r2_named)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == r1_named.naturaljoin(r2_named)


def test_union_unnamed():
    r1 = C_[AbstractSet](WrappedRelationalAlgebraSet([(1, 2), (7, 8)]))
    r2 = C_[AbstractSet](WrappedRelationalAlgebraSet([(5, 0), (7, 8)]))
    res = RelationalAlgebraSolver().walk(Union(r1, r2))
    assert res == C_[AbstractSet](
        WrappedRelationalAlgebraSet([(1, 2), (7, 8), (5, 0)])
    )
    assert (
        RelationalAlgebraSolver().walk(
            Union(r1, C_[AbstractSet](WrappedRelationalAlgebraSet()))
        )
        == r1
    )


def test_union_named():
    r1 = C_[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("x", "y"), [
            (1, "a"),
            (2, "b"),
            (3, "a"),
            (3, "b"),
        ])
    )
    r2 = C_[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("x", "y"), [
            (3, "b"),
            (3, "a"),
            (3, "c"),
        ])
    )
    empty = C_[AbstractSet](NamedRelationalAlgebraFrozenSet(('x', 'y'), []))
    res = RelationalAlgebraSolver().walk(Union(r1, r2))
    expected = C_[AbstractSet[Tuple[int, str]]](
        NamedRelationalAlgebraFrozenSet(('x', 'y'), [
            (1, "a"), (2, "b"), (3, "a"), (3, "b"), (3, "c")
        ])
    )
    assert res == expected
    assert RelationalAlgebraSolver().walk(Union(r1, empty)) == r1
    assert RelationalAlgebraSolver().walk(Union(empty, r1)) == r1


def test_intersection_unnamed():
    r1 = C_[AbstractSet](WrappedRelationalAlgebraSet([(1, 2), (7, 8)]))
    r2 = C_[AbstractSet](WrappedRelationalAlgebraSet([(5, 0), (7, 8)]))
    res = RelationalAlgebraSolver().walk(Union(r1, r2))
    assert res == C_[AbstractSet](
        WrappedRelationalAlgebraSet([(1, 2), (7, 8), (5, 0)])
    )
    assert (
        RelationalAlgebraSolver().walk(
            Union(r1, C_[AbstractSet](WrappedRelationalAlgebraSet()))
        )
        == r1
    )


def test_intersection_named():
    r1 = C_[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("x", "y"), [
            (1, "a"),
            (2, "b"),
            (3, "a"),
            (3, "b"),
        ])
    )
    r2 = C_[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("x", "y"), [
            (3, "b"),
            (3, "a"),
            (3, "c"),
        ])
    )
    empty  = C_[AbstractSet](NamedRelationalAlgebraFrozenSet(('x', 'y'), []))
    res = RelationalAlgebraSolver().walk(Intersection(r1, r2))
    expected = C_[AbstractSet[Tuple[int, str]]](
        NamedRelationalAlgebraFrozenSet(('x', 'y'), [(3, "a"), (3, "b")])
    )
    assert res == expected
    assert RelationalAlgebraSolver().walk(Intersection(r1, empty)) == empty
    assert RelationalAlgebraSolver().walk(Intersection(empty, r1)) == empty


def test_product():
    s = Product((C_(R1), C_(R2)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2)

    s = Product((C_(R1), C_(R2), C_(R1)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2).cross_product(R1)

    s = Product(tuple())
    sol = RelationalAlgebraSolver().walk(s).value
    assert len(sol) == 0


def test_selection_reorder():
    raop = RelationalAlgebraOptimiser()
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(1)))
    assert raop.walk(s) is s

    s1 = Selection(C_(R1), eq_(C_(1), C_(ColumnInt(0))))
    assert raop.walk(s1) == s

    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    assert raop.walk(s) is s

    s1 = Selection(C_(R1), eq_(C_(ColumnInt(1)), C_(ColumnInt(0))))
    assert raop.walk(s1) == s

    s_in = Selection(C_(R1), eq_(C_(ColumnInt(1)), C_(ColumnInt(1))))
    s_out = Selection(s_in, eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    assert raop.walk(s_out) is s_out

    s_in1 = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    s_out1 = Selection(s_in1, eq_(C_(ColumnInt(1)), C_(ColumnInt(1))))
    assert raop.walk(s_out1) == s_out


def test_push_selection_equijoins():
    raop = RelationalAlgebraOptimiser()
    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(0)), C_(1))
    )
    s2_res = EquiJoin(
        Selection(
            C_(R1),
            eq_(C_(ColumnInt(0)), C_(1))
        ),
        (C_(ColumnInt(0)),),
        C_(R2), (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(2)), C_(1))
    )
    s2_res = EquiJoin(
        C_(R1),
        (C_(ColumnInt(0)),),
        Selection(
            C_(R2),
            eq_(C_(ColumnInt(0)), C_(1))
        ),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    )
    s2_res = EquiJoin(
        Selection(
            C_(R1),
            eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
        ),
        (C_(ColumnInt(0)),),
        C_(R2),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(2)), C_(ColumnInt(3)))
    )
    s2_res = EquiJoin(
        C_(R1),
        (C_(ColumnInt(0)),),
        Selection(
            C_(R2),
            eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
        ),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(1)), C_(ColumnInt(2)))
    )
    assert raop.walk(s2) == s2


def test_push_and_infer_equijoins():
    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(R1), C_(R2)))
    formula1 = eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    s = Selection(inner, formula1)
    assert raop.walk(s) == Product((Selection(C_(R1), formula1), C_(R2)))

    inner = Product((C_(R1), C_(R2)))
    formula2 = eq_(C_(ColumnInt(2)), C_(ColumnInt(3)))
    s = Selection(inner, formula2)
    res = raop.walk(s)
    expected_res = Product((C_(R1), Selection(C_(R2), formula1)))
    assert res == expected_res

    inner = Product((C_(R1), C_(R2)))
    formula3 = eq_(C_(ColumnInt(0)), C_(ColumnInt(3)))
    s = Selection(inner, formula3)
    assert raop.walk(s) == EquiJoin(
        C_(R1),
        (C_(ColumnInt(0)),),
        C_(R2),
        (C_(ColumnInt(1)),),
    )

    inner = Product((C_(R1), C_(R2), C_(R1)))
    formula3 = eq_(C_(ColumnInt(0)), C_(ColumnInt(3)))
    s = Selection(inner, formula3)
    assert raop.walk(s) == Product((
        EquiJoin(
            C_(R1),
            (C_(ColumnInt(0)),),
            C_(R2),
            (C_(ColumnInt(1)),),
        ),
        C_(R1)
    ))

    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(R1), C_(R2)))
    formula4 = eq_(C_(ColumnInt(0)), C_(1))
    s = Selection(inner, formula4)
    assert raop.walk(s) == Product(
        (Selection(C_(R1), formula4), C_(R2))
    )

    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(R1), C_(R2)))
    formula5 = eq_(C_(ColumnInt(2)), C_(1))
    s = Selection(inner, formula5)
    res = raop.walk(s)
    theoretical_res = Product(
        (C_(R1), Selection(C_(R2), formula4))
    )
    assert res == theoretical_res


def test_named_columns_projection():
    s = NamedRelationalAlgebraFrozenSet(
        columns=("x", "y"), iterable=[("c", "g"), ("b", "h"), ("a", "a")]
    )

    op = Projection(
        Constant[AbstractSet[Tuple[str, str]]](s), (Constant(ColumnStr("x")),)
    )

    solver = RelationalAlgebraSolver()
    result = solver.walk(op)
    assert result == Constant[AbstractSet[Tuple[str, ]]](
        NamedRelationalAlgebraFrozenSet(
            columns=("x",), iterable=[("c",), ("b",), ("a",)]
        )
    )


def test_name_columns_after_projection():
    r = Constant[AbstractSet](
        RelationalAlgebraFrozenSet([(56, "bonjour"), (42, "second"),])
    )
    r = Projection(r, (Constant(ColumnInt(0)), Constant(ColumnInt(1))))
    r = NameColumns(r, (Constant(ColumnStr("x")), Constant(ColumnStr("n"))))
    solver = RelationalAlgebraSolver()
    result = solver.walk(r).value
    expected = NamedRelationalAlgebraFrozenSet(
        ("x", "n"), [(56, "bonjour"), (42, "second"),],
    )
    assert result == expected
    for tuple_result, tuple_expected in zip(result, expected):
        assert tuple_result == tuple_expected


def test_name_columns_symbolic_column_name():
    relation = Constant[AbstractSet](
        RelationalAlgebraSet([("hello", "world"), ("foo", "bar"),])
    )
    symbol_table = {
        Symbol("my_column_name_symbol"): Constant[ColumnStr](
            ColumnStr("a_column_name")
        )
    }
    solver = RelationalAlgebraSolver(symbol_table)
    assert solver.walk(Constant[ColumnStr](ColumnStr("test"))) == Constant[
        ColumnStr
    ](ColumnStr("test"))


def test_const_relation_type_is_known():
    type_ = AbstractSet[Tuple[int, str]]
    values = [(42, "bonjour"), (21, "galaxy")]
    relation = RelationalAlgebraFrozenSet(values)
    assert _const_relation_type_is_known(Constant[type_](relation))
    assert not _const_relation_type_is_known(Constant[AbstractSet](relation))
    assert not _const_relation_type_is_known(
        Constant[AbstractSet[Tuple]](relation)
    )


def test_sort_typed_const_named_relation_tuple_type_args():
    type_ = AbstractSet[Tuple[int, str]]
    sorted_type = AbstractSet[Tuple[str, int]]
    columns = ("b", "a")
    values = [(42, "bonjour"), (21, "galaxy")]
    sorted_named_relation = NamedRelationalAlgebraFrozenSet(
        sorted(columns), [t[::-1] for t in values]
    )
    not_sorted_named_relation = NamedRelationalAlgebraFrozenSet(
        columns, values
    )
    assert (
        _sort_typed_const_named_relation_tuple_type_args(
            Constant[type_](not_sorted_named_relation)
        )
        is sorted_type
    )
    assert (
        _sort_typed_const_named_relation_tuple_type_args(
            Constant[sorted_type](sorted_named_relation)
        )
        is sorted_type
    )


def test_infer_relation_type():
    assert (
        _infer_relation_type(RelationalAlgebraFrozenSet())
        is AbstractSet[Tuple]
    )
    assert (
        _infer_relation_type(RelationalAlgebraFrozenSet([(2, "hello")]))
        is AbstractSet[Tuple[int, str]]
    )


def test_get_const_relation_type():
    type_ = AbstractSet[Tuple[int, str]]
    values = [(42, "bonjour"), (21, "galaxy")]
    columns = ('y', 'z')
    relation = NamedRelationalAlgebraFrozenSet(columns, values)
    assert _get_const_relation_type(Constant[type_](relation)) is type_
    type_ = AbstractSet[Tuple[int, str]]
    sorted_type = AbstractSet[Tuple[str, int]]
    values = [(42, "bonjour"), (21, "galaxy")]
    columns = ('z', 'y')
    relation = NamedRelationalAlgebraFrozenSet(columns, values)
    assert _get_const_relation_type(Constant[type_](relation)) is sorted_type


def test_concatenate_constant_column():
    columns = ("x", "y")
    values = [("a", "b"), ("a", "c")]
    dst_column = Constant(ColumnStr("z"))
    cst = Constant(3)
    exp_columns = ("x", "y", dst_column.value)
    exp_values = list(val + (cst.value,) for val in values)
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns, values)
    )
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(exp_columns, exp_values)
    )
    concat_op = ConcatenateConstantColumn(relation, dst_column, cst)
    solver = RelationalAlgebraSolver()
    result = solver.walk(concat_op)
    assert result == expected


def test_concatenate_constant_column_already_existing_column():
    columns = ("x", "y")
    values = [("a", "b"), ("a", "c")]
    dst_column = Constant(ColumnStr("y"))
    cst = Constant(3)
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns, values)
    )
    concat_op = ConcatenateConstantColumn(relation, dst_column, cst)
    solver = RelationalAlgebraSolver()
    with pytest.raises(NeuroLangException, match=r"Cannot concatenate"):
        solver.walk(concat_op)


def test_extended_projection_divide_columns():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(50, 100), (20, 80),]
        )
    )
    dst_column = Constant(ColumnStr('z'))
    proj = ExtendedProjectionListMember(
        Constant(ColumnStr('y')) / Constant(ColumnStr('x')),
        dst_column
    )
    extended_proj_op = ExtendedProjection(relation, (proj, ))
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("z",), iterable=[(2.0, ), (4.0, )]
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(extended_proj_op)
    assert result == expected


def test_extended_projection_lambda_function():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[(50, 100), (20, 80),]
        )
    )
    lambda_fun = lambda df: (df.x + df.y) / 10.0
    extended_proj_op = ExtendedProjection(
        relation,
        (
            ExtendedProjectionListMember(
                Constant(lambda_fun), Constant(ColumnStr("z"))
            ),
            ExtendedProjectionListMember(
                Constant(RelationalAlgebraStringExpression("x")),
                Constant(ColumnStr("pomme_de_terre"))
            )
        )
    )
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("z", "pomme_de_terre"),
            iterable=[(15.0, 50), (10.0, 20)]
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(extended_proj_op)
    assert result == expected


def test_extended_projection_numeric_named_columns():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=(1, "y"), iterable=[(50, 100), (20, 80),]
        )
    )
    extended_proj_op = ExtendedProjection(
        relation,
        (
            ExtendedProjectionListMember(
                Constant(ColumnInt(1)),
                Constant(ColumnStr("pomme_de_terre"))
            ),
        )
    )
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("pomme_de_terre",),
            iterable=[(50,), (20,)]
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(extended_proj_op)
    assert result == expected


def test_extended_projection_other_relation_length():
    r1 = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"), iterable=[("a", 100), ("b", 80)]
        )
    )
    length = 2
    r2 = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("hello",), iterable=[(i,) for i in range(length)]
        )
    )
    proj = ExtendedProjectionListMember(
        Constant(ColumnStr("y")) / Constant(len)(r2), Constant(ColumnStr("y"))
    )
    extended_proj_op = ExtendedProjection(r1, (proj, ))
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("y",), iterable=[(50.0,), (40.0,)]
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(extended_proj_op)
    assert result == expected


def test_rename_columns():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("a", "b"),
            iterable=[("bonjour", "hello"), ("namaste", "ciao")],
        )
    )
    rename = RenameColumns(
        relation,
        (
            (Constant(ColumnStr("a")), Constant(ColumnStr("d"))),
            (Constant(ColumnStr("b")), Constant(ColumnStr("e"))),
        )
    )
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("d", "e"),
            iterable=[("bonjour", "hello"), ("namaste", "ciao")],
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(rename)
    assert result == expected
    with pytest.raises(ValueError):
        rename = RenameColumns(
            relation,
            (
                (str2columnstr_constant("a"), str2columnstr_constant("y")),
                (str2columnstr_constant("a"), str2columnstr_constant("z")),
            ),
        )
        solver.walk(rename)


def test_rename_columns_empty_relation():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns=("x", "y"))
    )
    rename = RenameColumns(
        relation, (
            (Constant(ColumnStr("x")), Constant(ColumnStr("z"))),
            (Constant(ColumnStr("y")), Constant(ColumnStr("x"))),
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(rename)
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns=("x", "z"))
    )
    assert result == expected


def test_rename_column_empty_relation():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns=("x", "y"))
    )
    rename = RenameColumn(
        relation, Constant(ColumnStr("x")), Constant(ColumnStr("z"))
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(rename)
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(columns=("z", "y"))
    )
    assert result == expected


def test_set_destroy():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=['x', 'y', 'z'],
            iterable=[
                (0, 1, frozenset({3, 4})),
                (0, 2, frozenset({5})),
            ]
        )
    )
    expected_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=['x', 'y', 'z', 'w'],
            iterable=[
                (0, 1, frozenset({3, 4}), 3),
                (0, 1, frozenset({3, 4}), 4),
                (0, 2, frozenset({5}), 5),
            ]
        )
    )

    destroy = Destroy(
        relation,
        Constant(ColumnStr('z')),
        Constant(ColumnStr('w'))
    )

    solver = RelationalAlgebraSolver()
    result = solver.walk(destroy)

    assert result == expected_relation


def test_set_destroy_no_grouping():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=['z'],
            iterable=[
                (frozenset({3, 4}),),
                (frozenset({5}),),
            ]
        )
    )
    expected_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=['z', 'w'],
            iterable=[
                (frozenset({3, 4}), 3,),
                (frozenset({3, 4}), 4,),
                (frozenset({5}), 5,),
            ]
        )
    )

    destroy = Destroy(
        relation,
        Constant(ColumnStr('z')),
        Constant(ColumnStr('w'))
    )

    solver = RelationalAlgebraSolver()
    result = solver.walk(destroy)

    assert result == expected_relation


def test_set_destroy_multiple_columns():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=['x', 'y', 'z'],
            iterable=[
                (0, 1, frozenset({(3, 4), (4, 5)})),
                (0, 2, frozenset({(5, 6)})),
            ]
        )
    )
    expected_relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=['x', 'y', 'z', 'v', 'w'],
            iterable=[
                (0, 1, frozenset({(3, 4), (4, 5)}), 3, 4),
                (0, 1, frozenset({(3, 4), (4, 5)}), 4, 5),
                (0, 2, frozenset({(5, 6)}), 5, 6),
            ]
        )
    )

    destroy = Destroy(
        relation,
        Constant(ColumnStr('z')),
        Constant[Tuple[ColumnStr, ColumnStr]](
            (ColumnStr('v'), ColumnStr('w'))
        )
    )

    solver = RelationalAlgebraSolver()
    result = solver.walk(destroy)

    assert result == expected_relation


def test_columns():
    r1 = Symbol('r1')
    r2 = Symbol('r2')
    a = Constant[ColumnStr](ColumnStr('a'))
    b = Constant[ColumnInt](ColumnInt(0))

    nj = NaturalJoin(r1, r2)
    assert len(nj.columns()) == 0

    rc = RenameColumn(r1, a, b)
    assert rc.columns() == set((a, b))

    assert NaturalJoin(rc, r2).columns() == set((a, b))


def test_push_in_optimiser():
    class Opt(RelationalAlgebraPushInSelections, ExpressionWalker):
        pass

    opt = Opt()

    r1 = Symbol('r1')
    r2 = Symbol('r2')
    a = Constant[ColumnStr](ColumnStr('a'))
    b = Constant[ColumnStr](ColumnStr('b'))
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


def test_eliminate_trivial_projections_optimiser():
    class Opt(EliminateTrivialProjections, ExpressionWalker):
        pass

    opt = Opt()

    r1 = Constant(R1)
    exp = Projection(
        r1,
        (Constant[ColumnInt](ColumnInt(0)), Constant[ColumnInt](ColumnInt(1)))
    )

    res = opt.walk(exp)
    assert res is r1

    a = Constant[ColumnStr](ColumnStr('a'))
    b = Constant[ColumnStr](ColumnStr('b'))
    R = NamedRelationalAlgebraFrozenSet(
        columns=('a', 'b'),
        iterable=R1
    )

    r = C_[AbstractSet[Tuple[int, int]]](R)

    exp1 = Projection(r, (a, b))
    res1 = opt.walk(exp1)
    assert res1 is r
