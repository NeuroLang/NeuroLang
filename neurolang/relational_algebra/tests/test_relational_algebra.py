import operator
from typing import AbstractSet, Tuple

import numpy
import pandas as pd
import pytest

from ...config import config
from ...datalog.basic_representation import WrappedRelationalAlgebraSet
from ...exceptions import NeuroLangException
from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import (
    ColumnInt,
    ColumnStr,
    ConcatenateConstantColumn,
    Destroy,
    EquiJoin,
    ExtendedProjection,
    FunctionApplicationListMember,
    GroupByAggregation,
    Intersection,
    NameColumns,
    NaturalJoin,
    Product,
    Projection,
    RelationalAlgebraSolver,
    RenameColumn,
    RenameColumns,
    Selection,
    Union,
    eq_,
    str2columnstr_constant
)
from ...utils import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet
)
from ...utils.relational_algebra_set import RelationalAlgebraStringExpression
from ..relational_algebra import (
    EVAL_OP_TO_STR,
    FullOuterNaturalJoin,
    LeftNaturalJoin,
    _const_relation_type_is_known,
    _get_const_relation_type,
    _infer_relation_type,
    _sort_typed_const_named_relation_tuple_type_args
)

C_ = Constant


@pytest.fixture
def R1():
    return WrappedRelationalAlgebraSet([(i, i * 2) for i in range(10)])


@pytest.fixture
def R2():
    return WrappedRelationalAlgebraSet([(i * 2, i * 3) for i in range(10)])


def test_selection(R1):
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(0)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection({0: 0})


def test_selection_columns(R1):
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection_columns({0: 1})


def test_selection_general(R1):
    gt_ = C_(operator.gt)
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    c = C_[AbstractSet[Tuple[int, int]]](r1_named)
    s = Selection(c, gt_(C_(ColumnStr('x')), C_(5)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == r1_named.selection(lambda t: t.x > 5)


def test_projections(R1):
    s = Projection(C_(R1), (C_(ColumnInt(0)),))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.projection(0)


def test_equijoin(R1, R2):
    s = EquiJoin(
        C_(R1), (C_(ColumnInt(0)),),
        C_(R2), (C_(ColumnInt(0)),)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.equijoin(R2, [(0, 0)])


def test_naturaljoin(R1, R2):
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    r2_named = NamedRelationalAlgebraFrozenSet(('x', 'z'), R2)
    s = NaturalJoin(
        C_[AbstractSet[Tuple[int, int]]](r1_named),
        C_[AbstractSet[Tuple[int, int]]](r2_named)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == r1_named.naturaljoin(r2_named)


def test_leftnaturaljoin(R1, R2):
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    r2_named = NamedRelationalAlgebraFrozenSet(('x', 'z'), R2)
    s = LeftNaturalJoin(
        C_[AbstractSet[Tuple[int, int]]](r1_named),
        C_[AbstractSet[Tuple[int, int]]](r2_named)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == r1_named.left_naturaljoin(r2_named)


def test_fullouterjoin(R1, R2):
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    r2_named = NamedRelationalAlgebraFrozenSet(('x', 'z'), R2)
    s = FullOuterNaturalJoin(
        C_[AbstractSet[Tuple[int, int]]](r1_named),
        C_[AbstractSet[Tuple[int, int]]](r2_named)
    )
    expected = (
        r1_named.left_naturaljoin(r2_named) |
        r2_named.left_naturaljoin(r1_named)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == expected


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


def test_product(R1, R2):
    s = Product((C_(R1), C_(R2)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2)

    s = Product((C_(R1), C_(R2), C_(R1)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2).cross_product(R1)

    s = Product(tuple())
    sol = RelationalAlgebraSolver().walk(s).value
    assert len(sol) == 0


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
    proj = FunctionApplicationListMember(
        Constant(ColumnStr('y')) / Constant(ColumnStr('x')),
        dst_column
    )
    extended_proj_op = ExtendedProjection(relation, (proj, ))
    if config["RAS"].get("backend") == "dask":
        expected = Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=("z",),
                iterable=pd.DataFrame([(2.0,), (4.0,)]).convert_dtypes(),
            )
        )
    else:
        expected = Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=("z",), iterable=[(2.0,), (4.0,)]
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
            FunctionApplicationListMember(
                Constant(lambda_fun), Constant(ColumnStr("z"))
            ),
            FunctionApplicationListMember(
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
            FunctionApplicationListMember(
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
    proj = FunctionApplicationListMember(
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


def test_groupby_aggregate_sum():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y", "z"),
            iterable=[(10, 20, 30), (10, 20, 50), (1, 2, 3)],
        )
    )
    agg_op = GroupByAggregation(
        relation,
        (Constant(ColumnStr("x")), Constant(ColumnStr("y"))),
        [
            FunctionApplicationListMember(
                FunctionApplication(Constant(sum), (Constant(ColumnStr("z")),), verify_type=False),
                Constant(ColumnStr("z_sum")),
            )
        ],
    )
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y", "z_sum"), iterable=[(10, 20, 80), (1, 2, 3)]
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(agg_op)
    assert result == expected


def test_groupby_aggregate_lambda():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y", "z"),
            iterable=[(10, 20, 30), (10, 20, 50), (10, 2, 50)],
        )
    )
    custom_lambda = lambda y: min(y) - 2
    agg_op = GroupByAggregation(
        relation,
        (Constant(ColumnStr("x")), Constant(ColumnStr("z"))),
        [
            FunctionApplicationListMember(
                FunctionApplication(
                    Constant(custom_lambda), (Constant(ColumnStr("y")),), verify_type=False
                ),
                Constant(ColumnStr("y_min")),
            )
        ],
    )
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "z", "y_min"), iterable=[(10, 30, 18), (10, 50, 0)]
        )
    )
    solver = RelationalAlgebraSolver()
    result = solver.walk(agg_op)
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

    rc = NameColumns(r1, (a, b))
    assert rc.columns() == set((a, b))

    rc2 = NameColumns(r2, (b,))
    assert NaturalJoin(rc, rc2).columns() == set((a, b))


def test_numpy_log():
    r = Constant[AbstractSet[Tuple[str, float]]](
        NamedRelationalAlgebraFrozenSet(
            columns=("x", "y"),
            iterable=[
                ("a", 1.0),
                ("b", 2.0),
            ],
        )
    )
    projlist = (
        FunctionApplicationListMember(
            fun_exp=Constant(RelationalAlgebraStringExpression("x")),
            dst_column=Constant(ColumnStr("x")),
        ),
        FunctionApplicationListMember(
            fun_exp=Constant(numpy.log)(Constant(ColumnStr("y"))),
            dst_column=Constant(ColumnStr("z")),
        ),
    )
    op = ExtendedProjection(r, projlist)
    solver = RelationalAlgebraSolver()
    result = solver.walk(op)
    expected = Constant[AbstractSet[Tuple[str, float]]](
        NamedRelationalAlgebraFrozenSet(
            iterable={
                ("a", 0.0),
                ("b", numpy.log(2.0)),
            },
            columns=("x", "z"),
        )
    )
    assert result == expected


def test_aggregation_translation():
    x = Constant(ColumnStr("x"))
    np_max = Constant(numpy.max, auto_infer_type=False)
    np_exp = Constant(numpy.exp, auto_infer_type=False)
    fun_exp = np_max(np_exp(x + Constant(2)))
    solver = RelationalAlgebraSolver()
    result = solver._saw.walk(fun_exp)
    expected = Constant(RelationalAlgebraStringExpression("max(exp((x + 2)))"))
    assert result == expected


def test_evaluatable_function_applications():
    x = Constant(ColumnStr("x"))
    for fun in EVAL_OP_TO_STR:
        fun_exp = Constant(fun)(x + Constant(2))
        solver = RelationalAlgebraSolver()
        result = solver._saw.walk(fun_exp)
        expected = Constant(
            RelationalAlgebraStringExpression(
                "{}((x + 2))".format(EVAL_OP_TO_STR[fun])
            )
        )
        assert result == expected
