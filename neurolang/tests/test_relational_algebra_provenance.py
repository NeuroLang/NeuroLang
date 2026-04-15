import operator
from typing import AbstractSet

import numpy as np
import pytest

from ..expressions import Constant, Symbol
from ..exceptions import RelationalAlgebraError
from ..probabilistic.cplogic import testing
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    NaturalJoin,
    Product,
    RelationalAlgebraSolver,
    RenameColumn,
    RenameColumns,
    Selection,
    eq_,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    IndependentDisjointProjectionsAndUnionMixin,
    IndependentProjection,
    ProvenanceAlgebraSet,
    ConcatenateConstantColumn,
    ExtendedProjection,
    FunctionApplicationListMember,
    NaturalJoinInverse,
    Projection,
    RelationalAlgebraProvenanceCountingSolver,
    Union,
    WeightedNaturalJoin,
    WeightedNaturalJoinSolverMixin,
)
from ..utils import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraStringExpression,
)

C_ = Constant
S_ = Symbol


@pytest.fixture
def R1():
    return NamedRelationalAlgebraFrozenSet(
        columns=("col1", "col2", "__provenance__"),
        iterable=[(i, i * 2, i) for i in range(10)],
    )


@pytest.fixture
def provenance_set_r1(R1):
    return testing.build_ra_provenance_set_from_named_ra_set(
        R1, ColumnStr("__provenance__")
    )


def test_selection(R1, provenance_set_r1):
    s = Selection(provenance_set_r1, eq_(C_(ColumnStr("col1")), C_(4)))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    assert sol == R1.selection({"col1": 4})
    assert "__provenance__" in sol.columns


def test_selection_columns(R1, provenance_set_r1):
    s = Selection(
        provenance_set_r1, eq_(C_(ColumnStr("col1")), C_(ColumnStr("col2")))
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    assert sol == R1.selection_columns({"col1": "col2"})
    assert sol == R1.selection({"col1": 0})
    assert "__provenance__" in sol.columns


def test_valid_rename(R1, provenance_set_r1):
    s = RenameColumn(
        provenance_set_r1, C_(ColumnStr("col1")), C_(ColumnStr("renamed"))
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    assert sol == R1.rename_column("col1", "renamed")
    assert "renamed" in sol.columns
    assert "col1" not in sol.columns
    assert "__provenance__" in sol.columns
    assert R1.projection("__provenance__") == sol.projection("__provenance__")


def test_provenance_rename(R1, provenance_set_r1):
    s = RenameColumn(
        provenance_set_r1,
        C_(ColumnStr("__provenance__")),
        C_(ColumnStr("renamed")),
    )

    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == str2columnstr_constant("renamed")
    sol = sol.relation.value
    assert sol == R1.rename_column("__provenance__", "renamed")
    assert "renamed" in sol.columns
    assert "__provenance__" not in sol.columns


def test_naturaljoin():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = testing.build_ra_provenance_set_from_named_ra_set(
        RA1, "__provenance__"
    )
    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = testing.build_ra_provenance_set_from_named_ra_set(
        RA2, "__provenance__"
    )

    s = NaturalJoin(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    RA1_np = NamedRelationalAlgebraFrozenSet(
        columns=("col1",), iterable=[(i * 2) for i in range(10)]
    )

    RA2_np = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA"), iterable=[(i % 5, i * 3) for i in range(20)]
    )

    R1njR2 = RA1_np.naturaljoin(RA2_np)
    sol_np = sol.projection(*["col1", "colA"])
    assert sol_np == R1njR2

    R1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )

    R2 = NamedRelationalAlgebraFrozenSet(
        columns=("colA", "__provenance__2"),
        iterable=[(i * 3, i) for i in range(20)],
    )

    R1cpR2 = R1.cross_product(R2)
    RnjR = R1cpR2.naturaljoin(R1njR2)

    res = ExtendedProjection(
        Constant[AbstractSet](RnjR),
        tuple(
            [
                FunctionApplicationListMember(
                    fun_exp=(
                        Constant(ColumnStr("__provenance__1")) *
                        Constant(ColumnStr("__provenance__2"))
                    ),
                    dst_column=Constant(ColumnStr("__provenance__")),
                )
            ]
        ),
    )

    res = RelationalAlgebraSolver().walk(res).value

    prov_sol = sol.projection("__provenance__")
    prov_res = res.projection("__provenance__")
    assert np.all(prov_sol == prov_res)


def test_naturaljoin_provenance_name():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = testing.build_ra_provenance_set_from_named_ra_set(
        RA1, ColumnStr("__provenance__1")
    )

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = testing.build_ra_provenance_set_from_named_ra_set(
        RA2, ColumnStr("__provenance__2")
    )

    s = NaturalJoin(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == str2columnstr_constant("__provenance__1")
    assert "__provenance__1" in sol.relation.value.columns
    assert "__provenance__2" not in sol.relation.value.columns


def test_product():
    R1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = testing.build_ra_provenance_set_from_named_ra_set(
        R1, ColumnStr("__provenance__")
    )

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("colA", "__provenance__"),
        iterable=[(i * 3, i) for i in range(20)],
    )
    pset_r2 = testing.build_ra_provenance_set_from_named_ra_set(
        RA2, ColumnStr("__provenance__")
    )

    s = Product((pset_r1, pset_r2))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    RA1_np = NamedRelationalAlgebraFrozenSet(
        columns=("col1",), iterable=[(i * 2) for i in range(10)]
    )

    RA2_np = NamedRelationalAlgebraFrozenSet(
        columns=("colA",), iterable=[(i * 3) for i in range(20)]
    )

    R1njR2 = RA1_np.cross_product(RA2_np)
    sol_np = sol.projection(*["col1", "colA"])
    assert sol_np == R1njR2

    R1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )

    R2 = NamedRelationalAlgebraFrozenSet(
        columns=("colA", "__provenance__2"),
        iterable=[(i * 3, i) for i in range(20)],
    )

    R1cpR2 = R1.cross_product(R2)
    RnjR = R1cpR2.naturaljoin(R1njR2)
    res = ExtendedProjection(
        Constant[AbstractSet](RnjR),
        tuple(
            [
                FunctionApplicationListMember(
                    fun_exp=(
                        Constant(ColumnStr("__provenance__1")) *
                        Constant(ColumnStr("__provenance__2"))
                    ),
                    dst_column=Constant(ColumnStr("__provenance__")),
                )
            ]
        ),
    )

    res = RelationalAlgebraSolver().walk(res)

    prov_sol = sol.projection("__provenance__")
    prov_res = res.value.projection("__provenance__")
    assert np.all(prov_sol == prov_res)


def test_product_provenance_name():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = testing.build_ra_provenance_set_from_named_ra_set(
        RA1, ColumnStr("__provenance__1")
    )

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col2", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = testing.build_ra_provenance_set_from_named_ra_set(
        RA2, ColumnStr("__provenance__2")
    )

    s = Product((pset_r1, pset_r2))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == str2columnstr_constant("__provenance__1")
    assert "__provenance__1" in sol.relation.value.columns
    assert "__provenance__2" not in sol.relation.value.columns


def test_union():
    relation1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                ("a", "b", 1),
                ("b", "a", 2),
                ("c", "a", 2),
                ("c", "a", 1),
                ("b", "a", 1),
            ],
            columns=["x", "y", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )
    relation2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                ("a", "b", 1),
                ("b", "a", 2),
                ("d", "a", 2),
                ("c", "a", 1),
                ("b", "a", 1),
            ],
            columns=["x", "y", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )

    expected = NamedRelationalAlgebraFrozenSet(
        iterable=[("a", "b", 2), ("b", "a", 6), ("c", "a", 4), ("d", "a", 2)],
        columns=["x", "y", "__provenance__"],
    )

    s = Union(relation1, relation2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    assert sol == expected


def test_union_different_prov_col_names():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p1_", "x"),
            [(0.1, "a"), (0.2, "b")]
        ),
        ColumnStr("_p1_")
    )

    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p2_", "x"),
            [(0.5, "a"), (0.9, "c")]
        ),
        ColumnStr("_p2_")
    )
    operation = Union(r1, r2)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(operation)

    expected = solver.walk(testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_whatever_", "x"), [(0.6, "a"), (0.2, "b"), (0.9, "c")]
        ),
        "_whatever_"
    ))

    assert testing.eq_prov_relations(result, expected)


def test_union_with_empty_set(provenance_set_r1):
    empty = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(("_p_", "col1", "col2")),
        ColumnStr("_p_")
    )
    operation = Union(provenance_set_r1, empty)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(operation)
    assert testing.eq_prov_relations(result, solver.walk(provenance_set_r1))


def test_projection():
    relation = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                ("a", "b", 1),
                ("b", "a", 2),
                ("c", "a", 2),
                ("c", "x", 1),
                ("b", "x", 1),
            ],
            columns=["x", "y", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )
    expected = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[("a", 1), ("b", 3), ("c", 3)],
            columns=["x", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )
    sum_agg_op = Projection(
        relation, (str2columnstr_constant("x"),)
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(sum_agg_op)
    expected = solver.walk(expected)

    assert result == expected


def test_concatenate_constant(provenance_set_r1):
    s = ConcatenateConstantColumn(
        provenance_set_r1, C_(ColumnStr("new_col")), C_(9)
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).relation.value

    expected = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "col2", "__provenance__", "new_col"),
        iterable=[(i, i * 2, i, 9) for i in range(10)],
    )

    assert expected == sol


def test_extended_projection():
    relation = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[(5, 1, 1), (6, 2, 2), (7, 3, 2), (1, 3, 1), (2, 1, 1)],
            columns=["x", "y", "__provenance__"],
        ),
        "__provenance__",
    )

    expected = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                (5, 1, 6, 1), (6, 2, 8, 2),
                (7, 3, 10, 2), (1, 3, 4, 1),
                (2, 1, 3, 1),
            ],
            columns=["x", "y", "sum_", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )

    x = str2columnstr_constant("x")
    y = str2columnstr_constant("y")

    res = ExtendedProjection(
        relation,
        tuple(
            [
                FunctionApplicationListMember(
                    fun_exp=x + y,
                    dst_column=Constant(ColumnStr("sum_")),
                ),
                FunctionApplicationListMember(x, x),
                FunctionApplicationListMember(y, y)
            ]
        ),
    )

    res = RelationalAlgebraProvenanceCountingSolver().walk(res)
    expected = RelationalAlgebraProvenanceCountingSolver().walk(expected)
    assert res == expected


def test_provenance_projection():
    relation = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            iterable=[
                (0.8, "a", 42),
                (0.7, "b", 84),
                (0.2, "a", 21),
                (0.1, "b", 128),
            ],
            columns=["myprov", "x", "y"],
        )),
        str2columnstr_constant("myprov"),
    )
    projection = Projection(relation, (Constant(ColumnStr("x")),))
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(projection)
    assert len(result.relation.value) == 2
    for exp_prob, exp_x in [(1.0, "a"), (0.8, "b")]:
        for tupl in result.relation.value:
            if tupl.x == exp_x:
                assert np.isclose(
                    exp_prob, getattr(tupl, result.provenance_column.value)
                )


def test_provenance_product_with_shared_non_prov_col_should_fail():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                (0.8, "a", 42),
                (0.7, "b", 84),
                (0.2, "a", 21),
                (0.1, "b", 128),
            ],
            columns=["myprov", "x", "y"],
        ),
        ColumnStr("myprov"),
    )
    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            iterable=[(0.2, "a", 42), (0.5, "b", 84)],
            columns=["myprov", "x", "z"],
        ),
        ColumnStr("myprov"),
    )
    product = Product((r1, r2))
    solver = RelationalAlgebraProvenanceCountingSolver()
    with pytest.raises(RelationalAlgebraError):
        solver.walk(product)


def test_concatenate_column_to_ra_relation():
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            iterable=[("a", "b"), ("c", "d")], columns=["x", "y"],
        )
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    concat_op = ConcatenateConstantColumn(
        relation, Constant(ColumnStr("z")), Constant[int](3)
    )
    result = solver.walk(concat_op)
    expected = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(
            iterable=[("a", "b", 3), ("c", "d", 3)], columns=["x", "y", "z"],
        )
    )
    assert result == expected


def test_rename_columns():
    prov_relation = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[(0.1, "a", 0), (1.0, "b", 44)],
        ),
        ColumnStr("_p_")
    )
    rename_columns = RenameColumns(
        prov_relation,
        ((str2columnstr_constant("x"), str2columnstr_constant("z")),),
    )
    expected = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "z", "y"),
            iterable=[(0.1, "a", 0), (1.0, "b", 44)],
        ),
        ColumnStr("_p_")
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(rename_columns)
    expected = solver.walk(expected)
    assert testing.eq_prov_relations(result, expected)


def test_njoin_inverse():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x"), iterable=[(1.0, "a"), (0.5, "b")],
        ),
        ColumnStr("_p_"),
    )
    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x"), iterable=[(0.5, "b")],
        ),
        ColumnStr("_p_"),
    )
    expected = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x"), iterable=[(1.0, "b")],
        ),
        ColumnStr("_p_"),
    )
    op = NaturalJoinInverse(r1, r2)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(op)
    expected = solver.walk(expected)
    assert testing.eq_prov_relations(result, expected)


def test_selection_between_columnints():
    r = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[
                (0.5, "a", "b"),
                (0.7, "a", "a"),
                (0.2, "b", "b"),
            ],
        )),
        str2columnstr_constant("_p_"),
    )
    col1 = Constant[ColumnInt](ColumnInt(0))
    col2 = Constant[ColumnInt](ColumnInt(1))
    op = Selection(r, Constant(operator.eq)(col1, col2))
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(op)
    expected = solver.walk(testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[
                (0.7, "a", "a"),
                (0.2, "b", "b"),
            ],
        ),
        ColumnStr("_p_"),
    ))
    assert testing.eq_prov_relations(result, expected)


def test_weightednaturaljoin_provenance_name():
    class TestRAPWeightedNaturalJoinSolver(
        WeightedNaturalJoinSolverMixin,
        RelationalAlgebraProvenanceCountingSolver,
    ):
        pass
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = testing.build_ra_provenance_set_from_named_ra_set(
        RA1, ColumnStr("__provenance__1")
    )

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = testing.build_ra_provenance_set_from_named_ra_set(
        RA2, ColumnStr("__provenance__2")
    )

    s = WeightedNaturalJoin((pset_r1, pset_r2), (Constant(1), Constant(-1)))
    sol = TestRAPWeightedNaturalJoinSolver().walk(s)

    expected = RA1.naturaljoin(RA2).extended_projection(
        {
            sol.provenance_column.value: RelationalAlgebraStringExpression(
                '__provenance__1 - __provenance__2'
            ),
            'col1': RelationalAlgebraStringExpression('col1'),
            'colA': RelationalAlgebraStringExpression('colA')
        }
    )

    assert sol.relation.value == expected

def test_independentdisjointprojections_provenance_name():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("_p_", "col1", "col2", "col3"),
        iterable=[(0.5, 1, 2, 5),
                  (1.0, 2, 1, 6),
                  (1.0, 1, 1, 7),
                  (0.2, 1, 2, 4)],
    )
    pset_r1 = testing.build_ra_provenance_set_from_named_ra_set(
        RA1, ColumnStr("_p_")
    )

    expected = NamedRelationalAlgebraFrozenSet(
        columns=("_p_", "col1", "col2"),
        iterable=[(1.0, 2, 1),
                  (0.6, 1, 2),
                  (1.0, 1, 1)],
    )

    s = IndependentProjection(pset_r1, (str2columnstr_constant("col1"),str2columnstr_constant("col2")))
    result = IndependentDisjointProjectionsAndUnionMixin().walk(s)
    res = RelationalAlgebraProvenanceCountingSolver().walk(result)

    assert res.relation.value == expected