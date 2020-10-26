import operator
from typing import AbstractSet

import numpy as np
import pytest

from ..exceptions import NeuroLangException
from ..expressions import Constant, Symbol
from ..probabilistic.cplogic import testing
from ..relational_algebra import (
    ColumnStr,
    ColumnInt,
    NaturalJoin,
    Product,
    RenameColumn,
    RenameColumns,
    Selection,
    eq_,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    ConcatenateConstantColumn,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NaturalJoinInverse,
    Projection,
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
    Union,
)
from ..utils import NamedRelationalAlgebraFrozenSet

C_ = Constant
S_ = Symbol

R1 = NamedRelationalAlgebraFrozenSet(
    columns=("col1", "col2", "__provenance__"),
    iterable=[(i, i * 2, i) for i in range(10)],
)
provenance_set_r1 = ProvenanceAlgebraSet(R1, ColumnStr("__provenance__"))


def test_selection():
    s = Selection(provenance_set_r1, eq_(C_(ColumnStr("col1")), C_(4)))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == R1.selection({"col1": 4})
    assert "__provenance__" in sol.columns


def test_selection_columns():
    s = Selection(
        provenance_set_r1, eq_(C_(ColumnStr("col1")), C_(ColumnStr("col2")))
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == R1.selection_columns({"col1": "col2"})
    assert sol == R1.selection({"col1": 0})
    assert "__provenance__" in sol.columns


def test_valid_rename():
    s = RenameColumn(
        provenance_set_r1, C_(ColumnStr("col1")), C_(ColumnStr("renamed"))
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == R1.rename_column("col1", "renamed")
    assert "renamed" in sol.columns
    assert "col1" not in sol.columns
    assert "__provenance__" in sol.columns
    assert R1.projection("__provenance__") == sol.projection("__provenance__")


def test_provenance_rename():
    s = RenameColumn(
        provenance_set_r1,
        C_(ColumnStr("__provenance__")),
        C_(ColumnStr("renamed")),
    )

    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == "renamed"
    sol = sol.value
    assert sol == R1.rename_column("__provenance__", "renamed")
    assert "renamed" in sol.columns
    assert "__provenance__" not in sol.columns


def test_naturaljoin():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, ColumnStr("__provenance__"))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, ColumnStr("__provenance__"))

    s = NaturalJoin(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    RA1_np = NamedRelationalAlgebraFrozenSet(
        columns=("col1",), iterable=[(i * 2) for i in range(10)]
    )

    RA2_np = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA"), iterable=[(i % 5, i * 3) for i in range(20)]
    )

    R1njR2 = RA1_np.naturaljoin(RA2_np)
    sol_np = sol.value.projection(*["col1", "colA"])
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
        ProvenanceAlgebraSet(RnjR, ColumnStr("__provenance__1")),
        tuple(
            [
                ExtendedProjectionListMember(
                    fun_exp=Constant(ColumnStr("__provenance__1"))
                    * Constant(ColumnStr("__provenance__2")),
                    dst_column=Constant(ColumnStr("__provenance__")),
                )
            ]
        ),
    )

    res = RelationalAlgebraProvenanceCountingSolver().walk(res)

    prov_sol = sol.value.projection("__provenance__")
    prov_res = res.value.projection("__provenance__")
    assert np.all(prov_sol == prov_res)


def test_naturaljoin_provenance_name():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, ColumnStr("__provenance__1"))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, ColumnStr("__provenance__2"))

    s = NaturalJoin(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == "__provenance__1"
    assert "__provenance__1" in sol.value.columns
    assert "__provenance__2" not in sol.value.columns


def test_product():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, ColumnStr("__provenance__"))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("colA", "__provenance__"),
        iterable=[(i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, ColumnStr("__provenance__"))

    s = Product((pset_r1, pset_r2))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

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
        ProvenanceAlgebraSet(RnjR, ColumnStr("__provenance__1")),
        tuple(
            [
                ExtendedProjectionListMember(
                    fun_exp=Constant(ColumnStr("__provenance__1"))
                    * Constant(ColumnStr("__provenance__2")),
                    dst_column=Constant(ColumnStr("__provenance__")),
                )
            ]
        ),
    )

    res = RelationalAlgebraProvenanceCountingSolver().walk(res)

    prov_sol = sol.projection("__provenance__")
    prov_res = res.value.projection("__provenance__")
    assert np.all(prov_sol == prov_res)


def test_product_provenance_name():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__1"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, ColumnStr("__provenance__1"))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col2", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, ColumnStr("__provenance__2"))

    s = Product((pset_r1, pset_r2))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == "__provenance__1"
    assert "__provenance__1" in sol.value.columns
    assert "__provenance__2" not in sol.value.columns


def test_union():
    relation1 = ProvenanceAlgebraSet(
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
    relation2 = ProvenanceAlgebraSet(
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
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == expected


def test_union_different_prov_col_names():
    r1 = testing.make_prov_set([(0.1, "a"), (0.2, "b")], ("_p1_", "x"))
    r2 = testing.make_prov_set([(0.5, "a"), (0.9, "c")], ("_p2_", "x"))
    expected = testing.make_prov_set(
        [(0.6, "a"), (0.2, "b"), (0.9, "c")], ("_whatever_", "x"),
    )
    operation = Union(r1, r2)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(operation)
    assert testing.eq_prov_relations(result, expected)


def test_union_with_empty_set():
    r = testing.make_prov_set([(0.1, "a"), (0.2, "b")], ("_p_", "x"))
    empty = testing.make_prov_set([], ("_p_", "x"))
    operation = Union(r, empty)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(operation)
    assert testing.eq_prov_relations(result, r)


def test_projection():
    relation = ProvenanceAlgebraSet(
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
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[("a", "b", 1), ("b", "a", 3), ("c", "a", 3)],
            columns=["x", "y", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )
    sum_agg_op = Projection(
        relation, tuple([Constant(ColumnStr("x")), Constant(ColumnStr("y"))]),
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(sum_agg_op)

    assert result == expected


def test_concatenate_constant():
    s = ConcatenateConstantColumn(
        provenance_set_r1, C_(ColumnStr("new_col")), C_(9)
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    expected = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "col2", "__provenance__", "new_col"),
        iterable=[(i, i * 2, i, 9) for i in range(10)],
    )

    assert expected == sol


def test_extended_projection():
    relation = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[(5, 1, 1), (6, 2, 2), (7, 3, 2), (1, 3, 1), (2, 1, 1)],
            columns=["x", "y", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )

    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[(6, 1), (8, 2), (10, 2), (4, 1), (3, 1),],
            columns=["sum", "__provenance__"],
        ),
        ColumnStr("__provenance__"),
    )

    res = ExtendedProjection(
        relation,
        tuple(
            [
                ExtendedProjectionListMember(
                    fun_exp=Constant(ColumnStr("x"))
                    + Constant(ColumnStr("y")),
                    dst_column=Constant(ColumnStr("sum")),
                )
            ]
        ),
    )

    res = RelationalAlgebraProvenanceCountingSolver().walk(res)
    assert res == expected


def test_provenance_projection():
    relation = ProvenanceAlgebraSet(
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
    projection = Projection(relation, (Constant(ColumnStr("x")),))
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(projection)
    assert len(result.value) == 2
    for exp_prob, exp_x in [(1.0, "a"), (0.8, "b")]:
        for tupl in result.value:
            if tupl.x == exp_x:
                assert np.isclose(
                    exp_prob, getattr(tupl, result.provenance_column)
                )


def test_provenance_product_with_shared_non_prov_col_should_fail():
    r1 = ProvenanceAlgebraSet(
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
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[(0.2, "a", 42), (0.5, "b", 84)],
            columns=["myprov", "x", "z"],
        ),
        ColumnStr("myprov"),
    )
    product = Product((r1, r2))
    solver = RelationalAlgebraProvenanceCountingSolver()
    with pytest.raises(
        NeuroLangException, match="Shared columns found: 'x'",
    ):
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
    prov_relation = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[(0.1, "a", 0), (1.0, "b", 44)],
        ),
        provenance_column=ColumnStr("_p_"),
    )
    rename_columns = RenameColumns(
        prov_relation,
        ((str2columnstr_constant("x"), str2columnstr_constant("z")),),
    )
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "z", "y"),
            iterable=[(0.1, "a", 0), (1.0, "b", 44)],
        ),
        provenance_column=ColumnStr("_p_"),
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(rename_columns)
    assert testing.eq_prov_relations(result, expected)


def test_njoin_inverse():
    r1 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x"), iterable=[(1.0, "a"), (0.5, "b")],
        ),
        ColumnStr("_p_"),
    )
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x"), iterable=[(0.5, "b")],
        ),
        ColumnStr("_p_"),
    )
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x"), iterable=[(1.0, "b")],
        ),
        ColumnStr("_p_"),
    )
    op = NaturalJoinInverse(r1, r2)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(op)
    assert testing.eq_prov_relations(result, expected)


def test_selection_between_columnints():
    r = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[
                (0.5, "a", "b"),
                (0.7, "a", "a"),
                (0.2, "b", "b"),
            ],
        ),
        ColumnStr("_p_"),
    )
    col1 = Constant[ColumnInt](ColumnInt(0))
    col2 = Constant[ColumnInt](ColumnInt(1))
    op = Selection(r, Constant(operator.eq)(col1, col2))
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[
                (0.7, "a", "a"),
                (0.2, "b", "b"),
            ],
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)
