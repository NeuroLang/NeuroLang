import typing

import numpy as np

from ..expressions import Constant, Symbol
from ..relational_algebra import (
    ColumnStr,
    NaturalJoin,
    Product,
    RelationalAlgebraSolver,
    RenameColumn,
    Selection,
    eq_,
)
from ..relational_algebra_provenance import (
    ConcatenateConstantColumn,
    ExtendedProjection,
    ExtendedProjectionListMember,
    Projection,
    ProjectionNonProvenance,
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
provenance_set_r1 = ProvenanceAlgebraSet(R1, C_(ColumnStr("__provenance__")))


def equal_prov_sets(first, second):
    # check that the non-provenance columns match
    if first.non_provenance_columns != second.non_provenance_columns:
        return False
    first_relation = Constant[typing.AbstractSet](first.value)
    second_relation = Constant[typing.AbstractSet](second.value)
    solver = RelationalAlgebraSolver()
    # check that the tuple values (without the provenance column) match
    if not (
        solver.walk(Projection(first_relation, first.non_provenance_columns))
        == solver.walk(
            Projection(second_relation, second.non_provenance_columns)
        )
    ):
        return False
    # temporarily rename provenance columns to apply natural join
    first_tmp_prov_col = Constant(ColumnStr(Symbol.fresh().name))
    second_tmp_prov_col = Constant(ColumnStr(Symbol.fresh().name))
    first_rename = RenameColumn(
        first_relation, first.provenance_column, first_tmp_prov_col
    )
    second_rename = RenameColumn(
        second_relation, second.provenance_column, second_tmp_prov_col
    )
    joined = solver.walk(NaturalJoin(first_rename, second_rename))
    projected = solver.walk(
        Projection(joined, (first_tmp_prov_col, second_tmp_prov_col))
    )
    # check that the provenance columns are numerically very close
    return np.all(
        np.isclose(
            projected.value._container[first_tmp_prov_col.value].values,
            projected.value._container[second_tmp_prov_col.value].values,
        )
    )


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

    assert sol.provenance_column == C_(ColumnStr("renamed"))
    sol = sol.value
    assert sol == R1.rename_column("__provenance__", "renamed")
    assert "renamed" in sol.columns
    assert "__provenance__" not in sol.columns


def test_projections_non_provenance():
    s = ProjectionNonProvenance(provenance_set_r1, (C_(ColumnStr("col1")),))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)
    R1proj = R1.projection("col1")

    assert sol == R1proj
    assert "__provenance__" not in sol.columns


def test_naturaljoin():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, C_(ColumnStr("__provenance__")))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, C_(ColumnStr("__provenance__")))

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
        ProvenanceAlgebraSet(RnjR, C_(ColumnStr("__provenance__"))),
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
    pset_r1 = ProvenanceAlgebraSet(RA1, C_(ColumnStr("__provenance__1")))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, C_(ColumnStr("__provenance__2")))

    s = NaturalJoin(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == C_(ColumnStr("__provenance__1"))
    assert "__provenance__1" in sol.value.columns
    assert "__provenance__2" not in sol.value.columns


def test_product():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "__provenance__"),
        iterable=[(i * 2, i) for i in range(10)],
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, C_(ColumnStr("__provenance__")))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("colA", "__provenance__"),
        iterable=[(i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, C_(ColumnStr("__provenance__")))

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
        ProvenanceAlgebraSet(RnjR, C_(ColumnStr("__provenance__"))),
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
    pset_r1 = ProvenanceAlgebraSet(RA1, C_(ColumnStr("__provenance__1")))

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=("col1", "colA", "__provenance__2"),
        iterable=[(i % 5, i * 3, i) for i in range(20)],
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, C_(ColumnStr("__provenance__2")))

    s = Product((pset_r1, pset_r2))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == C_(ColumnStr("__provenance__1"))
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
        C_(ColumnStr("__provenance__")),
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
        C_(ColumnStr("__provenance__")),
    )

    expected = NamedRelationalAlgebraFrozenSet(
        iterable=[("a", "b", 2), ("b", "a", 6), ("c", "a", 4), ("d", "a", 2)],
        columns=["x", "y", "__provenance__"],
    )

    s = Union(relation1, relation2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == expected


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
        C_(ColumnStr("__provenance__")),
    )
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[("a", "b", 1), ("b", "a", 3), ("c", "a", 3)],
            columns=["x", "y", "__provenance__"],
        ),
        C_(ColumnStr("__provenance__")),
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
            iterable=[(5, 1, 1), (6, 2, 2), (7, 3, 2), (1, 3, 1), (2, 1, 1),],
            columns=["x", "y", "__provenance__"],
        ),
        C_(ColumnStr("__provenance__")),
    )

    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                (5, 1, 6, 1),
                (6, 2, 8, 2),
                (7, 3, 10, 2),
                (1, 3, 4, 1),
                (2, 1, 3, 1),
            ],
            columns=["x", "y", "sum", "__provenance__"],
        ),
        C_(ColumnStr("__provenance__")),
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
        Constant(ColumnStr("myprov")),
    )
    projection = Projection(relation, (Constant(ColumnStr("x")),))
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(projection)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[(1.0, "a"), (0.8, "b"),], columns=["myprov", "x"],
        ),
        Constant(ColumnStr("myprov")),
    )
    assert equal_prov_sets(result, expected)
