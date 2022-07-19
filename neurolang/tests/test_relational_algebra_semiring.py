import operator
from typing import AbstractSet

import pytest

from ..config import config
from ..expressions import Constant
from ..probabilistic.cplogic import testing
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    Difference,
    ExtendedProjection,
    FunctionApplicationListMember,
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    NumberColumns,
    Projection,
    RelationalAlgebraSet,
    RenameColumn,
    Selection,
    Union,
    int2columnint_constant,
    str2columnstr_constant
)
from ..relational_algebra_provenance import (
    BuildProvenanceAlgebraSetWalkIntoMixin,
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolverMixin,
    RelationalAlgebraSolver
)

EQ = Constant(operator.eq)


class RelationalAlgebraProvenanceExpressionSemringSolver(
    BuildProvenanceAlgebraSetWalkIntoMixin,
    RelationalAlgebraProvenanceExpressionSemringSolverMixin,
    RelationalAlgebraSolver
):
    pass


def test_integer_addition_semiring():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(2, "a"), (3, "b")]),
        ColumnStr("_p_"),
    )
    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(5, "a"), (10, "c")]),
        ColumnStr("_p_"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)

    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(10, "a")])
        ),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


class SetType(frozenset):
    def __add__(self, other):
        return self.union(other)

    def __mul__(self, other):
        return self.union(other)


@pytest.mark.skipif(
    config["RAS"].get("backend") == "dask",
    reason="multiplication of sets not yet implemented in dask backend",
)
def test_set_type_semiring():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"),
            [(SetType({"a", "b"}), "hello"), (SetType({"c", "a"}), "bonjour")],
        ),
        ColumnStr("_p_"),
    )
    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"),
            [(SetType({"c"}), "hello"), (SetType({"c", "a"}), "zoo")],
        ),
        ColumnStr("_p_"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(SetType({"a", "b", "c"}), "hello")]
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


class StringTestType:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return StringTestType(self.value + other.value)

    def __mul__(self, other):
        return StringTestType(self.value * len(other.value))

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


@pytest.mark.skipif(
    config["RAS"].get("backend") == "dask",
    reason="multiplication of strings not yet implemented in dask backend",
)
def test_string_semiring():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "word"),
            [
                (StringTestType("walter"), "say"),
                (StringTestType("white"), "my"),
                (StringTestType("heisenberg"), "name"),
            ],
        ),
        ColumnStr("_p_"),
    )
    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "word"),
            [(StringTestType("he"), "my"), (StringTestType("the"), "name")],
        ),
        ColumnStr("_p_"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "word"),
            [
                (StringTestType("white" * 2), "my"),
                (StringTestType("heisenberg" * 3), "name"),
            ],
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_multiple_columns():
    r = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [(42, "a", "b"), (21, "a", "z"), (12, "b", "y"), (89, "b", "h"),],
        ),
        ColumnStr("_p_"),
    )
    op = Projection(r, (str2columnstr_constant("x"),))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(63, "a"), (101, "b"),],
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_renaming():
    r = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [(42, "a", "b"), (21, "a", "z"), (12, "b", "y"), (89, "b", "h"),],
        ),
        ColumnStr("_p_"),
    )
    op = RenameColumn(
        r, str2columnstr_constant("x"), str2columnstr_constant("z"),
    )
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "z", "y"),
            [(42, "a", "b"), (21, "a", "z"), (12, "b", "y"), (89, "b", "h"),],
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_selection():
    r = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [(42, "a", "b"), (21, "a", "z"), (12, "b", "y"), (89, "b", "h"),],
        ),
        ColumnStr("_p_"),
    )
    op = Selection(
        r, Constant(operator.eq)(str2columnstr_constant("x"), Constant("a"))
    )
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"), [(42, "a", "b"), (21, "a", "z")],
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_union():
    r1 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(("_p1_", "x"), [(2, "a"), (3, "b")]),
        ColumnStr("_p1_"),
    )
    r2 = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(("_p2_", "x"), [(5, "b"), (10, "c")]),
        ColumnStr("_p2_"),
    )
    op = Union(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(2, "a"), (8, "b"), (10, "c")]
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_selection_by_columnint():
    r = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"), [(0.2, "a", "b"), (0.5, "b", "a")]
        ),
        ColumnStr("_p_"),
    )
    op = Selection(r, EQ(Constant(ColumnInt(0)), Constant("a")))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                ("_p_", "x", "y"), [(0.2, "a", "b")]
            )
        ),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)
    op = Selection(r, EQ(Constant(ColumnInt(0)), Constant("a")))
    op = Selection(op, EQ(Constant(ColumnInt(1)), Constant("b")))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [(0.2, "a", "b")])
        ),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_projection_columnint():
    r = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"), [(0.2, "a", "b"), (0.5, "b", "a")]
        ),
        ColumnStr("_p_"),
    )

    r = NumberColumns(r, (str2columnstr_constant("x"), str2columnstr_constant("y")))
    op = Projection(r, (Constant(ColumnInt(0)),))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](RelationalAlgebraSet(
            [(0.2, "a"), (0.5, "b")]
        )),
        int2columnint_constant(0),
    )
    assert result == expected
    op = Projection(r, (Constant(ColumnInt(0)), Constant(ColumnInt(1))))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    assert result == solver.walk(r)
    op = Projection(r, (Constant(ColumnInt(1)), Constant(ColumnInt(0))))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](RelationalAlgebraSet(
            [(0.2, "b", "a"), (0.5, "a", "b")]
        )),
        int2columnint_constant(0)
    )
    assert result == expected


def test_selection_between_columnints():
    r = testing.build_ra_provenance_set_from_named_ra_set(
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
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y"),
            iterable=[
                (0.7, "a", "a"),
                (0.2, "b", "b"),
            ],
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_difference():
    r_left = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p1_", "x", "y", "w"),
            iterable=[
                (0.5, "a", "b", "a"),
                (0.7, "a", "a", "b"),
                (0.2, "b", "b", "c"),
            ],
        ),
        ColumnStr("_p1_"),
    )

    r_right = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p2_", "x", "y", "z"),
            iterable=[
                (0.2, "a", "b", "a"),
                (0.3, "b", "a", "b"),
                (0.1, "b", "b", "c"),
            ],
        ),
        ColumnStr("_p2_"),
    )

    r_expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            columns=("_p1_", "x", "y", "w"),
            iterable=[
                (0.4, "a", "b", "a"),
                (0.7, "a", "a", "b"),
                (0.18, "b", "b", "c"),
            ],
        )),
        str2columnstr_constant("_p1_"),
    )

    op = Projection(
        Difference(r_left, r_right),
        tuple(str2columnstr_constant(c) for c in ("x", "y", "w"))
    )
    result = RelationalAlgebraProvenanceExpressionSemringSolver().walk(op)
    assert testing.eq_prov_relations(result, r_expected)


def test_difference_same_provenance_column():
    r_left = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y", "w"),
            iterable=[
                (0.5, "a", "b", "a"),
                (0.3, "a", "a", "b"),
                (0.2, "b", "b", "c"),
            ],
        ),
        ColumnStr("_p_"),
    )

    r_right = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y", "z"),
            iterable=[
                (0.7, "a", "b", "a"),
                (0.8, "b", "a", "b"),
                (0.9, "b", "b", "c"),
            ],
        ),
        ColumnStr("_p_"),
    )

    r_expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            columns=("_p_", "x", "y", "w"),
            iterable=[
                (0.15, "a", "b", "a"),
                (0.3, "a", "a", "b"),
                (0.02, "b", "b", "c"),
            ],
        )),
        str2columnstr_constant("_p_"),
    )

    op = Projection(
        Difference(r_left, r_right),
        tuple(str2columnstr_constant(c) for c in ("x", "y", "w"))
    )
    result = RelationalAlgebraProvenanceExpressionSemringSolver().walk(op)
    assert testing.eq_prov_relations(result, r_expected)


def test_extended_proj():
    provset = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [
                (0.2, "a", "b"),
                (0.3, "b", "a"),
                (0.5, "c", "c"),
            ],
        ),
        ColumnStr("_p_"),
    )
    proj_list = [
        FunctionApplicationListMember(
            str2columnstr_constant("x"), str2columnstr_constant("x")
        ),
        FunctionApplicationListMember(
            str2columnstr_constant("y"), str2columnstr_constant("y")
        ),
        FunctionApplicationListMember(
            Constant("d"), str2columnstr_constant("z")
        ),
    ]
    proj = ExtendedProjection(provset, proj_list)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(proj)
    expected = ProvenanceAlgebraSet(
        Constant[AbstractSet](NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y", "z"),
            [
                (0.2, "a", "b", "d"),
                (0.3, "b", "a", "d"),
                (0.5, "c", "c", "d"),
            ],
        )),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_forbidden_extended_proj_missing_nonprov_cols():
    provset = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [
                (0.2, "a", "b"),
                (0.3, "b", "a"),
                (0.5, "c", "c"),
            ],
        ),
        ColumnStr("_p_"),
    )
    proj_list = [
        FunctionApplicationListMember(
            str2columnstr_constant("x"), str2columnstr_constant("x")
        ),
        FunctionApplicationListMember(
            Constant("d"), str2columnstr_constant("z")
        ),
    ]
    proj = ExtendedProjection(provset, proj_list)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    with pytest.raises(ValueError):
        solver.walk(proj)


def test_forbidden_extended_proj_on_provcol():
    provset = testing.build_ra_provenance_set_from_named_ra_set(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [
                (0.2, "a", "b"),
                (0.3, "b", "a"),
                (0.5, "c", "c"),
            ],
        ),
        ColumnStr("_p_"),
    )
    proj_list = [
        FunctionApplicationListMember(
            str2columnstr_constant("x"), str2columnstr_constant("x")
        ),
        FunctionApplicationListMember(
            str2columnstr_constant("y"), str2columnstr_constant("y")
        ),
        FunctionApplicationListMember(
            Constant("d"), str2columnstr_constant("_p_")
        ),
    ]
    proj = ExtendedProjection(provset, proj_list)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    with pytest.raises(ValueError):
        solver.walk(proj)
