import operator

from ..expressions import Constant
from ..probabilistic.cplogic import testing
from ..relational_algebra import (
    ColumnInt,
    ColumnStr,
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    RenameColumn,
    Selection,
    Union,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolver,
)

EQ = Constant(operator.eq)


def test_integer_addition_semiring():
    r1 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(2, "a"), (3, "b")]),
        ColumnStr("_p_"),
    )
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(5, "a"), (10, "c")]),
        ColumnStr("_p_"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(10, "a")]),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


class SetType(frozenset):
    def __add__(self, other):
        return self.union(other)

    def __mul__(self, other):
        return self.union(other)


def test_set_type_semiring():
    r1 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"),
            [(SetType({"a", "b"}), "hello"), (SetType({"c", "a"}), "bonjour")],
        ),
        ColumnStr("_p_"),
    )
    r2 = ProvenanceAlgebraSet(
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
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(SetType({"a", "b", "c"}), "hello")]
        ),
        ColumnStr("_p_"),
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


def test_string_semiring():
    r1 = ProvenanceAlgebraSet(
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
    r2 = ProvenanceAlgebraSet(
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
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "word"),
            [
                (StringTestType("white" * 2), "my"),
                (StringTestType("heisenberg" * 3), "name"),
            ],
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_multiple_columns():
    r = ProvenanceAlgebraSet(
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
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(63, "a"), (101, "b"),],
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_renaming():
    r = ProvenanceAlgebraSet(
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
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "z", "y"),
            [(42, "a", "b"), (21, "a", "z"), (12, "b", "y"), (89, "b", "h"),],
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_selection():
    r = ProvenanceAlgebraSet(
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
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"), [(42, "a", "b"), (21, "a", "z")],
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_union():
    r1 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p1_", "x"), [(2, "a"), (3, "b")]),
        ColumnStr("_p1_"),
    )
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p2_", "x"), [(5, "b"), (10, "c")]),
        ColumnStr("_p2_"),
    )
    op = Union(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(2, "a"), (8, "b"), (10, "c")]
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_selection_by_columnint():
    r = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"), [(0.2, "a", "b"), (0.5, "b", "a")]
        ),
        ColumnStr("_p_"),
    )
    op = Selection(r, EQ(Constant(ColumnInt(0)), Constant("a")))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x", "y"), [(0.2, "a", "b")]),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)
    op = Selection(r, EQ(Constant(ColumnInt(0)), Constant("a")))
    op = Selection(op, EQ(Constant(ColumnInt(1)), Constant("b")))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x", "y"), [(0.2, "a", "b")]),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_projection_columnint():
    r = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"), [(0.2, "a", "b"), (0.5, "b", "a")]
        ),
        ColumnStr("_p_"),
    )
    op = Projection(r, (Constant(ColumnInt(0)),))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(0.2, "a"), (0.5, "b")]
        ),
        ColumnStr("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)
    op = Projection(r, (Constant(ColumnInt(0)), Constant(ColumnInt(1))))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    assert testing.eq_prov_relations(result, r)
    op = Projection(r, (Constant(ColumnInt(1)), Constant(ColumnInt(0))))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "y", "x"), [(0.2, "b", "a"), (0.5, "a", "b")]
        ),
        ColumnStr("_p_"),
    )
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
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
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
