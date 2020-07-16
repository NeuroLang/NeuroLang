from ..probabilistic.cplogic import testing
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
    Projection,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceExpressionSemringSolver,
)


def test_integer_addition_semiring():
    r1 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(2, "a"), (3, "b")]),
        str2columnstr_constant("_p_"),
    )
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(5, "a"), (10, "c")]),
        str2columnstr_constant("_p_"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(("_p_", "x"), [(10, "a")]),
        str2columnstr_constant("_p_"),
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
        str2columnstr_constant("_p_"),
    )
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"),
            [(SetType({"c"}), "hello"), (SetType({"c", "a"}), "zoo")],
        ),
        str2columnstr_constant("_p_"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(SetType({"a", "b", "c"}), "hello")]
        ),
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
        str2columnstr_constant("_p_"),
    )
    r2 = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "word"),
            [(StringTestType("he"), "my"), (StringTestType("the"), "name")],
        ),
        str2columnstr_constant("_p_"),
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
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_multiple_columns():
    r = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x", "y"),
            [(42, "a", "b"), (21, "a", "z"), (12, "b", "y"), (89, "b", "h"),],
        ),
        str2columnstr_constant("_p_"),
    )
    op = Projection(r, (str2columnstr_constant("x"),))
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            ("_p_", "x"), [(63, "a"), (101, "b"),],
        ),
        str2columnstr_constant("_p_"),
    )
    assert testing.eq_prov_relations(result, expected)
