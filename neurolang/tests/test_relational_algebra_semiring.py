from ..probabilistic.cplogic import testing
from ..relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    NaturalJoin,
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
