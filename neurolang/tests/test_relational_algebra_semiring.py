from ..probabilistic.cplogic import testing
from ..relational_algebra import NaturalJoin
from ..relational_algebra_provenance import (
    RelationalAlgebraProvenanceExpressionSemringSolver,
)


def test_integer_addition_semiring():
    r1 = testing.make_prov_set([(2, "a"), (3, "b"),], ("_p_", "x"))
    r2 = testing.make_prov_set([(5, "a"), (10, "c"),], ("_p_", "x"))
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = testing.make_prov_set([(10, "a")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


class SetType(frozenset):
    def __add__(self, other):
        return self.union(other)

    def __mul__(self, other):
        return self.union(other)


def test_set_type_semiring():
    r1 = testing.make_prov_set(
        [(SetType({"a", "b"}), "hello"), (SetType({"c", "a"}), "bonjour")],
        ("_p_", "x"),
    )
    r2 = testing.make_prov_set(
        [(SetType({"c"}), "hello"), (SetType({"c", "a"}), "zoo")],
        ("_p_", "x"),
    )
    op = NaturalJoin(r1, r2)
    solver = RelationalAlgebraProvenanceExpressionSemringSolver()
    result = solver.walk(op)
    expected = testing.make_prov_set(
        [(SetType({"a", "b", "c"}), "hello")], ("_p_", "x")
    )
    assert testing.eq_prov_relations(result, expected)
