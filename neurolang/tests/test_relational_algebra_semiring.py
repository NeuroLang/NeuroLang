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
    expected = testing.make_prov_set([(7, "a")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)
