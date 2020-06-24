from ....expressions import Symbol
from ....logic import Conjunction, Implication
from ..flatquery import flatten_query
from ..program import CPLogicProgram

P = Symbol("P")
R = Symbol("R")
Z = Symbol("Z")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")


def test_flatten_extensional_predicate_query():
    cpl = CPLogicProgram()
    cpl.add_extensional_predicate_from_tuples(P, {("a",), ("b",)})
    result = flatten_query(P(x), cpl.intensional_database())
    assert result == Conjunction((P(x),))


def test_flatten_with_renaming():
    cpl = CPLogicProgram()
    cpl.add_extensional_predicate_from_tuples(P, {("a",), ("b",)})
    cpl.walk(Implication(Z(x), P(x)))
    cpl.walk(Implication(R(y), P(y)))
    cpl.walk(Implication(Q(y), Conjunction((Z(y), R(x)))))
    result = flatten_query(Q(x), cpl.intensional_database())
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert any(
        pred.functor == P and pred.args[0] != x for pred in result.formulas
    )
