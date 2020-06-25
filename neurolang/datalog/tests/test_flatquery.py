from ...expressions import Symbol
from ...logic import Conjunction, Implication
from ..expression_processing import flatten_query

P = Symbol("P")
R = Symbol("R")
Z = Symbol("Z")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")


def test_flatten_extensional_predicate_query():
    result = flatten_query(P(x), dict())
    assert result == Conjunction((P(x),))


def test_flatten_with_renaming():
    idb = {
        Z: Implication(Z(x), P(x)),
        R: Implication(R(y), P(y)),
        Q: Implication(Q(y), Conjunction((Z(y), R(x)))),
    }
    result = flatten_query(Q(x), idb)
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert any(
        pred.functor == P and pred.args[0] != x for pred in result.formulas
    )


def test_flatten_query_multiple_preds():
    query = Conjunction((P(x), Z(y)))
    result = flatten_query(query, set())
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert Z(y) in result.formulas
