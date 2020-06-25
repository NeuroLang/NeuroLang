from ...expressions import Symbol
from ...logic import Conjunction, Implication, Union
from ...expression_walker import ExpressionWalker
from ..basic_representation import DatalogProgram
from ..expression_processing import flatten_query

P = Symbol("P")
R = Symbol("R")
Z = Symbol("Z")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")


class TestDatalogProgram(DatalogProgram, ExpressionWalker):
    pass


def test_flatten_extensional_predicate_query():
    result = flatten_query(P(x), TestDatalogProgram())
    assert result == Conjunction((P(x),))


def test_flatten_with_renaming():
    code = Union(
        (
            Implication(Z(x), P(x)),
            Implication(R(y), P(y)),
            Implication(Q(y), Conjunction((Z(y), R(x)))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    result = flatten_query(Q(x), program)
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert any(
        pred.functor == P and pred.args[0] != x for pred in result.formulas
    )


def test_flatten_query_multiple_preds():
    query = Conjunction((P(x), Z(y)))
    result = flatten_query(query, TestDatalogProgram())
    assert len(result.formulas) == 2
    assert P(x) in result.formulas
    assert Z(y) in result.formulas
