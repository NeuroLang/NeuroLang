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
z = Symbol("z")


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


def test_existential():
    code = Union(
        (
            Implication(P(x), Conjunction((Z(x, y), Z(y, x)))),
            Implication(Q(x), Conjunction((P(x), P(y), R(z)))),
        )
    )
    program = TestDatalogProgram()
    program.walk(code)
    query = Q(z)
    result = flatten_query(query, program)
    assert any(
        formula.functor == R and formula.args[0] != z
        for formula in result.formulas
    )
    assert any(
        formula.functor == Z
        and formula.args[0] == z
        and "fresh" in formula.args[1].name
        for formula in result.formulas
    )
    match = next(
        formula for formula in result.formulas
        if formula.functor == Z
        and "fresh" in formula.args[0].name
        and "fresh" in formula.args[1].name
    )
    expected = match.functor(*reversed(match.args))
    assert expected in result.formulas
