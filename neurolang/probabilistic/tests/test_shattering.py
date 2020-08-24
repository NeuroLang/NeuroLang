import pytest

from ...exceptions import UnexpectedExpressionError
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction
from ..cplogic.program import CPLogicProgram
from ..shattering import shatter_easy_probfacts

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")


def test_query_shattering_single_predicate():
    query = P(x, y)
    shattered = shatter_easy_probfacts(query, CPLogicProgram())
    assert shattered == query
    query = P(a, x)
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    shattered = shatter_easy_probfacts(query, cpl)
    assert isinstance(shattered, FunctionApplication)
    assert shattered.functor.name.startswith("fresh_")
    assert shattered.args == (x,)
    assert shattered.functor in cpl.symbol_table
    assert shattered.functor in cpl.pfact_pred_symbs


def test_query_shattering_self_join():
    query = Conjunction((P(a, x), P(b, x)))
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    shattered = shatter_easy_probfacts(query, CPLogicProgram())
    assert isinstance(shattered, Conjunction)


def test_query_shattering_not_easy():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(x), P(y)))
        shatter_easy_probfacts(query, cpl)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(a, x), P(a, y)))
        shatter_easy_probfacts(query, cpl)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(a, x), P(y, a)))
        shatter_easy_probfacts(query, cpl)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(a), P(x)))
        shatter_easy_probfacts(query, cpl)
