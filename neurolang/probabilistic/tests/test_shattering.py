import pytest

from ...exceptions import UnexpectedExpressionError
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction
from ..cplogic.program import CPLogicProgram
from ..probabilistic_ra_utils import (
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query,
)
from ..shattering import shatter_easy_probfacts

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")


def test_no_constant():
    query = P(x, y)
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)
    assert len(shattered.formulas) == 1
    assert isinstance(shattered.formulas[0], FunctionApplication)
    assert isinstance(shattered.formulas[0].functor, ProbabilisticFactSet)
    assert shattered.formulas[0].args == (x, y)


def test_one_constant_one_var():
    query = P(a, x)
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)
    assert len(shattered.formulas) == 1
    assert isinstance(shattered.formulas[0], FunctionApplication)
    assert isinstance(shattered.formulas[0].functor, ProbabilisticFactSet)
    assert shattered.formulas[0].args == (x,)
    assert shattered.formulas[0].functor.relation in symbol_table


def test_query_shattering_self_join():
    query = Conjunction((P(a, x), P(b, x)))
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)


def test_query_shattering_not_easy():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(x), P(y)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(a, x), P(a, y)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(a, x), P(y, a)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(a), P(x)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
