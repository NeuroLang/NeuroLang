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
d = Constant("d")


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
    assert all(shattered.formulas[i].args == (x,) for i in (0, 1))


def test_query_shattering_self_join_different_variables():
    query = Conjunction((P(a, x), P(b, y)))
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)
    assert len(shattered.formulas) == 2
    assert any(shattered.formulas[i].args == (x,) for i in (0, 1))
    assert any(shattered.formulas[i].args == (y,) for i in (0, 1))


def test_query_shattering_not_easy():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
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
        query = Conjunction((P(a, x), P(x, a)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(x, y), P(a, z)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Conjunction((P(x, y), P(z, z)))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)


def test_shattering_duplicated_predicate():
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    query = Conjunction((P(x, y), P(x, y)))
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)
    assert len(shattered.formulas) == 1
    assert shattered.formulas[0].args == (x, y)


def test_predicates_with_more_than_two_parameters():
    query = Conjunction((P(a, x, b, y), P(b, x, d, y)))
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [
            (0.2, "a", "b", "c", "d"),
            (1.0, "a", "c", "b", "f"),
            (0.7, "b", "b", "d", "g"),
        ],
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)


def test_predicates_with_more_than_one_self_join():
    query = Conjunction((P(a, x, b), P(b, x, d), P(d, x, a)))
    cpl = CPLogicProgram()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [(0.2, "a", "b", "c",), (1.0, "a", "c", "b",), (0.7, "b", "b", "d",),],
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Conjunction)
