import operator

import pytest

from ...datalog.expression_processing import UnifyVariableEqualitiesMixin
from ...exceptions import UnexpectedExpressionError
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction, Implication
from ..cplogic.program import CPLogicProgram
from ..probabilistic_ra_utils import (
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query,
)
from ..shattering import shatter_easy_probfacts

EQ = Constant(operator.eq)

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
ans = Symbol("ans")
a = Constant("a")
b = Constant("b")
d = Constant("d")


class CPLogicProgramWithVarEqUnification(
    UnifyVariableEqualitiesMixin,
    CPLogicProgram,
):
    pass


def test_no_constant():
    query = Implication(ans(x, y), P(x, y))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 1
    assert isinstance(shattered.antecedent.formulas[0], FunctionApplication)
    assert isinstance(
        shattered.antecedent.formulas[0].functor, ProbabilisticFactSet
    )
    assert shattered.antecedent.formulas[0].args == (x, y)


def test_one_constant_one_var():
    query = Implication(ans(y), P(a, y))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 1
    assert isinstance(shattered.antecedent.formulas[0], FunctionApplication)
    assert isinstance(
        shattered.antecedent.formulas[0].functor, ProbabilisticFactSet
    )
    assert shattered.antecedent.formulas[0].args == (y,)
    assert shattered.antecedent.formulas[0].functor.relation in symbol_table


def test_query_shattering_self_join():
    query = Implication(ans(x), Conjunction((P(a, x), P(b, x))))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert all(shattered.antecedent.formulas[i].args == (x,) for i in (0, 1))


def test_query_shattering_self_join_different_variables():
    query = Implication(ans(x, y), Conjunction((P(a, x), P(b, y))))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 2
    assert any(shattered.antecedent.formulas[i].args == (x,) for i in (0, 1))
    assert any(shattered.antecedent.formulas[i].args == (y,) for i in (0, 1))


def test_query_shattering_not_easy():
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    with pytest.raises(UnexpectedExpressionError):
        query = Implication(ans(x, y), Conjunction((P(a, x), P(a, y))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Implication(ans(x, y), Conjunction((P(a, x), P(y, a))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Implication(ans(x), Conjunction((P(a, x), P(x, a))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Implication(ans(x, y, z), Conjunction((P(x, y), P(a, z))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(UnexpectedExpressionError):
        query = Implication(ans(x, y, z), Conjunction((P(x, y), P(z, z))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)


def test_shattering_duplicated_predicate():
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    query = Implication(ans(x, y), Conjunction((P(x, y), P(x, y))))
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 1
    assert shattered.antecedent.formulas[0].args == (x, y)


def test_predicates_with_more_than_two_parameters():
    query = Implication(ans(x, y), Conjunction((P(a, x, b, y), P(b, x, d, y))))
    cpl = CPLogicProgramWithVarEqUnification()
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
    assert isinstance(shattered, Implication)


def test_predicates_with_more_than_one_self_join():
    query = Implication(
        ans(x), Conjunction((P(a, x, b), P(b, x, d), P(d, x, a)))
    )
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [
            (0.2, "a", "b", "c"),
            (1.0, "a", "c", "b"),
            (0.7, "b", "b", "d"),
        ],
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)


def test_shattering_with_variable_equality():
    query = Implication(ans(x, y, z), Conjunction((P(x, y, z), EQ(x, a))))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [
            (0.2, "a", "b", "c"),
            (1.0, "a", "c", "b"),
            (0.7, "b", "b", "d"),
        ],
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 2
    assert any(
        isinstance(formula.functor, ProbabilisticFactSet)
        and formula.args == (x, y, z)
        for formula in shattered.antecedent.formulas
    )
    assert any(
        formula.functor == EQ and formula.args == (x, a)
        for formula in shattered.antecedent.formulas
    )


def test_shattering_with_reversed_variable_equality():
    query = Implication(ans(x, y, z), Conjunction((P(x, y, z), EQ(a, x))))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [
            (0.2, "a", "b", "c"),
            (1.0, "a", "c", "b"),
            (0.7, "b", "b", "d"),
        ],
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 2
    assert any(
        isinstance(formula.functor, ProbabilisticFactSet)
        and formula.args == (x, y, z)
        for formula in shattered.antecedent.formulas
    )
    assert any(
        formula.functor == EQ and formula.args == (a, x)
        for formula in shattered.antecedent.formulas
    )


def test_shattering_between_symbol_equalities():
    query = Implication(
        ans(x, y, z), Conjunction((P(x, y, z), EQ(x, y), EQ(y, z)))
    )
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [
            (0.2, "a", "b", "c"),
            (1.0, "a", "c", "b"),
            (0.7, "b", "b", "b"),
        ],
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert len(shattered.antecedent.formulas) == 3
    assert any(
        isinstance(formula.functor, ProbabilisticFactSet)
        and len(set(formula.args)) == 3
        for formula in shattered.antecedent.formulas
    )
    assert any(
        formula.functor == EQ and formula.args == (x, y)
        for formula in shattered.antecedent.formulas
    )
    assert any(
        formula.functor == EQ and formula.args == (y, z)
        for formula in shattered.antecedent.formulas
    )
