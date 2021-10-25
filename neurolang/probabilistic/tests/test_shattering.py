import operator

import pytest
from neurolang.probabilistic.exceptions import NotEasilyShatterableError

from ...datalog.expression_processing import UnifyVariableEqualitiesMixin
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Negation,
)
from ..cplogic.program import CPLogicProgram
from ..probabilistic_ra_utils import (
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query,
)
from ..shattering import shatter_easy_probfacts

EQ = Constant(operator.eq)
NE = Constant(operator.ne)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
w = Symbol("w")
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
    assert isinstance(shattered.antecedent, FunctionApplication)
    assert isinstance(shattered.antecedent.functor, ProbabilisticFactSet)
    assert shattered.antecedent.args == (x, y)


def test_one_constant_one_var():
    query = Implication(ans(y), P(a, y))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered, Implication)
    assert isinstance(shattered.antecedent, FunctionApplication)
    assert isinstance(shattered.antecedent.functor, ProbabilisticFactSet)
    assert shattered.antecedent.args == (y,)
    assert shattered.antecedent.functor.relation in symbol_table


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
    with pytest.raises(NotEasilyShatterableError):
        query = Implication(ans(x, y), Conjunction((P(a, x), P(a, y))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(NotEasilyShatterableError):
        query = Implication(ans(x, y), Conjunction((P(a, x), P(y, a))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(NotEasilyShatterableError):
        query = Implication(ans(x), Conjunction((P(a, x), P(x, a))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(NotEasilyShatterableError):
        query = Implication(ans(x, y, z), Conjunction((P(x, y), P(a, z))))
        symbol_table = generate_probabilistic_symbol_table_for_query(
            cpl, query
        )
        shatter_easy_probfacts(query, symbol_table)
    with pytest.raises(NotEasilyShatterableError):
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
    assert shattered.antecedent.args == (x, y)


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


def test_cannot_shatter_dependent_disjuncts():
    """
    In this query

    ans(x, y) :- ( P(a, x) ^ Q(x, y) ) v ( P(a, x) ^ P(b, x) ^ Q(y, hello) )

    the two disjuncts are not independent because they have an overlapping
    ground atom Q(x, hello) with x == y.

    Therefore the shattering should raise a NotEasilyShatterableError
    """
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P,
        [
            (0.2, 'a', 42),
            (0.7, 'b', 19),
            (0.1, 'a', 10),
            (0.9, 'b', 99),
            (0.888, 'b', 42),
        ],
    )
    cpl.add_probabilistic_facts_from_tuples(
        Q,
        [
            (0.777, 42, 'hello'),
            (0.666, 42, 'bonjour'),
            (0.999, 19, 'ciao'),
        ],
    )
    query = Implication(
        ans(x, y),
        Disjunction(
            (
                Conjunction((P(a, x), Q(x, y))),
                Conjunction((Q(y, Constant('hello')), P(b, y), P(a, x))),
            )
        )
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    with pytest.raises(NotEasilyShatterableError):
        shatter_easy_probfacts(query, symbol_table)


def test_shattering_negation():
    query = Implication(ans(x, y), Negation(P(x, y)))
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered.antecedent, Negation)
    assert isinstance(
        shattered.antecedent.formula.functor, ProbabilisticFactSet
    )
    assert shattered.antecedent.formula.functor.relation.is_fresh


def test_shatter_disjunction_same_shattering_relation():
    query = Implication(
        ans(x, y),
        Disjunction(
            (
                Conjunction((P(a, y), Q(x))),
                Conjunction((P(a, y), R(x))),
            )
        )
    )
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        P, [(0.2, "a", "b"), (1.0, "a", "c"), (0.7, "b", "b")]
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    shattered = shatter_easy_probfacts(query, symbol_table)
    assert isinstance(shattered.antecedent, Disjunction)
    assert len(shattered.antecedent.formulas) == 2
    R_formula = next(
        formula for formula in shattered.antecedent.formulas
        if isinstance(formula, Conjunction)
        and any(
            isinstance(f, FunctionApplication)
            and f.functor == R
            for f in formula.formulas
        )
    )
    assert len(R_formula.formulas) == 2
    Q_formula = next(
        formula for formula in shattered.antecedent.formulas
        if isinstance(formula, Conjunction)
        and any(
            isinstance(f, FunctionApplication)
            and f.functor == Q
            for f in formula.formulas
        )
    )
    assert len(Q_formula.formulas) == 2
    shattered_in_R = next(
        formula for formula in R_formula.formulas
        if isinstance(formula.functor, ProbabilisticFactSet)
        and formula.functor.relation.is_fresh
    )
    shattered_in_Q = next(
        formula for formula in Q_formula.formulas
        if isinstance(formula.functor, ProbabilisticFactSet)
        and formula.functor.relation.is_fresh
    )
    assert shattered_in_R.functor.relation == shattered_in_Q.functor.relation


def test_shatter_negexist_segregation_query():
    SegQuery = Symbol("SegQuery")
    NetworkReported = Symbol("NetworkReported")
    SelectedStudy = Symbol("SelectedStudy")
    n = Symbol("n")
    n2 = Symbol("n2")
    s = Symbol("s")
    cpl = CPLogicProgramWithVarEqUnification()
    cpl.add_probabilistic_facts_from_tuples(
        NetworkReported,
        [
            (0.1, "A", "s1"),
            (0.2, "B", "s1"),
            (0.7, "C", "s1"),
            (0.8, "A", "s2"),
            (0.3, "B", "s2"),
            (0.4, "C", "s2"),
        ],
    )
    cpl.add_probabilistic_choice_from_tuples(
        SelectedStudy,
        [
            (1 / 3, "s1"),
            (1 / 3, "s2"),
            (1 / 3, "s3"),
        ],
    )
    query = Implication(
        SegQuery(n),
        Conjunction(
            (
                NetworkReported(n, s),
                SelectedStudy(s),
                Negation(
                    ExistentialPredicate(
                        n2,
                        Conjunction(
                            (
                                NE(n2, n),
                                NetworkReported(n2, s),
                            )
                        ),
                    )
                )
            )
        )
    )
    symbol_table = generate_probabilistic_symbol_table_for_query(cpl, query)
    res = shatter_easy_probfacts(query, symbol_table)
    breakpoint()


def test_shatter_negexist_shatterable_cross_product():
    query = Implication(
        ans(x, y, w),
        Conjunction(
            (
                R(x, y),
                Negation(
                    ExistentialPredicate(
                        z,
                        Conjunction(
                            (
                                R(w, z),
                                NE(z, y),
                            )
                        ),
                    )
                )
            )
        )
    )
