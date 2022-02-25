import operator
from typing import AbstractSet

import pytest
from pytest import fixture

from ...datalog.expression_processing import (
    UnifyVariableEqualitiesMixin,
    extract_logic_atoms
)
from ...datalog.translate_to_named_ra import TranslateToNamedRA
from ...datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ...expression_walker import add_match
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction, Disjunction, Implication, Negation
from ...relational_algebra import Projection, Selection, int2columnint_constant
from ...relational_algebra.optimisers import (
    PushUnnamedSelectionsUp,
    RelationalAlgebraOptimiser
)
from ...relational_algebra.relational_algebra import RelationalAlgebraSolver
from ..cplogic.program import CPLogicProgram
from ..exceptions import NotEasilyShatterableError
from ..probabilistic_ra_utils import (
    ProbabilisticFactSet,
    generate_probabilistic_symbol_table_for_query
)
from ..shattering import (
    sets_per_symbol,
    shatter_constants,
    shatter_easy_probfacts
)

RAO = RelationalAlgebraOptimiser()


EQ = Constant(operator.eq)
NE = Constant(operator.ne)

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
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


def test_symbols_per_set_unitary():
    query = Conjunction((Q(x), Q(a), Q(b), P(x), Negation(Q(a))))

    res = sets_per_symbol(query)

    c0 = int2columnint_constant(0)
    expected = {
        Q: set([
            Projection(Selection(Q, EQ(c0, a)), (c0,)),
            Projection(Selection(Q, EQ(c0, b)), (c0,)),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b)))),
                (int2columnint_constant(0),)
            )
        ])
    }

    assert res == expected


def test_symbols_per_set_unitary_ne():
    query = Conjunction((Q(x), Q(a), Q(b), P(x), Negation(Q(a)), NE(d, x)))

    res = sets_per_symbol(query)

    c0 = int2columnint_constant(0)
    expected = {
        Q: set([
            Projection(Selection(Q, EQ(c0, a)), (c0,)),
            Projection(Selection(Q, EQ(c0, b)), (c0,)),
            Projection(Selection(Q, EQ(c0, d)), (c0,)),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b), NE(c0, d)))),
                (c0,)
            ),
        ]),
        P: set([
            Projection(Selection(P, EQ(c0, d)), (c0,)),
            Projection(
                Selection(P, NE(c0, d)),
                (c0,)
            )

        ])
    }

    assert res == expected


def test_symbols_per_set_binary():
    query = Conjunction((Q(x, b), Q(a, x), Q(b, y), NE(y, b)))

    res = sets_per_symbol(query)

    c0 = int2columnint_constant(0)
    c1 = int2columnint_constant(1)
    expected = {
        Q: set([
            Projection(
                Selection(Q, Conjunction((EQ(c0, a), EQ(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((EQ(c0, b), EQ(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((EQ(c0, a), NE(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((EQ(c0, b), NE(c1, b)))), (c0, c1,)
            ),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b), EQ(c1, b)))),
                (c0, c1)
            ),
            Projection(
                Selection(Q, Conjunction((NE(c0, a), NE(c0, b), NE(c1, b)))),
                (c0, c1)
            )
        ])
    }

    assert res == expected


@fixture
def R1():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([(chr(ord('a') + i),) for i in range(10)])
    )


@fixture
def R2():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (chr(ord('a') + i * 2), i)
            for i in range(10)
        ])
    )


@fixture
def R3():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (i,)
            for i in range(10)
        ])
    )


@fixture
def R4():
    return Constant[AbstractSet](
        WrappedRelationalAlgebraSet([
            (
                chr(ord('a') + i * 2),
                chr(ord('a') + i)
            )
            for i in range(10)
        ])
    )


class RelationalAlgebraSelectionConjunction(
    PushUnnamedSelectionsUp, RelationalAlgebraSolver
):
    @add_match(Selection(..., Conjunction))
    def selection_conjunction(self, expression):
        and_ = Constant(operator.and_)
        formulas = expression.formula.formulas
        new_formula = formulas[0]
        for f in formulas[1:]:
            new_formula = and_(new_formula, f)
        return Selection(
            expression.relation,
            new_formula
        )


def test_shatter_unitary(R1, R2, R3):
    query = Conjunction((Q(x), Q(a), Q(b), R(y), Negation(P(b, y))))

    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R1, P: R2, R: R3})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table)

    ras = RelationalAlgebraSelectionConjunction(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_1(R4):
    query = Conjunction((Q(x, a), Q(y, a), Q(x, y)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table)

    ras = RelationalAlgebraSelectionConjunction(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_2(R1, R4):
    query = Conjunction((Q(x, b), Q(a, x), Q(x, y), R(z)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4, R: R1})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table)

    ras = RelationalAlgebraSelectionConjunction(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_3(R4):
    query = Conjunction((Q(x, b), Q(y, y)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table)

    ras = RelationalAlgebraSelectionConjunction(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol


def test_shatter_binary_inequality(R1, R4):
    query = Conjunction((Q(x, a), Q(a, x), Q(y, y), NE(x, b), R(z)))
    ra_query = TranslateToNamedRA().walk(query)

    symbol_table = dict({Q: R4, R: R1})
    ras = RelationalAlgebraSolver(symbol_table)
    ra_sol = ras.walk(ra_query)

    res = shatter_constants(query, symbol_table)

    ras = RelationalAlgebraSelectionConjunction(symbol_table)
    ra_query = TranslateToNamedRA().walk(res)
    shattered_sol = ras.walk(ra_query)

    assert all(
        all(isinstance(arg, Symbol) for arg in atom.args)
        for atom in extract_logic_atoms(res)
    )
    assert ra_sol == shattered_sol
