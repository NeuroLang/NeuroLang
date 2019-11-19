from typing import AbstractSet

import pytest

from ... import expression_walker, solver_datalog_extensional_db
from ...expressions import (Constant, ExpressionBlock, FunctionApplication,
                            Lambda, NeuroLangException, Query, Symbol,
                            is_leq_informative)
from ...logic import Disjunction, ExistentialPredicate, Implication
from ...solver_datalog_naive import NULL, SolverNonRecursiveDatalogNaive
from .. import Fact

S_ = Symbol
C_ = Constant
Imp_ = Implication
F_ = FunctionApplication
L_ = Lambda
EP_ = ExistentialPredicate
Q_ = Query
T_ = Fact


class Datalog(
    SolverNonRecursiveDatalogNaive,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_null_constant_resolves_to_undefined():
    dl = Datalog()
    P = S_('P')  # noqa: N806
    dl.walk(T_(P(C_('a'))))

    dl.walk(T_(P(C_('b'))))
    assert 'P' in dl.symbol_table
    res = dl.walk(P(NULL))
    assert isinstance(res, Constant)
    assert not res.value


def test_no_facts():
    dl = Datalog()
    x = S_('x')
    Q = S_('Q')  # noqa: N806

    res = dl.walk(Query(x, Q(x)))

    assert len(res.value) == 0


def test_facts_constants():
    dl = Datalog()

    f1 = T_(S_('Q')(C_(1), C_(2)))

    dl.walk(f1)

    assert 'Q' in dl.symbol_table
    assert isinstance(dl.symbol_table['Q'], Constant[AbstractSet])
    fact_set = dl.symbol_table['Q']
    assert isinstance(fact_set, Constant)
    assert is_leq_informative(fact_set.type, AbstractSet)
    expected_result = {C_((C_(1), C_(2)))}
    assert expected_result == fact_set.value

    f2 = T_(S_('Q')(C_(3), C_(4)))
    dl.walk(f2)
    assert (
        {C_((C_(1), C_(2))), C_((C_(3), C_(4)))} ==
        fact_set.value
    )

    f = S_('Q')(C_(1), C_(2))
    g = S_('Q')(C_(18), C_(23))

    assert dl.walk(f).value is True
    assert dl.walk(g).value is False


def test_atoms_variables():
    dl = Datalog()

    eq = S_('equals')
    x = S_('x')
    y = S_('y')
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    T = S_('T')  # noqa: N806

    f1 = Imp_(Q(x,), eq(x, x))

    dl.walk(f1)

    assert 'Q' in dl.symbol_table
    assert isinstance(dl.symbol_table['Q'], Disjunction)
    fact = dl.symbol_table['Q'].formulas[-1]
    assert isinstance(fact, Implication)
    assert isinstance(fact.consequent, FunctionApplication)
    assert fact.consequent.functor is Q
    assert fact.consequent.args == (x,)
    assert fact.antecedent == eq(x, x)

    f2 = Imp_(T(x, y), eq(x, y))

    dl.walk(f2)

    assert 'T' in dl.symbol_table
    assert isinstance(dl.symbol_table['T'], Disjunction)
    fact = dl.symbol_table['T'].formulas[-1]
    assert isinstance(fact, Implication)
    assert isinstance(fact.consequent, FunctionApplication)
    assert fact.consequent.functor is T
    assert fact.consequent.args == (x, y)
    assert fact.antecedent == eq(x, y)

    f3 = Imp_(R(x, C_(1)), eq(x, x))
    dl.walk(f3)

    assert 'R' in dl.symbol_table
    assert isinstance(dl.symbol_table['R'], Disjunction)
    fact = dl.symbol_table['R'].formulas[-1]
    assert isinstance(fact, Implication)
    assert isinstance(fact.consequent, FunctionApplication)
    assert fact.consequent.functor is R
    assert fact.consequent.args == (x, C_(1))
    assert fact.antecedent == eq(x, x)

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x), ...))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), eq(x, y)))

    f = Q(C_(10))
    g = T(C_(1), C_(5))
    h = T(C_(1), C_(1))
    i = R(C_(2), C_(1))
    g = R(C_(2), C_(2))

    assert dl.walk(f).value is True
    assert dl.walk(g).value is False
    assert dl.walk(h).value is True
    assert dl.walk(i).value is True
    assert dl.walk(g).value is False


def test_facts_intensional():
    dl = Datalog()

    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock((
        T_(Q(C_(1), C_(1))),
        T_(Q(C_(1), C_(2))),
        T_(Q(C_(1), C_(4))),
        T_(Q(C_(2), C_(4))),
    ))

    intensional = ExpressionBlock((
        Imp_(R(x, y, z), Q(x, y) & Q(y, z)),
        Imp_(T(x, z), EP_(y, Q(x, y) & Q(y, z))),
    ))

    dl.walk(extensional)
    dl.walk(intensional)

    res = dl.walk(R(C_(1), C_(2), C_(4)))
    assert res.value is True

    res = dl.walk(R(C_(1), C_(2), C_(5)))
    assert res.value is False

    res = dl.walk(T(C_(1), C_(4)))
    assert res.value is True

    res = dl.walk(T(C_(1), C_(5)))
    assert res.value is False

    with pytest.raises(NeuroLangException):
        res = dl.walk(Imp_(Q(x, y), Q(x)))


def test_query_single_element():
    solver = Datalog()
    # free variables
    x = S_('x')

    # constants
    a, b = C_('a'), C_('b')

    # predicates
    Q = S_('Q')  # noqa: N806

    extensional = ExpressionBlock((
        T_(Q(a)),
        T_(Q(b)),
    ))

    solver.walk(extensional)

    query = Query(x, Q(x))

    result = solver.walk(query)

    assert isinstance(result, Constant)
    assert result.value is not None
    assert result.value == {a, b}


def test_query_tuple():
    dl = Datalog()

    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock((
        T_(Q(C_(1), C_(1))),
        T_(Q(C_(1), C_(2))),
        T_(Q(C_(1), C_(4))),
        T_(Q(C_(2), C_(4))),
    ))

    intensional = ExpressionBlock((
        Imp_(R(x, y, z), Q(x, y) & Q(y, z)),
        Imp_(T(x, z), Q(x, y) & Q(y, z)),
        Imp_(S(C_(100), x), Q(x, x))
    ))

    dl.walk(extensional)
    dl.walk(intensional)

    query = Q_((x, y), T(x, y))
    res = dl.walk(query)

    assert res.value == set((
        C_((C_(1), C_(1))),
        C_((C_(1), C_(2))),
        C_((C_(1), C_(4))),
    ))

    query = Q_((x, y), S(x, y))
    res = dl.walk(query)

    assert res.value == set((
        C_((C_(100), C_(1))),
    ))


def test_extensional_database():

    dl = Datalog()

    Q = S_('Q')  # noqa: N806
    R0 = S_('R0')  # noqa: N806
    R = S_('R')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock((
        T_(Q(C_(1), C_(1))),
        T_(Q(C_(1), C_(2))),
        T_(Q(C_(1), C_(4))),
        T_(Q(C_(2), C_(4))),
        T_(R0(C_('a'), C_(1), C_(3))),
    ))

    intensional = ExpressionBlock((
        Imp_(R(x, y, z), R0(x, y, z)),
        Imp_(R(x, y, z), Q(x, y) & Q(y, z)),
        Imp_(T(x, z), Q(x, y) & Q(y, z)),
    ))

    dl.walk(extensional)

    edb = dl.extensional_database()

    assert edb.keys() == {'R0', 'Q'}

    assert edb['Q'] == C_(frozenset((
        C_((C_(1), C_(1))),
        C_((C_(1), C_(2))),
        C_((C_(1), C_(4))),
        C_((C_(2), C_(4))),
    )))

    assert edb['R0'] == C_(frozenset((
        C_((C_('a'), C_(1), C_(3))),
    )))

    dl.walk(intensional)
    edb = dl.extensional_database()

    assert edb.keys() == {'R0', 'Q'}

    assert edb['Q'] == C_(frozenset((
        C_((C_(1), C_(1))),
        C_((C_(1), C_(2))),
        C_((C_(1), C_(4))),
        C_((C_(2), C_(4))),
    )))

    assert edb['R0'] == C_(frozenset((
        C_((C_('a'), C_(1), C_(3))),
    )))


@pytest.mark.xfail(
    reason="The naive solver can't handle recursion",
    raises=RecursionError
)
def test_intensional_recursive():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock(tuple(
        T_(Q(C_(i), C_(2 * i)))
        for i in range(4)
    ))

    intensional = ExpressionBlock((
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), R(x, z) & R(z, y))
    ))

    dl = Datalog()
    dl.walk(extensional)
    dl.walk(intensional)

    res = dl.walk(Q_((x, y), R(x, y)))

    assert res == {
        (i, 2 * i)
        for i in range(4)
    } | {
        (i, 4 * i)
        for i in range(4)
    }


def test_not_conjunctive():

    dl = Datalog()

    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), R(x) | R(y)))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), R(x) & R(y) | R(x)))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), ~R(x)))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), R(Q(x))))


def test_equality_operation():
    dl = Datalog()

    assert dl.walk(S_('equals')(C_(1), C_(1))).value is True
    assert dl.walk(S_('equals')(C_(1), C_(2))).value is False


def test_existential_predicate():
    solver = Datalog()
    a, b = C_('a'), C_('b')
    x = S_('x')
    Q = S_('Q')  # noqa: N806
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)

    exp = EP_(x, Q(x))
    result = solver.walk(exp)

    assert result.value is True


def test_and_query_resolution():
    solver = Datalog()
    a, b = C_('a'), C_('b')
    x, y = S_('x'), S_('y')
    P, Q = S_('P'), S_('Q')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
        Fact(P(a)),
    ))
    solver.walk(extensional)

    query = Query(x, EP_(y, P(x) & Q(y)))
    result = solver.walk(query)

    assert result.value == {a}
