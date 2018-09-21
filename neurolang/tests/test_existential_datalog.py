import pytest
from typing import Any

from .. import solver_datalog_extensional_db
from .. import expression_walker
from .. import expressions
from ..expressions import ExpressionBlock, Query
from ..existential_datalog import (
    NonRecursiveExistentialDatalog, SolverNonRecursiveExistentialDatalog,
    Implication
)
from ..solver_datalog_naive import NaiveDatalog, Fact, UNDEFINED, NULL

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
EP_ = expressions.ExistentialPredicate


class SolverWithoutExistentialResolution(
    NonRecursiveExistentialDatalog,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    expression_walker.ExpressionBasicEvaluator,
):
    pass


class SolverWithExistentialResolution(
    SolverNonRecursiveExistentialDatalog,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    expression_walker.ExpressionBasicEvaluator,
):
    pass


def test_existential_intensional_database():
    solver = SolverWithoutExistentialResolution()
    x, y, P, Q = S_('x'), S_('y'), S_('P'), S_('Q')
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    assert 'P' in solver.existential_intensional_database()
    assert 'P' not in solver.intensional_database()


def test_bad_existential_formulae():

    solver = SolverWithoutExistentialResolution()

    x, y = S_('x'), S_('y')
    P, Q = S_('P'), S_('Q')

    with pytest.raises(expressions.NeuroLangException):
        solver.walk(Implication(EP_(y, P(x, y)), Q(x, y)))

    with pytest.raises(expressions.NeuroLangException):
        solver.walk(Implication(EP_(y, P(x)), Q(x)))

    exp = Implication(EP_(y, P(x, y)), Q(x))
    solver.walk(exp)


def test_existential_statement_added_to_symbol_table():
    solver = SolverWithoutExistentialResolution()
    x, y = S_('x'), S_('y')
    a, b, c = C_('a'), C_('b'), C_('c')
    P, Q = S_('P'), S_('Q')
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    assert len(solver.symbol_table['P'].expressions) == 1
    assert isinstance(
        solver.symbol_table['P'].expressions[0].consequent,
        expressions.ExistentialPredicate
    )
    solver = SolverWithoutExistentialResolution()
    z = S_('z')
    solver.walk(Implication(EP_(x, EP_(y, P(x, y, z))), Q(z)))
    assert 'P' in solver.symbol_table


def test_existential_statement_resolution():
    solver = SolverWithExistentialResolution()
    x, y = S_('x'), S_('y')
    a, b, c = C_('a'), C_('b'), C_('c')
    P, Q = S_('P'), S_('Q')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)
    assert 'Q' in solver.symbol_table
    query = Query(x, Q(x))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})

    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))

    query = Query(x, EP_(y, P(x, y)))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})

    u, v = S_('u'), S_('v')
    query = Query(u, EP_(v, P(u, v)))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})


def test_existential_statement_resolution_undefined():
    solver = SolverWithExistentialResolution()
    x, y = S_('x'), S_('y')
    a, b, c = C_('a'), C_('b'), C_('c')
    P, Q = S_('P'), S_('Q')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    u, v = S_('u'), S_('v')
    query = Query(u, EP_(u, P(v, u)))
    result = solver.walk(query)
    assert result is UNDEFINED


def test_very_simple():
    solver = SolverWithExistentialResolution()
    P, Q = S_('P'), S_('Q')
    x, y = S_('x'), S_('y')
    f = Implication(EP_(y, P(x, y)), Q(x))
    assert not solver.walk(f(NULL, NULL))


def test_and_query_resolution():
    solver = SolverWithExistentialResolution()
    a, b = C_('a'), C_('b')
    x, y = S_('x'), S_('y')
    P, Q = S_('P'), S_('Q')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
        Fact(P(a)),
    ))
    solver.walk(extensional)

    query = Query(x, P(x) & EP_(y, Q(y)))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a'})


def test_multiple_eq_variables_in_consequent():
    solver = SolverWithExistentialResolution()
    a, b = C_('a'), C_('b')
    x, y, z = S_('x'), S_('y'), S_('z')
    P, Q = S_('P'), S_('Q')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)
    solver.walk(Implication(EP_(x, EP_(y, P(x, y, z))), Q(z)))
    query = Query(z, EP_(x, EP_(y, P(x, y, z))))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})


def test_multiple_eq_variables_in_consequent_undefined():
    solver = SolverWithExistentialResolution()
    a, b = C_('a'), C_('b')
    x, y, z = S_('x'), S_('y'), S_('z')
    P, Q = S_('P'), S_('Q')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)
    solver.walk(Implication(EP_(x, EP_(y, P(x, y, z))), Q(z)))
    query = Query(x, EP_(y, EP_(z, P(x, y, z))))
    result = solver.walk(query)
    assert result is UNDEFINED


def test_cannot_mix_existential_and_non_existential_rule_definitions():
    solver = SolverWithoutExistentialResolution()
    x, y = S_('x'), S_('y')
    P, Q, R = S_('P'), S_('Q'), S_('R')

    solver.walk(Implication(P(x, y), Q(x)))
    assert 'P' in solver.symbol_table
    assert 'P' in solver.existential_intensional_database()

    with pytest.raises(expressions.NeuroLangException):
        solver.walk(Implication(P(x, y), Q(x, y)))
