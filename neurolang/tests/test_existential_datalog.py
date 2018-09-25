import pytest
from typing import Any

from .. import solver_datalog_extensional_db
from .. import expression_walker
from .. import expressions
from ..expressions import ExpressionBlock, Query
from ..existential_datalog import (
    ExistentialDatalog, SolverNonRecursiveExistentialDatalog, Implication
)
from ..solver_datalog_naive import Fact, UNDEFINED, NULL

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
EP_ = expressions.ExistentialPredicate


class SolverWithoutExistentialResolution(
    ExistentialDatalog,
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
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    assert 'P' in solver.existential_intensional_database()
    assert 'P' not in solver.intensional_database()


def test_bad_existential_formulae():
    solver = SolverWithoutExistentialResolution()
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')
    with pytest.raises(expressions.NeuroLangException):
        solver.walk(Implication(EP_(y, P(x, y)), Q(x, y)))
    with pytest.raises(expressions.NeuroLangException):
        solver.walk(Implication(EP_(y, P(x)), Q(x)))
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))


def test_existential_statement_added_to_symbol_table():
    solver = SolverWithoutExistentialResolution()
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    assert len(solver.symbol_table['P'].expressions) == 1
    assert isinstance(
        solver.symbol_table['P'].expressions[0].consequent,
        expressions.ExistentialPredicate
    )
    solver = SolverWithoutExistentialResolution()
    solver.walk(Implication(EP_(x, EP_(y, P(x, y, z))), Q(z)))
    assert 'P' in solver.symbol_table


def test_existential_statement_resolution():
    solver = SolverWithExistentialResolution()
    x = S_('x')
    P = S_('P')
    Q = S_('Q')
    a = C_('a')
    b = C_('b')
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

    y = S_('y')
    P = S_('P')
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    query = Query(x, EP_(y, P(x, y)))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})

    u = S_('u')
    v = S_('v')
    query = Query(u, EP_(v, P(u, v)))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})


def test_existential_statement_resolution_undefined():
    solver = SolverWithExistentialResolution()
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')
    a = C_('a')
    b = C_('b')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)
    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    query = Query(y, P(x, y))
    result = solver.walk(query)
    assert result is UNDEFINED


def test_function_application_on_null_returns_false():
    solver = SolverWithExistentialResolution()
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')
    a = C_('a')
    b = C_('b')
    extensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))
    solver.walk(extensional)
    f = Implication(EP_(y, P(x, y)), Q(x))
    res = solver.walk(f(NULL, NULL))
    assert isinstance(res, expressions.Constant)
    assert not res.value


def test_existential_and_query_resolution():
    solver = SolverWithExistentialResolution()
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')
    a = C_('a')
    b = C_('b')
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


def test_multiple_existential_variables_in_consequent():
    solver = SolverWithExistentialResolution()
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    a = C_('a')
    b = C_('b')
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


def test_multiple_existential_variables_in_consequent_undefined():
    solver = SolverWithExistentialResolution()
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    a = C_('a')
    b = C_('b')
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
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    R = S_('R')

    solver.walk(Implication(EP_(y, P(x, y)), Q(x)))
    assert 'P' in solver.symbol_table
    assert 'P' in solver.existential_intensional_database()

    with pytest.raises(expressions.NeuroLangException):
        solver.walk(Implication(P(x, y), R(x, y)))
