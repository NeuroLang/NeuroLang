import pytest
from typing import Any

from .. import solver_datalog_extensional_db
from .. import expression_walker
from .. import expressions
from ..expressions import ExpressionBlock, Query
from ..existential_datalog import (
    ExistentialDatalog, SolverExistentialDatalog, Implication
)
from ..solver_datalog_naive import NaiveDatalog
from ..solver_datalog_naive import Fact

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
EP_ = expressions.ExistentialPredicate


class Solver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    ExistentialDatalog, expression_walker.ExpressionBasicEvaluator
):
    pass


class SolverWithExistentialResolution(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    SolverExistentialDatalog, expression_walker.ExpressionBasicEvaluator
):
    pass


def test_bad_existential_formulae():

    solver = Solver()

    x, y = S_('x'), S_('y')
    P, Q = S_('P'), S_('Q')

    with pytest.raises(expressions.NeuroLangException):
        exp = Implication(EP_(y, P(x, y)), Q(x, y))
        solver.walk(exp)

    with pytest.raises(expressions.NeuroLangException):
        exp = Implication(EP_(y, P(x)), Q(x))
        solver.walk(exp)

    exp = Implication(EP_(y, P(x, y)), Q(x))
    solver.walk(exp)


def test_existential_statement_added_to_symbol_table():
    solver = Solver()
    x, y = S_('x'), S_('y')
    a, b, c = C_('a'), C_('b'), C_('c')
    P, Q = S_('P'), S_('Q')
    intensional = ExpressionBlock((Implication(EP_(y, P(x, y)), Q(x)), ))
    solver.walk(intensional)
    assert 'P' in solver.symbol_table
    assert len(solver.symbol_table['P'].expressions) == 1
    assert isinstance(
        solver.symbol_table['P'].expressions[0].consequent,
        expressions.ExistentialPredicate
    )
    solver = Solver()
    z = S_('z')
    intensional = ExpressionBlock(
        (Implication(EP_(x, EP_(y, P(x, y, z))), Q(z)), )
    )
    solver.walk(intensional)
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
    intensional = ExpressionBlock((Implication(EP_(y, P(x, y)), Q(x)), ))
    solver.walk(intensional)
    query = Query(x, P(x, y))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})

    u, v = S_('u'), S_('v')
    query = Query(u, P(u, v))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})

    # query = Query(u, P(v, u))
    # result = solver.walk(query)
    # assert not isinstance(result, expressions.Constant)


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

    query = Query(x, P(x) & Q(y))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a'})

    # query = Query(x, P(x) & Q(y))
    # import pdb; pdb.set_trace()
    # result = solver.walk(query)
    # assert isinstance(result, expressions.Constant)
    # assert result.value is not None
    # assert result.value == frozenset({'a'})


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
    query = Query(z, P(x, y, z))
    result = solver.walk(query)
    assert isinstance(result, expressions.Constant)
    assert result.value is not None
    assert result.value == frozenset({'a', 'b'})
