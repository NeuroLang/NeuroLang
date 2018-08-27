import pytest
from typing import Any

from .. import solver_datalog_extensional_db
from .. import expression_walker
from .. import expressions
from ..expressions import ExpressionBlock, Query
from ..existential_datalog import ExistentialDatalog, SolverExistentialDatalog
from ..solver_datalog_naive import NaiveDatalog
from ..solver_datalog_naive import Fact

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
St_ = expressions.Statement
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
        exp = St_(EP_(y, P(x, y)), Q(x, y))
        solver.walk(exp)

    with pytest.raises(expressions.NeuroLangException):
        exp = St_(EP_(y, P(x)), Q(x))
        solver.walk(exp)

    exp = St_(EP_(y, P(x, y)), Q(x))
    solver.walk(exp)


def test_existential_statement_added_to_symbol_table():
    solver = Solver()
    x, y = S_('x'), S_('y')
    a, b, c = C_('a'), C_('b'), C_('c')
    P, Q = S_('P'), S_('Q')
    intensional = ExpressionBlock((St_(EP_(y, P(x, y)), Q(x)), ))

    solver.walk(intensional)

    assert 'P' in solver.symbol_table
    assert len(solver.symbol_table['P'].expressions) == 1
    assert isinstance(
        solver.symbol_table['P'].expressions[0].lhs,
        expressions.ExistentialPredicate
    )


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
    intensional = ExpressionBlock((St_(EP_(y, P(x, y)), Q(x)), ))
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
