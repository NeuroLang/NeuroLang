import pytest

from .. import solver_datalog_extensional_db
from .. import expression_walker
from .. import expressions
from ..expressions import ExpressionBlock, Query
from ..existential_datalog import ExistentialDatalog
from ..solver_datalog_naive import Fact

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
St_ = expressions.Statement
EP_ = expressions.ExistentialPredicate


class Solver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    ExistentialDatalog,
    expression_walker.ExpressionBasicEvaluator
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

def test_existential_statement_resolution():

    solver = Solver()

    x, y = S_('x'), S_('y')
    a, b, c = C_('a'), C_('b'), C_('c')
    P, Q = S_('P'), S_('Q')

    intensional = ExpressionBlock((
        Fact(Q(a)),
        Fact(Q(b)),
    ))

    extensional = ExpressionBlock((
        St_(EP_(y, P(x, y)), Q(x)),
    ))

    solver.walk(intensional)
    solver.walk(extensional)

    query = Query(x, EP_(y, P(x, y)))

    result = solver.walk(query)

    assert isinstance(result, expressions.Constant)
    assert result.value is not None

    assert result.value == frozenset({'a', 'b'})
