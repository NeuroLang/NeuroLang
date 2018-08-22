import pytest

from .. import expressions
from ..existential_datalog import ExistentialDatalog

S_ = expressions.Symbol
F_ = expressions.FunctionApplication
St_ = expressions.Statement
EP_ = expressions.ExistentialPredicate


def test_existential_rule_constructor():

    solver = ExistentialDatalog()

    x, y = S_('x'), S_('y')
    P, Q = S_('P'), S_('Q')

    with pytest.raises(expressions.NeuroLangException):
        exp = St_(EP_(y, P(x, y)), F_(Q, (x, y)))
        solver.walk(exp)

    with pytest.raises(expressions.NeuroLangException):
        exp = St_(EP_(y, P(x)), F_(Q, (x, )))
        solver.walk(exp)
