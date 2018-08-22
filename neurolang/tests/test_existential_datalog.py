import pytest

from .. import expressions
from ..existential_datalog import ExistentialRule

S_ = expressions.Symbol
F_ = expressions.FunctionApplication
St_ = expressions.Statement


def test_existential_rule_constructor():
    x, y = S_('x'), S_('y')
    P, Q = S_('P'), S_('Q')
    with pytest.raises(expressions.NeuroLangException):
        rule = ExistentialRule(y, St_(F_(P, (x, y)), F_(Q, (y, ))))
    with pytest.raises(expressions.NeuroLangException):
        rule = ExistentialRule(y, St_(F_(P, (x, )), F_(Q, (x, ))))
    rule = ExistentialRule(y, St_(F_(P, (x, y)), F_(Q, (x, ))))
