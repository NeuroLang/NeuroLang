import pytest

from .. import expressions

from operator import and_, or_, invert

from ..stratified_datalog import StratifiedDatalog

from ..existential_datalog import (
    Implication
)

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
E_ = expressions.ExistentialPredicate
U_ = expressions.UniversalPredicate
Eb_ = expressions.ExpressionBlock


def test_graph():
    x = S_('x')
    y = S_('y')

    imp = Implication(x, and_(y, y))

    program = Eb_((
        imp
    ))

    sDatalog = StratifiedDatalog()
    sDatalog.solve(program)
