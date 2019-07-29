from .. import expressions
from ..solver_datalog_naive import (
    Implication, Fact
)

from ..warded_datalog import WardedDatalog

C_ = expressions.Constant
S_ = expressions.Symbol
Fa_ = expressions.FunctionApplication
Ep_ = expressions.ExistentialPredicate
Eb_ = expressions.ExpressionBlock
I_ = Implication
F_ = Fact

def test_warded_walker():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    R = S_('R')

    P1 = F_(Q(x, z))
    P2 = I_(R(y, z), P(x, y))
    #P3 = F_(P(z, y))
    P4 = I_(R(x, z), Q(x,z) & P(x))

    program = Eb_((
        P1,
        P2,
        #P3,
        P4,
    ))

    wd = WardedDatalog()
    result = wd.walk(program)
    print(result)