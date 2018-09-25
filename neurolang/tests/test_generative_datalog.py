from .. import solver_datalog_extensional_db
from .. import expression_walker
from ..expressions import ExpressionBlock, Constant, Symbol, Query, Statement
from ..existential_datalog import Implication
from ..generative_datalog import (
    SolverNonRecursiveGenerativeDatalog, DeltaTerm, DeltaAtom
)

C_ = Constant
S_ = Symbol
St_ = Statement


class GenerativeDatalogTestSolver(
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    SolverNonRecursiveGenerativeDatalog,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_burglar():
    solver = GenerativeDatalogTestSolver()
    City = S_('City')
    House = S_('House')
    Business = S_('Business')
    Unit = S_('Unit')
    Earthquake = S_('Earthquake')
    Burglary = S_('Burglary')
    Trig = S_('Trig')
    Alarm = S_('Alarm')
    Flip = C_('Flip')
    x, h, b, c, r = S_('x'), S_('h'), S_('b'), S_('c'), S_('r')

    extensional = ExpressionBlock(())

    intensional = ExpressionBlock((
        St_(Unit(h, c), House(h, c)), St_(Unit(b, c), Business(b, c)),
        Implication(
            Earthquake(
                c, DeltaTerm(Flip, (C_(0.01), ), (C_('Earthquake'), c))
            ), City(c, r)
        ),
        Implication(
            Burglary(x, c, DeltaTerm(Flip, (r, ), (C_('Burglary'), x, c))),
            Unit(x, c) & City(c, r)
        ),
        Implication(
            Trig(x, DeltaTerm(Flip, (C_(0.6), ), (C_('Trig'), x))),
            Unit(x, c) & Earthquake(c, C_(1))
        ),
        Implication(
            Trig(x, DeltaTerm(Flip, (C_(0.9), ), (C_('Trig'), x))),
            Burglary(x, c, C_(1))
        ), St_(Alarm(x), Trig(x, C_(1)))
    ))
