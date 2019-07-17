import pytest

from .. import expressions, exceptions
from .. import solver_datalog_negation as sdn
from .. import solver_datalog_extensional_db
from .. import expression_walker as ew
from .. import datalog_negative_chase as dc

from operator import and_, or_, invert

from ..solver_datalog_naive import (
    Implication,
    Fact,
)

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
E_ = expressions.ExistentialPredicate
U_ = expressions.UniversalPredicate
Eb_ = expressions.ExpressionBlock

class Datalog(
    sdn.DatalogBasicNegation,
    solver_datalog_extensional_db.ExtensionalDatabaseSolver,
    ew.ExpressionBasicEvaluator
):
    def function_gt(self, x: int, y: int)->bool:
        return x > y

def test_stratified_and_chase():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    G = S_('G')
    T = S_('T')
    V = S_('V')
    NG = S_('NG')

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(V(C_(4))),
        Fact(G(C_(1), C_(2))),
        Fact(G(C_(2), C_(3))),
        Implication(T(x, y), V(x) & V(y) & G(x, y)),
        Implication(T(x, y), G(x, z) & T(z, y)),
        Implication(NG(x, y), V(x) & V(y) & invert(G(x, y))),
    ))

    dl = Datalog()
    dl.walk(program)

    solution_instance = dc.build_chase_solution(dl)

    final_instance = {
        G: C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        V: C_({
            C_((C_(2),)),
            C_((C_(3),)),
            C_((C_(1),)),
            C_((C_(4),)),
        }),
        NG: C_({
            C_((C_(3), C_(2))),
            C_((C_(1), C_(3))),
            C_((C_(4), C_(1))),
            C_((C_(3), C_(3))),
            C_((C_(3), C_(1))),
            C_((C_(4), C_(4))),
            C_((C_(2), C_(1))),
            C_((C_(1), C_(4))),
            C_((C_(1), C_(1))),
            C_((C_(4), C_(3))),
            C_((C_(2), C_(2))),
            C_((C_(4), C_(2))),
            C_((C_(3), C_(4))),
            C_((C_(2), C_(4))),
        }),
        T: C_({
            C_((C_(1), C_(2))),
            C_((C_(1), C_(3))),
            C_((C_(2), C_(3)))
        })
    }

    assert solution_instance == final_instance