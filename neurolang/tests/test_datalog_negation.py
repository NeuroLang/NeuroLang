import pytest

from .. import expressions, exceptions
from .. import solver_datalog_negation as sdn
from .. import solver_datalog_extensional_db
from .. import expression_walker as ew
from ..solver_datalog_naive import (
    Implication,
    Fact,
)

from ..datalog_chase_negation import DatalogChaseNegation

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
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


def test_stratified_and_chase():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    G = S_('G')
    T = S_('T')
    V = S_('V')
    NT = S_('NT')
    ET = S_('ET')
    equals = S_('equals')

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(V(C_(4))),
        Fact(G(C_(1), C_(2))),
        Fact(G(C_(2), C_(3))),
        Implication(T(x, y),
                    V(x) & V(y) & G(x, y)),
        Implication(T(x, y),
                    G(x, z) & T(z, y)),
        Implication(ET(x, y),
                    V(x) & V(y) & ~(G(x, y)) & equals(x, y)),
        Implication(NT(x, y),
                    V(x) & V(y) & ~(G(x, y)) & ~(equals(x, y))),
    ))

    dl = Datalog()
    dl.walk(program)

    dc = DatalogChaseNegation()
    solution_instance = dc.build_chase_solution(dl)

    final_instance = {
        G:
        C_({
            C_((C_(1), C_(2))),
            C_((C_(2), C_(3))),
        }),
        V:
        C_({
            C_((C_(2), )),
            C_((C_(3), )),
            C_((C_(1), )),
            C_((C_(4), )),
        }),
        ET:
        C_({
            C_((C_(1), C_(1))),
            C_((C_(3), C_(3))),
            C_((C_(4), C_(4))),
            C_((C_(2), C_(2))),
        }),
        NT:
        C_({
            C_((C_(3), C_(2))),
            C_((C_(1), C_(3))),
            C_((C_(4), C_(1))),
            C_((C_(3), C_(1))),
            C_((C_(2), C_(1))),
            C_((C_(1), C_(4))),
            C_((C_(4), C_(3))),
            C_((C_(4), C_(2))),
            C_((C_(3), C_(4))),
            C_((C_(2), C_(4))),
        }),
        T:
        C_({C_((C_(1), C_(2))),
            C_((C_(1), C_(3))),
            C_((C_(2), C_(3)))})
    }

    assert solution_instance == final_instance


def test_negative_fact():
    V = S_('V')
    G = S_('G')
    T = S_('T')
    CT = S_('CT')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    program = Eb_((
        Fact(V(C_(1))),
        Fact(V(C_(2))),
        Fact(V(C_(3))),
        Fact(V(C_(4))),
        Fact(G(C_(1), C_(2))),
        Fact(G(C_(2), C_(3))),
        sdn.NegativeFact(G(C_(1), C_(2))),
        Implication(T(x, y),
                    V(x) & V(y) & G(x, y)),
        Implication(T(x, y),
                    G(x, z) & T(z, y)),
        Implication(CT(x, y),
                    V(x) & V(y) & ~(G(x, y))),
    ))

    dl = Datalog()
    dl.walk(program)
    dc = DatalogChaseNegation()
    with pytest.raises(
        exceptions.NeuroLangException, match=r'There is a contradiction .*'
    ):
        dc.build_chase_solution(dl)
