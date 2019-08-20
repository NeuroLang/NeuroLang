import pytest

from .. import expressions, exceptions
from .. import expression_walker as ew
from ..existential_datalog import ExistentialDatalog
from ..solver_datalog_naive import (
    Implication,
    Fact,
    NullConstant,
    Any,
    Unknown,
)
from ..datalog_chase_existential import (
    DatalogExistentialChaseRestricted,
    DatalogExistentialChaseOblivious,
)
from ..datalog_chase import NeuroLangRecursionException

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
Ep_ = expressions.ExistentialPredicate
Eb_ = expressions.ExpressionBlock


class Datalog(ExistentialDatalog, ew.ExpressionBasicEvaluator):
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


def test_finite_chase():
    P = S_('P')
    R = S_('R')
    K = S_('K')

    z = S_('z')
    x = S_('x')
    y = S_('y')

    datalog_program = Eb_((
        Fact(P(C_(1))),
        Fact(P(C_(2))),
        Fact(R(C_(1), C_(2))),
        Implication(Ep_(x, K(y, x)), P(y)),
        Implication(K(y, x), R(z, y) & K(z, x))
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dcw = DatalogExistentialChaseRestricted(dl)
    solution_instance = dcw.build_chase_solution()

    NULL = NullConstant('NULL', auto_infer_type=False)

    final_instance = {
        P:
        C_({
            C_((C_[int](1),)),
            C_((C_[int](2),)),
        }),
        K:
        C_({
            C_((NULL, C_[int](1))),
        }),
        R:
        C_({
            C_((C_[int](1), C_[int](2))),
        })
    }

    assert solution_instance[P] == final_instance[P]
    assert solution_instance[R] == final_instance[R]

    for value in solution_instance[K].value:
        assert value.value[0] == C_[int](1) or value.value[0] == C_[int](2)
        assert isinstance(value.value[1], NullConstant)


def test_infinite_chase():
    '''
    Example 2.9 of Cali et. al. [1]_.


    .. [1] Calì, A., G. Gottlob, and M. Kifer. “Taming the Infinite Chase:
    Query Answering under Expressive Relational Constraints.”
    Journal of Artificial Intelligence Research 48 (October 22, 2013):
    115–74. https://doi.org/10.1613/jair.3873.
    '''
    x = S_('x')
    y = S_('y')
    z = S_('z')
    R1 = S_('R1')
    R2 = S_('R2')
    R3 = S_('R3')
    R1e = S_('R1e')

    datalog_program = Eb_((
        Fact(R1e(C_(1), C_(2))),
        Implication(R1(x, y), R1e(x, y)),
        Implication(R2(x), R3(x, y)),
        Implication(Ep_(z, R3(y, z)), R1(x, y)),
        Implication(Ep_(z, R1(y, z)),
                    R1(x, y) & R2(y)),
        Implication(R2(y), R1(x, y)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    dcw = DatalogExistentialChaseOblivious(dl)
    solution_instance = dcw.build_chase_solution()

    NULL = NullConstant('NULL 4', auto_infer_type=False)

    final_instance = {
        R1:
            C_({
                C_((C_[int](1), C_[int](2),)),
                C_((C_[int](2), NULL,)),
                C_((NULL, NULL,)),
            }),
        R2:
            C_({
                C_((C_[int](2),)),
                C_((NULL,)),
            }),
        R3:
            C_({
                C_((C_[int](2), NULL,)),
                C_((NULL, NULL,)),
            }),
    }

    for key, value in enumerate(solution_instance[R1].value):
        if key == 0:
            assert value.value[0] == C_[int](1) and value.value[1] == C_[int](2)
        elif key == 1:
            assert value.value[0] == C_[int](2) and isinstance(value.value[1], NullConstant)
        else:
            isinstance(value.value[0], NullConstant) and isinstance(value.value[1], NullConstant)

    for key, value in enumerate(solution_instance[R2].value):
        if key == 0:
            assert value.value[0] == C_[int](2)
        else:
            isinstance(value.value[0], NullConstant)

    for key, value in enumerate(solution_instance[R3].value):
        if key == 0:
            assert value.value[0] == C_[int](2) and isinstance(value.value[1], NullConstant)
        else:
            isinstance(value.value[0], NullConstant) and isinstance(value.value[1], NullConstant)

