import pytest

from .. import expressions, exceptions
from .. import expression_walker as ew
from ..existential_datalog import ExistentialDatalog
from ..solver_datalog_naive import (
    Implication,
    Fact,
)

from ..datalog_chase_existential import DatalogExistentialChaseOblivious
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


def test_stratified_and_chase():
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
        Implication(Ep_(z, R1(y, z)), R1(x, y) & R2(y)),
        Implication(R2(y), R1(x, y)),
    ))

    dl = Datalog()
    dl.walk(datalog_program)

    rules = []

    dcw = DatalogExistentialChaseOblivious(dl, max_iterations=10)
    with pytest.raises(
        NeuroLangRecursionException,
    ):
        dcw.build_chase_solution()