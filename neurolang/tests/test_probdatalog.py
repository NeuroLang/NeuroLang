import pytest

from ..expressions import Symbol, Constant, ExpressionBlock
from ..exceptions import NeuroLangException
from ..datalog.expressions import Fact, Implication, Disjunction
from ..expression_walker import ExpressionBasicEvaluator
from ..probabilistic.probdatalog import (
    ProbDatalogProgram, ProbFact, ProbChoice
)

S_ = Symbol
C_ = Constant

P = Symbol('P')
Q = Symbol('Q')
Z = Symbol('Z')
x = Symbol('x')
a = Constant('a')
b = Constant('b')


class ProbDatalog(ProbDatalogProgram, ExpressionBasicEvaluator):
    pass


def test_probfact():
    probfact = ProbFact(Constant[float](0.2), P(x))
    assert probfact.probability == Constant[float](0.2)
    assert probfact.consequent == P(x)


def test_probchoice():
    probfacts = [
        ProbFact(Constant[float](0.3), P(a)),
        ProbFact(Constant[float](0.7), P(b)),
    ]
    probchoice = ProbChoice(probfacts)


def test_probchoice_sum_probs_gt_1():
    probfacts = [
        ProbFact(Constant[float](0.3), P(a)),
        ProbFact(Constant[float](0.9), P(b)),
    ]
    with pytest.raises(
        NeuroLangException, match=r'.*cannot be greater than 1.*'
    ) as exception:
        probchoice = ProbChoice(probfacts)


def test_probdatalog_program():
    pd = ProbDatalog()

    block = ExpressionBlock((
        ProbFact(Constant[float](0.5), P(x)),
        Implication(Q(x),
                    P(x) & Z(x)),
        Fact(Z(a)),
        Fact(Z(b)),
    ))

    pd.walk(block)

    assert pd.extensional_database() == {
        Z: C_(frozenset({C_((a, )), C_((b, ))})),
    }
    assert pd.intensional_database() == {
        Q: Disjunction([Implication(Q(x),
                                    P(x) & Z(x))]),
    }
    assert pd.probabilistic_database() == {
        P: ProbFact(Constant[float](0.5), P(x)),
    }
