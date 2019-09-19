import pytest

from ...expressions import Symbol, Constant
from ...exceptions import NeuroLangException
from ...datalog.expressions import Fact, Implication, Disjunction
from ...expression_walker import ExpressionBasicEvaluator
from ..probdatalog import (
    ProbDatalogProgram, ProbFact, ProbChoice, GDatalogToProbDatalog
)
from ..ppdl import DeltaTerm

S_ = Symbol
C_ = Constant

P = Symbol('P')
Q = Symbol('Q')
Z = Symbol('Z')
x = Symbol('x')
a = Constant('a')
b = Constant('b')
bernoulli = Symbol('bernoulli')


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

    block = Disjunction((
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


def test_gdatalog_translation():
    program = Disjunction((
        Implication(Q(x, DeltaTerm(bernoulli, (C_(0.2), ))), P(x)),
        Fact(P(a)),
        Fact(P(b)),
    ))
    translator = GDatalogToProbDatalog()
    translated = translator.walk(program)
    matching_exps = [
        exp for exp in translated.formulas if isinstance(exp, ProbFact)
    ]
    assert len(matching_exps) == 1
    probfact = matching_exps[0]
    assert x in probfact.consequent.args
    assert probfact.probability == C_(0.2)
