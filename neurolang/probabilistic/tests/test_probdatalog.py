import pytest

from ...expressions import Symbol, Constant, ExpressionBlock
from ...exceptions import NeuroLangException
from ...datalog.expressions import Fact, Implication, Disjunction, Conjunction
from ...expression_walker import ExpressionBasicEvaluator
from ...datalog.instance import SetInstance
from ..probdatalog import (
    ProbDatalogProgram, ProbFact, ProbChoice, GDatalogToProbDatalog,
    get_possible_ground_substitutions
)
from ..ppdl import DeltaTerm

S_ = Symbol
C_ = Constant

P = Symbol('P')
Q = Symbol('Q')
Z = Symbol('Z')
Y = Symbol('Y')
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
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
    ):
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


def test_gdatalog_translation():
    probabilistic_rule = Implication(
        Q(x, DeltaTerm(bernoulli, (C_(0.2), ))), P(x)
    )
    deterministic_rule = Implication(Z(x), P(x))
    program = ExpressionBlock((
        probabilistic_rule,
        deterministic_rule,
        Fact(P(a)),
        Fact(P(b)),
    ))
    translator = GDatalogToProbDatalog()
    translated = translator.walk(program)
    matching_exps = [
        exp for exp in translated.expressions if isinstance(exp, ProbFact)
    ]
    assert len(matching_exps) == 1
    probfact = matching_exps[0]
    assert x in probfact.consequent.args
    assert probfact.probability == C_(0.2)
    program = ProbDatalog()
    program.walk(translated)
    assert deterministic_rule in program.intensional_database()[Z].formulas
    with pytest.raises(NeuroLangException, match=r'.*bernoulli.*'):
        bad_rule = Implication(
            Q(x, DeltaTerm(S_('bad_distrib'), tuple())), P(x)
        )
        translator = GDatalogToProbDatalog()
        translator.walk(bad_rule)


def test_get_possible_ground_substitutions():
    probfact = ProbFact(C_(0.2), Z(x))
    rule = Implication(Q(x), Conjunction([Z(x), P(x)]))
    interpretation = SetInstance([
        Fact(fa) for fa in [P(a), P(b), Z(a),
                            Z(b), Q(a), Q(b)]
    ])
    substitutions = get_possible_ground_substitutions(
        probfact, rule, interpretation
    )
    assert substitutions == frozenset({
        frozenset({(x, a)}), frozenset({(x, b)})
    })

    probfact = ProbFact(C_(0.5), Z(x, y))
    rule = Implication(Q(x), Conjunction([Z(x, y), P(x), Y(y)]))
    interpretation = SetInstance([
        Fact(fa) for fa in
        [P(a), P(b), Y(a),
         Y(b), Z(a, b), Q(a),
         Z(b, a), Q(b)]
    ])
    substitutions = get_possible_ground_substitutions(
        probfact, rule, interpretation
    )
    assert substitutions == frozenset({
        frozenset({(x, a), (y, a)}),
        frozenset({(x, a), (y, b)}),
        frozenset({(x, b), (y, a)}),
        frozenset({(x, b), (y, b)}),
    })
