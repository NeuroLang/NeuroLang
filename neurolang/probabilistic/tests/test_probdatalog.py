import pytest
import numpy as np

from ...expressions import Symbol, Constant, ExpressionBlock
from ...exceptions import NeuroLangException
from ...datalog.expressions import Fact, Implication, Disjunction, Conjunction
from ...expression_walker import ExpressionBasicEvaluator
from ...datalog.instance import SetInstance
from ..probdatalog import (
    ProbDatalogProgram, ProbFact, ProbChoice, GDatalogToProbDatalog,
    get_possible_ground_substitutions, full_observability_parameter_estimation,
    infer_pfact_typing_predicate_symbols, ProbfactAsFactWalker
)
from ..ppdl import DeltaTerm

C_ = Constant

P = Symbol('P')
Q = Symbol('Q')
R = Symbol('R')
Z = Symbol('Z')
Y = Symbol('Y')
p = Symbol('p')
p_1 = Symbol('p_1')
p_2 = Symbol('p_2')
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
    ProbChoice(probfacts)


def test_probchoice_sum_probs_gt_1():
    probfacts = [
        ProbFact(Constant[float](0.3), P(a)),
        ProbFact(Constant[float](0.9), P(b)),
    ]
    with pytest.raises(
        NeuroLangException, match=r'.*cannot be greater than 1.*'
    ):
        ProbChoice(probfacts)


def test_probdatalog_program():
    pd = ProbDatalog()

    code = ExpressionBlock((
        ProbFact(Constant[float](0.5), P(x)),
        Implication(Q(x),
                    P(x) & Z(x)),
        Fact(Z(a)),
        Fact(Z(b)),
    ))

    pd.walk(code)

    assert pd.extensional_database() == {
        Z: C_(frozenset({C_((a, )), C_((b, ))})),
    }
    assert pd.intensional_database() == {
        Q: Disjunction([Implication(Q(x),
                                    P(x) & Z(x))]),
    }
    assert pd.probabilistic_facts() == {
        P: ExpressionBlock((ProbFact(Constant[float](0.5), P(x)), )),
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
            Q(x, DeltaTerm(Symbol('bad_distrib'), tuple())), P(x)
        )
        translator = GDatalogToProbDatalog()
        translator.walk(bad_rule)


def test_get_possible_ground_substitutions_constant_probfact():
    probfact = ProbFact(C_(0.2), Z(a))
    typing = dict()
    interpretation = SetInstance([Fact(fa) for fa in [P(a), Z(a), P(b), Q(a)]])
    substitutions = get_possible_ground_substitutions(
        probfact, typing, interpretation
    )
    assert substitutions == frozenset({frozenset()})

    probfact = ProbFact(C_(0.2), Z(x, a))
    typing = {Z: {0: {P}}}
    interpretation = SetInstance({
        R: frozenset({(a, ), (b, )}),
        P: frozenset({(a, )}),
        Z: frozenset({(a, a)}),
        Q: frozenset({(a, )}),
    })
    substitutions = get_possible_ground_substitutions(
        probfact, typing[Z], interpretation
    )
    assert substitutions == frozenset({frozenset({(x, a)})})


def test_get_possible_ground_substitutions():
    probfact = ProbFact(C_(0.2), Z(x))
    interpretation = SetInstance([
        Fact(fa) for fa in [P(a), P(b), Z(a),
                            Z(b), Q(a), Q(b)]
    ])
    typing = {Z: {0: {P}}}
    substitutions = get_possible_ground_substitutions(
        probfact, typing[Z], interpretation
    )
    assert substitutions == frozenset({
        frozenset({(x, a)}), frozenset({(x, b)})
    })

    probfact = ProbFact(C_(0.5), Z(x, y))
    interpretation = SetInstance([
        Fact(fa) for fa in
        [P(a), P(b), Y(a),
         Y(b), Z(a, b), Q(a),
         Z(b, a), Q(b)]
    ])
    typing = {Z: {0: {P}, 1: {Y}}}
    substitutions = get_possible_ground_substitutions(
        probfact, typing[Z], interpretation
    )
    assert substitutions == frozenset({
        frozenset({(x, a), (y, a)}),
        frozenset({(x, a), (y, b)}),
        frozenset({(x, b), (y, a)}),
        frozenset({(x, b), (y, b)}),
    })


def test_full_observability_parameter_estimation():
    code = ExpressionBlock((
        ProbFact(p, Z(x)),
        Implication(Q(x), Conjunction([Z(x), P(x)])),
        Fact(P(a)),
        Fact(P(b)),
    ))
    program = ProbDatalog()
    program.walk(code)
    assert program.parametric_probfacts() == {p: ProbFact(p, Z(x))}
    interpretations = frozenset([
        SetInstance({
            P: frozenset({(a, ), (b, )}),
            Z: frozenset({(a, )}),
            Q: frozenset({(a, )}),
        }),
        SetInstance({
            P: frozenset({(a, ), (b, )}),
            Z: frozenset({(b, )}),
            Q: frozenset({(b, )}),
        }),
    ])
    estimations = full_observability_parameter_estimation(
        program,
        interpretations,
    )
    assert p in estimations
    assert np.isclose(estimations[p], 0.5)

    probfact_1 = ProbFact(p_1, Z(x))
    probfact_2 = ProbFact(p_2, Y(y))
    rule = Implication(Q(x, y), Conjunction([Z(x), Y(y), P(x), R(y)]))
    code = ExpressionBlock((
        probfact_1,
        probfact_2,
        rule,
        Fact(P(a)),
        Fact(P(b)),
    ))
    program = ProbDatalog()
    program.walk(code)
    assert program.parametric_probfacts() == {p_1: probfact_1, p_2: probfact_2}
    assert program.probabilistic_rules() == {Z: {rule}, Y: {rule}}
    interpretations = frozenset([
        SetInstance({
            P: frozenset({(a, ), (b, )}),
            R: frozenset({(a, )}),
            Z: frozenset({(a, )}),
            Y: frozenset({(a, )}),
            Q: frozenset({(a, a)})
        }),
        SetInstance({
            P: frozenset({(a, ), (b, )}),
            R: frozenset({(a, )}),
            Y: frozenset({(a, )})
        }),
        SetInstance({
            P: frozenset({(a, ), (b, )}),
            R: frozenset({(a, )}),
            Z: frozenset({(b, )}),
            Y: frozenset({(a, )})
        }),
        SetInstance({
            P: frozenset({(a, ), (b, )}),
            R: frozenset({(a, )})
        })
    ])
    estimations = full_observability_parameter_estimation(
        program,
        interpretations,
    )
    assert p_1 in estimations
    assert p_2 in estimations
    assert np.isclose(estimations[p_1], 0.25)
    assert np.isclose(estimations[p_2], 0.75)


def test_program_const_probfact_in_antecedent():
    code = ExpressionBlock([
        Implication(Q(a), Z(a)),
        ProbFact(p, Z(a)),
    ])
    program = ProbDatalog()
    program.walk(code)
    interpretations = [
        SetInstance({
            Z: frozenset({(a, )}),
            Q: frozenset({(a, )}),
        }),
        SetInstance({
            Z: frozenset(),
            Q: frozenset(),
        }),
    ]
    estimations = full_observability_parameter_estimation(
        program, interpretations
    )
    assert np.isclose(estimations[p], 0.5)


def test_program_with_twice_occurring_probfact_in_antecedent():
    code = ExpressionBlock([
        Implication(Q(x, y), Conjunction([Z(x), Z(y), P(x),
                                          P(y)])),
        ProbFact(p, Z(x)),
        Fact(P(a)),
        Fact(P(b)),
    ])
    program = ProbDatalog()
    program.walk(code)


def test_infer_pfact_typing_predicate_symbols():
    Pfact = Symbol('Pfact')
    rule = Implication(Q(x), Conjunction([P(x), Z(x), Pfact(x)]))
    assert infer_pfact_typing_predicate_symbols(Pfact, rule) == {0: {P, Z}}

    nopfact_rule = Implication(Q(x), P(x))
    with pytest.raises(NeuroLangException, match=r'Expected rule with atom'):
        infer_pfact_typing_predicate_symbols(Pfact, nopfact_rule)

    rule = Implication(Q(x, y), Conjunction([P(x), Q(y), Pfact(x), Pfact(y)]))
    with pytest.raises(NeuroLangException, match=r'Inconsistent'):
        infer_pfact_typing_predicate_symbols(Pfact, rule)


def test_probfact_as_fact():
    code = ExpressionBlock([
        ProbFact(p, Z(a)),
        Implication(Q(x), Conjunction([P(x), Z(x)])),
        Fact(P(a)),
        Fact(P(b)),
    ])
    walker = ProbfactAsFactWalker()
    new_code = walker.walk(code)
    assert Fact(Z(a)) in new_code.expressions
    assert Fact(P(a)) in new_code.expressions


def test_program_with_existential_raises_exception():
    code = ExpressionBlock([
        Implication(Q(x), P(x, y)),
    ])
    program = ProbDatalog()
    with pytest.raises(NeuroLangException, match=r'Existentially'):
        program.walk(code)
