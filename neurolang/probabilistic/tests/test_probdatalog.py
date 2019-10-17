import numpy as np
import pytest

from ...datalog.expressions import Conjunction, Disjunction, Fact, Implication
from ...datalog.instance import FrozenSetInstance, SetInstance
from ...exceptions import NeuroLangException
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import (
    Constant,
    ExpressionBlock,
    Symbol,
    ExistentialPredicate,
)
from ..ppdl import DeltaTerm
from ..expressions import ProbabilisticPredicate
from ..probdatalog import (
    GDatalogToProbDatalog,
    ProbDatalogProgram,
    ProbfactAsFactWalker,
    conjunct_formulas,
    full_observability_parameter_estimation,
    get_possible_ground_substitutions,
    get_rule_groundings,
    ground_probdatalog_program,
    _infer_pfact_typing_pred_symbs,
    is_probabilistic_fact,
)

C_ = Constant

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
Y = Symbol("Y")
p = Symbol("p")
p_1 = Symbol("p_1")
p_2 = Symbol("p_2")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")
c = Constant("c")
bernoulli = Symbol("bernoulli")


class ProbDatalog(ProbDatalogProgram, ExpressionBasicEvaluator):
    pass


def test_probfact():
    probfact = Implication(
        ProbabilisticPredicate(Constant[float](0.2), P(x)),
        Constant[bool](True),
    )
    assert probfact.consequent.probability == Constant[float](0.2)
    assert probfact.consequent.body == P(x)

    with pytest.raises(
        NeuroLangException, match=r"must be a symbol or constant"
    ):
        Implication(ProbabilisticPredicate(0.3, P(x)), Constant[bool](True))


def test_probdatalog_program():
    pd = ProbDatalog()

    code = ExpressionBlock(
        (
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(x)),
                Constant[bool](True),
            ),
            Implication(Q(x), P(x) & Z(x)),
            Fact(Z(a)),
            Fact(Z(b)),
        )
    )

    pd.walk(code)

    assert pd.extensional_database() == {
        Z: C_(frozenset({C_((a,)), C_((b,))}))
    }
    assert pd.intensional_database() == {
        Q: Disjunction([Implication(Q(x), P(x) & Z(x))])
    }
    assert pd.probabilistic_facts() == {
        P: ExpressionBlock(
            [
                Implication(
                    ProbabilisticPredicate(Constant[float](0.5), P(x)),
                    Constant[bool](True),
                )
            ]
        )
    }


def test_gdatalog_translation():
    probabilistic_rule = Implication(
        Q(x, DeltaTerm(bernoulli, (C_(0.2),))), P(x)
    )
    deterministic_rule = Implication(Z(x), P(x))
    program = ExpressionBlock(
        (probabilistic_rule, deterministic_rule, Fact(P(a)), Fact(P(b)))
    )
    translator = GDatalogToProbDatalog()
    translated = translator.walk(program)
    matching_exps = [
        exp for exp in translated.expressions if is_probabilistic_fact(exp)
    ]
    assert len(matching_exps) == 1
    probfact = matching_exps[0]
    assert x in probfact.consequent.body.args
    assert probfact.consequent.probability == C_(0.2)
    program = ProbDatalog()
    program.walk(translated)
    assert deterministic_rule in program.intensional_database()[Z].formulas
    with pytest.raises(NeuroLangException, match=r".*bernoulli.*"):
        bad_rule = Implication(
            Q(x, DeltaTerm(Symbol("bad_distrib"), tuple())), P(x)
        )
        translator = GDatalogToProbDatalog()
        translator.walk(bad_rule)


def test_get_possible_ground_substitutions_constant_probfact():
    probfact = Implication(
        ProbabilisticPredicate(C_(0.2), Z(a)), Constant[bool](True)
    )
    typing = dict()
    interpretation = SetInstance([P(a), Z(a), P(b), Q(a)])
    substitutions = get_possible_ground_substitutions(
        probfact, typing, interpretation
    )
    assert substitutions == frozenset({frozenset()})

    probfact = Implication(
        ProbabilisticPredicate(C_(0.2), Z(x, a)), Constant[bool](True)
    )
    typing = {Z: {0: {P}}}
    interpretation = SetInstance(
        {
            R: frozenset({(a,), (b,)}),
            P: frozenset({(a,)}),
            Z: frozenset({(a, a)}),
            Q: frozenset({(a,)}),
        }
    )
    substitutions = get_possible_ground_substitutions(
        probfact, typing[Z], interpretation
    )
    assert substitutions == frozenset({frozenset({(x, a)})})


def test_get_possible_ground_substitutions():
    probfact = Implication(
        ProbabilisticPredicate(C_(0.2), Z(x)), Constant[bool](True)
    )
    interpretation = SetInstance([P(a), P(b), Z(a), Z(b), Q(a), Q(b)])
    typing = {Z: {0: {P}}}
    substitutions = get_possible_ground_substitutions(
        probfact, typing[Z], interpretation
    )
    assert substitutions == frozenset(
        {frozenset({(x, a)}), frozenset({(x, b)})}
    )

    probfact = Implication(
        ProbabilisticPredicate(C_(0.5), Z(x, y)), Constant[bool](True)
    )
    interpretation = SetInstance(
        [P(a), P(b), Y(a), Y(b), Z(a, b), Q(a), Z(b, a), Q(b)]
    )
    typing = {Z: {0: {P}, 1: {Y}}}
    substitutions = get_possible_ground_substitutions(
        probfact, typing[Z], interpretation
    )
    assert substitutions == frozenset(
        {
            frozenset({(x, a), (y, a)}),
            frozenset({(x, a), (y, b)}),
            frozenset({(x, b), (y, a)}),
            frozenset({(x, b), (y, b)}),
        }
    )


def test_full_observability_parameter_estimation():
    code = ExpressionBlock(
        (
            Implication(ProbabilisticPredicate(p, Z(x)), Constant[bool](True)),
            Implication(Q(x), Conjunction([Z(x), P(x)])),
            Fact(P(a)),
            Fact(P(b)),
        )
    )
    program = ProbDatalog()
    program.walk(code)
    assert program.parametric_probfacts() == {
        p: Implication(ProbabilisticPredicate(p, Z(x)), Constant[bool](True))
    }
    interpretations = frozenset(
        [
            FrozenSetInstance(
                {
                    P: frozenset({(a,), (b,)}),
                    Z: frozenset({(a,)}),
                    Q: frozenset({(a,)}),
                }
            ),
            FrozenSetInstance(
                {
                    P: frozenset({(a,), (b,)}),
                    Z: frozenset({(b,)}),
                    Q: frozenset({(b,)}),
                }
            ),
        ]
    )
    estimations = full_observability_parameter_estimation(
        program, interpretations
    )
    assert p in estimations
    assert np.isclose(estimations[p], 0.5)

    probfact_1 = Implication(
        ProbabilisticPredicate(p_1, Z(x)), Constant[bool](True)
    )
    probfact_2 = Implication(
        ProbabilisticPredicate(p_2, Y(y)), Constant[bool](True)
    )
    rule = Implication(Q(x, y), Conjunction([Z(x), Y(y), P(x), R(y)]))
    code = ExpressionBlock(
        (probfact_1, probfact_2, rule, Fact(P(a)), Fact(P(b)))
    )
    program = ProbDatalog()
    program.walk(code)
    assert program.parametric_probfacts() == {p_1: probfact_1, p_2: probfact_2}
    assert program.probabilistic_rules() == {Z: {rule}, Y: {rule}}
    interpretations = frozenset(
        [
            FrozenSetInstance(
                {
                    P: frozenset({(a,), (b,)}),
                    R: frozenset({(a,)}),
                    Z: frozenset({(a,)}),
                    Y: frozenset({(a,)}),
                    Q: frozenset({(a, a)}),
                }
            ),
            FrozenSetInstance(
                {
                    P: frozenset({(a,), (b,)}),
                    R: frozenset({(a,)}),
                    Y: frozenset({(a,)}),
                }
            ),
            FrozenSetInstance(
                {
                    P: frozenset({(a,), (b,)}),
                    R: frozenset({(a,)}),
                    Z: frozenset({(b,)}),
                    Y: frozenset({(a,)}),
                }
            ),
            FrozenSetInstance(
                {P: frozenset({(a,), (b,)}), R: frozenset({(a,)})}
            ),
        ]
    )
    estimations = full_observability_parameter_estimation(
        program, interpretations
    )
    assert p_1 in estimations
    assert p_2 in estimations
    assert np.isclose(estimations[p_1], 0.25)
    assert np.isclose(estimations[p_2], 0.75)


def test_program_const_probfact_in_antecedent():
    code = ExpressionBlock(
        [
            Implication(Q(a), Z(a)),
            Implication(ProbabilisticPredicate(p, Z(a)), Constant[bool](True)),
        ]
    )
    program = ProbDatalog()
    program.walk(code)
    interpretations = [
        SetInstance({Z: frozenset({(a,)}), Q: frozenset({(a,)})}),
        SetInstance({Z: frozenset(), Q: frozenset()}),
    ]
    estimations = full_observability_parameter_estimation(
        program, interpretations
    )
    assert np.isclose(estimations[p], 0.5)


def test_program_with_twice_occurring_probfact_in_antecedent():
    code = ExpressionBlock(
        [
            Implication(Q(x, y), Conjunction([Z(x), Z(y), P(x), P(y)])),
            Implication(ProbabilisticPredicate(p, Z(x)), Constant[bool](True)),
            Fact(P(a)),
            Fact(P(b)),
        ]
    )
    program = ProbDatalog()
    program.walk(code)


def test_infer_pfact_typing_pred_symbs():
    Pfact = Symbol("Pfact")
    rule = Implication(Q(x), Conjunction([P(x), Z(x), Pfact(x)]))
    assert _infer_pfact_typing_pred_symbs(Pfact, rule) == {0: {P, Z}}

    nopfact_rule = Implication(Q(x), P(x))
    with pytest.raises(NeuroLangException, match=r"Expected rule with atom"):
        _infer_pfact_typing_pred_symbs(Pfact, nopfact_rule)

    rule = Implication(Q(x, y), Conjunction([P(x), Q(y), Pfact(x), Pfact(y)]))
    with pytest.raises(NeuroLangException, match=r"Inconsistent"):
        _infer_pfact_typing_pred_symbs(Pfact, rule)


def test_probfact_as_fact():
    code = ExpressionBlock(
        [
            Implication(ProbabilisticPredicate(p, Z(a)), Constant[bool](True)),
            Implication(Q(x), Conjunction([P(x), Z(x)])),
            Fact(P(a)),
            Fact(P(b)),
        ]
    )
    walker = ProbfactAsFactWalker()
    new_code = walker.walk(code)
    assert Fact(Z(a)) in new_code.expressions
    assert Fact(P(a)) in new_code.expressions

    code = ExpressionBlock(
        [Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))]
    )
    with pytest.raises(NeuroLangException):
        walker.walk(code)

    existential_pfact = Implication(
        ExistentialPredicate(p, ProbabilisticPredicate(p, Z(a))),
        Constant[bool](True),
    )
    walker = ProbfactAsFactWalker()
    assert walker.walk(existential_pfact) == Fact(Z(a))


def test_program_with_existential_raises_exception():
    code = ExpressionBlock([Implication(Q(x), P(x, y))])
    program = ProbDatalog()
    with pytest.raises(NeuroLangException, match=r"Existentially"):
        program.walk(code)


def test_ground_probdatalog_program():
    rule = Implication(Q(x), Conjunction([P(x), Z(x)]))
    code = ExpressionBlock(
        [
            Implication(
                ProbabilisticPredicate(C_(0.3), Z(a)), Constant[bool](True)
            ),
            Implication(
                ProbabilisticPredicate(C_(0.3), Z(b)), Constant[bool](True)
            ),
            Fact(P(a)),
            Fact(P(b)),
            rule,
        ]
    )
    assert get_rule_groundings(rule, SetInstance({})) == set()
    assert get_rule_groundings(
        rule,
        SetInstance(
            {
                P: frozenset({(a,), (b,)}),
                Z: frozenset({(a,), (b,)}),
                Q: frozenset({(a,), (b,)}),
            }
        ),
    ) == {
        Implication(Q(a), Conjunction([P(a), Z(a)])),
        Implication(Q(b), Conjunction([P(b), Z(b)])),
    }
    grounded = ground_probdatalog_program(code)
    assert Implication(Q(a), Conjunction([P(a), Z(a)])) in grounded.expressions

    code = ExpressionBlock(
        [
            Implication(
                ProbabilisticPredicate(C_(0.5), R(a)), Constant[bool](True)
            ),
            Implication(
                ProbabilisticPredicate(C_(0.5), R(c)), Constant[bool](True)
            ),
            Fact(P(a)),
            Fact(P(b)),
            Fact(Q(a)),
            Fact(Q(b)),
            Implication(Z(x, y), Conjunction([P(x), Q(y)])),
            Implication(Y(x), R(x)),
        ]
    )
    grounded = ground_probdatalog_program(code)
    assert (
        Implication(Z(a, b), Conjunction([P(a), Q(b)])) in grounded.expressions
    )
    assert (
        Implication(Z(b, a), Conjunction([P(b), Q(a)])) in grounded.expressions
    )
    assert (
        Implication(Z(a, a), Conjunction([P(a), Q(a)])) in grounded.expressions
    )
    assert (
        Implication(Z(b, b), Conjunction([P(b), Q(b)])) in grounded.expressions
    )
    assert Implication(Y(a), R(a)) in grounded.expressions
    assert Implication(Y(c), R(c)) in grounded.expressions

    code = ExpressionBlock([])
    grounded = ground_probdatalog_program(code)
    assert not grounded.expressions

    code = ExpressionBlock([Fact(P(a))])
    grounded = ground_probdatalog_program(code)
    assert Fact(P(a)) in grounded.expressions


def test_conjunct_formulas():
    assert conjunct_formulas(P(x), Q(x)) == Conjunction([P(x), Q(x)])
    a = P(x)
    b = Conjunction([Q(x), Z(x)])
    c = Conjunction([P(x), Q(x), Z(x)])
    d = Conjunction([Q(x), Z(x), P(x)])
    assert conjunct_formulas(P(x), b) == c
    assert conjunct_formulas(b, P(x)) == d
    assert conjunct_formulas(c, d) == Conjunction(c.formulas + d.formulas)


def test_program_with_eprobfact():
    code = ExpressionBlock(
        [
            Implication(
                ExistentialPredicate(
                    p, ProbabilisticPredicate(Symbol[float](p), P(x))
                ),
                Constant[bool](True),
            )
        ]
    )
    program = ProbDatalog()
    program.walk(code)

    code = ExpressionBlock(
        [
            Implication(
                ExistentialPredicate(
                    z, ProbabilisticPredicate(Constant(0.2), P(z))
                ),
                Constant[bool](True),
            )
        ]
    )
    program = ProbDatalog()
    with pytest.raises(NeuroLangException, match=r"can only be used"):
        program.walk(code)
