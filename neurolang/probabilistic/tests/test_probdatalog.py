from typing import Mapping, AbstractSet

import numpy as np
import pytest

from ...datalog.expressions import Fact
from ...logic import Union, Conjunction, Implication, ExistentialPredicate
from ...exceptions import NeuroLangException
from ...expressions import (
    Constant,
    ExpressionBlock,
    Symbol,
)
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ..ppdl import DeltaTerm
from ..expressions import ProbabilisticPredicate, Grounding
from ..probdatalog import (
    GDatalogToProbDatalog,
    ProbDatalogProgram,
    conjunct_formulas,
    _combine_typings,
    _check_typing_consistency,
    _infer_pfact_typing_pred_symbs,
    is_probabilistic_fact,
    RemoveProbabilitiesWalker,
    ground_probdatalog_program,
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
    pd = ProbDatalogProgram()

    code = ExpressionBlock(
        [
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(x)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction([P(x), Z(x)])),
            Fact(Z(a)),
            Fact(Z(b)),
        ]
    )

    pd.walk(code)

    assert pd.probabilistic_rules() == {
        P: ExpressionBlock([Implication(Q(x), Conjunction([P(x), Z(x)]))])
    }

    assert not pd.parametric_probfacts()

    assert pd.extensional_database() == {
        Z: C_(frozenset({C_((a,)), C_((b,))}))
    }
    assert pd.intensional_database() == {
        Q: Union([Implication(Q(x), Conjunction([P(x), Z(x)]))])
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

    pfact = Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    code = ExpressionBlock(
        [pfact, Implication(Q(x), Conjunction([P(x), Z(x)]))]
    )
    program = ProbDatalogProgram()
    program.walk(code)
    assert program.parametric_probfacts() == {p: pfact}


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
    program = ProbDatalogProgram()
    program.walk(translated)
    assert deterministic_rule in program.intensional_database()[Z].formulas
    with pytest.raises(NeuroLangException, match=r".*bernoulli.*"):
        bad_rule = Implication(
            Q(x, DeltaTerm(Symbol("bad_distrib"), tuple())), P(x)
        )
        translator = GDatalogToProbDatalog()
        translator.walk(bad_rule)


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
            ),
            Implication(Z(x), Conjunction([P(x), Q(x)])),
        ]
    )
    program = ProbDatalogProgram()
    program.walk(code)
    assert program.symbol_table[program.typing_symbol].value[P] == Constant[
        Mapping
    ]({Constant[int](0): Constant[AbstractSet]({Q})})

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
    program = ProbDatalogProgram()
    with pytest.raises(NeuroLangException, match=r"can only be used"):
        program.walk(code)


def test_remove_probabilities():
    assert RemoveProbabilitiesWalker({}).walk(
        Implication(ProbabilisticPredicate(p, P(a)), Constant[bool](True))
    ) == Fact(P(a))
    assert RemoveProbabilitiesWalker({}).walk(
        Implication(
            ExistentialPredicate(p, ProbabilisticPredicate(p, P(a))),
            Constant[bool](True),
        )
    ) == Fact(P(a))
    assert RemoveProbabilitiesWalker(
        {
            ProbDatalogProgram.typing_symbol: Constant(
                {
                    P: Constant[Mapping](
                        {Constant[int](0): Constant[AbstractSet]({Q})}
                    )
                }
            )
        }
    ).walk(
        Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    ) == Implication(
        P(x), Q(x)
    )
    assert RemoveProbabilitiesWalker(
        {
            ProbDatalogProgram.typing_symbol: Constant(
                {
                    P: Constant[Mapping](
                        {Constant[int](0): Constant[AbstractSet]({Q})}
                    )
                }
            )
        }
    ).walk(
        Implication(
            ExistentialPredicate(p, ProbabilisticPredicate(p, P(x))),
            Constant[bool](True),
        )
    ) == Implication(
        P(x), Q(x)
    )
    assert RemoveProbabilitiesWalker(
        {
            ProbDatalogProgram.typing_symbol: Constant(
                {
                    P: Constant[Mapping](
                        {
                            Constant[int](0): Constant[AbstractSet]({Q}),
                            Constant[int](1): Constant[AbstractSet]({Z}),
                        }
                    )
                }
            )
        }
    ).walk(
        ExpressionBlock(
            [
                Implication(
                    ProbabilisticPredicate(p, P(x, y)), Constant[bool](True)
                ),
                Implication(R(x, y), Conjunction([P(x, y), Q(x), Z(y)])),
            ]
        )
    ) == ExpressionBlock(
        [
            Implication(P(x, y), Conjunction([Q(x), Z(y)])),
            Implication(R(x, y), Conjunction([P(x, y), Q(x), Z(y)])),
        ]
    )


def test_combine_typings():
    typing_a = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({P, Q})}
    )
    typing_b = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Z, Q})}
    )
    combined = _combine_typings(typing_a, typing_b)
    assert combined == Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Q})}
    )

    typing_a = Constant[Mapping](dict())
    typing_b = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Q})}
    )
    combined = _combine_typings(typing_a, typing_b)
    assert combined == typing_b


def test_check_typing_consistency():
    typing_a = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({P, Q})}
    )
    typing_b = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Z})}
    )
    with pytest.raises(NeuroLangException):
        _check_typing_consistency(typing_a, typing_b)


def test_infer_pfact_typing_pred_symbs():
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    typing = _infer_pfact_typing_pred_symbs(P, rule)
    assert typing == Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Q})}
    )
    with pytest.raises(NeuroLangException):
        rule = Implication(Z(x), Q(x))
        _infer_pfact_typing_pred_symbs(P, rule)


def test_probdatalog_pfact_type_inference():
    code = ExpressionBlock(
        [
            Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True)),
            Implication(Q(x), Conjunction([P(x), Z(x), R(x)])),
            Implication(Q(x), Conjunction([P(x), Y(x), R(x)])),
        ]
    )
    program = ProbDatalogProgram()
    program.walk(code)
    assert program.symbol_table[program.typing_symbol] == Constant[Mapping](
        {P: Constant[Mapping]({Constant[int](0): Constant[AbstractSet]({R})})}
    )


def test_probdatalog_pfact_cant_infer_type():
    pfact = Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    rule = Implication(Q(x), Conjunction([P(x), Z(x), R(x)]))
    program = ProbDatalogProgram()
    program.walk(pfact)
    program.walk(rule)
    assert program.symbol_table[program.typing_symbol] == Constant[Mapping](
        {
            P: Constant[Mapping](
                {Constant[int](0): Constant[AbstractSet]({R, Z})}
            )
        }
    )
    program = ProbDatalogProgram()
    with pytest.raises(NeuroLangException):
        program.walk(ExpressionBlock([pfact, rule]))


def test_probdatalog_grounding():
    pfact = Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    code = ExpressionBlock([pfact, rule, Fact(Q(a)), Fact(Q(b))])
    grounded = ground_probdatalog_program(code)
    expected = Grounding(
        pfact,
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=["a", "b"], columns=["x"])
        ),
    )
    assert expected in grounded.expressions
    expected = Grounding(
        rule,
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=["a", "b"], columns=["x"])
        ),
    )
    assert expected in grounded.expressions

    pfact = Implication(
        ExistentialPredicate(p, ProbabilisticPredicate(p, P(x))),
        Constant[bool](True),
    )
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    code = ExpressionBlock([pfact, rule, Fact(Q(a)), Fact(Q(b))])
    grounded = ground_probdatalog_program(code)
    expected = Grounding(
        pfact,
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=["a", "b"], columns=["x"])
        ),
    )
    assert expected in grounded.expressions

    code = ExpressionBlock(
        [Fact(P(a, b)), Fact(P(b, b)), Fact(Q(a)), Fact(Q(b))]
    )
    grounded = ground_probdatalog_program(code)
    assert len(grounded.expressions) == 2
    for grounding in grounded.expressions:
        if grounding.expression.functor == P:
            assert np.all(
                np.vstack(list(grounding.relation.value.itervalues()))
                == np.vstack(
                    [
                        np.array(["a", "b"], dtype=str),
                        np.array(["b", "b"], dtype=str),
                    ]
                )
            )
        elif grounding.expression.functor == Q:
            assert np.all(
                np.array(list(grounding.relation.value.itervalues()))
                == np.array([["a"], ["b"]], dtype=str)
            )


def test_unsupported_grounding_program_with_disjunction():
    code = ExpressionBlock([Implication(Q(x), P(x)), Implication(Q(x), R(x))])
    with pytest.raises(NeuroLangException, match=r"supported"):
        ground_probdatalog_program(code)
