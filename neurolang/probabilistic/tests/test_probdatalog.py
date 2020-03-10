from typing import Mapping, AbstractSet

import numpy as np
import pytest

from ...datalog.expressions import Fact
from ...logic import Union, Conjunction, Implication, ExistentialPredicate
from ...exceptions import NeuroLangException
from ...expressions import Constant, ExpressionBlock, Symbol
from ...utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ..ppdl import DeltaTerm
from ..expressions import ProbabilisticPredicate, Grounding
from ..probdatalog import (
    GDatalogToProbDatalog,
    ProbDatalogProgram,
    conjunct_formulas,
    is_probabilistic_fact,
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


def test_multiple_probfact_same_pred_symb():
    pd = ProbDatalogProgram()
    code = ExpressionBlock(
        [
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction([P(x), Z(x)])),
            Fact(Z(a)),
            Fact(Z(b)),
        ]
    )
    pd.walk(code)
    assert pd.extensional_database() == {
        Z: C_(frozenset({C_((a,)), C_((b,))}))
    }
    assert pd.intensional_database() == {
        Q: Union([Implication(Q(x), Conjunction([P(x), Z(x)]))])
    }
    assert len(pd.probabilistic_facts()) == 1
    assert P in pd.probabilistic_facts()
    probfacts = pd.probabilistic_facts()[P]
    assert isinstance(probfacts, Constant[AbstractSet])


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


@pytest.mark.skip(reason="existential probfacts not working yet")
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


@pytest.mark.skip()
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


@pytest.mark.skip()
def test_check_typing_consistency():
    typing_a = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({P, Q})}
    )
    typing_b = Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Z})}
    )
    with pytest.raises(NeuroLangException):
        _check_typing_consistency(typing_a, typing_b)


@pytest.mark.skip()
def test_infer_pfact_typing_pred_symbs():
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    typing = _infer_pfact_typing_pred_symbs(P, rule)
    assert typing == Constant[Mapping](
        {Constant[int](0): Constant[AbstractSet]({Q})}
    )
    with pytest.raises(NeuroLangException):
        rule = Implication(Z(x), Q(x))
        _infer_pfact_typing_pred_symbs(P, rule)


@pytest.mark.skip()
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


@pytest.mark.skip()
def test_probdatalog_pfact_cant_infer_type():
    pfact = Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    rule = Implication(Q(x), Conjunction([P(x), Z(x), R(x)]))
    program = ProbDatalogProgram()
    with pytest.raises(NeuroLangException, match="could not be inferred"):
        program.walk(ExpressionBlock([pfact, rule]))


def test_probdatalog_grounding():
    pfact1 = Implication(ProbabilisticPredicate(p, P(a)), Constant[bool](True))
    pfact2 = Implication(ProbabilisticPredicate(p, P(b)), Constant[bool](True))
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    code = ExpressionBlock([pfact1, pfact2, rule, Fact(Q(a)), Fact(Q(b))])
    grounded = ground_probdatalog_program(code)
    matching_groundings = [
        grounding
        for grounding in grounded.expressions
        if is_probabilistic_fact(grounding.expression)
        and grounding.expression.consequent.body.functor == P
    ]
    assert len(matching_groundings) == 1
    expected = Grounding(
        rule,
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
        if grounding.expression.consequent.functor == P:
            assert np.all(
                np.vstack(list(grounding.relation.value.itervalues()))
                == np.vstack(
                    [
                        np.array(["a", "b"], dtype=str),
                        np.array(["b", "b"], dtype=str),
                    ]
                )
            )
        elif grounding.expression.consequent.functor == Q:
            assert np.all(
                np.array(list(grounding.relation.value.itervalues()))
                == np.array([["a"], ["b"]], dtype=str)
            )


@pytest.mark.skip()
def test_probdatalog_grounding_general():
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

    code = ExpressionBlock(
        [Fact(P(a, b)), Fact(P(b, b)), Fact(Q(a)), Fact(Q(b))]
    )
    grounded = ground_probdatalog_program(code)
    assert len(grounded.expressions) == 2
    for grounding in grounded.expressions:
        if grounding.expression.consequent.functor == P:
            assert np.all(
                np.vstack(list(grounding.relation.value.itervalues()))
                == np.vstack(
                    [
                        np.array(["a", "b"], dtype=str),
                        np.array(["b", "b"], dtype=str),
                    ]
                )
            )
        elif grounding.expression.consequent.functor == Q:
            assert np.all(
                np.array(list(grounding.relation.value.itervalues()))
                == np.array([["a"], ["b"]], dtype=str)
            )


def test_unsupported_grounding_program_with_disjunction():
    code = ExpressionBlock([Implication(Q(x), P(x)), Implication(Q(x), R(x))])
    with pytest.raises(NeuroLangException, match=r"supported"):
        ground_probdatalog_program(code)


def test_add_probfacts_from_tuple():
    pd = ProbDatalogProgram()
    pd.walk(ExpressionBlock([]))
    pd.add_probfacts_from_tuples(
        P, {(0.3, "hello", "gaston"), (0.7, "hello", "antonia"),},
    )
    assert P in pd.pfact_pred_symbs
    assert (
        Constant[float](0.7),
        Constant[str]("hello"),
        Constant[str]("antonia"),
    ) in pd.symbol_table[P].value


def test_add_probfacts_from_tuple_no_probability():
    pd = ProbDatalogProgram()
    pd.walk(ExpressionBlock([]))
    with pytest.raises(NeuroLangException, match=r"probability"):
        pd.add_probfacts_from_tuples(
            P, {("hello", "gaston"), ("hello", "antonia"),},
        )


def test_add_probchoice_from_tuple():
    probchoice_as_tuples_iterable = [
        (0.5, "a", "a"),
        (0.2, "a", "b"),
        (0.3, "b", "b"),
    ]
    pd = ProbDatalogProgram()
    pd.add_probchoice_from_tuples(P, probchoice_as_tuples_iterable)
    assert P in pd.symbol_table
    assert (
        Constant[float](0.2),
        Constant[str]("a"),
        Constant[str]("b"),
    ) in pd.symbol_table[P].value


def test_add_probchoice_from_tuple_no_probability():
    pd = ProbDatalogProgram()
    with pytest.raises(NeuroLangException, match=r"probability"):
        pd.add_probchoice_from_tuples(P, [("a", "b"), ("b", "b"),])

def test_add_probchoice_does_not_sum_to_one():
    probchoice_as_tuples_iterable = [
        (0.5, "a", "a"),
        (0.2, "a", "b"),
        (0.1, "b", "b"),
    ]
    pd = ProbDatalogProgram()
    with pytest.raises(NeuroLangException, match=r"sum"):
        pd.add_probchoice_from_tuples(P, probchoice_as_tuples_iterable)
