from typing import Mapping, AbstractSet

import pytest

from ....datalog.expressions import Fact
from ....logic import Union, Conjunction, Implication, ExistentialPredicate
from ....exceptions import NeuroLangException
from ....expressions import Constant, ExpressionBlock, Symbol
from ...expressions import ProbabilisticPredicate
from ..program import CPLogicProgram

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


def test_cplogic_program():
    cpl = CPLogicProgram()

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

    cpl.walk(code)

    assert cpl.extensional_database() == {
        Z: Constant(frozenset({Constant((a,)), Constant((b,))}))
    }
    assert cpl.intensional_database() == {
        Q: Union([Implication(Q(x), Conjunction([P(x), Z(x)]))])
    }
    assert cpl.probabilistic_facts() == {
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
    cpl = CPLogicProgram()
    cpl.walk(code)


def test_multiple_probfact_same_pred_symb():
    cpl = CPLogicProgram()
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
    cpl.walk(code)
    assert cpl.extensional_database() == {
        Z: Constant(frozenset({Constant((a,)), Constant((b,))}))
    }
    assert cpl.intensional_database() == {
        Q: Union([Implication(Q(x), Conjunction([P(x), Z(x)]))])
    }
    assert len(cpl.probabilistic_facts()) == 1
    assert P in cpl.probabilistic_facts()
    probfacts = cpl.probabilistic_facts()[P]
    assert isinstance(probfacts, Constant[AbstractSet])


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
    cpl = CPLogicProgram()
    cpl.walk(code)

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
    cpl = CPLogicProgram()
    with pytest.raises(NeuroLangException, match=r"can only be used"):
        cpl.walk(code)


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
def test_cplogic_pfact_type_inference():
    code = ExpressionBlock(
        [
            Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True)),
            Implication(Q(x), Conjunction([P(x), Z(x), R(x)])),
            Implication(Q(x), Conjunction([P(x), Y(x), R(x)])),
        ]
    )
    cpl = CPLogicProgram()
    cpl.walk(code)
    assert cpl.symbol_table[cpl.typing_symbol] == Constant[Mapping](
        {P: Constant[Mapping]({Constant[int](0): Constant[AbstractSet]({R})})}
    )


@pytest.mark.skip()
def test_cplogic_pfact_cant_infer_type():
    pfact = Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    rule = Implication(Q(x), Conjunction([P(x), Z(x), R(x)]))
    cpl = CPLogicProgram()
    with pytest.raises(NeuroLangException, match="could not be inferred"):
        cpl.walk(ExpressionBlock([pfact, rule]))


def test_add_probfacts_from_tuple():
    cpl = CPLogicProgram()
    cpl.walk(ExpressionBlock([]))
    cpl.add_probfacts_from_tuples(
        P, {(0.3, "hello", "gaston"), (0.7, "hello", "antonia"),},
    )
    assert P in cpl.pfact_pred_symbs
    assert (
        Constant[float](0.7),
        Constant[str]("hello"),
        Constant[str]("antonia"),
    ) in cpl.symbol_table[P].value


def test_add_probfacts_from_tuple_no_probability():
    cpl = CPLogicProgram()
    cpl.walk(ExpressionBlock([]))
    with pytest.raises(NeuroLangException, match=r"probability"):
        cpl.add_probfacts_from_tuples(
            P, {("hello", "gaston"), ("hello", "antonia"),},
        )


def test_add_probchoice_from_tuple():
    probchoice_as_tuples_iterable = [
        (0.5, "a", "a"),
        (0.2, "a", "b"),
        (0.3, "b", "b"),
    ]
    cpl = CPLogicProgram()
    cpl.add_probchoice_from_tuples(P, probchoice_as_tuples_iterable)
    assert P in cpl.symbol_table
    assert (
        Constant[float](0.2),
        Constant[str]("a"),
        Constant[str]("b"),
    ) in cpl.symbol_table[P].value


def test_add_probchoice_from_tuple_no_probability():
    cpl = CPLogicProgram()
    with pytest.raises(NeuroLangException, match=r"probability"):
        cpl.add_probchoice_from_tuples(P, [("a", "b"), ("b", "b"),])


def test_add_probchoice_does_not_sum_to_one():
    probchoice_as_tuples_iterable = [
        (0.5, "a", "a"),
        (0.2, "a", "b"),
        (0.1, "b", "b"),
    ]
    cpl = CPLogicProgram()
    with pytest.raises(NeuroLangException, match=r"sum"):
        cpl.add_probchoice_from_tuples(P, probchoice_as_tuples_iterable)
