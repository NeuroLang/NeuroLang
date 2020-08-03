import typing

import numpy
import pytest

from ....datalog import Fact
from ....exceptions import ForbiddenDisjunctionError
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from ....utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expression_processing import add_to_union, is_probabilistic_fact
from ...expressions import (
    Grounding,
    ProbabilisticChoiceGrounding,
    ProbabilisticPredicate,
)
from ..grounding import ground_cplogic_program
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
x = Symbol("x")
y = Symbol("y")
p = Symbol("p")
a = Constant("a")
b = Constant("b")


def test_cplogic_grounding():
    pfact1 = Implication(ProbabilisticPredicate(p, P(a)), Constant[bool](True))
    pfact2 = Implication(ProbabilisticPredicate(p, P(b)), Constant[bool](True))
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    code = Union((pfact1, pfact2, rule, Fact(Q(a)), Fact(Q(b))))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    grounded = ground_cplogic_program(cpl_program)
    matching_groundings = [
        grounding
        for grounding in grounded.expressions
        if is_probabilistic_fact(grounding.expression)
        and grounding.expression.consequent.body.functor == P
    ]
    assert len(matching_groundings) == 1
    expected = Grounding(
        rule,
        Constant[typing.AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=["a", "b"], columns=["x"])
        ),
    )
    assert expected in grounded.expressions

    code = Union((Fact(P(a, b)), Fact(P(b, b)), Fact(Q(a)), Fact(Q(b))))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    grounded = ground_cplogic_program(cpl_program)
    assert len(grounded.expressions) == 2
    for grounding in grounded.expressions:
        if grounding.expression.consequent.functor == P:
            assert numpy.all(
                numpy.vstack(list(grounding.relation.value.itervalues()))
                == numpy.vstack(
                    [
                        numpy.array(["a", "b"], dtype=str),
                        numpy.array(["b", "b"], dtype=str),
                    ]
                )
            )
        elif grounding.expression.consequent.functor == Q:
            assert numpy.all(
                numpy.array(
                    [tuple(nt) for nt in grounding.relation.value.itervalues()]
                )
                == numpy.array([["a"], ["b"]], dtype=str)
            )


def test_cplogic_grounding_general():
    pfacts = Union(
        tuple(
            Implication(
                ProbabilisticPredicate(prob, P(const)), Constant[bool](True)
            )
            for (prob, const) in {
                (Constant[float](0.2), a),
                (Constant[float](0.6), b),
            }
        )
    )
    facts = (Fact(Q(a)),)
    rule = Implication(Z(x), Conjunction((P(x), Q(x))))
    code = add_to_union(pfacts, (rule,))
    code = add_to_union(pfacts, facts)
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    grounded = ground_cplogic_program(cpl_program)
    pfact_grounding = next(
        grounding
        for grounding in grounded.expressions
        if isinstance(grounding.expression, Implication)
        and isinstance(grounding.expression.consequent, ProbabilisticPredicate)
        and grounding.expression.consequent.body.functor == P
    )
    assert set(pfact_grounding.relation.value) == {(0.2, "a"), (0.6, "b")}


def test_cplogic_grounding_with_pchoice():
    probchoice_set = {(0.5, "a"), (0.25, "b"), (0.25, "c")}
    code = Union(tuple())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    cpl_program.add_probabilistic_choice_from_tuples(P, probchoice_set)
    grounded = ground_cplogic_program(cpl_program)
    assert isinstance(grounded.expressions[0], ProbabilisticChoiceGrounding)


def test_forbidden_disjunction():
    rule_a = Implication(P(x), Q(x))
    rule_b = Implication(P(y), Z(y))
    code = Union((rule_a, rule_b))
    cpl = CPLogicProgram()
    pfact_sets = {
        Q: {(1.0, "a"), (0.5, "b")},
        Z: {(0.2, "z")},
    }
    for pred_symb, pfact_set in pfact_sets.items():
        cpl.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl.walk(code)
    with pytest.raises(ForbiddenDisjunctionError):
        ground_cplogic_program(cpl)
