import typing

import pytest
import numpy

from ....exceptions import NeuroLangException
from ....expressions import Symbol, Constant, ExpressionBlock
from ....logic import Implication, Conjunction
from ....datalog import Fact
from ....utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expressions import ProbabilisticPredicate, Grounding
from ...expression_processing import is_probabilistic_fact
from ..grounding import ground_cplogic_program

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
x = Symbol("x")
p = Symbol("p")
a = Constant("a")
b = Constant("b")


def test_cplogic_grounding():
    pfact1 = Implication(ProbabilisticPredicate(p, P(a)), Constant[bool](True))
    pfact2 = Implication(ProbabilisticPredicate(p, P(b)), Constant[bool](True))
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    code = ExpressionBlock([pfact1, pfact2, rule, Fact(Q(a)), Fact(Q(b))])
    grounded = ground_cplogic_program(code)
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

    code = ExpressionBlock(
        [Fact(P(a, b)), Fact(P(b, b)), Fact(Q(a)), Fact(Q(b))]
    )
    grounded = ground_cplogic_program(code)
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
                numpy.array(list(grounding.relation.value.itervalues()))
                == numpy.array([["a"], ["b"]], dtype=str)
            )


@pytest.mark.skip()
def test_cplogic_grounding_general():
    pfact = Implication(ProbabilisticPredicate(p, P(x)), Constant[bool](True))
    rule = Implication(Z(x), Conjunction([P(x), Q(x)]))
    code = ExpressionBlock([pfact, rule, Fact(Q(a)), Fact(Q(b))])
    grounded = ground_cplogic_program(code)
    expected = Grounding(
        pfact,
        Constant[typing.AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=["a", "b"], columns=["x"])
        ),
    )
    assert expected in grounded.expressions
    expected = Grounding(
        rule,
        Constant[typing.AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=["a", "b"], columns=["x"])
        ),
    )
    assert expected in grounded.expressions

    code = ExpressionBlock(
        [Fact(P(a, b)), Fact(P(b, b)), Fact(Q(a)), Fact(Q(b))]
    )
    grounded = ground_cplogic_program(code)
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
                numpy.array(list(grounding.relation.value.itervalues()))
                == numpy.array([["a"], ["b"]], dtype=str)
            )


def test_unsupported_grounding_program_with_disjunction():
    code = ExpressionBlock([Implication(Q(x), P(x)), Implication(Q(x), R(x))])
    with pytest.raises(NeuroLangException, match=r"supported"):
        ground_cplogic_program(code)
