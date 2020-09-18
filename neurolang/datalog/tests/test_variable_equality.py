import operator

from ...expressions import Constant, Symbol
from ..expression_processing import VariableEqualityPropagator
from ...logic import Conjunction, Disjunction


P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")
c = Constant("c")

EQ = Constant(operator.eq)


def test_propagation_to_one_conjunct():
    conjunction = Conjunction((P(x, y), EQ(y, a)))
    propagator = VariableEqualityPropagator()
    result = propagator.walk(conjunction)
    assert result == Conjunction((P(x, a),))


def test_propagation_to_two_conjuncts():
    conjunction = Conjunction((P(x, y), EQ(y, a), Q(y, y)))
    propagator = VariableEqualityPropagator()
    result = propagator.walk(conjunction)
    assert result == Conjunction((P(x, a), Q(a, a)))


def test_propagation_ucq_two_conjunctions():
    conjunction_a = Conjunction((P(x, y), EQ(y, a), Q(y, y)))
    conjunction_b = Conjunction((P(x, y), EQ(y, b), Q(y, x)))
    ucq = Disjunction((conjunction_a, conjunction_b))
    propagator = VariableEqualityPropagator()
    result = propagator.walk(ucq)
    assert result == Disjunction(
        (Conjunction((P(x, a), Q(a, a))), Conjunction((P(x, b), Q(b, x))))
    )


def test_propagation_disjunction_with_nested_disjunction():
    conjunction_a = Conjunction((P(x), Q(x, y), EQ(y, a)))
    conjunction_b = Conjunction((Q(z, z), P(y), P(z), R(y, x), EQ(z, a)))
    conjunction_c = Conjunction((R(z, y), EQ(z, c)))
    expression = Conjunction(
        (
            (
                conjunction_a,
                Disjunction((conjunction_b, conjunction_c)),
                EQ(x, b),
            )
        )
    )
    propagator = VariableEqualityPropagator()
    result = propagator.walk(expression)
    assert result == Conjunction(
        (
            Conjunction((P(b), Q(b, a))),
            Disjunction(
                (
                    Conjunction((Q(a, a), P(a), P(a), R(a, b))),
                    Conjunction((R(c, a),)),
                )
            ),
        )
    )
