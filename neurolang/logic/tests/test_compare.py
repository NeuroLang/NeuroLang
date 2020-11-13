import pytest

from ...expressions import Constant, Symbol
from .. import Conjunction, Implication
from ..compare import logic_exps_equal

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
x = Symbol("x")
y = Symbol("y")
a = Constant("a")


def test_compare_simple_expressions():
    assert logic_exps_equal(P, P)
    assert not logic_exps_equal(P, Q)
    with pytest.raises(ValueError):
        logic_exps_equal(P, 3)


def test_compare_conjunctions():
    conj1 = Conjunction((P(x, y), Q(x)))
    conj2 = Conjunction((Q(x), P(x, y)))
    assert logic_exps_equal(conj1, conj2)
    conj3 = Conjunction((Q(y), P(x, y)))
    assert not logic_exps_equal(conj1, conj3)
    conj4 = Conjunction((P(x, y), Q(y)))
    assert not logic_exps_equal(conj1, conj4)


def test_compare_implications():
    impl1 = Implication(R(x, a), Conjunction((Q(a, a), P(x, a))))
    impl2 = Implication(R(x, a), Conjunction((P(x, a), Q(a, a))))
    assert logic_exps_equal(impl1, impl2)
