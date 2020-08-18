from ...expressions import Symbol
from ...logic import Conjunction
from ..dichotomy_theorem_based_solver import is_hierarchical_without_self_joins

Q = Symbol('Q')
R = Symbol('R')
S = Symbol('S')
T = Symbol('T')
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')


def test_hierarchical_without_self_joins():
    q1 = Conjunction((Q(x), R(y), T(x)))
    q2 = Conjunction((Q(x),))
    q3 = Conjunction((Q(x), R(x, y)))
    q4 = Conjunction((Q(x), R(x, y, z), T(x, y)))

    assert is_hierarchical_without_self_joins(q1)
    assert is_hierarchical_without_self_joins(q2)
    assert is_hierarchical_without_self_joins(q3)
    assert is_hierarchical_without_self_joins(q4)


def test_not_hierarchical_without_self_joins():
    q1 = Conjunction((Q(x), T(x, y), R(y)))
    q2 = Conjunction((Q(x), Q(y), T(x, y)))

    assert not is_hierarchical_without_self_joins(q1)
    assert not is_hierarchical_without_self_joins(q2)
