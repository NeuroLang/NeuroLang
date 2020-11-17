import typing

from ...datalog.expression_processing import EQ
from ...expressions import Constant, Symbol
from ...logic import Conjunction
from ..dichotomy_theorem_based_solver import is_hierarchical_without_self_joins

Q = Symbol('Q')
R = Symbol('R')
S = Symbol('S')
T = Symbol('T')
w = Symbol('w')
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


def test_hierarchical_despite_multiple_equalities():
    q = Conjunction((Q(x), R(y), EQ(y, z), EQ(x, w)))
    assert is_hierarchical_without_self_joins(q)


def test_non_hierarchical_builtin():
    some_builtin = Constant[typing.Callable[[typing.Any, typing.Any], bool]](
        "some_builtin",
        verify_type=False,
        auto_infer_type=False,
    )
    q = Conjunction((Q(x), R(y), some_builtin(y, z), some_builtin(x, w)))
    assert not is_hierarchical_without_self_joins(q)
