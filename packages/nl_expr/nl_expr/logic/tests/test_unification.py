from ... import expressions
from ...logic import Conjunction, Disjunction, Union
from ..unification import (
    apply_substitution,
    merge_substitutions,
    most_general_unifier,
)

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


def test_unification():
    a = S_("a")
    b = S_("b")
    x = S_("x")
    y = S_("y")
    z = S_("z")

    assert most_general_unifier(a(x), a(x)) == (dict(), a(x))
    assert most_general_unifier(a(x), a(y)) == (dict(x=y), a(y))
    assert most_general_unifier(a(x, y), a(y, y)) == (dict(x=y), a(y, y))
    assert most_general_unifier(a(x, z), a(y, y)) == (dict(x=y, z=y), a(y, y))
    assert most_general_unifier(a(C_(1), z), a(y, y)) == (
        dict(y=C_(1), z=C_(1)),
        a(C_(1), C_(1)),
    )
    assert most_general_unifier(a(C_(1), C_(2)), a(y, y)) is None
    assert most_general_unifier(a(b(x)), a(b(y))) == (dict(x=y), a(b(y)))
    assert most_general_unifier(a(b(x), z), a(b(y), y)) == (
        dict(x=y, z=y),
        a(b(y), y),
    )

    assert most_general_unifier(a(x, y), a(x)) is None
    assert most_general_unifier(a(b(x), y), a(x, y)) is None


def test_apply_substitution_naryoperator():
    P = S_("P")
    Q = S_("Q")
    x = S_("x")
    y = S_("y")
    z = S_("z")

    union = Union((P(x), Q(x)))
    expected = Union((P(y), Q(y)))
    assert apply_substitution(union, {x: y}) == expected
    assert apply_substitution(union, {y: x}) == union

    union = Union((P(y), Q(x)))
    expected_1 = Union((P(y), Q(y)))
    expected_2 = Union((P(x), Q(x)))

    assert apply_substitution(union, {x: y}) == expected_1
    assert apply_substitution(union, {y: x}) == expected_2

    union = Union((P(y), Q(x, z)))
    expected_1 = Union((P(y), Q(y, z)))
    expected_2 = Union((P(x), Q(x, z)))

    assert apply_substitution(union, {x: y}) == expected_1
    assert apply_substitution(union, {y: x}) == expected_2

    union = Conjunction((P(x), Q(x)))
    expected = Conjunction((P(y), Q(y)))
    assert apply_substitution(union, {x: y}) == expected
    assert apply_substitution(union, {y: x}) == union

    union = Conjunction((P(y), Q(x)))
    expected_1 = Conjunction((P(y), Q(y)))
    expected_2 = Conjunction((P(x), Q(x)))

    assert apply_substitution(union, {x: y}) == expected_1
    assert apply_substitution(union, {y: x}) == expected_2

    union = Conjunction((P(y), Q(x, z)))
    expected_1 = Conjunction((P(y), Q(y, z)))
    expected_2 = Conjunction((P(x), Q(x, z)))

    assert apply_substitution(union, {x: y}) == expected_1
    assert apply_substitution(union, {y: x}) == expected_2

    union = Disjunction((P(x), Q(x)))
    expected = Disjunction((P(y), Q(y)))
    assert apply_substitution(union, {x: y}) == expected
    assert apply_substitution(union, {y: x}) == union

    union = Disjunction((P(y), Q(x)))
    expected_1 = Disjunction((P(y), Q(y)))
    expected_2 = Disjunction((P(x), Q(x)))

    assert apply_substitution(union, {x: y}) == expected_1
    assert apply_substitution(union, {y: x}) == expected_2

    union = Disjunction((P(y), Q(x, z)))
    expected_1 = Disjunction((P(y), Q(y, z)))
    expected_2 = Disjunction((P(x), Q(x, z)))

    assert apply_substitution(union, {x: y}) == expected_1
    assert apply_substitution(union, {y: x}) == expected_2


def test_merge_substitutions():
    assert merge_substitutions({1: 2}, {3: 4}) == {1: 2, 3: 4}
    assert merge_substitutions({1: 2}, {1: 2}) == {1: 2}
    assert merge_substitutions({1: 2}, {1: 4}) is None
