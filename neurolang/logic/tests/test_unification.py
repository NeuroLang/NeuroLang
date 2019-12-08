from ..unification import most_general_unifier, merge_substitutions
from ... import expressions

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


def test_unification():
    a = S_('a')
    b = S_('b')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    assert most_general_unifier(a(x), a(x)) == (dict(), a(x))
    assert most_general_unifier(a(x), a(y)) == (dict(x=y), a(y))
    assert most_general_unifier(a(x, y), a(y, y)) == (dict(x=y), a(y, y))
    assert most_general_unifier(a(x, z), a(y, y)) == (dict(x=y, z=y), a(y, y))
    assert most_general_unifier(a(C_(1), z), a(y, y)) ==\
        (dict(y=C_(1), z=C_(1)), a(C_(1), C_(1)))
    assert most_general_unifier(a(C_(1), C_(2)), a(y, y)) is None
    assert most_general_unifier(a(b(x)), a(b(y))) == (dict(x=y), a(b(y)))
    assert most_general_unifier(a(b(x), z), a(b(y), y)) ==\
        (dict(x=y, z=y), a(b(y), y))

    assert most_general_unifier(a(x, y), a(x)) is None
    assert most_general_unifier(a(b(x), y), a(x, y)) is None


def test_merge_substitutions():
    assert merge_substitutions({1: 2}, {3: 4}) == {1: 2, 3: 4}
    assert merge_substitutions({1: 2}, {1: 2}) == {1: 2}
    assert merge_substitutions({1: 2}, {1: 4}) is None
