from ..unification import (
    most_general_unifier, merge_substitutions,
    apply_substitution_to_delta_term, apply_substitution
)
from .. import expressions
from ..generative_datalog import DeltaTerm, DeltaAtom

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


def test_unification():
    a = S_('a')
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


def test_merge_substitutions():
    assert merge_substitutions({1: 2}, {3: 4}) == {1: 2, 3: 4}
    assert merge_substitutions({1: 2}, {1: 2}) == {1: 2}
    assert merge_substitutions({1: 2}, {1: 4}) is None


def test_apply_substitution_to_delta_term():
    dist_name = C_('bernoulli')
    parameter_symbol = S_('p')
    new_parameter_symbol = S_('q')
    dterm = DeltaTerm(dist_name, parameter_symbol)
    substitution = {parameter_symbol: new_parameter_symbol}
    new_dterm = apply_substitution_to_delta_term(dterm, substitution)
    assert new_dterm == DeltaTerm(dist_name, new_parameter_symbol)

    substitution = {S_('random_symbol'): S_('another_random_symbol')}
    new_dterm = apply_substitution_to_delta_term(dterm, substitution)
    assert new_dterm == dterm


def test_apply_substitution_to_delta_atom():
    dist_name = C_('bernoulli')
    parameter_symbol = S_('p')
    new_parameter_symbol = S_('q')
    P = S_('P')
    x = S_('x')
    dterm = DeltaTerm(dist_name, parameter_symbol)
    datom = DeltaAtom(P, (x, dterm))
    substitution = {parameter_symbol: new_parameter_symbol}
    new_datom = apply_substitution(datom, substitution)
    assert new_datom == DeltaAtom(
        P, (x, DeltaTerm(dist_name, new_parameter_symbol))
    )


def test_unification_of_delta_atom():
    a = DeltaAtom(S_('P'), (S_('x'), DeltaTerm(C_('bernoulli'), S_('p'))))
    b = DeltaAtom(S_('P'), (S_('y'), DeltaTerm(C_('bernoulli'), S_('q'))))
    mgu = most_general_unifier(a, b)
    assert mgu is not None
    unifier, new_exp = mgu
    assert unifier == {S_('x'): S_('y'), S_('p'): S_('q')}
