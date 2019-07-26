from ..probabilistic.unification import (
    most_general_unifier, merge_substitutions,
    apply_substitution_to_delta_term, apply_substitution
)
from .. import expressions
from ..probabilistic.ppdl import DeltaTerm

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication

bernoulli = S_('bernoulli')
p = S_('p')
q = S_('q')
P = S_('P')
x = S_('x')
y = S_('y')

def test_apply_substitution_to_delta_term():
    dterm = DeltaTerm(bernoulli, (p, ))
    new_dterm = apply_substitution_to_delta_term(dterm, {p: q})
    assert new_dterm == DeltaTerm(bernoulli, (q, ))

    substitution = {S_('random_symbol'): S_('another_random_symbol')}
    new_dterm = apply_substitution_to_delta_term(dterm, substitution)
    assert new_dterm == dterm


def test_apply_substitution_to_delta_atom():
    datom = P(x, DeltaTerm(bernoulli, (p, )))
    new_datom = apply_substitution(datom, {p: q})
    assert new_datom == P(x, DeltaTerm(bernoulli, (q, )))


def test_unification_of_delta_atom():
    a = P(x, DeltaTerm(bernoulli, (p, )))
    b = P(y, DeltaTerm(bernoulli, (q, )))
    mgu = most_general_unifier(a, b)
    assert mgu is not None
    unifier, new_exp = mgu
    assert unifier == {x: y, p: q}
