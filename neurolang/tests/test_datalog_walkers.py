import pytest

from .. import datalog_walkers as dw
from ..neurolang import (
    Constant, Symbol, FunctionApplication,
    ExistentialPredicate, UniversalPredicate
)

C_ = Constant
S_ = Symbol
F_ = FunctionApplication
E_ = ExistentialPredicate
A_ = UniversalPredicate


def test_atom():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')

    f = F_[bool](a, (b, C_(1)))

    restrictors = srv.walk(f)
    assert restrictors == {b: {f}}

    f = F_[bool](a, (b, c, C_(1)))

    restrictors = srv.walk(f)
    assert restrictors == {
        b: dw.Intersection({f}),
        c: dw.Intersection({f})
    }


def test_conjunction():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')
    d = C_(lambda x: x % 2 == 0)

    f = F_[bool](a, (b, c, C_(1)))
    g = F_[bool](d, (b,))
    e = f & g
    restrictors = srv.walk(e)
    assert restrictors == {
        b: dw.Intersection({f, g}),
        c: dw.Intersection({f})
    }


def test_disjunction():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')
    d = C_(lambda x: x % 2 == 0)

    f = F_[bool](a, (b, c, C_(1)))
    g = F_[bool](d, (b,))
    e = f | g
    restrictors = srv.walk(e)
    assert restrictors == {b: {
        dw.Union((dw.Intersection({f}), dw.Intersection({g})))
    }}


def test_inversion():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')

    f = F_[bool](a, (b, c, C_(1)))
    e = ~f

    restrictors = srv.walk(e)

    assert len(restrictors) == 0


def test_existential():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')

    f = F_[bool](a, (b, c, C_(1)))
    e = E_[bool](b, f)

    restrictors = srv.walk(e)
    assert restrictors == {c: dw.Intersection({f})}

    e = E_[bool](S_('x'), f)

    restrictors = srv.walk(e)
    assert restrictors is dw.undefined


def test_not_srnf():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')

    f = F_[bool](a, (b, c, C_(1)))
    e = A_[bool](b, f)

    with pytest.raises(dw.NeuroLangException):
        srv.walk(e)

    e2 = ~(f & E_[bool](b, f))

    with pytest.raises(dw.NeuroLangException):
        srv.walk(e2)
