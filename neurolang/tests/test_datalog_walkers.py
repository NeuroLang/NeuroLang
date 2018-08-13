import pytest

from operator import or_, and_, invert
import typing

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

and_binary_functor = C_[typing.Callable[[bool, bool], bool]](and_)
or_binary_functor = C_[typing.Callable[[bool, bool], bool]](or_)
invert_functor = C_[typing.Callable[[bool], bool]](invert)


def test_undefined():
    a = set()

    assert (a + dw.undefined) is dw.undefined
    assert (dw.undefined + a) is dw.undefined
    assert (a - dw.undefined) is dw.undefined
    assert (dw.undefined - a) is dw.undefined
    assert (a | dw.undefined) is dw.undefined
    assert (dw.undefined | a) is dw.undefined
    assert (a & dw.undefined) is dw.undefined
    assert (dw.undefined & a) is dw.undefined
    assert (~dw.undefined) is dw.undefined


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

    e = f & E_[bool](S_('x'), g)
    restrictors = srv.walk(e)

    assert restrictors is dw.undefined

    e = E_[bool](S_('x'), g) & f
    restrictors = srv.walk(e)

    assert restrictors is dw.undefined


def test_disjunction():
    srv = dw.SafeRangeVariablesWalker()

    a = C_(sum)
    b = S_('b')
    c = S_('c')
    d = C_(any)
    e = C_(all)

    f = F_[bool](a, (b, c, C_(1)))
    g = F_[bool](d, (b,))

    exp = f | g
    restrictors = srv.walk(exp)
    res = {b: dw.Union((f, g))}

    assert restrictors == res

    h = F_[bool](e, (b,))
    exp = (f & g) | h
    restrictors = srv.walk(exp)
    res = {b: dw.Union((dw.Intersection((f, g)), h))}

    assert restrictors == res

    h = F_[bool](e, (b,))
    exp = f | (g & h)
    restrictors = srv.walk(exp)
    res = {b: dw.Union((dw.Intersection((h, g)), f))}

    assert restrictors == res

    e = f | E_[bool](S_('x'), g)
    restrictors = srv.walk(e)

    assert restrictors is dw.undefined

    e = E_[bool](S_('x'), g) | f
    restrictors = srv.walk(e)

    assert restrictors is dw.undefined


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


def test_replace_variables():
    vsw = dw.VariableSubstitutionWalker()

    a = C_(sum)
    b = S_('b')
    b_ = S_('b_')
    c = S_('c')

    f = F_[bool](a, (b, c, C_(1)))
    e = E_[bool](b, f)

    exp = f & e
    exp_new = vsw.walk(exp)
    expected_result = (f & E_[bool](b_, F_(a, (b_, c, C_(1))))).cast(bool)

    assert exp_new == expected_result

    vsw = dw.VariableSubstitutionWalker()

    exp_new_2 = vsw.walk(exp_new)

    assert exp_new is exp_new_2


def test_push_neg_double_neg():
    csnrf = dw.ConvertToSNRFWalker()

    a = S_('a')
    e = (~(~a).cast(bool)).cast(bool)

    res = csnrf.walk(e)

    assert res == a


def test_push_neg_and():
    csnrf = dw.ConvertToSNRFWalker()

    a = S_[bool]('a')
    b = S_[bool]('b')
    e = (~(a & b).cast(bool)).cast(bool)

    res = csnrf.walk(e)
    assert res == or_binary_functor(invert_functor(a), invert_functor(b))


def test_push_neg_or():
    csnrf = dw.ConvertToSNRFWalker()

    a = S_[bool]('a')
    b = S_[bool]('b')
    e = (~(a | b).cast(bool)).cast(bool)

    res = csnrf.walk(e)
    assert res == and_binary_functor(invert_functor(a), invert_functor(b))


def test_univ_to_ex():
    csnrf = dw.ConvertToSNRFWalker()

    a = S_[typing.Callable[[bool], bool]]('a')
    b = S_[bool]('b')
    e = A_[bool](b, a(b))

    exp_res = invert_functor(E_[bool](b, invert_functor(a(b))))

    res = csnrf.walk(e)

    assert res == exp_res


def test_flatten_and():
    fw = dw.FlattenMultipleLogicalOperators()

    args = [S_[bool](f'a{i}') for i in range(5)]

    exp = and_binary_functor(
        and_binary_functor(
            args[0],
            and_binary_functor(
                args[1],
                and_binary_functor(args[2], args[3])
            ),
        ),
        args[4]
    )

    res = fw.walk(exp)

    assert res.functor == C_(and_)
    assert set(res.args) == set(args)


def test_flatten_or():
    fw = dw.FlattenMultipleLogicalOperators()

    args = [S_[bool](f'a{i}') for i in range(5)]

    exp = or_binary_functor(
        or_binary_functor(
            args[0],
            or_binary_functor(
                args[1],
                or_binary_functor(args[2], args[3])
            ),
        ),
        args[4]
    )

    res = fw.walk(exp)

    assert res.functor == C_(or_)
    assert set(res.args) == set(args)


def test_SNRF_and_range():
    R = C_[typing.Callable[[bool], bool]](any)
    Q = C_[typing.Callable[[bool], bool]](all)

    x = S_[bool]('x')
    y = S_[bool]('y')

    exp = R(x)

    res, r = dw.expression_to_SRNF_and_range(exp)
    assert r == {x: dw.Intersection((exp,))}
    assert res is exp

    exp = A_[bool](x, invert_functor(Q(x, y)))
    res, r = dw.expression_to_SRNF_and_range(exp)
    exp_res = invert_functor(E_[bool](x, Q(x, y)))
    assert r == {}
    assert res == exp_res
