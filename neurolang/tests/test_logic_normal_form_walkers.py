import pytest

from operator import or_, and_, invert, eq
import typing

from .. import logic_normal_form_walkers as nfw
from ..neurolang import (
    Constant, Symbol, FunctionApplication,
    ExistentialPredicate, UniversalPredicate
)

C_ = Constant
S_ = Symbol
F_ = FunctionApplication
EP_ = ExistentialPredicate
UP_ = UniversalPredicate

and_binary_functor = C_[typing.Callable[[bool, bool], bool]](and_)
or_binary_functor = C_[typing.Callable[[bool, bool], bool]](or_)
invert_functor = C_[typing.Callable[[bool], bool]](invert)


def test_undefined():
    a = set()

    assert (a + nfw.undefined) is nfw.undefined
    assert (nfw.undefined + a) is nfw.undefined
    assert (a - nfw.undefined) is nfw.undefined
    assert (nfw.undefined - a) is nfw.undefined
    assert (a | nfw.undefined) is nfw.undefined
    assert (nfw.undefined | a) is nfw.undefined
    assert (a & nfw.undefined) is nfw.undefined
    assert (nfw.undefined & a) is nfw.undefined
    assert (~nfw.undefined) is nfw.undefined


def test_atom():
    srv = nfw.SafeRangeVariablesWalker()

    a = C_(sum)
    x = S_('x')
    y = S_('y')

    f = F_[bool](a, (x, C_(1)))

    restrictors = srv.walk(f)
    assert restrictors == {x: {f}}

    f = F_[bool](a, (x, y, C_(1)))

    restrictors = srv.walk(f)
    assert restrictors == {
        x: nfw.Intersection({f}),
        y: nfw.Intersection({f})
    }


def test_conjunction():
    srv = nfw.SafeRangeVariablesWalker()

    a = C_(sum)
    x = S_('x')
    y = S_('y')
    c = C_[int](1)
    d = C_(lambda x: x % 2 == 0)

    f = F_[bool](a, (x, y, C_(1)))
    g = F_[bool](d, (x,))
    e = f & g
    restrictors = srv.walk(e)
    assert restrictors == {
        x: nfw.Intersection({f, g}),
        y: nfw.Intersection({f})
    }

    e = f & g & F_[bool](C_(eq), (x, c))
    restrictors = srv.walk(e)
    assert restrictors == {
        x: nfw.Intersection({f, g, c}),
        y: nfw.Intersection({f})
    }

    z = S_('z')
    e = f & g & F_[bool](C_(eq), (x, z))
    with pytest.raises(nfw.NeuroLangException):
        restrictors = srv.walk(e)

    e = f & EP_[bool](S_('z'), g)
    restrictors = srv.walk(e)

    assert restrictors is nfw.undefined

    e = EP_[bool](S_('z'), g) & f
    restrictors = srv.walk(e)

    assert restrictors is nfw.undefined


def test_disjunction():
    srv = nfw.SafeRangeVariablesWalker()

    a = C_(sum)
    x = S_('x')
    y = S_('y')
    d = C_(any)
    e = C_(all)

    f = F_[bool](a, (x, y, C_(1)))
    g = F_[bool](d, (x,))

    exp = f | g
    restrictors = srv.walk(exp)
    res = {x: nfw.Union((f, g))}

    assert restrictors == res

    h = F_[bool](e, (x,))
    exp = (f & g) | h
    restrictors = srv.walk(exp)
    res = {x: nfw.Union((nfw.Intersection((f, g)), h))}

    assert restrictors == res

    h = F_[bool](e, (x,))
    exp = f | (g & h)
    restrictors = srv.walk(exp)
    res = {x: nfw.Union((nfw.Intersection((h, g)), f))}

    assert restrictors == res

    e = f | EP_[bool](S_('z'), g)
    restrictors = srv.walk(e)

    assert restrictors is nfw.undefined

    e = EP_[bool](S_('z'), g) | f
    restrictors = srv.walk(e)

    assert restrictors is nfw.undefined


def test_inversion():
    srv = nfw.SafeRangeVariablesWalker()

    a = C_(sum)
    x = S_('x')
    y = S_('y')

    f = F_[bool](a, (x, y, C_(1)))
    e = ~f

    restrictors = srv.walk(e)

    assert len(restrictors) == 0


def test_existential():
    srv = nfw.SafeRangeVariablesWalker()

    a = C_(sum)
    x = S_('x')
    y = S_('y')

    f = F_[bool](a, (x, y, C_(1)))
    e = EP_[bool](x, f)

    restrictors = srv.walk(e)
    assert restrictors == {y: nfw.Intersection({f})}

    e = EP_[bool](S_('z'), f)

    restrictors = srv.walk(e)
    assert restrictors is nfw.undefined


def test_not_srnf():
    srv = nfw.SafeRangeVariablesWalker()

    a = C_(sum)
    x = S_('x')
    y = S_('y')

    f = F_[bool](a, (x, y, C_(1)))
    e = UP_[bool](x, f)

    with pytest.raises(nfw.NeuroLangException):
        srv.walk(e)

    e2 = ~(f & EP_[bool](x, f))

    with pytest.raises(nfw.NeuroLangException):
        srv.walk(e2)


def test_replace_variables():
    vsw = nfw.VariableSubstitutionWalker()

    a = C_(sum)
    x = S_('x')
    x_ = S_('x_')
    y = S_('y')

    f = F_[bool](a, (x, y, C_(1)))
    e = EP_[bool](x, f)

    exp = f & e
    exp_new = vsw.walk(exp)
    expected_result = (f & EP_[bool](x_, F_(a, (x_, y, C_(1))))).cast(bool)

    assert exp_new == expected_result

    vsw = nfw.VariableSubstitutionWalker()

    exp_new_2 = vsw.walk(exp_new)

    assert exp_new is exp_new_2


def test_push_neg_double_neg():
    csnrf = nfw.ConvertToSNRFWalker()

    a = S_('a')
    e = (~(~a).cast(bool)).cast(bool)

    res = csnrf.walk(e)

    assert res == a


def test_push_neg_and():
    csnrf = nfw.ConvertToSNRFWalker()

    a = S_[bool]('a')
    b = S_[bool]('b')
    e = (~(a & b).cast(bool)).cast(bool)

    res = csnrf.walk(e)
    assert res == or_binary_functor(invert_functor(a), invert_functor(b))


def test_push_neg_or():
    csnrf = nfw.ConvertToSNRFWalker()

    a = S_[bool]('a')
    b = S_[bool]('b')
    e = (~(a | b).cast(bool)).cast(bool)

    res = csnrf.walk(e)
    assert res == and_binary_functor(invert_functor(a), invert_functor(b))


def test_univ_to_ex():
    csnrf = nfw.ConvertToSNRFWalker()

    a = S_[typing.Callable[[bool], bool]]('a')
    b = S_[bool]('b')
    e = UP_[bool](b, a(b))

    exp_res = invert_functor(EP_[bool](b, invert_functor(a(b))))

    res = csnrf.walk(e)

    assert res == exp_res


def test_flatten_and():
    fw = nfw.FlattenMultipleLogicalOperators()

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
    fw = nfw.FlattenMultipleLogicalOperators()

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

    res, r = nfw.expression_to_SRNF_and_range(exp)
    assert r == {x: nfw.Intersection((exp,))}
    assert res is exp

    exp = UP_[bool](x, invert_functor(Q(x, y)))
    res, r = nfw.expression_to_SRNF_and_range(exp)
    exp_res = invert_functor(EP_[bool](x, Q(x, y)))
    assert r == {}
    assert res == exp_res
