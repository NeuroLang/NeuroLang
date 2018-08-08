import pytest

from .. import expressions
from ..expressions import (
    Expression, ToBeInferred, expressions_behave_as_objects
)

from ..expression_walker import ExpressionBasicEvaluator

import operator as op
import inspect
from typing import Set, Callable

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication


def evaluate(expression, **kwargs):
    ebe = ExpressionBasicEvaluator()
    for k, v in kwargs.items():
        if not isinstance(v, Expression):
            v = C_(v)
        ebe.symbol_table[k] = v
    return ebe.walk(expression)


def test_expression_behaves_as_object():
    a = S_('a')

    with pytest.raises(AttributeError):
        c = a.b

    with expressions_behave_as_objects():
        c = a.b

    assert c.functor.value is getattr
    assert c.args[0] is a
    assert c.args[1] == C_('b')


def test_symbol_application():
    a = S_('a')
    b = S_('b')

    oadd = C_[Callable[[int, int], int]](op.add)
    osub = C_[Callable[[int, int], int]](op.sub)
    omul = C_[Callable[[int, int], int]](op.mul)
    c = oadd(C_[int](2), C_[int](3))
    assert c.functor == oadd
    assert all((e1.value == e2 for e1, e2 in zip(c.args, (2, 3))))
    assert evaluate(c) == 5

    fva = oadd(a, C_[int](3))
    fvb = osub(fva, C_[int](10))
    fvc = omul(fvb, b)
    fvd = F_(a, None, kwargs={'c': b})
    fve = F_(a, None, kwargs={'c': op.add(b, C_(2))})

    assert a in fva._symbols and (len(fva._symbols) == 1)
    assert evaluate(fva, a=C_(2)) == 5
    assert a in fvb._symbols and (len(fvb._symbols) == 1)
    assert evaluate(fvb, a=C_(2)) == -5
    assert b in fvc._symbols and (len(fvc._symbols) == 2)
    assert isinstance(evaluate(fvc, b=C_(2)), expressions.FunctionApplication)
    assert evaluate(
        evaluate(fvc, b=C_(3)), a=C_(2)
    ) == evaluate(
        fvc, a=C_(2), b=C_(3)
    ) == -15
    return
    assert evaluate(
        evaluate(fvd, a=C_(lambda *args, **kwargs: kwargs['c'])), b=C_(2)
    ) == 2
    assert evaluate(
        evaluate(fvd, b=C_(2)), a=lambda *args, **kwargs: kwargs['c']
    ) == 2
    assert evaluate(
        evaluate(fve, b=C_(2)), a=lambda *args, **kwargs: kwargs['c']
    ) == 4


def test_symbol_method_and_operator():
    with expressions_behave_as_objects():
        a = S_('a')
        fva = a.__len__()
        fvb = a - C_[int](4)
        fvc = C_[int](4) - a
        fve = a[C_(2)]

    assert evaluate(a, a=C_(1)) == 1
    assert evaluate(fva, a=C_[Set[int]]({1})) == 1
    assert evaluate(fvb, a=C_(1)) == -3
    assert evaluate(fvc, a=C_(1)) == 3
    assert evaluate(fvc * C_(2), a=C_(1)) == 6
    assert evaluate(a | C_(False), a=True)
    assert evaluate(fve, a=C_([C_(x) for x in (0, 1, 2)])) == 2


def test_constant_method_and_operator():
    with expressions_behave_as_objects():
        a = C_[int](1)
        fva = a + C_(1)
        b = C_[Set[int]]({1})
        fvb = b.__len__()
        fbc = b.union(C_({C_(1), C_(2)}))

    assert a == 1
    assert evaluate(a) == 1
    assert evaluate(fva) == 2
    assert evaluate(fvb) == 1
    assert evaluate(fbc).value == {1, 2}


def test_symbol_wrapping():
    def f(a: int) -> float:
        '''
        test help
        '''
        return 2. * int(a)

    fva = C_(f)
    x = S_[int]('x')
    fvb = fva(x)
    assert fva.__annotations__ == f.__annotations__
    assert fva.__doc__ == f.__doc__
    assert fva.__name__ == f.__name__
    assert fva.__qualname__ == f.__qualname__
    assert fva.__module__ == f.__module__
    assert inspect.signature(fva) == inspect.signature(f)
    assert fvb.type == float
    assert x.type == int


def test_compatibility_for_pattern_matching():
    for symbol_name in dir(expressions):
        symbol = getattr(expressions, symbol_name)
        if not (
            type(symbol) == type and
            issubclass(symbol, expressions.Expression) and
            symbol != expressions.Expression and
            symbol != expressions.Definition
        ):
            continue
        signature = inspect.signature(symbol)
        argnames = [
            name for name, param in signature.parameters.items()
            if param.default == signature.empty
        ]
        args = (..., ) * len(argnames)
        instance = symbol(*args)

        for argname in argnames:
            assert getattr(instance, argname) == ...


def test_instance_check():
    c = C_[int](2)
    assert isinstance(c, C_)
    assert isinstance(c, C_[int])
    assert issubclass(c.__class__, C_)
    assert c.type is int


def test_typecast_check():
    s = S_('a')
    s_float = s.cast(float)

    assert isinstance(s, S_)
    assert s.type is ToBeInferred
    assert isinstance(s_float, S_)
    assert isinstance(s_float, S_[float])
    assert issubclass(s.__class__, S_)
    assert s_float.type is float
    assert s_float.name == 'a'


def test_fa_composition_symbols_correctly_propagated():
    fa1 = S_('fa1')
    fa2 = S_('fa2')
    fa3 = S_('fa3')
    a = S_('a')
    b = S_('b')
    c = S_('c')
    d = S_('d')
    e = S_('e')

    expression = fa1(a, fa2(b, fa3(c, d), e))

    for symbol in [a, b, c, d, e]:
        assert symbol in expression._symbols
