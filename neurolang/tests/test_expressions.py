from ..expressions import (
    Symbol, Constant,
    SymbolApplication,
    evaluate
)

import operator as op
import inspect


def test_symbol_application():
    a = Symbol('a')
    b = Symbol('b')

    oadd = SymbolApplication(op.add)
    c = oadd(2, 3)
    assert c.__wrapped__ == oadd
    assert c.args == (2, 3)
    assert evaluate(c) == 5

    fva = SymbolApplication(op.add)(a, 3)
    fvb = SymbolApplication(op.sub)(fva, 10)
    fvc = SymbolApplication(op.mul)(fvb, b)
    fvd = SymbolApplication(a, kwargs={'c': b})
    fve = SymbolApplication(a, kwargs={'c': op.add(b, 2)})

    assert a in fva._symbols and (len(fva._symbols) == 1)
    assert evaluate(fva, a=2) == 5
    assert a in fvb._symbols and (len(fvb._symbols) == 1)
    assert evaluate(fvb, a=2) == -5
    assert b in fvc._symbols and (len(fvc._symbols) == 2)
    assert isinstance(evaluate(fvc, b=2), SymbolApplication)
    assert evaluate(evaluate(fvc, b=3), a=2) == evaluate(fvc, a=2, b=3) == -15
    assert evaluate(
        evaluate(fvd, a=lambda *args, **kwargs: kwargs['c']),
        b=2
    ) == 2
    assert evaluate(
        evaluate(fvd, b=2),
        a=lambda *args, **kwargs: kwargs['c']
    ) == 2
    assert evaluate(
        evaluate(fve, b=2),
        a=lambda *args, **kwargs: kwargs['c']
    ) == 4


def test_symbol_method_and_operator():
    a = Symbol('a')
    fva = a.__len__()
    fvb = a - 4
    fvc = 4 - a
    fve = a[2]

    assert evaluate(a, a=1) == 1
    assert evaluate(fva, a={1}) == 1
    assert evaluate(fvb, a=1) == -3
    assert evaluate(fvc, a=1) == 3
    assert evaluate(fvc * 2, a=1) == 6
    assert evaluate(a | False, a=True)
    assert evaluate(fve, a=[0, 1, 2]) == 2


def test_constant_method_and_operator():
    a = Constant(1, int)
    fva = a + 1
    b = Constant({1}, int)
    fvb = b.__len__()
    fbc = b.union({1, 2})

    assert a == 1
    assert evaluate(a) == 1
    assert evaluate(fva) == 2
    assert evaluate(fvb) == 1
    assert evaluate(fbc) == {1, 2}


def test_symbol_wrapping():
    def f(a: int)->float:
        '''
        test help
        '''
        return 2. * int(a)

    fva = SymbolApplication(f)
    x = Symbol('x', type_=int)
    fvb = fva(x)
    assert fva.__annotations__ == f.__annotations__
    assert fva.__doc__ == f.__doc__
    assert fva.__name__ == f.__name__
    assert fva.__qualname__ == f.__qualname__
    assert fva.__module__ == f.__module__
    assert inspect.signature(fva) == inspect.signature(f)
    assert fvb.type == float
    assert x.type == int
