import inspect
import operator as op
from typing import AbstractSet, Callable, Mapping, Sequence, Tuple

import pytest

from .. import expressions, logic
from ..expression_walker import (
    ExpressionBasicEvaluator,
    TypedSymbolTableEvaluator,
)
from ..expressions import Expression, Unknown, expressions_behave_as_objects

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
L_ = expressions.Lambda
E_ = logic.ExistentialPredicate
U_ = logic.UniversalPredicate


class Evaluator(ExpressionBasicEvaluator, TypedSymbolTableEvaluator):
    pass


def test_build_constants():
    a = C_('ab')
    assert a.type is str and a.value == 'ab'

    b = C_(['a'])
    assert b.type is Sequence[str]
    assert b.value[0].type is str and b.value[0].value == 'a'

    b = C_(('a', 1))
    assert b.type is Tuple[str, int]
    assert b.value[0].type is str and b.value[0].value == 'a'
    assert b.value[1].type is int and b.value[1].value == 1

    b = C_({'a'})
    assert b.type is AbstractSet[str]
    v = b.value.pop()
    assert v.type is str and v.value == 'a'

    b = C_({'a': 1})
    assert b.type is Mapping[str, int]
    assert len(b.value) == 1
    assert C_('a') in b.value
    assert C_(1) in b.value.values()


def test_fresh_symbol():
    s1 = S_.fresh()
    s2 = S_.fresh()
    s3 = S_[int].fresh()
    s4 = S_('a')

    assert isinstance(s1, S_) and s1.type is Unknown
    assert s1 != s2
    assert s1 != s2 and s2 != s3 and s3.type is int
    assert s1.is_fresh
    assert s2.is_fresh
    assert s3.is_fresh
    assert not s4.is_fresh


def test_fresh_symbol_2():
    # Reset sequence generators as if the program just started
    if hasattr(S_, "_fresh_generator_"):
        del S_._fresh_generator_
    if hasattr(S_[str], "_fresh_generator_"):
        del S_[str]._fresh_generator_

    s2 = S_[str].fresh()
    s1 = S_.fresh()

    assert s1.name != s2.name


def evaluate(expression, **kwargs):
    ebe = Evaluator()
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


def test_symbol_method_and_operator():
    with expressions_behave_as_objects():

        a = S_('a')
        fva = a.__len__()
        fvb = a - C_[int](4)
        fvc = C_[int](4) - a
        fve = a[C_(2)]

    assert evaluate(a, a=C_(1)) == 1
    assert evaluate(fva, a=C_[AbstractSet[int]](set((C_(1),)))) == 1
    assert evaluate(fvb, a=C_(1)) == -3
    assert evaluate(fvc, a=C_(1)) == 3
    assert evaluate(fvc * C_(2), a=C_(1)) == 6
    assert evaluate(a | C_(False), a=True)
    assert evaluate(fve, a=C_([C_(x) for x in (0, 1, 2)])) == 2


def test_constant_method_and_operator():
    with expressions_behave_as_objects():
        a = C_[int](1)
        fva = a + C_(1)
        b = C_[AbstractSet[int]]({C_(1)})
        fvb = b.__len__()
        fbc = b.union(C_({C_(1), C_(2)}))

    assert a == 1
    assert evaluate(a) == 1
    assert evaluate(fva) == 2
    assert evaluate(fvb) == 1
    assert evaluate(fbc).value == {1, 2}


def test_lambda_expression():
    l_ = L_[Callable[[], int]](tuple(), C_[int](2))
    fa = F_[int](l_, tuple())
    assert evaluate(fa) == 2

    x = S_[int]('x')
    l_ = L_[Callable[[int], int]]((x,), C_[int](2) + x)
    fa = F_[int](l_, (C_[int](2),))

    assert evaluate(fa) == 4

    y = S_[int]('y')
    l_ = L_[Callable[[int, int], int]]((x, y), x + y)
    fa = F_[int](l_, (C_[int](2), C_[int](3),))

    assert evaluate(fa) == 5


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


def test_equality():
    assert C_(1) == C_(1)
    assert C_(1) != C_(2)
    assert S_('a') == S_('a')
    assert S_('a') != S_('b')
    assert S_('a')(C_(1)) == S_('a')(C_(1))
    assert S_('a')(C_(1)) != S_('b')(C_(1))
    assert S_('a')(C_(1)) != S_('a')(C_(2))

    assert C_((C_(1), C_(2))) == C_((C_(1), C_(2)))
    assert C_((C_(1), C_(2))) != C_((C_(1), C_(3)))

    assert C_((C_(1), C_(2))) != C_((C_(1)))
    assert S_('a')(C_(1)) != S_('a')(C_(1), C_(2))


def test_instance_check():
    c = C_[int](2)
    assert isinstance(c, expressions.Constant)
    assert isinstance(c, expressions.Constant[int])
    assert issubclass(c.__class__, C_)
    assert c.type is int


def test_typecast_check():
    s = S_('a')
    s_float = s.cast(float)

    assert isinstance(s, expressions.Symbol)
    assert s.type is Unknown
    assert isinstance(s_float, expressions.Symbol)
    assert isinstance(s_float, expressions.Symbol[float])
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
    expression_block = expressions.ExpressionBlock((
        fa1(a), fa2(b, fa3(c, d), e)
    ))

    for symbol in [a, b, c, d, e]:
        assert symbol in expression._symbols
        assert symbol in expression_block._symbols


def test_apply_unapply():
    a = C_(1)
    b = a.apply(*a.unapply())
    assert a is not b and a == b

    a = F_(S_('a'), (C_(1),))
    b = a.apply(*a.unapply())
    assert a is not b and a == b


def test_nested_existentials():
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')

    exp = E_(x, E_(y, P(x) & Q(y)))

    assert exp._symbols == {Q, P}


def test_nested_universals():
    x = S_('x')
    y = S_('y')
    P = S_('P')
    Q = S_('Q')

    exp = U_(x, U_(y, P(x) & Q(y)))

    assert exp._symbols == {Q, P}


def test_fresh_symbol_subclass():
    class TestSymbol(expressions.Symbol):
        pass

    assert isinstance(TestSymbol.fresh(), TestSymbol)
    assert isinstance(TestSymbol[int].fresh(), TestSymbol[int])
