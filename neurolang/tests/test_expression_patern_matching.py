from pytest import raises

from ..expression_pattern_matching import PatternMatcher, add_match
from ..expressions import (
    Constant, Symbol, FunctionApplication, Projection,
    Statement, Query
)


example_expressions = dict(
    c_int=Constant(1, type_=int),
    c_str=Constant('a', type_=str),
    s_a=Symbol('a', type_=str),
    d_a=Statement(Symbol('a'), Constant(2, type_=int)),
    f_d=FunctionApplication(
        Constant(lambda x: 2 * x),
        (Constant(2, type_=int),)
    ),
    q_a=Query(
        Symbol('a'),
        FunctionApplication(
            Constant(lambda x: x % 2 == 0), (Constant(2, type_=int),)
        )
    ),
    t_a=(Constant(1., type_=float), Symbol('a')),
    t_b=(Constant('a', type_=str), Symbol('a')),
    p_a=Projection(
        Constant((Constant('a', type_=str), Symbol('a'))),
        Constant(1, type_=int)
    ),
)


def test_construction():
    pm = PatternMatcher()
    assert isinstance(pm, PatternMatcher)


def test_default():
    class PM(PatternMatcher):
        @add_match(...)
        def _(self, expression):
            return expression
    pm = PM()

    for e in example_expressions.values():
        assert pm.match(e) is e


def test_match_expression_type():
    class PM(PatternMatcher):
        @add_match(Constant)
        def _(self, expression):
            return expression
    pm = PM()

    for e in example_expressions.values():
        if isinstance(e, Constant):
            assert pm.match(e) is e
        else:
            with raises(ValueError):
                pm.match(e)


def test_match_expression():
    class PM(PatternMatcher):
        @add_match(Constant(..., type_=int))
        def _(self, expression):
            return expression
    pm = PM()

    for e in example_expressions.values():
        if isinstance(e, Constant) and e == 1:
            assert pm.match(e) is e
        else:
            with raises(ValueError):
                pm.match(e)


def test_match_expression_value():
    class PM(PatternMatcher):
        @add_match(Constant(..., type_=int))
        def _(self, expression):
            return expression

        @add_match(Statement(..., Constant(..., type_=int)))
        def __(self, expression):
            return expression

        @add_match(Query(..., FunctionApplication))
        def ___(self, expression):
            return expression

    pm = PM()

    for k, e in example_expressions.items():
        if k in ('c_int', 'd_a', 'q_a'):
            assert pm.match(e) is e
        else:
            with raises(ValueError):
                pm.match(e)


def test_match_expression_tuple():
    class PM(PatternMatcher):
        @add_match((Constant(..., type_=int),))
        def __(self, expression):
            return False

        @add_match((Constant(..., type_=int), ...))
        def _(self, expression):
            return expression
    pm = PM()

    for k, e in example_expressions.items():
        if k == 't_a':
            assert pm.match(e) is e
        else:
            with raises(ValueError):
                pm.match(e)
