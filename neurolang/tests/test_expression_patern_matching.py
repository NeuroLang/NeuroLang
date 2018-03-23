from pytest import raises

from ..expression_pattern_matching import PatternMatcher, add_match
from ..expressions import (
    Constant, Symbol, FunctionApplication, Projection,
    Statement, Query
)


example_expressions = dict(
    c_int=Constant[int](1),
    c_str=Constant[str]('a'),
    s_a=Symbol[str]('a'),
    d_a=Statement(Symbol('a'), Constant[int](2)),
    f_d=FunctionApplication(
        Constant(lambda x: 2 * x),
        (Constant[int](2),)
    ),
    q_a=Query(
        Symbol('a'),
        FunctionApplication(
            Constant(lambda x: x % 2 == 0), (Constant[int](2),)
        )
    ),
    t_a=Constant((Constant[float](1.), Symbol('a'))),
    t_b=Constant((Constant[str]('a'), Symbol('a'))),
    p_a=Projection(
        Constant((Constant[str]('a'), Symbol('a'))),
        Constant[int](1)
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
        @add_match(Constant[int](...))
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
        @add_match(Constant[int](...))
        def _(self, expression):
            return expression

        @add_match(Statement(..., Constant[int](...)))
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
        @add_match(Constant((Constant[float](...),)))
        def __(self, expression):
            return False

        @add_match(Constant((Constant[float](...), ...)))
        def _(self, expression):
            return expression
    pm = PM()

    for k, e in example_expressions.items():
        if k == 't_a':
            assert pm.match(e) is e
        else:
            with raises(ValueError):
                pm.match(e)
