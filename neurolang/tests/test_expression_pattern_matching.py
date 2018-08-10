from pytest import raises
import typing

from .. import expressions
from ..expression_pattern_matching import (
    PatternMatcher, add_match, NeuroLangPatternMatchingNoMatch
)
from ..expressions import Projection, Statement, Query

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication

example_expressions = dict(
    c_int=C_[int](1),
    c_str=C_[str]('a'),
    s_a=S_[str]('a'),
    d_a=Statement(S_('a'), C_[int](2)),
    f_d=F_(C_(lambda x: 2 * x), (C_[int](2), )),
    q_a=Query(S_('a'), F_(C_(lambda x: x % 2 == 0), (C_[int](2), ))),
    t_a=C_((C_[float](1.), S_('a'))),
    t_b=C_((C_[str]('a'), S_('a'))),
    p_a=Projection(C_((C_[str]('a'), S_('a'))), C_[int](1)),
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
        @add_match(expressions.Constant)
        def _(self, expression):
            return expression

    pm = PM()

    for e in example_expressions.values():
        if isinstance(e, expressions.Constant):
            assert pm.match(e) is e
        else:
            with raises(NeuroLangPatternMatchingNoMatch):
                pm.match(e)


def test_match_expression():
    class PM(PatternMatcher):
        @add_match(expressions.Constant[int](...))
        def _(self, expression):
            return expression

    pm = PM()

    for e in example_expressions.values():
        if isinstance(e, expressions.Constant) and e == 1:
            assert pm.match(e) is e
        else:
            with raises(NeuroLangPatternMatchingNoMatch):
                pm.match(e)


def test_match_expression_value():
    class PM(PatternMatcher):
        @add_match(expressions.Constant[int](...))
        def _(self, expression):
            return expression

        @add_match(Statement(..., expressions.Constant[int](...)))
        def __(self, expression):
            return expression

        @add_match(Query(..., expressions.FunctionApplication))
        def ___(self, expression):
            return expression

    pm = PM()

    for k, e in example_expressions.items():
        if k in ('c_int', 'd_a', 'q_a'):
            assert pm.match(e) is e
        else:
            with raises(NeuroLangPatternMatchingNoMatch):
                pm.match(e)


def test_match_expression_tuple():
    class PM(PatternMatcher):
        @add_match(expressions.Constant((expressions.Constant[float](...), )))
        def __(self, expression):
            return False

        @add_match(
            expressions.Constant((expressions.Constant[float](...), ...))
        )
        def _(self, expression):
            return expression

    pm = PM()

    for k, e in example_expressions.items():
        if k == 't_a':
            assert pm.match(e) is e
        else:
            with raises(NeuroLangPatternMatchingNoMatch):
                pm.match(e)


def test_pattern_matching_parametric_type():
    T = typing.TypeVar('T')

    class PM(PatternMatcher[T]):
        @add_match(expressions.Constant[T])
        def _(self, expression):
            return expression

        @add_match(
            expressions.FunctionApplication(..., (expressions.Constant[T], ))
        )
        def __(self, expression):
            return expression

        @add_match(expressions.Constant)
        def ___(self, expression):
            return expression

    PM_int = PM[int]
    assert PM_int.__patterns__[0][0] == expressions.Constant[int]
    assert isinstance(
        PM_int.__patterns__[1][0], expressions.FunctionApplication
    )
    assert PM_int.__patterns__[1][0].args[0] is expressions.Constant[int]
    assert PM_int.__patterns__[2][0] == expressions.Constant
