from inspect import signature
from itertools import product
from operator import add, eq, mul, pow, sub, truediv
import logging

import pytest

from .... import config
from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....datalog.aggregation import AggregationApplication
from ....expression_pattern_matching import add_match
from ....expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ....expressions import Constant, Query, Symbol
from ....logic import (
    ExistentialPredicate,
    LogicOperator,
    NaryLogicOperator,
    UniversalPredicate
)
from ....probabilistic.expressions import ProbabilisticPredicate, Condition, PROB
from ..squall import LogicSimplifier
from ..squall_syntax_lark import parser
from ..standard_syntax import ExternalSymbol
from ...probabilistic_frontend import RegionFrontendCPLogicSolver, Chase


LOGGER = logging.getLogger()


config.disable_expression_type_printing()


EQ = Constant(eq)


class LogicWeakEquivalence(ExpressionWalker):
    @add_match(EQ(UniversalPredicate, UniversalPredicate))
    def eq_universal_predicates(self, expression):
        left, right = expression.args
        if left.head != right.head:
            new_head = Symbol.fresh()
            rew = ReplaceExpressionWalker(
                {left.head: new_head, right.head: new_head}
            )
            left = rew.walk(left)
            right = rew.walk(right)

        return self.walk(EQ(left.body, right.body))

    @add_match(EQ(ExistentialPredicate, ExistentialPredicate))
    def eq_existential_predicates(self, expression):
        return self.eq_universal_predicates(expression)

    @add_match(
        EQ(NaryLogicOperator, NaryLogicOperator),
        lambda exp: (
            len(exp.args[0].formulas) == len(exp.args[1].formulas) and
            type(exp.args[0]) == type(exp.args[1])
        )
    )
    def n_ary_sort(self, expression):
        left, right = expression.args

        return all(
            self.walk(EQ(a1, a2))
            for a1, a2 in zip(
                sorted(left.formulas, key=repr),
                sorted(right.formulas, key=repr)
            )
        )

    @add_match(EQ(Condition, Condition))
    def eq_condition(self, expression):
        left, right = expression.args
        return (
            self.walk(EQ(left.conditioned, right.conditioned)) and
            self.walk(EQ(left.conditioning, right.conditioning))
        )

    @add_match(EQ(LogicOperator, LogicOperator))
    def eq_logic_operator(self, expression):
        left, right = expression.args
        for l, r in zip(left.unapply(), right.unapply()):
            if isinstance(l, tuple) and isinstance(r, tuple):
                return (
                    (len(l) == len(r)) and
                    all(
                        self.walk(EQ(ll, rr))
                        for ll, rr in zip(l, r)
                    )
                )
            else:
                return self.walk(EQ(l, r))

    @add_match(EQ(..., ...))
    def eq_expression(self, expression):
        return expression.args[0] == expression.args[1]


def weak_logic_eq(left, right):
    left = LogicSimplifier().walk(left)
    right = LogicSimplifier().walk(right)
    return LogicWeakEquivalence().walk(EQ(left, right))


@pytest.fixture
def datalog_simple():
    datalog = RegionFrontendCPLogicSolver()

    datalog.add_extensional_predicate_from_tuples(
        Symbol("item"),
        [('a',), ('b',), ('c',), ('d',)]
    )

    datalog.add_extensional_predicate_from_tuples(
        Symbol("item_count"),
        [('a', 0), ('a', 1), ('b', 2), ('c', 3)]
    )

    datalog.add_extensional_predicate_from_tuples(
        Symbol("quantity"),
        [(i,) for i in range(5)]
    )
    return datalog


def test_lark_semantics_item_selection(datalog_simple):
    code = (
        "define as Large every Item "
        "that has an item_count greater equal than 2 ."
    )
    logic_code = parser(code)
    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()['large'].value
    expected = set([('b',), ('c',)])

    assert solution == expected


def test_lark_semantics_item_selection_the_operator(datalog_simple):
    code = (
        "define as Large the Item "
        "that has the item_count greater equal than 2 ."
    )
    logic_code = parser(code)
    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()['large'].value
    expected = set([('b',), ('c',)])

    assert solution == expected


def test_lark_semantics_join(datalog_simple):
    code = (
        "define as merge for every Item ?i ;"
        " with every Quantity that ?i item_counts."
    )
    logic_code = parser(code)
    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()['merge'].value
    expected = set([('a', 0), ('a', 1), ('b', 2), ('c', 3)])

    assert solution == expected


def test_lark_semantics_aggregation(datalog_simple):
    code = """
        define as max_items for every Item ?i ;
            where every Max of the Quantity where ?i item_count per ?i.
    """
    logic_code = parser(code)

    datalog_simple.walk(logic_code)
    chase = Chase(datalog_simple)
    solution = chase.build_chase_solution()["max_item"].value
    expected = set([('a', 1), ('b', 2), ('c', 3)])
    assert solution == expected


def test_intransitive_per_conditional(datalog_simple):
    code = """
        define as probably Active every Focus (@x; @y; @z) that a Study @s reports
            conditioned to @s mentions a Term  @t that is Synonym of 'pain'.
    """
    active = Symbol('active')
    mention = Symbol('mention')
    term = Symbol('term')
    focus = Symbol('focus')
    study = Symbol('study')
    synonym = Symbol('synonym')
    report = Symbol('report')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    s = Symbol('s')
    t = Symbol('t')

    expected = Implication(
        active(x, y, z, PROB(x, y, z)),
        Condition(
            Conjunction((focus(x, y, z), study(s), report(s, x, y, z))),
            Conjunction((mention(s, t), term(t), synonym(t, Constant('pain'))))
        )
    )

    logic_code = parser(code)
    datalog_simple.walk(logic_code)

    assert weak_logic_eq(logic_code.formulas[0], expected)


def test_server_example_VWFA():
    code = """
        define as VWFA every Focus (@x; @y; @z) 
            such that ((@x - (-45)) ** 2 + (@y - (-57)) ** 2 + (@z - (-12)) ** 2) is lower than 5 ** 2 .

        define as `VWFA image` the `Created Region` of the VWFA in 3D.

        define as `Mentions VWFA` every Study that reports the VWFA in 3D.

        define as choice `Given study` with probability 1 / (the Count of the Studies)
        every Study.

        define as probably `Linked to VWFA`
            every Term that a Study @s mentions
            conditioned to @s is a `Given study` that `Mentions VWFA`.

        define as probably Universal every Term that a `Given study` mentions.

        define as `specific to the VWFA` every Term
            that is `Linked to VWFA` with a Probability @p and  Universal with a Probability @p0;
            by every Quantity @lor that is equal to log10(@p / @p0).

        obtain every Term @t with every Quantity @lor such that @t `specific to the VWFA` with @lor.
    """
    logic_code = parser(code)
    datalog_simple.walk(logic_code)

    assert True