import logging
from inspect import signature
from itertools import product
from operator import add, attrgetter, eq, mul, pow, sub, truediv

import pytest

from .... import config
from ....datalog import Conjunction, Fact, Implication, Negation, Union
from ....datalog.aggregation import AggregationApplication
from ....expression_pattern_matching import add_match
from ....expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ....expressions import (
    Constant,
    Definition,
    FunctionApplication,
    Query,
    Symbol
)
from ....logic import (
    ExistentialPredicate,
    LogicOperator,
    NaryLogicOperator,
    UniversalPredicate
)
from ....probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticPredicate
)
from ...probabilistic_frontend import Chase, RegionFrontendCPLogicSolver
from ..squall import LogicSimplifier
from ..squall_syntax_lark import parser
from ..standard_syntax import ExternalSymbol

LOGGER = logging.getLogger()


config.disable_expression_type_printing()


EQ = Constant(eq)


class LogicWeakEquivalence(ExpressionWalker):
    def __init__(self):
        self.fresh_equivalences = dict()

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
        return self._compare_composite_equality(expression)

    def _compare_composite_equality(self, expression):
        left, right = expression.args
        for l, r in zip(left.unapply(), right.unapply()):
            if isinstance(l, tuple) and isinstance(r, tuple):
                equal = (
                    (len(l) == len(r)) and
                    all(
                        self.walk(EQ(ll, rr))
                        for ll, rr in zip(l, r)
                    )
                )
            else:
                equal = self.walk(EQ(l, r))
            if not equal:
                return False
        return True

    @add_match(EQ(Definition, Definition))
    def eq_definition(self, expression):
        return self._compare_composite_equality(expression)

    @add_match(EQ(Symbol, Symbol))
    def eq_symbols(self, expression):
        left, right = expression.args
        if left.is_fresh and right.is_fresh:
            left, right = sorted((left, right), key=attrgetter('name'))
            if left not in self.fresh_equivalences:
                self.fresh_equivalences[left] = right
            return self.fresh_equivalences[left] == right
        else:
            return left == right

    @add_match(EQ(..., ...))
    def eq_expression(self, expression):
        return expression.args[0] == expression.args[1]


def weak_logic_eq(left, right):
    left = LogicSimplifier().walk(left)
    right = LogicSimplifier().walk(right)
    return LogicWeakEquivalence().walk(EQ(left, right))


@pytest.fixture(scope="module")
def datalog_base():
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
    datalog.add_extensional_predicate_from_tuples(
        Symbol("voxel"),
        [(0, 0, 0)]
    )
    datalog.add_extensional_predicate_from_tuples(
        Symbol("focus"),
        [(0, 0, 0)]
    )

    datalog.add_extensional_predicate_from_tuples(
        Symbol("report"),
        [(0, 0, 0, 0)]
    )

    return datalog


@pytest.fixture(scope="function")
def datalog_simple(datalog_base):
    datalog_base.push_scope()
    yield datalog_base
    datalog_base.pop_scope()


def test_rules():
    A = Symbol('a')
    B = Symbol('b')
    C = Symbol('c')
    f = Symbol('f')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    res = parser(
        """
        define as A every Element @x
            whose b is an Element @y and
            such that 3 c an Element @z.
        """,
        type_predicate_symbols={"element"}
    )
    expected = Union((
        Implication(A(x), Conjunction((B(x, y), C(Constant(3), z)))),
    ))
    assert weak_logic_eq(res, expected)

    res = parser(
        """
            define as A every Element @x that is not B.
        """,
        type_predicate_symbols={"element"}
    )
    expected = Union((
        Implication(A(x), Negation(B(x))),
    ))
    assert weak_logic_eq(res, expected)

    res = parser(
        """
        define as A every Element @x
        whose b is an Element @y and
        such that 3 c an Element @z that is 4 .
        """,
        type_predicate_symbols={"element"}
    )
    expected = Union((
        Implication(
            A(x),
            Conjunction((
                B(x, y), C(Constant(3), z), Constant(eq)(z, Constant(4))
            ))
        ),
    ))
    assert weak_logic_eq(res, expected)

    res = parser(
        """
        define as A the Element @x
            such that f(@x + 5 * 2) is whose b is an Element @y and
            such that 3 c an Element @z that is 4 .
        """,
        type_predicate_symbols={"element"}
    )
    fresh = Symbol.fresh()
    expected = Union((
        (Implication(
            A(x),
            Conjunction((
                B(fresh, y),
                C(Constant(3), z), Constant(eq)(z, Constant(4)),
                EQ(fresh,
                    FunctionApplication(
                        f,
                        (
                            Constant(add)(
                                x,
                                Constant(mul)(Constant(5), Constant(2))
                            ),
                        )
                    )
                )
            )),
        ),)
    ))

    assert weak_logic_eq(res, expected)


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


def test_lark_aggregation_with_parameter(datalog_simple):
    code = """
        define as Top the Percentile, by 95, from the Items.
    """
    logic_code = parser(code)
    datalog_simple.walk(logic_code)
    top = Symbol("top")
    aggregation_symbol = Symbol.fresh()
    x = Symbol.fresh()
    y = Symbol.fresh()
    item = Symbol("item")
    expected = Union((
        Implication(
            aggregation_symbol(
                AggregationApplication(Symbol("percentile"), (y, Constant(95)))
            ),
            item(y)
        ),
        Implication(top(x), aggregation_symbol(x))
    ))
    assert weak_logic_eq(logic_code, expected)


def test_intransitive_per_conditional(datalog_simple):

    active = Symbol('active')
    mention = Symbol('mention')
    term = Symbol('term')
    focus = Symbol('focus')
    study = Symbol('study')
    synonym = Symbol('synonym')
    syn_active = Symbol('syn act')
    report = Symbol('report')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    s = Symbol('s')
    t = Symbol('t')

    code = """
        define as probably Active every Focus (@x; @y; @z) that a Study @s reports
            conditioned to @s mentions a Term  @t that is Synonym of 'pain'.
    """
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

    code = """
        define as probably `Syn active` a Study @s reports a Focus (@x; @y; @z)
            conditioned to every Term @t that @s mentions and
            that is Synonym of 'pain'.
    """
    expected = Implication(
        syn_active(t, PROB(t)),
        Condition(
            Conjunction((focus(x, y, z), study(s), report(s, x, y, z))),
            Conjunction((mention(s, t), term(t), synonym(t, Constant('pain'))))
        )
    )

    logic_code = parser(code)
    datalog_simple.walk(logic_code)

    assert weak_logic_eq(logic_code.formulas[0], expected)


def test_transitive_per_conditional(datalog_simple):

    related = Symbol('relate')
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
    t = Symbol.fresh()

    code = """
        define as probably related every Focus (@x; @y; @z)
        that a Study @s reports
            conditioned to every Term
                that is Synonym of 'pain' and
                that @s mentions.
    """
    expected = Implication(
        related(x, y, z, t, PROB(x, y, z, t)),
        Condition(
            Conjunction((focus(x, y, z), study(s), report(s, x, y, z))),
            Conjunction((mention(s, t), term(t), synonym(t, Constant('pain'))))
        )
    )

    logic_code = parser(code)
    datalog_simple.walk(logic_code)

    assert weak_logic_eq(logic_code.formulas[0], expected)


def test_server_example_VWFA(datalog_simple):
    code = """
        define as VWFA every Focus (@x; @y; @z)
            such that ((@x - (-45)) ** 2 + (@y - (-57)) ** 2 + (@z - (-12)) ** 2)
            is lower than 5 ** 2 .

        define as `VWFA image` the `Created Region` of the VWFA in 3D.

        define as `Mentions VWFA` every Study that reports the VWFA in 3D.

        define as choice `Given study` with probability 1 / (the Count of the Studies)
        every Study.

        define as probably `Linked to VWFA`
            every Term that a Study @s mentions
            conditioned to @s is a `Given study` that `Mentions VWFA`.

        define as probably Universal every Term that a `Given study` mentions.

        define as `specific to the VWFA` every Term
            that is `Linked to VWFA` with a Probability @p and
            Universal with a Probability @p0;
                by every Quantity @lor that is equal to log10(@p / @p0).

        obtain every Term @t with every Quantity @lor such that @t `specific to the VWFA` with @lor.
    """
    logic_code = parser(code)
    datalog_simple.walk(logic_code)

    assert True


def test_server_example_activation_map(datalog_simple):
    code = """
      define as `Study of interest` every Study @s,
        that mentions a Term --that is 'language' or that is 'memory' -- with a TfIdf greater than 0.0001,
        and that does not mention a Term that is 'auditory' with a TfIdf greater than 0 .

      define as `reports as active` every Study
        that reports a Focus (@fx; @fy; @fz);
        with every Voxel (@x; @y; @z) such that `euclidean distance`(@x,@y,@z,@fx,@fy,@fz) is lower than 4 .

      define as choice `Given study` with probability 1 / (the Count of the Studies)
        every Study.

      define as probably Active every Voxel in 3D
        that @s `reports as active` conditioned to a `Given Study` @s is a `Study of interest`.

      define as Images
        every `Created region overlay` from the Tuples (@x; @y; @z; @p)
        such that (@x; @y; @z) is a Voxel that is Active with a Probability @p .

      obtain every Image.
        obtain every Term @t with every Quantity @lor such that @t `specific to the VWFA` with @lor.
    """
    logic_code = parser(code)
    datalog_simple.walk(logic_code)

    assert True
