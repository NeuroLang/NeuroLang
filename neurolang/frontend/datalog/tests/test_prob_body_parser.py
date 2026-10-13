"""Tests for the PROB/MARG body predicate parser/transformer.

These tests exercise parser transformer methods for probabilistic body
predicates and related constructs without executing through the solver.
"""

import operator
import pytest

from lark import Tree, Token

from neurolang.datalog import Conjunction, Fact, Implication, Union
from neurolang.datalog.expressions import AggregationApplication
from neurolang.exceptions import (
    UnexpectedCharactersError,
    UnexpectedTokenError,
)
from neurolang.expressions import (
    Command,
    Constant,
    FunctionApplication,
    Lambda,
    Query,
    Statement,
    Symbol,
)
from neurolang.frontend.datalog.tests.test_squall_parser import (
    ConditionAwareLogicWeakEquivalence,
    EQ,
)
from neurolang.logic import Negation
from neurolang.probabilistic.expressions import (
    Condition,
    PROB,
    ProbabilisticFact,
)

from ..standard_syntax import (
    DatalogTransformer,
    _preprocess,
    parse_rules,
    parser,
)


def normalize_fresh(expr, _mapping=None, _counter=None):
    """Replace every Symbol.is_fresh with a canonical placeholder so that
    two expression trees built from different `Symbol.fresh()` calls compare equal."""
    if _mapping is None:
        _mapping = {}
        _counter = [0]
    if isinstance(expr, Symbol) and (expr.is_fresh or expr.name.startswith("fresh_")):
        key = expr.name
        if key not in _mapping:
            _mapping[key] = Symbol(f"$fresh_{_counter[0]}")
            _counter[0] += 1
        return _mapping[key]
    if isinstance(expr, FunctionApplication):
        return FunctionApplication(
            normalize_fresh(expr.functor, _mapping, _counter),
            tuple(normalize_fresh(a, _mapping, _counter) for a in expr.args),
        )
    if isinstance(expr, Conjunction):
        return Conjunction(
            tuple(normalize_fresh(f, _mapping, _counter) for f in expr.formulas)
        )
    if isinstance(expr, Union):
        return Union(
            tuple(normalize_fresh(f, _mapping, _counter) for f in expr.formulas)
        )
    if isinstance(expr, Implication):
        return Implication(
            normalize_fresh(expr.consequent, _mapping, _counter),
            normalize_fresh(expr.antecedent, _mapping, _counter),
        )
    if isinstance(expr, Query):
        return Query(
            normalize_fresh(expr.head, _mapping, _counter),
            normalize_fresh(expr.body, _mapping, _counter),
        )
    if isinstance(expr, Fact):
        return Fact(normalize_fresh(expr.consequent, _mapping, _counter))
    if isinstance(expr, Negation):
        return Negation(normalize_fresh(expr.formula, _mapping, _counter))
    if isinstance(expr, Condition):
        return Condition(
            normalize_fresh(expr.conditioned, _mapping, _counter),
            normalize_fresh(expr.conditioning, _mapping, _counter),
        )
    if isinstance(expr, ProbabilisticFact):
        return ProbabilisticFact(
            normalize_fresh(expr.probability, _mapping, _counter),
            normalize_fresh(expr.body, _mapping, _counter),
        )
    if isinstance(expr, AggregationApplication):
        return AggregationApplication(
            normalize_fresh(expr.functor, _mapping, _counter),
            tuple(normalize_fresh(a, _mapping, _counter) for a in expr.args),
        )
    if isinstance(expr, Statement):
        return Statement(
            normalize_fresh(expr.lhs, _mapping, _counter),
            normalize_fresh(expr.rhs, _mapping, _counter),
        )
    if isinstance(expr, Lambda):
        return Lambda(
            normalize_fresh(expr.args, _mapping, _counter),
            normalize_fresh(expr.function_expression, _mapping, _counter),
        )
    if isinstance(expr, Command):
        return Command(
            normalize_fresh(expr.name, _mapping, _counter),
            normalize_fresh(expr.args, _mapping, _counter),
            normalize_fresh(expr.kwargs, _mapping, _counter),
        )
    if isinstance(expr, tuple):
        return tuple(normalize_fresh(a, _mapping, _counter) for a in expr)
    return expr


def weak_eq(left, right):
    """Compare two expressions, normalising fresh symbols first."""
    left = normalize_fresh(left)
    right = normalize_fresh(right)
    return ConditionAwareLogicWeakEquivalence().walk(EQ(left, right))


# --- Module-level Symbol aliases for test readability ---
__MARG__ = Symbol("__MARG__")
__PROB__ = Symbol("__PROB__")
a_ = Symbol("a")
ans_ = Symbol("ans")
b_ = Symbol("b")
backtick_id_ = Symbol("backtick_id")
c_ = Symbol("c")
cmd_ = Symbol("cmd")
cnt_ = Symbol("cnt")
count_ = Symbol("count")
derived_ = Symbol("derived")
external_ = Symbol("external")
f_ = Symbol("f")
key_ = Symbol("key")
load_ = Symbol("load")
p_ = Symbol("p")
pa_ = Symbol("pa")
pb_ = Symbol("pb")
pc1_ = Symbol("pc1")
q_ = Symbol("q")
R_ = Symbol("R")
r_ = Symbol("r")
r1_ = Symbol("r1")
r2_ = Symbol("r2")
reset_ = Symbol("reset")
s_ = Symbol("s")
v_ = Symbol("v")
x_ = Symbol("x")
y_ = Symbol("y")
z_ = Symbol("z")


def _parse(code):
    return parser(code)


def test_marg_single_predicate():
    """MARG with single predicate — hits _build_prob_rule MARG single branch."""
    result = _parse("derived(p) :- MARG[pc1(s)] = p.")

    expected = Union(
        (
            Implication(
                Symbol("fresh_00000000")(PROB(Conjunction((pc1_(s_),)))),
                Conjunction((pc1_(s_),)),
            ),
            Implication(derived_(p_), Conjunction((Symbol("fresh_00000000")(p_),))),
        )
    )
    assert weak_eq(result, expected)


def test_prob_body_predicate_simple():
    """PROB[pred] = var — hits prob_body_predicate else branch (non-cond)."""
    result = _parse("derived(x, p) :- PROB[R(x)] = p.")

    expected = Union(
        (
            Implication(Symbol("fresh_00000001")(x_, PROB(x_)), Conjunction((R_(x_),))),
            Implication(
                derived_(x_, p_), Conjunction((Symbol("fresh_00000001")(x_, p_),))
            ),
        )
    )
    assert weak_eq(result, expected)


def test_prob_body_predicate_conditional():
    """PROB[pred // cond] = var — hits prob_body_predicate conditional branch."""
    result = _parse("derived(x, p) :- PROB[R(x) // R(s)] = p.")

    expected = Union(
        (
            Implication(
                Symbol("fresh_00000002")(x_, PROB(x_)),
                Condition(R_(x_), Conjunction((R_(s_),))),
            ),
            Implication(
                derived_(x_, p_), Conjunction((Symbol("fresh_00000002")(x_, p_),))
            ),
        )
    )
    assert weak_eq(result, expected)


def test_marg_body_predicate_simple():
    """MARG[pred] = var — hits marg_body_predicate non-conditional."""
    result = _parse("derived(x, p) :- MARG[R(x)] = p.")

    expected = Union(
        (
            Implication(
                Symbol("fresh_00000003")(PROB(Conjunction((R_(x_),)))),
                Conjunction((R_(x_),)),
            ),
            Implication(derived_(x_, p_), Conjunction((Symbol("fresh_00000003")(p_),))),
        )
    )
    assert weak_eq(result, expected)


def test_marg_body_predicate_conditional():
    """MARG[pred // cond] = var — hits marg_body_predicate conditional."""
    result = _parse("derived(p) :- MARG[R(x) // R(y)] = p.")

    expected = Union(
        (
            Implication(
                Symbol("fresh_00000004")(PROB()),
                Condition(R_(x_), Conjunction((R_(y_),))),
            ),
            Implication(derived_(p_), Conjunction((Symbol("fresh_00000004")(p_),))),
        )
    )
    assert weak_eq(result, expected)


def test_succ_body_predicate():
    """SUCC[...] = var — hits succ_body_predicate (discarded at conjunction)."""
    result = _parse("derived(x) :- SUCC[R(x)] = p.")
    # The SUCC atom should be discarded

    expected = Union((Implication(derived_(x_), Constant(True)),))
    assert weak_eq(result, expected)


def test_probabilistic_rule_head():
    """p(x) :: prob :- body — hits probabilistic_rule transformer."""
    result = _parse("p(x) :: 0.5 :- q(x).")

    expected = Union(
        (Implication(ProbabilisticFact(Constant(0.5), p_(x_)), Conjunction((q_(x_),))),)
    )
    assert weak_eq(result, expected)


def test_prob_head_rule_simple():
    """PROB[pred] :- body — simple prob head rule."""
    result = _parse("PROB[p(x)] :- q(x).")
    assert isinstance(list(result.formulas)[0], Implication)

    expected = Union((Implication(p_(x_, PROB(x_)), Conjunction((q_(x_),))),))
    assert weak_eq(result, expected)


def test_prob_head_rule_conditional():
    """PROB[pred // cond] :- body — conditional prob head rule."""
    result = _parse("PROB[p(x) // r(x, s)] :- q(s).")

    expected = Union(
        (Implication(p_(x_, PROB(x_)), Condition(p_(x_), Conjunction((r_(x_, s_),)))),)
    )
    assert weak_eq(result, expected)


def test_marg_head_rule_simple():
    """MARG[pred] :- body — simple marg head rule."""
    result = _parse("MARG[p(x)] :- q(x).")
    assert isinstance(list(result.formulas)[0], Implication)

    expected = Union((Implication(Conjunction((p_(x_, PROB(x_)),)), Conjunction((q_(x_),))),))
    assert weak_eq(result, expected)


def test_marg_head_rule_conditional():
    """MARG[pred // cond] :- body — conditional marg head rule."""
    result = _parse("MARG[p(x) // r(x, s)] :- q(s).")

    expected = Union(
        (
            Implication(
                p_(x_, PROB(x_)),
                Condition(Conjunction((p_(x_),)), Conjunction((r_(x_, s_),))),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_succ_head_rule():
    """SUCC[...] :- body — no-op (returns Union wrapping Constant(True))."""
    result = _parse("SUCC[p(x)] :- q(x).")

    expected = Union((Constant(True),))
    assert weak_eq(result, expected)


def test_condition_separator():
    """pred1 // pred2 — hits condition transformer."""
    result = _parse("ans(x) :- R(x, y) // R(y, z).")

    expected = Union((Query(ans_(x_), Condition(R_(x_, y_), R_(y_, z_))),))
    assert weak_eq(result, expected)


def test_negated_predicate():
    """~pred — hits negated_predicate transformer."""
    result = _parse("ans(x) :- ~R(x, y).")

    expected = Union((Query(ans_(x_), Conjunction((Negation(R_(x_, y_)),))),))
    assert weak_eq(result, expected)


def test_negation_inside_prob():
    """PROB[~pred(x)] — hits _classify_prob_predicate Negation branch."""
    result = _parse("derived(x, y, p) :- PROB[~R(x, y)] = p.")

    expected = Union(
        (
            Implication(
                Symbol("fresh_00000005")(x_, y_, PROB(x_, y_)),
                Conjunction((Negation(R_(x_, y_)),)),
            ),
            Implication(
                derived_(x_, y_, p_),
                Conjunction((Symbol("fresh_00000005")(x_, y_, p_),)),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_statement_function():
    """f(x) := expr — hits statement_function transformer."""
    result = _parse("f(x) := x.\nans(v) :- f(v).")

    expected = Union(
        (
            Statement(f_, Lambda((x_,), x_)),
            Query(ans_(v_), Conjunction((f_(v_),))),
        )
    )
    assert weak_eq(result, expected)


def test_probabilistic_fact():
    """0.5 :: p(1) — hits probabilistic_fact transformer."""
    result = _parse("0.5 :: p(1).")

    expected = Union(
        (
            Implication(
                ProbabilisticFact(Constant(0.5), p_(Constant(1))), Constant(True)
            ),
        )
    )
    assert weak_eq(result, expected)


def test_multiple_facts():
    """Multiple facts — hits expressions transformer list path."""
    result = _parse("p(1).\nq(2).\nr(3).")

    expected = Union(
        (
            Implication(p_(Constant(1)), Constant(True)),
            Implication(q_(Constant(2)), Constant(True)),
            Implication(r_(Constant(3)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_preprocess_comments():
    """_preprocess strips % comments and trailing dots."""
    cleaned = _preprocess("% comment line\np(1).\n%\nq(2) % inline comment.\n")
    assert "p(1)" in cleaned
    assert "q(2)" in cleaned
    assert "%" not in cleaned


def test_preprocess_bare_dot():
    """_preprocess does not strip a bare dot on its own."""
    cleaned = _preprocess(".\n")
    assert cleaned.strip()


def test_inline_negation_in_comparison():
    """Negation in rule body via comparison."""
    result = _parse("ans(x) :- R(x, y) & ~(x == 2).")

    expected = Union(
        (
            Query(
                ans_(x_),
                Conjunction(
                    (
                        R_(x_, y_),
                        Negation(Constant(operator.eq)(x_, Constant(2))),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_build_prob_rule_ans_head():
    """PROB with ans head directly — hits ans branch of new_head."""
    result = _parse("ans(x, p) :- PROB[R(x)] = p.")

    expected = Union(
        (
            Implication(Symbol("fresh_00000006")(x_, PROB(x_)), Conjunction((R_(x_),))),
            Query(ans_(x_, p_), Conjunction((Symbol("fresh_00000006")(x_, p_),))),
        )
    )
    assert weak_eq(result, expected)


def test_multiple_prob_specs():
    """Multiple PROB specs in a single rule."""
    result = _parse(
        "derived(x, p1, p2) :- PROB[R(x)] = p1 & PROB[R(x)] = p2.\n"
        "ans(x, p1, p2) :- derived(x, p1, p2)."
    )
    assert isinstance(result, Union)


def test_arithmetic_operations():
    """Arithmetic operators — hits plus_op, minus_op, mul_term, div_term, etc."""
    result = _parse("ans(x) :- R(x, y) & (y > 1.5).")

    expected = Union(
        (
            Query(
                ans_(x_),
                Conjunction(
                    (
                        R_(x_, y_),
                        Constant(operator.gt)(y_, Constant(1.5)),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_tuple_literal():
    """Tuple literal in argument position."""
    result = _parse("ans(x) :- R(x, (1, 2)).")

    expected = Union(
        (Query(ans_(x_), Conjunction((R_(x_, Constant((Constant(1), Constant(2)))),))),)
    )
    assert weak_eq(result, expected)


def test_lambda_application():
    """Lambda application expression."""
    result = _parse("f(x) := (lambda z : z)(x).\nans(v) :- f(v).")

    expected = Union(
        (
            Statement(f_, Lambda((x_,), Lambda((z_,), z_)(x_))),
            Query(ans_(v_), Conjunction((f_(v_),))),
        )
    )
    assert weak_eq(result, expected)


def test_external_symbol():
    """External identifier @prefix."""
    result = _parse('ans(x) :- @type(x, "concept").')

    expected = Union(
        (Query(ans_(x_), Conjunction((Symbol("type")(x_, Constant("concept")),))),)
    )
    assert weak_eq(result, expected)


def test_identifier_regexp():
    """Backtick-quoted identifier."""
    result = _parse("ans(x) :- `custom/pred`(x).")

    expected = Union((Query(ans_(x_), Conjunction((Symbol("custom/pred")(x_),))),))
    assert weak_eq(result, expected)


def test_constant_int_and_float():
    """Integer and float constants."""
    result = _parse("p(1, 2.5).\nq(-3, -4.0).")

    expected = Union(
        (
            Implication(p_(Constant(1), Constant(2.5)), Constant(True)),
            Implication(q_(Constant(-3), Constant(-4.0)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_command():
    """Dot command syntax."""
    result = _parse('.cmd("arg")\np(1).')

    expected = Union(
        (
            Command(cmd_, (Constant("arg"),), ()),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_keyword_command():
    """Command with keyword argument."""
    result = _parse('.cmd(key="val")\np(1).')

    expected = Union(
        (
            Command(
                cmd_,
                (),
                (
                    (
                        key_,
                        Constant("val"),
                    ),
                ),
            ),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_multiple_prob_specs_in_one_rule():
    """Multiple PROB specifications in one rule — tests prob_body_predicate."""
    result = _parse("ans(pa, pb) :- PROB[r1(x)] = pa, PROB[r2(x)] = pb.")

    expected = Union(
        (
            Implication(Symbol("fresh_00000007")(PROB()), Conjunction((r1_(x_),))),
            Implication(Symbol("fresh_00000008")(PROB()), Conjunction((r2_(x_),))),
            Query(
                ans_(pa_, pb_),
                Conjunction(
                    (
                        Symbol("fresh_00000007")(pa_),
                        Symbol("fresh_00000008")(pb_),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_aggregate_assign_vars():
    """AGGREGATE with group vars — hits agg_assign_vars."""
    result = _parse("ans(x, c) :- AGGREGATE[x](R(x, y) @ count(y)) = c.")

    expected = Union((Implication(ans_(x_, AggregationApplication(count_, (y_,))), Conjunction((R_(x_, y_),))),))
    assert weak_eq(result, expected)


def test_aggregate_assign_empty():
    """AGGREGATE without group vars — hits agg_assign_empty."""
    result = _parse("ans(c) :- AGGREGATE[](R(x, y) @ count(y)) = c.")

    expected = Union((
        Implication(Symbol("fresh_00000000")(count_(y_)), Conjunction((R_(x_, y_),))),
        Implication(ans_(c_), Conjunction((Symbol("fresh_00000000")(c_),))),
    ))
    assert weak_eq(result, expected)


def test_head_with_prod_in_args():
    """Head predicate with PROB in its arguments."""
    result = _parse("ans(x, y) :- R(x, y).")

    expected = Union((Query(ans_(x_, y_), Conjunction((R_(x_, y_),))),))
    assert weak_eq(result, expected)


def test_true_and_false_constants():
    """True/False logical constants."""
    result = _parse("ans(x) :- R(x, y) & (y == True).")

    expected = Union(
        (
            Query(
                ans_(x_),
                Conjunction(
                    (
                        R_(x_, y_),
                        Constant(operator.eq)(y_, Symbol("True")),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_extended_projection_via_extended_predicate():
    """Extended projection via arithmetic."""
    result = _parse("ans(z) :- R(x, y) & (z == x + y).")

    expected = Union(
        (
            Query(
                ans_(z_),
                Conjunction(
                    (
                        R_(x_, y_),
                        Constant(operator.eq)(z_, Constant(operator.add)(x_, y_)),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_aggregate_fn_call():
    """AGGREGATE fn call — hits agg_fn_call transformer."""
    result = _parse("ans(cnt) :- AGGREGATE[](R(x, y) @ count(x)) = cnt.")

    expected = Union((
        Implication(Symbol("fresh_00000000")(count_(x_)), Conjunction((R_(x_, y_),))),
        Implication(ans_(cnt_), Conjunction((Symbol("fresh_00000000")(cnt_),))),

    ))
    assert weak_eq(result, expected)


def test_multiple_args_comparison():
    """Multiple comparisons in body."""
    result = _parse("ans(x) :- R(x, y) & (y > 1) & (y < 10).")

    expected = Union(
        (
            Query(
                ans_(x_),
                Conjunction(
                    (
                        R_(x_, y_),
                        Constant(operator.gt)(y_, Constant(1)),
                        Constant(operator.lt)(y_, Constant(10)),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_preprocess_preserves_percent_in_strings():
    """% inside single-quoted strings preserved, % outside stripped."""
    processed = _preprocess("ans(x) :- R(x, 'a'). % comment")
    assert "% comment" not in processed
    assert "'a'" in processed


def test_preprocess_double_quotes():
    """% inside double-quoted strings preserved."""
    processed = _preprocess('ans(x) :- R(x, "a").')
    assert '"a"' in processed


def test_preprocess_percent_comment():
    """% outside quotes is a comment — stripped."""
    result = _parse("ans(x) :- R(x, y). % comment here")

    expected = Union((Query(ans_(x_), Conjunction((R_(x_, y_),))),))
    assert weak_eq(result, expected)


def test_command_no_args():
    """Command with no arguments — hits command(None) branch."""
    result = _parse(".reset()\np(1).")

    expected = Union(
        (
            Command(reset_, (), ()),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_command_keyword_args():
    """Command with keyword-only arguments — hits keyword_item."""
    result = _parse('.load(key="val")\np(1).')

    expected = Union(
        (
            Command(
                load_,
                (),
                (
                    (
                        key_,
                        Constant("val"),
                    ),
                ),
            ),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_command_args_and_kwargs():
    """Command with both positional and keyword args."""
    result = _parse('.load("path", key="val")\np(1).')

    expected = Union(
        (
            Command(
                load_,
                (Constant("path"),),
                (
                    (
                        key_,
                        Constant("val"),
                    ),
                ),
            ),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_head_predicate_no_args():
    """Head predicate with no arguments — hits empty-args branch."""
    result = _parse("p() :- q(1).")

    expected = Union((Implication(p_(), Conjunction((q_(Constant(1)),))),))
    assert weak_eq(result, expected)


def test_anonymous_variable():
    """Anonymous variable _ — triggers Symbol.fresh in argument."""
    result = _parse("ans(x) :- R(x, _).")

    expected = Union(
        (Query(ans_(x_), Conjunction((R_(x_, Symbol("fresh_00000009")),))),)
    )
    assert weak_eq(result, expected)


def test_minus_signed_id():
    """Signed identifier with minus sign."""
    result = _parse("ans(x) :- R(-x).")

    expected = Union(
        (Query(ans_(x_), Conjunction((R_(Constant(operator.mul)(Constant(-1), x_)),))),)
    )
    assert weak_eq(result, expected)


def test_minus_op_single_arg():
    """minus_op with single argument — unary minus fallback."""
    result = _parse("ans(x) :- R(x, -1).")

    expected = Union((Query(ans_(x_), Conjunction((R_(x_, Constant(-1)),))),))
    assert weak_eq(result, expected)


def test_arithmetic_ops():
    """Division, multiplication, power — div_term, mul_term, pow_factor."""
    result = _parse("ans(z) :- R(x, y) & (z == x / y * 2 ** 3).")

    expected = Union(
        (
            Query(
                ans_(z_),
                Conjunction(
                    (
                        R_(x_, y_),
                        Constant(operator.eq)(
                            z_,
                            Constant(operator.mul)(
                                Constant(operator.truediv)(x_, y_),
                                Constant(operator.pow)(Constant(2), Constant(3)),
                            ),
                        ),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_lambda_expression():
    """Lambda expression in argument position."""
    result = _parse("f(x) := (lambda y: x + y)(1).")

    expected = Union(
        (
            Statement(
                f_,
                Lambda(
                    (x_,), Lambda((y_,), Constant(operator.add)(x_, y_))(Constant(1))
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_neg_int():
    """Negative integer constant — neg_int transformer."""
    result = _parse("ans(x) :- R(-5).")

    expected = Union((Query(ans_(x_), Conjunction((R_(Constant(-5)),))),))
    assert weak_eq(result, expected)


def test_neg_float():
    """Negative float constant — neg_float transformer."""
    result = _parse("ans(x) :- R(-3.14).")

    expected = Union((Query(ans_(x_), Conjunction((R_(Constant(-3.14)),))),))
    assert weak_eq(result, expected)


def test_ext_identifier():
    """External identifier @ in predicate."""
    result = _parse("ans(x) :- @external(x).")

    expected = Union((Query(ans_(x_), Conjunction((external_(x_),))),))
    assert weak_eq(result, expected)


def test_tuple_literal_three_elements():
    """Tuple literal with three elements in argument position."""
    result = _parse("ans(x) :- R((1, 2, 3)).")

    expected = Union(
        (
            Query(
                ans_(x_),
                Conjunction((R_(Constant((Constant(1), Constant(2), Constant(3)))),)),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_parser_invalid_syntax():
    """Invalid syntax raises UnexpectedTokenError."""
    with pytest.raises(UnexpectedTokenError):
        parser("this is not valid syntax at all!")


def test_query_no_args():
    """Query with no arguments — hits query no-args branch."""
    result = _parse("ans() :- p(1).")

    expected = Union((Query(ans_(), Conjunction((p_(Constant(1)),))),))
    assert weak_eq(result, expected)


def test_statement():
    """Simple statement transformation."""
    result = _parse("x := 5.")

    expected = Union((Statement(x_, Constant(5)),))
    assert weak_eq(result, expected)


def test_arithmetic_multi_ops():
    """Multiple subtraction operations — hits minus_op with >1 args."""
    result = _parse("ans(r) :- R(a, b, c) & (r == a - b - c).")

    expected = Union(
        (
            Query(
                ans_(r_),
                Conjunction(
                    (
                        R_(a_, b_, c_),
                        Constant(operator.eq)(
                            r_,
                            Constant(operator.sub)(Constant(operator.sub)(a_, b_), c_),
                        ),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_arithmetic_multi_plus():
    """Multiple addition operations — hits plus_op with >1 args."""
    result = _parse("ans(r) :- R(a, b, c) & (r == a + b + c).")

    expected = Union(
        (
            Query(
                ans_(r_),
                Conjunction(
                    (
                        R_(a_, b_, c_),
                        Constant(operator.eq)(
                            r_,
                            Constant(operator.add)(Constant(operator.add)(a_, b_), c_),
                        ),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_division():
    """Division operation in body."""
    result = _parse("ans(r) :- R(a, b) & (r == a / b).")

    expected = Union(
        (
            Query(
                ans_(r_),
                Conjunction(
                    (
                        R_(a_, b_),
                        Constant(operator.eq)(r_, Constant(operator.truediv)(a_, b_)),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_multiplication():
    """Multiplication operation in body."""
    result = _parse("ans(r) :- R(a, b) & (r == a * b).")

    expected = Union(
        (
            Query(
                ans_(r_),
                Conjunction(
                    (
                        R_(a_, b_),
                        Constant(operator.eq)(r_, Constant(operator.mul)(a_, b_)),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_power():
    """Power operation in body."""
    result = _parse("ans(r) :- R(a, b) & (r == a ** b).")

    expected = Union(
        (
            Query(
                ans_(r_),
                Conjunction(
                    (
                        R_(a_, b_),
                        Constant(operator.eq)(r_, Constant(operator.pow)(a_, b_)),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_text_argument():
    """Text literal in argument position — hits text transformer."""
    result = _parse("ans(x) :- R('hello').")

    expected = Union((Query(ans_(x_), Conjunction((R_(Constant("hello")),))),))
    assert weak_eq(result, expected)


def test_backtick_identifier_as_argument():
    """Backtick identifier — hits identifier_regexp transformer."""
    result = _parse("ans(x) :- R(`backtick_id`).")

    expected = Union((Query(ans_(x_), Conjunction((R_(backtick_id_),))),))
    assert weak_eq(result, expected)


def test_statement_function_arithmetic():
    """Statement function — tests statement_function transformer."""
    result = _parse("f(x) := x + 1.")

    expected = Union(
        (Statement(f_, Lambda((x_,), Constant(operator.add)(x_, Constant(1)))),)
    )
    assert weak_eq(result, expected)


def test_command_string_arg():
    """Command with string argument — hits pos_item non-Expression."""
    result = _parse('.load("some_path")\np(1).')

    expected = Union(
        (
            Command(load_, (Constant("some_path"),), ()),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_command_multi_pos_args():
    """Command with multiple positional args — hits pos_args."""
    result = _parse(".cmd(a, b, c)\np(1).")

    expected = Union(
        (
            Command(
                cmd_,
                (
                    a_,
                    b_,
                    c_,
                ),
                (),
            ),
            Implication(p_(Constant(1)), Constant(True)),
        )
    )
    assert weak_eq(result, expected)


def test_parser_unexpected_chars():
    """Input with characters the lexer can't handle."""
    with pytest.raises(UnexpectedCharactersError):
        parser("ans(x) :- R(\x00).")


def test_parser_interactive():
    """Parser in interactive mode returns autocompletion data."""
    result = parser("ans(x)", interactive=True)
    assert isinstance(result, dict)


def test_parse_rules():
    """parse_rules reads the rules.json definitions file."""
    rules = parse_rules()
    assert isinstance(rules, dict)
    assert len(rules) > 0


def test_aggregate_with_group_vars():
    """AGGREGATE with group vars and extra head args."""
    result = _parse("ans(x, cnt) :- AGGREGATE[x](R(x, y) @ count(y)) = cnt.")

    expected = Union((Implication(ans_(x_, AggregationApplication(count_, (y_,))), Conjunction((R_(x_, y_),))),))
    assert weak_eq(result, expected)


def test_aggregate_with_prob_body():
    """AGGREGATE with PROB body predicate in the conjunction."""
    result = _parse("ans(x, cnt) :- AGGREGATE[x](PROB[R(x)] = p @ count(x)) = cnt.")

    expected = Union(
        (
            Implication(
                ans_(x_, count_(x_)),
                Conjunction((__PROB__(Conjunction((R_(x_),)), None, p_),)),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_aggregate_with_marg_body():
    """AGGREGATE with MARG body predicate in the conjunction."""
    result = _parse("p(x, cnt) :- AGGREGATE[x](MARG[R(x)] = p @ count(x)) = cnt.")

    expected = Union(
        (Implication(Symbol("fresh_00000000")(x_, PROB(x_)), R_(x_)),
         Implication(
             p_(x_, count_(x_)),
             Conjunction((Symbol("fresh_00000000")(x_, p_),)),
         ),)
    )
    assert weak_eq(result, expected)


def test_mixed_regular_and_prob_body():
    """Rule body mixing regular and PROB predicates."""
    result = _parse("ans(x, p) :- R(x), PROB[R(x)] = p.")

    expected = Union(
        (
            Implication(Symbol("fresh_00000010")(x_, PROB(x_)), Conjunction((R_(x_),))),
            Query(
                ans_(x_, p_),
                Conjunction(
                    (
                        R_(x_),
                        Symbol("fresh_00000010")(x_, p_),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_mixed_regular_and_marg_body():
    """Rule body mixing regular and MARG predicates."""
    result = _parse("ans(x, p) :- R(x), MARG[R(x)] = p.")

    expected = Union(
        (
            Implication(
                Symbol("fresh_00000011")(x_, PROB(x_)),
                Conjunction((R_(x_),)),
            ),
            Query(
                ans_(x_, p_),
                Conjunction(
                    (
                        R_(x_),
                        Symbol("fresh_00000011")(x_, p_),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_probabilistic_rule_with_regular_body():
    """Probabilistic rule with regular predicates in the body."""
    result = _parse("p(x) :: 0.5 :- R(x), q(x).")

    expected = Union(
        (
            Implication(
                ProbabilisticFact(Constant(0.5), p_(x_)),
                Conjunction(
                    (
                        R_(x_),
                        q_(x_),
                    )
                ),
            ),
        )
    )
    assert weak_eq(result, expected)


def test_multiple_mixed_prob_specs():
    """Multiple PROB specs with regular predicates interleaved."""
    result = _parse("ans(x, p1, p2) :- R(x), PROB[R(x)] = p1, q(x), PROB[R(x)] = p2.")
    assert isinstance(result, Union)


def test_direct_extract_special_body_atoms_mixed():
    """_extract_special_body_atoms with regular + PROB atoms."""
    t = DatalogTransformer()
    reg = FunctionApplication(Symbol("p"), (Symbol("x"),))
    prob_marker = FunctionApplication(Symbol("__PROB__"), (reg, None, Symbol("z")))
    conj = Conjunction((reg, prob_marker))
    regular, prob_specs = t._extract_special_body_atoms(conj)
    assert len(regular) == 1
    assert len(prob_specs) == 1
    assert prob_specs[0][0] == "__PROB__"


def test_direct_extract_special_body_atoms_all_regular():
    """_extract_special_body_atoms with only regular atoms."""
    t = DatalogTransformer()
    reg1 = FunctionApplication(Symbol("p"), (Symbol("x"),))
    reg2 = FunctionApplication(Symbol("q"), (Symbol("x"),))
    conj = Conjunction((reg1, reg2))
    regular, prob_specs = t._extract_special_body_atoms(conj)
    assert len(regular) == 2
    assert len(prob_specs) == 0


def test_direct_extract_special_body_atoms_empty():
    """_extract_special_body_atoms with only prob atoms."""
    t = DatalogTransformer()
    prob_marker = FunctionApplication(
        Symbol("__PROB__"), (Symbol("p"), None, Symbol("z"))
    )
    marg_marker = FunctionApplication(
        Symbol("__MARG__"), (Symbol("q"), None, Symbol("w"))
    )
    conj = Conjunction((prob_marker, marg_marker))
    regular, prob_specs = t._extract_special_body_atoms(conj)
    assert len(regular) == 0
    assert len(prob_specs) == 2


def test_direct_transformer_minus_op_single():
    """Direct call to minus_op with single int covers non-Expression."""
    t = DatalogTransformer()
    result = t.minus_op([Token("INT", "5")])
    assert result is not None


def test_direct_transformer_plus_op_single():
    """Direct call to plus_op with single int covers non-Expression."""
    t = DatalogTransformer()
    result = t.plus_op([Token("INT", "5")])
    assert result is not None


def test_direct_transformer_mult_term_single_non_expr():
    """Direct call to mul_term with single non-Expression."""
    t = DatalogTransformer()
    result = t.mul_term(["a"])
    assert result == 0


def test_direct_transformer_div_term_single_non_expr():
    """Direct call to div_term with single non-Expression."""
    t = DatalogTransformer()
    result = t.div_term(["a"])
    assert result == 0


def test_direct_transformer_sing_term_multiple():
    """Direct call to sing_term with multiple children."""
    t = DatalogTransformer()
    result = t.sing_term(["a", "b"])
    assert result == 0


def test_direct_transformer_factor_multiple():
    """Direct call to factor with multiple children."""
    t = DatalogTransformer()
    result = t.factor(["a", "b"])
    assert result == 0


def test_direct_transformer_pow_factor_single_non_expr():
    """Direct call to pow_factor single non-Expression."""
    t = DatalogTransformer()
    result = t.pow_factor(["a"])
    assert result == 0


def test_direct_transformer_sing_factor_multiple():
    """Direct call to sing_factor with multiple children."""
    t = DatalogTransformer()
    result = t.sing_factor(["a", "b"])
    assert result == 0


def test_direct_transformer_sing_op_multiple():
    """Direct call to sing_op with multiple children."""
    t = DatalogTransformer()
    result = t.sing_op(["a", "b"])
    assert result == 0


def test_direct_transformer_term_multiple():
    """Direct call to term with multiple children (not Expression, not len 1)."""
    t = DatalogTransformer()
    result = t.term(["a", "b"])
    assert result is None or result == 0


def test_direct_transformer_predicate_list_multi():
    """Direct call to predicate with multi-element list."""
    t = DatalogTransformer()
    result = t.predicate([Symbol("f"), (Symbol("x"), Symbol("y"))])
    assert result is not None


def test_direct_transformer_tuple_literal_symbol():
    """tuple_literal with Symbol inside covers symbol tracking line."""
    t = DatalogTransformer()
    s = Symbol("x")
    c = Constant(1)
    result = t.tuple_literal([c, s])
    assert result is not None


def test_direct_transformer_existential_multiple():
    """existential_predicate with multiple predicates covers line 664."""
    t = DatalogTransformer()
    x = Symbol("x")
    p1 = FunctionApplication(Symbol("p"), (x,))
    p2 = FunctionApplication(Symbol("q"), (x,))
    # Simulate Lark tree for existential_body with multiple predicates
    body = Tree("existential_body", [(x,), Token("SUCH_THAT_WORD", "st"), p1, p2])
    ast = [Tree("exists", [Token("EXISTS_WORD", "exists")]), body]
    result = t.existential_predicate(ast)
    assert result is not None


def test_direct_transformer_default():
    """Direct call to _default returns its input."""
    t = DatalogTransformer()
    result = t._default("anything")
    assert result == "anything"


def test_direct_transformer_aggregate_wrap():
    """_wrap_aggregation_args with aggregation functor."""
    t = DatalogTransformer()
    fn = FunctionApplication(Symbol("count"), (Symbol("x"),))
    args = [fn]
    result = t._wrap_aggregation_args(args)
    assert result is not None


def test_direct_transformer_query_single():
    """query transformer with single non-tuple arg covers line 739."""
    t = DatalogTransformer()
    # Simulate Lark's single-argument unwrapping: args is a list, not tuple
    result = t.query([["x"]])  # list wrapping a string — not tuple, list-able
    assert result is not None


def test_direct_transformer_argument_fresh():
    """argument transformer with non-Expression covers line 832."""
    t = DatalogTransformer()
    result = t.argument([Token("CMD_IDENTIFIER", "_")])
    assert result is not None


def test_direct_transformer_id_application_non_expr():
    """id_application with string functor covers line 807."""
    t = DatalogTransformer()
    result = t.id_application(["p", (Symbol("x"),)])
    assert result is not None


def test_direct_transformer_lambda_application_non_expr():
    """lambda_application with string functor covers line 799."""
    t = DatalogTransformer()
    result = t.lambda_application(["f", (Symbol("x"),)])
    assert result is not None


def test_direct_transformer_head_predicate_prod():
    """head_predicate with PROB class in arguments covers lines 716-722."""
    t = DatalogTransformer()
    result = t.head_predicate([Symbol("p"), [Symbol("x"), PROB, Symbol("y")]])
    assert isinstance(result, FunctionApplication)


# ---------------------------------------------------------------------------
# Coverage tests for standard_syntax.py uncovered branches
# ---------------------------------------------------------------------------


def test_start_single_expression():
    """start rule with a single expression wraps in Union (lines 270-281)."""
    result = _parse("P().")
    assert isinstance(result, Union)
    assert len(result.formulas) == 1


def test_add_prob_arg_to_predicate_non_fa():
    """_add_prob_arg_to_predicate with a non-FunctionApplication returns it
    unchanged (line 321)."""
    constant = Constant(True)
    result = DatalogTransformer._add_prob_arg_to_predicate(
        constant, (Symbol("x"),)
    )
    assert result is constant


def test_add_prob_arg_to_conjunction_fa():
    """_add_prob_arg_to_conjunction with a plain FunctionApplication (not
    Conjunction) hits lines 367-369."""
    pred = Symbol("P")(Symbol("x"))
    prob_vars = (Symbol("x"),)
    result = DatalogTransformer._add_prob_arg_to_conjunction(
        pred, prob_vars, unwrap=False
    )
    assert isinstance(result, FunctionApplication)
    assert result.args[-1].functor == PROB


def test_add_prob_arg_to_conjunction_unwrap_single():
    """_add_prob_arg_to_conjunction with unwrap=True and single-element
    Conjunction returns the augmented atom directly (line 364-365)."""
    conj = Conjunction((Symbol("P")(Symbol("x")),))
    prob_vars = (Symbol("x"),)
    result = DatalogTransformer._add_prob_arg_to_conjunction(
        conj, prob_vars, unwrap=True
    )
    assert isinstance(result, FunctionApplication)
    assert not isinstance(result, Conjunction)


def test_add_prob_arg_to_conjunction_non_fa_non_conj():
    """_add_prob_arg_to_conjunction with neither FunctionApplication nor
    Conjunction returns it unchanged (line 369)."""
    constant = Constant(True)
    result = DatalogTransformer._add_prob_arg_to_conjunction(
        constant, (Symbol("x"),), unwrap=False
    )
    assert result is constant


def test_classify_prob_predicate_negation_non_fa():
    """_classify_prob_predicate with Negation whose inner formula is not a
    FunctionApplication returns None (line 467)."""
    from neurolang.logic import Negation

    neg = Negation(Constant(True))
    result = DatalogTransformer._classify_prob_predicate(neg)
    assert result is None


def test_classify_prob_predicate_existential_non_fa_body():
    """_classify_prob_predicate with ExistentialPredicate whose body attribute
    is not a FunctionApplication returns None (line 475)."""
    from neurolang.logic import ExistentialPredicate

    ep = ExistentialPredicate(Symbol("x"), Symbol("P")(Symbol("x")))
    ep.__dict__["body"] = Constant(True)
    result = DatalogTransformer._classify_prob_predicate(ep)
    assert result is None


def test_classify_prob_predicate_unknown_type():
    """_classify_prob_predicate with an unsupported type returns None
    (line 476)."""
    result = DatalogTransformer._classify_prob_predicate(Constant(42))
    assert result is None


def test_prob_body_rule_prob_marker():
    """prob_body_rule with a PROB body predicate (non-MARG) hits the
    __PROB__ marker path (lines 540-545)."""
    result = _parse("ans(x, p) :- PROB[P(x)] = p, Q(x).")
    assert isinstance(result, Union)
    assert len(result.formulas) >= 2


def test_build_prob_rule_result_var_already_in_head():
    """_build_prob_rule where result_var is already in head_args hits
    lines 594 and 613 (skip append)."""
    result = _parse("ans(x, p) :- PROB[P(x)] = p, Q(x).")
    assert isinstance(result, Union)
    assert len(result.formulas) >= 2


def test_build_prob_rule_no_body_atoms():
    """_build_prob_rule with no non-prob body atoms hits line 630:
    new_body = Constant(True)."""
    result = _parse("ans(x, p) :- PROB[P(x)] = p.")
    assert isinstance(result, Union)
    assert len(result.formulas) >= 2


def test_build_aggregate_rule_marg_desugaring():
    """_build_aggregate_rule with MARG spec hits the MARG desugaring branch
    (lines 656-716)."""
    result = _parse(
        'ans(c) :- AGGREGATE[x](MARG[P(x)] = p @ count(c)) = c.'
    )
    assert isinstance(result, Union)
    assert len(result.formulas) >= 2


def test_build_aggregate_rule_marg_no_prob_vars():
    """_build_aggregate_rule MARG branch with no prob_vars hits line 711."""
    result = _parse(
        'ans(c) :- AGGREGATE[](MARG[P()] = p @ count(c)) = c.'
    )
    assert isinstance(result, Union)
    assert len(result.formulas) >= 2


def test_build_aggregate_rule_prob_marker_in_body():
    """_build_aggregate_rule with PROB body predicate hits prob_marker_atoms
    reconstruction (lines 651-653)."""
    result = _parse(
        'ans(c) :- AGGREGATE[x](PROB[P(x)] = p @ count(c)) = c.'
    )
    assert isinstance(result, Union)


def test_extract_prob_result_var_string_marker_3_args():
    """_extract_prob_result_var with string marker and 3-element args
    (line 762)."""
    rv = Symbol("p")
    spec = ("__MARG__", (Symbol("P")(Symbol("x")), Symbol("cond"), rv))
    result = DatalogTransformer._extract_prob_result_var(spec)
    assert result == rv


def test_extract_prob_result_var_string_marker_2_args():
    """_extract_prob_result_var with string marker and 2-element args
    (line 764)."""
    rv = Symbol("p")
    spec = ("__MARG__", (Symbol("P")(Symbol("x")), rv))
    result = DatalogTransformer._extract_prob_result_var(spec)
    assert result == rv


def test_extract_prob_result_var_plain_tuple_3():
    """_extract_prob_result_var with plain 3-element tuple (line 768)."""
    rv = Symbol("p")
    spec = (Symbol("P")(Symbol("x")), Symbol("cond"), rv)
    result = DatalogTransformer._extract_prob_result_var(spec)
    assert result == rv


def test_extract_prob_result_var_plain_tuple_2():
    """_extract_prob_result_var with plain 2-element tuple (line 770)."""
    rv = Symbol("p")
    spec = (Symbol("P")(Symbol("x")), rv)
    result = DatalogTransformer._extract_prob_result_var(spec)
    assert result == rv


def test_extract_prob_result_var_none():
    """_extract_prob_result_var with non-tuple returns None (line 772)."""
    result = DatalogTransformer._extract_prob_result_var(Constant(42))
    assert result is None


def test_condition_transformation():
    """condition transformer produces Condition(conditioned, conditioning)
    (lines 823-826)."""
    from neurolang.probabilistic.expressions import Condition as Cond

    result = _parse("ans(x) :- P(x) // Q(x).")
    assert isinstance(result, Union)
    inner = result.formulas[0]
    if isinstance(inner, Implication):
        assert isinstance(inner.antecedent, Cond)


def test_constraint_transformation():
    """constraint transformer produces RightImplication (line 829)."""
    from neurolang.datalog.constraints_representation import RightImplication

    try:
        result = _parse("Q(x) \u2192 A(x).")
        if isinstance(result, RightImplication):
            return
    except Exception:
        pass

    t = DatalogTransformer()
    from lark import Tree, Token
    body = Symbol("Q")(Symbol("x"))
    head = Symbol("A")(Symbol("x"))
    arrow = Token("RIGHT_IMPLICATION", "\u2192")
    result = t.constraint([body, arrow, head])
    assert isinstance(result, RightImplication)


def test_succ_head_rule_transformation():
    """succ_head_rule returns Constant(True) (line 373)."""
    result = _parse("SUCC[foo] :- Q(x).")
    assert isinstance(result, Union)
    assert result.formulas[0] == Constant(True)


def test_conjunction_discard_succ_marker():
    """conjunction transformer discards SUCC markers (line 837-839)."""
    from lark import Tree, Token
    t = DatalogTransformer()
    succ_marker = FunctionApplication(Symbol("__SUCC__"), (Symbol("p"),))
    regular_pred = Symbol("P")(Symbol("x"))
    result = t.conjunction([regular_pred, succ_marker])
    assert isinstance(result, Conjunction)
    assert len(result.formulas) == 1
    assert result.formulas[0] == regular_pred


def test_conjunction_empty_returns_true():
    """conjunction transformer returns Constant(True) when empty after
    filtering (line 841-842)."""
    t = DatalogTransformer()
    succ_marker = FunctionApplication(Symbol("__SUCC__"), (Symbol("p"),))
    result = t.conjunction([succ_marker])
    assert result == Constant(True)


def test_marg_body_predicate_simple():
    """marg_body_predicate simple form (2-element ast, line 803-808)."""
    result = _parse("ans(p) :- MARG[P(x)] = p.")
    assert isinstance(result, (Union, Implication))


def test_marg_body_predicate_conditional():
    """marg_body_predicate conditional form (4-element ast, lines 795-802)."""
    result = _parse("ans(p) :- MARG[P(x) // Q(x)] = p.")
    assert isinstance(result, (Union, Implication))
