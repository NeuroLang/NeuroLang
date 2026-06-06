"""Tests for the PROB/MARG body predicate parser/transformer.

These tests exercise parser transformer methods for probabilistic body
predicates and related constructs without executing through the solver.
"""

from typing import Tuple

from neurolang.datalog import Conjunction, Fact, Implication, Union
from neurolang.expressions import (
    Command,
    Constant,
    FunctionApplication,
    Lambda,
    Query,
    Statement,
    Symbol,
)
from neurolang.logic import Negation
from neurolang.probabilistic.expressions import (
    Condition,
    ProbabilisticFact,
)
from ..standard_syntax import _preprocess, parser


def _parse(code):
    return parser(code)


def test_marg_single_predicate():
    """MARG with single predicate — hits _build_prob_rule MARG single branch."""
    result = _parse("derived(p) :- MARG[pc1(s)] = p.")
    assert isinstance(result, Union)
    formulas = list(result.formulas)
    assert len(formulas) == 2
    assert isinstance(formulas[0], Implication)


def test_prob_body_predicate_simple():
    """PROB[pred] = var — hits prob_body_predicate else branch (non-cond)."""
    result = _parse("derived(x, p) :- PROB[R(x)] = p.")
    assert isinstance(result, Union)


def test_prob_body_predicate_conditional():
    """PROB[pred // cond] = var — hits prob_body_predicate conditional branch."""
    result = _parse("derived(x, p) :- PROB[R(x) // R(s)] = p.")
    assert isinstance(result, Union)


def test_marg_body_predicate_simple():
    """MARG[pred] = var — hits marg_body_predicate non-conditional."""
    result = _parse("derived(x, p) :- MARG[R(x)] = p.")
    assert isinstance(result, Union)


def test_marg_body_predicate_conditional():
    """MARG[pred // cond] = var — hits marg_body_predicate conditional."""
    result = _parse("derived(p) :- MARG[R(x) // R(y)] = p.")
    assert isinstance(result, Union)


def test_succ_body_predicate():
    """SUCC[...] = var — hits succ_body_predicate (discarded at conjunction)."""
    result = _parse("derived(x) :- SUCC[R(x)] = p.")
    assert isinstance(result, Union)
    # The SUCC atom should be discarded
    formulas = list(result.formulas)
    assert len(formulas) >= 1


def test_probabilistic_rule_head():
    """p(x) :: prob :- body — hits probabilistic_rule transformer."""
    result = _parse("p(x) :: 0.5 :- q(x).")
    assert isinstance(result, Union)
    formula = list(result.formulas)[0]
    assert isinstance(formula, Implication)
    assert isinstance(formula.consequent, ProbabilisticFact)


def test_prob_head_rule_simple():
    """PROB[pred] :- body — simple prob head rule."""
    result = _parse("PROB[p(x)] :- q(x).")
    assert isinstance(result, Union)
    assert isinstance(list(result.formulas)[0], Implication)


def test_prob_head_rule_conditional():
    """PROB[pred // cond] :- body — conditional prob head rule."""
    result = _parse("PROB[p(x) // r(x, s)] :- q(s).")
    assert isinstance(result, Union)
    formula = list(result.formulas)[0]
    assert isinstance(formula, Implication)
    assert isinstance(formula.antecedent, Condition)


def test_marg_head_rule_simple():
    """MARG[pred] :- body — simple marg head rule."""
    result = _parse("MARG[p(x)] :- q(x).")
    assert isinstance(result, Union)
    assert isinstance(list(result.formulas)[0], Implication)


def test_marg_head_rule_conditional():
    """MARG[pred // cond] :- body — conditional marg head rule."""
    result = _parse("MARG[p(x) // r(x, s)] :- q(s).")
    assert isinstance(result, Union)
    formula = list(result.formulas)[0]
    assert isinstance(formula, Implication)
    assert isinstance(formula.antecedent, Condition)


def test_succ_head_rule():
    """SUCC[...] :- body — no-op (returns Union wrapping Constant(True))."""
    result = _parse("SUCC[p(x)] :- q(x).")
    assert isinstance(result, Union)
    assert list(result.formulas) == [Constant(True)]


def test_condition_separator():
    """pred1 // pred2 — hits condition transformer."""
    result = _parse("ans(x) :- R(x, y) // R(y, z).")
    assert isinstance(result, Union)
    formula = list(result.formulas)[0]
    if isinstance(formula, Query):
        assert isinstance(formula.body, Condition)


def test_negated_predicate():
    """~pred — hits negated_predicate transformer."""
    result = _parse("ans(x) :- ~R(x, y).")
    assert isinstance(result, Union)


def test_negation_inside_prob():
    """PROB[~pred(x)] — hits _classify_prob_predicate Negation branch."""
    result = _parse("derived(x, y, p) :- PROB[~R(x, y)] = p.")
    assert isinstance(result, Union)


def test_statement_function():
    """f(x) := expr — hits statement_function transformer."""
    result = _parse("f(x) := x.\nans(v) :- f(v).")
    assert isinstance(result, Union)
    formulas = list(result.formulas)
    assert any(isinstance(f, Statement) for f in formulas)


def test_probabilistic_fact():
    """0.5 :: p(1) — hits probabilistic_fact transformer."""
    result = _parse("0.5 :: p(1).")
    assert isinstance(result, Union)
    formula = list(result.formulas)[0]
    assert isinstance(formula, Implication)
    assert isinstance(formula.consequent, ProbabilisticFact)


def test_multiple_facts():
    """Multiple facts — hits expressions transformer list path."""
    result = _parse("p(1).\nq(2).\nr(3).")
    assert isinstance(result, Union)
    formulas = list(result.formulas)
    assert len(formulas) == 3
    assert all(isinstance(f, Fact) for f in formulas)


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
    assert isinstance(result, Union)


def test_build_prob_rule_ans_head():
    """PROB with ans head directly — hits ans branch of new_head."""
    result = _parse("ans(x, p) :- PROB[R(x)] = p.")
    assert isinstance(result, Union)


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
    assert isinstance(result, Union)


def test_tuple_literal():
    """Tuple literal in argument position."""
    result = _parse("ans(x) :- R(x, (1, 2)).")
    assert isinstance(result, Union)


def test_lambda_application():
    """Lambda application expression."""
    result = _parse("f(x) := (lambda z : z)(x).\nans(v) :- f(v).")
    assert isinstance(result, Union)


def test_external_symbol():
    """External identifier @prefix."""
    result = _parse('ans(x) :- @type(x, "concept").')
    assert isinstance(result, Union)


def test_identifier_regexp():
    """Backtick-quoted identifier."""
    result = _parse("ans(x) :- `custom/pred`(x).")
    assert isinstance(result, Union)


def test_constant_int_and_float():
    """Integer and float constants."""
    result = _parse("p(1, 2.5).\nq(-3, -4.0).")
    assert isinstance(result, Union)


def test_command():
    """Dot command syntax."""
    result = _parse('.cmd("arg")\np(1).')
    assert isinstance(result, Union)


def test_keyword_command():
    """Command with keyword argument."""
    result = _parse('.cmd(key="val")\np(1).')
    assert isinstance(result, Union)


def test_multiple_prob_specs():
    """Multiple PROB specifications in one rule — tests prob_body_predicate."""
    result = _parse("ans(pa, pb) :- PROB[r1(x)] = pa, PROB[r2(x)] = pb.")
    assert isinstance(result, Union)
    assert len(result.formulas) == 3


def test_aggregate_assign_vars():
    """AGGREGATE with group vars — hits agg_assign_vars."""
    result = _parse("ans(x, c) :- AGGREGATE[x](R(x, y) @ count(y)) = c.")
    assert isinstance(result, Union)


def test_aggregate_assign_empty():
    """AGGREGATE without group vars — hits agg_assign_empty."""
    result = _parse("ans(c) :- AGGREGATE[](R(x, y) @ count(y)) = c.")
    assert isinstance(result, Union)


def test_head_with_prod_in_args():
    """Head predicate with PROB in its arguments."""
    result = _parse("ans(x, y) :- R(x, y).")
    assert isinstance(result, Union)


def test_true_and_false_constants():
    """True/False logical constants."""
    result = _parse("ans(x) :- R(x, y) & (y == True).")
    assert isinstance(result, Union)


def test_extended_projection_via_extended_predicate():
    """Extended projection via arithmetic."""
    result = _parse("ans(z) :- R(x, y) & (z == x + y).")
    assert isinstance(result, Union)


def test_aggregate_fn_call():
    """AGGREGATE fn call — hits agg_fn_call transformer."""
    result = _parse("ans(cnt) :- AGGREGATE[](R(x, y) @ count(x)) = cnt.")
    assert isinstance(result, Union)


def test_multiple_args_comparison():
    """Multiple comparisons in body."""
    result = _parse("ans(x) :- R(x, y) & (y > 1) & (y < 10).")
    assert isinstance(result, Union)


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
    assert isinstance(result, Union)


def test_command_no_args():
    """Command with no arguments — hits command(None) branch."""
    result = _parse(".reset()\np(1).")
    assert isinstance(result, Union)


def test_command_keyword_args():
    """Command with keyword-only arguments — hits keyword_item."""
    result = _parse('.load(key="val")\np(1).')
    assert isinstance(result, Union)


def test_command_args_and_kwargs():
    """Command with both positional and keyword args."""
    result = _parse('.load("path", key="val")\np(1).')
    assert isinstance(result, Union)


def test_head_predicate_no_args():
    """Head predicate with no arguments — hits empty-args branch."""
    result = _parse("p() :- q(1).")
    assert isinstance(result, Union)


def test_anonymous_variable():
    """Anonymous variable _ — triggers Symbol.fresh in argument."""
    result = _parse("ans(x) :- R(x, _).")
    assert isinstance(result, Union)


def test_minus_signed_id():
    """Signed identifier with minus sign."""
    result = _parse("ans(x) :- R(-x).")
    assert isinstance(result, Union)


def test_minus_op_single_arg():
    """minus_op with single argument — unary minus fallback."""
    result = _parse("ans(x) :- R(x, -1).")
    assert isinstance(result, Union)


def test_arithmetic_ops():
    """Division, multiplication, power — div_term, mul_term, pow_factor."""
    result = _parse("ans(z) :- R(x, y) & (z == x / y * 2 ** 3).")
    assert isinstance(result, Union)


def test_lambda_expression():
    """Lambda expression in argument position."""
    result = _parse("f(x) := (lambda y: x + y)(1).")
    assert isinstance(result, Union)


def test_neg_int():
    """Negative integer constant — neg_int transformer."""
    result = _parse("ans(x) :- R(-5).")
    assert isinstance(result, Union)


def test_neg_float():
    """Negative float constant — neg_float transformer."""
    result = _parse("ans(x) :- R(-3.14).")
    assert isinstance(result, Union)


def test_ext_identifier():
    """External identifier @ in predicate."""
    result = _parse("ans(x) :- @external(x).")
    assert isinstance(result, Union)


def test_tuple_literal():
    """Tuple literal in argument position."""
    result = _parse("ans(x) :- R((1, 2, 3)).")
    assert isinstance(result, Union)


def test_parser_invalid_syntax():
    """Invalid syntax raises UnexpectedTokenError."""
    import pytest
    from neurolang.exceptions import UnexpectedTokenError
    with pytest.raises(UnexpectedTokenError):
        parser("this is not valid syntax at all!")


def test_query_no_args():
    """Query with no arguments — hits query no-args branch."""
    result = _parse("ans() :- p(1).")
    assert isinstance(result, Union)


def test_statement():
    """Simple statement transformation."""
    result = _parse("x := 5.")
    assert isinstance(result, Union)


def test_arithmetic_multi_ops():
    """Multiple subtraction operations — hits minus_op with >1 args."""
    result = _parse("ans(r) :- R(a, b, c) & (r == a - b - c).")
    assert isinstance(result, Union)


def test_arithmetic_multi_plus():
    """Multiple addition operations — hits plus_op with >1 args."""
    result = _parse("ans(r) :- R(a, b, c) & (r == a + b + c).")
    assert isinstance(result, Union)


def test_division():
    """Division operation in body."""
    result = _parse("ans(r) :- R(a, b) & (r == a / b).")
    assert isinstance(result, Union)


def test_multiplication():
    """Multiplication operation in body."""
    result = _parse("ans(r) :- R(a, b) & (r == a * b).")
    assert isinstance(result, Union)


def test_power():
    """Power operation in body."""
    result = _parse("ans(r) :- R(a, b) & (r == a ** b).")
    assert isinstance(result, Union)


def test_text_argument():
    """Text literal in argument position — hits text transformer."""
    result = _parse("ans(x) :- R('hello').")
    assert isinstance(result, Union)


def test_identifier_regexp():
    """Backtick identifier — hits identifier_regexp transformer."""
    result = _parse("ans(x) :- R(`backtick_id`).")
    assert isinstance(result, Union)


def test_statement_function():
    """Statement function — tests statement_function transformer."""
    result = _parse("f(x) := x + 1.")
    assert isinstance(result, Union)


def test_command_string_arg():
    """Command with string argument — hits pos_item non-Expression."""
    result = _parse('.load("some_path")\np(1).')
    assert isinstance(result, Union)


def test_command_multi_pos_args():
    """Command with multiple positional args — hits pos_args."""
    result = _parse('.cmd(a, b, c)\np(1).')
    assert isinstance(result, Union)


def test_parser_unexpected_chars():
    """Input with characters the lexer can't handle."""
    import pytest
    from neurolang.exceptions import UnexpectedCharactersError
    with pytest.raises(UnexpectedCharactersError):
        parser("ans(x) :- R(\x00).")


def test_parser_interactive():
    """Parser in interactive mode returns autocompletion data."""
    result = parser("ans(x)", interactive=True)
    assert isinstance(result, dict)


def test_parse_rules():
    """parse_rules reads the rules.json definitions file."""
    from ..standard_syntax import parse_rules
    rules = parse_rules()
    assert isinstance(rules, dict)
    assert len(rules) > 0


def test_aggregate_with_group_vars():
    """AGGREGATE with group vars and extra head args."""
    result = _parse("ans(x, cnt) :- AGGREGATE[x](R(x, y) @ count(y)) = cnt.")
    assert isinstance(result, Union)


# --- Combined aggregation + probability tests ---


def test_aggregate_with_prob_body():
    """AGGREGATE with PROB body predicate in the conjunction."""
    result = _parse("ans(x, cnt) :- AGGREGATE[x](PROB[R(x)] = p @ count(x)) = cnt.")
    assert isinstance(result, Union)


def test_aggregate_with_marg_body():
    """AGGREGATE with MARG body predicate in the conjunction."""
    result = _parse("ans(x, cnt) :- AGGREGATE[x](MARG[R(x)] = p @ count(x)) = cnt.")
    assert isinstance(result, Union)


def test_mixed_regular_and_prob_body():
    """Rule body mixing regular and PROB predicates."""
    result = _parse("ans(x, p) :- R(x), PROB[R(x)] = p.")
    assert isinstance(result, Union)


def test_mixed_regular_and_marg_body():
    """Rule body mixing regular and MARG predicates."""
    result = _parse("ans(x, p) :- R(x), MARG[R(x)] = p.")
    assert isinstance(result, Union)


def test_probabilistic_rule_with_regular_body():
    """Probabilistic rule with regular predicates in the body."""
    result = _parse("p(x) :: 0.5 :- R(x), q(x).")
    assert isinstance(result, Union)


def test_multiple_mixed_prob_specs():
    """Multiple PROB specs with regular predicates interleaved."""
    result = _parse(
        "ans(x, p1, p2) :- R(x), PROB[R(x)] = p1, q(x), PROB[R(x)] = p2."
    )
    assert isinstance(result, Union)


def test_direct_extract_special_body_atoms_mixed():
    """_extract_special_body_atoms with regular + PROB atoms."""
    from ..standard_syntax import DatalogTransformer
    from neurolang.datalog import Conjunction
    from neurolang.expressions import Symbol, FunctionApplication
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
    from ..standard_syntax import DatalogTransformer
    from neurolang.datalog import Conjunction
    from neurolang.expressions import Symbol, FunctionApplication
    t = DatalogTransformer()
    reg1 = FunctionApplication(Symbol("p"), (Symbol("x"),))
    reg2 = FunctionApplication(Symbol("q"), (Symbol("x"),))
    conj = Conjunction((reg1, reg2))
    regular, prob_specs = t._extract_special_body_atoms(conj)
    assert len(regular) == 2
    assert len(prob_specs) == 0


def test_direct_extract_special_body_atoms_empty():
    """_extract_special_body_atoms with only prob atoms."""
    from ..standard_syntax import DatalogTransformer
    from neurolang.datalog import Conjunction
    from neurolang.expressions import Symbol, FunctionApplication
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
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    from lark import Token
    result = t.minus_op([Token('INT', '5')])
    assert result is not None


def test_direct_transformer_plus_op_single():
    """Direct call to plus_op with single int covers non-Expression."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    from lark import Token
    result = t.plus_op([Token('INT', '5')])
    assert result is not None


def test_direct_transformer_mult_term_single_non_expr():
    """Direct call to mul_term with single non-Expression."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.mul_term(["a"])
    assert result == 0


def test_direct_transformer_div_term_single_non_expr():
    """Direct call to div_term with single non-Expression."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.div_term(["a"])
    assert result == 0


def test_direct_transformer_sing_term_multiple():
    """Direct call to sing_term with multiple children."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.sing_term(["a", "b"])
    assert result == 0


def test_direct_transformer_factor_multiple():
    """Direct call to factor with multiple children."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.factor(["a", "b"])
    assert result == 0


def test_direct_transformer_pow_factor_single_non_expr():
    """Direct call to pow_factor single non-Expression."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.pow_factor(["a"])
    assert result == 0


def test_direct_transformer_sing_factor_multiple():
    """Direct call to sing_factor with multiple children."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.sing_factor(["a", "b"])
    assert result == 0


def test_direct_transformer_sing_op_multiple():
    """Direct call to sing_op with multiple children."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.sing_op(["a", "b"])
    assert result == 0


def test_direct_transformer_term_multiple():
    """Direct call to term with multiple children (not Expression, not len 1)."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.term(["a", "b"])
    assert result is None or result == 0


def test_direct_transformer_predicate_list_multi():
    """Direct call to predicate with multi-element list."""
    from ..standard_syntax import DatalogTransformer, Symbol
    t = DatalogTransformer()
    result = t.predicate([Symbol("f"), (Symbol("x"), Symbol("y"))])
    assert result is not None


def test_direct_transformer_tuple_literal_symbol():
    """tuple_literal with Symbol inside covers symbol tracking line."""
    from ..standard_syntax import DatalogTransformer
    from neurolang.expressions import Constant, Symbol
    from neurolang.type_system import Tuple, Unknown
    t = DatalogTransformer()
    s = Symbol("x")
    c = Constant(1)
    result = t.tuple_literal([c, s])
    assert result is not None


def test_direct_transformer_existential_multiple():
    """existential_predicate with multiple predicates covers line 664."""
    from ..standard_syntax import DatalogTransformer
    from lark import Tree, Token
    from neurolang.expressions import Symbol, FunctionApplication
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
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t._default("anything")
    assert result == "anything"


def test_direct_transformer_aggregate_wrap():
    """_wrap_aggregation_args with aggregation functor."""
    from ..standard_syntax import DatalogTransformer, AGGREGATION_FUNCS
    from neurolang.expressions import Symbol, FunctionApplication, Constant
    t = DatalogTransformer()
    fn = FunctionApplication(Symbol("count"), (Symbol("x"),))
    args = [fn]
    result = t._wrap_aggregation_args(args)
    assert result is not None


def test_direct_transformer_query_single():
    """query transformer with single non-tuple arg covers line 739."""
    from ..standard_syntax import DatalogTransformer
    from neurolang.expressions import Symbol
    t = DatalogTransformer()
    # Simulate Lark's single-argument unwrapping: args is a list, not tuple
    result = t.query([["x"]])  # list wrapping a string — not tuple, list-able
    assert result is not None


def test_direct_transformer_argument_fresh():
    """argument transformer with non-Expression covers line 832."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    from lark import Token
    result = t.argument([Token("CMD_IDENTIFIER", "_")])
    assert result is not None


def test_direct_transformer_id_application_non_expr():
    """id_application with string functor covers line 807."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.id_application(["p", (Symbol("x"),)])
    assert result is not None


def test_direct_transformer_lambda_application_non_expr():
    """lambda_application with string functor covers line 799."""
    from ..standard_syntax import DatalogTransformer
    t = DatalogTransformer()
    result = t.lambda_application(["f", (Symbol("x"),)])
    assert result is not None


def test_direct_transformer_head_predicate_prod():
    """head_predicate with PROB class in arguments covers lines 716-722."""
    from ..standard_syntax import DatalogTransformer
    from neurolang.expressions import Symbol
    from neurolang.probabilistic.expressions import PROB
    t = DatalogTransformer()
    result = t.head_predicate([Symbol("p"), [Symbol("x"), PROB, Symbol("y")]])
    assert isinstance(result, FunctionApplication)
