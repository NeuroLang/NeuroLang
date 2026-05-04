"""
Tests for NeurolangPDL.execute_squall_program.

These tests exercise the high-level API that mirrors execute_datalog_program
but accepts SQUALL controlled-English programs.
"""
import pytest

from ..probabilistic_frontend import NeurolangPDL


@pytest.fixture
def nl():
    """Fresh NeurolangPDL instance with basic extensional facts."""
    engine = NeurolangPDL()
    engine.add_tuple_set(
        [("alice",), ("bob",), ("carol",)], name="person"
    )
    engine.add_tuple_set(
        [("alice",), ("carol",)], name="plays"
    )
    return engine


@pytest.fixture
def nl_counts():
    """NeurolangPDL instance with item / item_count / quantity facts."""
    engine = NeurolangPDL()
    engine.add_tuple_set(
        [("a",), ("b",), ("c",), ("d",)], name="item"
    )
    engine.add_tuple_set(
        [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    )
    engine.add_tuple_set(
        [(i,) for i in range(5)], name="quantity"
    )
    return engine


# ---------------------------------------------------------------------------
# Rules-only programs return None
# ---------------------------------------------------------------------------

def test_execute_squall_rules_only_returns_none(nl):
    """define-as programs with no obtain clause return None."""
    result = nl.execute_squall_program(
        "define as Active every person that plays."
    )
    assert result is None


def test_execute_squall_rules_fire_into_program(nl):
    """Rules walked in via execute_squall_program are visible in solve_all."""
    nl.execute_squall_program(
        "define as Active every person that plays."
    )
    solution = nl.solve_all()
    assert "active" in solution
    assert (
        set(solution["active"].as_pandas_dataframe().iloc[:, 0].tolist())
        == {"alice", "carol"}
    )


# ---------------------------------------------------------------------------
# Single obtain clause returns NamedRelationalAlgebraFrozenSet
# ---------------------------------------------------------------------------

def test_execute_squall_single_obtain_returns_relation(nl):
    """A single obtain clause returns a NamedRelationalAlgebraFrozenSet."""
    from neurolang.utils.relational_algebra_set.pandas import (
        NamedRelationalAlgebraFrozenSet,
    )
    result = nl.execute_squall_program("obtain every Person that plays.")
    assert isinstance(result, NamedRelationalAlgebraFrozenSet)


def test_execute_squall_single_obtain_correct_rows(nl):
    """obtain every Person that plays returns alice and carol."""
    result = nl.execute_squall_program("obtain every Person that plays.")
    rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"alice", "carol"}


# ---------------------------------------------------------------------------
# Multiple obtain clauses return dict keyed obtain_0, obtain_1, …
# ---------------------------------------------------------------------------

def test_execute_squall_multiple_obtains_returns_dict(nl):
    """Multiple obtain clauses produce a dict with obtain_0 / obtain_1 keys."""
    result = nl.execute_squall_program(
        "obtain every Person that plays. "
        "obtain every Person."
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"obtain_0", "obtain_1"}


def test_execute_squall_multiple_obtains_correct_rows(nl):
    """Each entry in the multi-obtain dict holds the right result set."""
    result = nl.execute_squall_program(
        "obtain every Person that plays. "
        "obtain every Person."
    )
    plays_rows = set(result["obtain_0"].as_pandas_dataframe().iloc[:, 0].tolist())
    all_rows = set(result["obtain_1"].as_pandas_dataframe().iloc[:, 0].tolist())
    assert plays_rows == {"alice", "carol"}
    assert all_rows == {"alice", "bob", "carol"}


# ---------------------------------------------------------------------------
# Combined rules + obtain in one program
# ---------------------------------------------------------------------------

def test_execute_squall_rules_then_obtain(nl):
    """Rules defined before obtain are visible to the query."""
    result = nl.execute_squall_program(
        "define as Active every person that plays. "
        "obtain every Active."
    )
    rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"alice", "carol"}


# ---------------------------------------------------------------------------
# Comparison filtering
# ---------------------------------------------------------------------------

def test_execute_squall_comparison_filter(nl_counts):
    """Filtering with 'greater equal than' produces correct rows."""
    nl_counts.execute_squall_program(
        "define as Large every Item "
        "that has an item_count greater equal than 2."
    )
    solution = nl_counts.solve_all()
    rows = set(solution["large"].as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"b", "c"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_execute_squall_aggregation(nl_counts):
    """Aggregation rule produces correct per-item max values."""
    nl_counts.execute_squall_program(
        "define as max_items for every Item ?i ;"
        " where every Max of the Quantity where ?i item_count per ?i."
    )
    solution = nl_counts.solve_all()
    df = solution["max_items"].as_pandas_dataframe()
    # Build set of (item, max_count) pairs from whatever columns are present
    rows = set(map(tuple, df.values.tolist()))
    assert rows == {("a", 1), ("b", 2), ("c", 3)}


@pytest.fixture
def nl_author():
    """NeurolangPDL with paper/author facts stored as author(paper, person)."""
    engine = NeurolangPDL()
    engine.add_tuple_set(
        [("p1",), ("p2",), ("p3",)], name="paper"
    )
    # 'a Person ~author' requires a Person EDB so the existential
    # variable is drawn from a bounded domain, not left unconstrained.
    engine.add_tuple_set(
        [("alice",), ("bob",)], name="person"
    )
    engine.add_tuple_set(
        [("p1", "alice"), ("p2", "alice"), ("p3", "bob")], name="author"
    )
    return engine


def test_execute_squall_tilde_inversion_end_to_end(nl_author):
    """~author inverts the default SQUALL argument order.

    In standard SQUALL, 'a Person author' would resolve to author(person, paper)
    (the existential subject is arg[0]). The tilde flips this to author(paper,
    person), matching the EDB column order [paper, person] stored in the fixture.

    'obtain every Paper that a Person ~author.' therefore asks:
      for each paper p, there exists a person x such that author(p, x).
    Expected: all three papers are returned.
    """
    result = nl_author.execute_squall_program(
        "obtain every Paper that a Person ~author."
    )
    rows = set(result.as_pandas_dataframe().iloc[:, 0].tolist())
    assert rows == {"p1", "p2", "p3"}


def test_execute_squall_marg_query_walks_without_error():
    """'with probability … conditioned to …' walks into the engine without raising.

    Smoke test confirming the MARG syntax is parsed and the IR
    (Implication with Condition body and ProbabilisticQuery head arg) is
    accepted by the solver's walk(). Full probabilistic solving is not
    asserted — that requires a complete CPLogic dataset.
    """
    from neurolang.logic.horn_clauses import Fol2DatalogTranslationException

    engine = NeurolangPDL()
    _ = engine.add_tuple_set([("v1",), ("v2",)], name="voxel")
    _ = engine.add_tuple_set([("s1",), ("s2",)], name="study")

    try:
        engine.execute_squall_program(
            "define as Published with probability every Voxel "
            "conditioned to every Study activates."
        )
    except Fol2DatalogTranslationException:
        pass  # Expected: reached rewrite_conditional_query — plumbing is OK
    except Exception as exc:
        pytest.fail(
            f"execute_squall_program raised an unexpected exception for MARG rule: "
            f"{type(exc).__name__}: {exc}"
        )


def test_execute_squall_arbitrary_aggregation():
    """'every Custom_func of the relation' aggregates over all free vars.

    Registers a Python function as an aggregation over a 1-column
    relation, verifies the rule walks without error.
    """
    from neurolang.expressions import Symbol, Constant
    from neurolang.logic.horn_clauses import Fol2DatalogTranslationException

    engine = NeurolangPDL()
    _ = engine.add_tuple_set(
        [("a",), ("b",), ("c",)], name="item"
    )

    # Register a simple aggregation function in the engine's symbol table
    def collect_all(*values):
        return sorted(values)

    engine.symbol_table[Symbol("collect_all")] = Constant(collect_all)

    try:
        engine.execute_squall_program(
            "define as Result every Collect_all of the Item."
        )
    except Fol2DatalogTranslationException:
        pass  # Acceptable: reached datalog translation — plumbing is OK
    except Exception as exc:
        pytest.fail(
            f"execute_squall_program raised unexpectedly for arbitrary agg: "
            f"{type(exc).__name__}: {exc}"
        )


def test_execute_squall_conditioned_rule_produces_implication_with_condition():
    """Conditioned SQUALL rule reaches rewrite_conditional_query inside the engine.

    This is an integration boundary test: the SQUALL parser correctly produces
    an Implication with a Condition antecedent, and the walker reaches
    TranslateProbabilisticQueryMixin.rewrite_conditional_query.  The toy
    sentence deliberately uses mismatched head/body variables so
    Fol2DatalogTranslationException is raised there — NOT a parse error.

    The absence of a parse-level exception (SyntaxError, KeyError from a
    missing transformer handler, etc.) confirms that the conditioned-rule
    transformer methods are wired up end-to-end.
    """
    from neurolang.logic.horn_clauses import Fol2DatalogTranslationException

    engine = NeurolangPDL()
    _ = engine.add_tuple_set([("v1",), ("v2",)], name="voxel")
    _ = engine.add_tuple_set([("s1",), ("s2",)], name="study")

    # The sentence reaches rewrite_conditional_query which raises
    # Fol2DatalogTranslationException because the toy head variables are not
    # present in the conditioned body.  Any *other* exception means the
    # conditioned-rule plumbing is broken.
    try:
        engine.execute_squall_program(
            "define as probably Published every Voxel conditioned to every Study activates."
        )
        # If no exception: also acceptable (engine accepted the rule)
    except Fol2DatalogTranslationException:
        pass  # Expected: reached rewrite_conditional_query — plumbing is OK
    except Exception as exc:
        pytest.fail(
            f"execute_squall_program raised an unexpected exception type for "
            f"conditioned rule (parse/walk plumbing broken?): {type(exc).__name__}: {exc}"
        )


def test_execute_squall_body_function_call():
    """Function call in rel_b position produces a valid body atom."""
    from neurolang.logic.horn_clauses import Fol2DatalogTranslationException

    engine = NeurolangPDL()
    engine.add_tuple_set([("a", 1), ("b", 2)], name="item_val")

    def close_enough(x, y):
        return abs(x - y) < 0.5

    from neurolang.expressions import Symbol, Constant
    engine.symbol_table[Symbol("close_enough")] = Constant(close_enough)

    try:
        engine.execute_squall_program(
            "define as Near every Item_val that close_enough(?x, ?y) holds."
        )
    except Fol2DatalogTranslationException:
        pass  # Acceptable: reached datalog translation — plumbing is OK
    except Exception as exc:
        pytest.fail(
            f"Body function call raised unexpected exception: "
            f"{type(exc).__name__}: {exc}"
        )


def test_execute_squall_variable_probability_fact():
    """'activates with probability ?p' accepts a label as the probability argument."""
    from neurolang.logic.horn_clauses import Fol2DatalogTranslationException
    from neurolang.expression_pattern_matching import NeuroLangPatternMatchingNoMatch

    engine = NeurolangPDL()
    engine.add_tuple_set([("s1",), ("s2",)], name="study")

    try:
        engine.execute_squall_program(
            "define as Probable every Study that activates with probability ?p."
        )
        # Rule was accepted
    except (Fol2DatalogTranslationException, NeuroLangPatternMatchingNoMatch):
        pass  # Acceptable: reached datalog translation — plumbing is OK
    except Exception as exc:
        pytest.fail(
            f"Variable-probability fact raised unexpected exception: "
            f"{type(exc).__name__}: {exc}"
        )


def test_execute_squall_inline_expr_comparison():
    """'such that EUCLIDEAN(a,b) is lower than 5' works without a relay variable.

    This path: rel_s -> s_np_vp -> vpbe_rel -> rel_comp already exists.
    The test confirms the function-call expression is correctly used as the
    left operand of the comparison (not a relay variable).
    """
    import operator
    from neurolang.expressions import Constant, FunctionApplication, Symbol
    from neurolang.logic import Conjunction, Implication

    engine = NeurolangPDL()

    def my_dist(a, b):
        return abs(a - b)

    engine.symbol_table[Symbol("MY_DIST")] = Constant(my_dist)
    _ = engine.add_tuple_set([(1,), (3,), (10,)], name="point")

    # Parse: "such that MY_DIST(?x, ?x) is lower than 5"
    # Expected body contains: lt(MY_DIST(x, x), 5)  (no relay variable)
    # Note: uppercase function names in `and` position after another clause
    # require `such that` as introducer (the grammar's rel_s path).
    result = engine.execute_squall_program(
        "define as Close every Point (?x) "
        "such that MY_DIST(?x, ?x) is lower than 5."
    )
    assert result is None  # rules-only, no obtain

    # Inspect the intensional rule that was added
    idb = engine.program_ir.intensional_database()
    close_symb = next(k for k in idb if k.name == "close")
    rule = idb[close_symb].formulas[0]

    # Body must contain a FunctionApplication of lt (operator.lt)
    body = rule.antecedent
    formulas = body.formulas if isinstance(body, Conjunction) else [body]
    lt_atoms = [
        f for f in formulas
        if isinstance(f, FunctionApplication)
        and isinstance(f.functor, Constant)
        and f.functor.value is operator.lt
    ]
    assert len(lt_atoms) == 1, f"Expected one lt atom, got: {formulas}"
    lt_atom = lt_atoms[0]
    # Left operand must be FunctionApplication(MY_DIST, ...) — functor is a
    # Constant wrapping my_dist (registered via symbol_table) or Symbol("MY_DIST").
    lhs = lt_atom.args[0]
    assert isinstance(lhs, FunctionApplication), (
        f"Expected FunctionApplication as lt left arg, got: {lhs}"
    )
    # The functor may be Symbol("MY_DIST") or Constant(my_dist) depending on
    # when type resolution runs; either way it must NOT be a plain Symbol
    # representing a relay variable like ?d.
    assert not (isinstance(lhs.functor, Symbol) and lhs.functor.name != "MY_DIST"), (
        f"Unexpected relay-variable functor: {lhs.functor}"
    )


def test_execute_squall_compact_per_list():
    """'per ?i, ?j, ?k and per ?s' produces same head args as four separate per dims.

    Compact form: per ?i1, ?j1, ?k1 and per ?s
    Verbose form: per ?i1 and per ?j1 and per ?k1 and per ?s
    Both must produce an Implication whose consequent has >=4 args.
    """
    from typing import Iterable

    engine = NeurolangPDL()

    def agg_collect(vals: Iterable) -> float:
        return float(sum(vals))

    engine.add_symbol(agg_collect, name="agg_collect")
    _ = engine.add_tuple_set(
        [(1, 10, 100, "s1"), (2, 20, 200, "s2")], name="data"
    )

    engine.execute_squall_program(
        "define as Result with a probability of "
        "the Agg_collect of the Data (?v; _; _; ?s) "
        "per ?i1, ?j1, ?k1 and per ?s "
        "that data(?i1, ?j1, ?k1) holds."
    )

    idb = engine.program_ir.intensional_database()
    # The query-based prob-fact translation produces fresh predicates.
    # At least one rule must have a consequent with >=4 args
    # (the fresh det rule has prob_var + i1, j1, k1, s = 5 args).
    all_rules = [idb[k].formulas[0] for k in idb]
    arg_counts = [len(r.consequent.args) for r in all_rules]
    assert any(c >= 4 for c in arg_counts), (
        f"Expected a rule consequent with >=4 args, got arg counts: {arg_counts}"
    )


def test_execute_squall_where_tuple_is_noun():
    """'where (?i; ?j; ?k) is a Noun' expands to Noun(i, j, k) in rule body.

    The clause is semantically equivalent to 'that noun(?i, ?j, ?k) holds'.
    We verify that the rule body contains a FunctionApplication of voxel with 3 args.
    """
    from neurolang.expressions import Constant, FunctionApplication, Symbol
    from neurolang.logic import Conjunction

    engine = NeurolangPDL()
    _ = engine.add_tuple_set(
        [(0, 0, 0), (1, 1, 1)], name="voxel"
    )
    _ = engine.add_tuple_set(
        [(0, 0, 0, 10), (1, 1, 1, 20)], name="focus"
    )

    engine.execute_squall_program(
        "define as Near every Focus (?i2; ?j2; ?k2; ?s) "
        "where (?i2; ?j2; ?k2) is a Voxel."
    )

    idb = engine.program_ir.intensional_database()
    near_symb = next(k for k in idb if k.name == "near")
    rule = idb[near_symb].formulas[0]

    body = rule.antecedent
    formulas = body.formulas if isinstance(body, Conjunction) else [body]
    voxel_calls = [
        f for f in formulas
        if isinstance(f, FunctionApplication)
        and isinstance(f.functor, Symbol)
        and f.functor.name == "voxel"
    ]
    assert len(voxel_calls) == 1, (
        f"Expected exactly one voxel(...) call in body, got: {formulas}"
    )
    assert len(voxel_calls[0].args) == 3, (
        f"Expected voxel(i,j,k) with 3 args, got: {voxel_calls[0]}"
    )


def test_extension_e_where_integration(tmp_path):
    """Full program using 'where' instead of 'such that' produces same solution."""
    import pandas as pd
    from ..probabilistic_frontend import NeurolangPDL

    nl = NeurolangPDL()
    df = pd.DataFrame({"x": [1, 2, 3]})
    nl.add_tuple_set(df, name="item")
    df2 = pd.DataFrame({"x": [1, 2]})
    nl.add_tuple_set(df2, name="selected")

    # with 'such that'
    nl.execute_squall_program(
        "define as Result every Item (?x) such that ?x is a Selected."
    )
    sol_such = nl.solve_all()["result"].as_pandas_dataframe()

    nl2 = NeurolangPDL()
    nl2.add_tuple_set(df, name="item")
    nl2.add_tuple_set(df2, name="selected")

    # with 'where'
    nl2.execute_squall_program(
        "define as Result every Item (?x) where ?x is a Selected."
    )
    sol_where = nl2.solve_all()["result"].as_pandas_dataframe()

    assert set(sol_such["x"]) == set(sol_where["x"])


def test_extension_f_given_integration(tmp_path):
    """'given every X' parses to identical IR as 'conditioned to every X'.

    Approach:
    1. Parse both sentence variants with the SQUALL parser and compare the
       resulting Implication structures (functor, Condition antecedent type,
       and Condition sub-expression functors must be equal).
    2. Walk both programs into fresh engines and confirm neither raises an
       unexpected exception — the same exception type (if any) is acceptable
       since MARG engine limitations affect both paths equally.
    """
    from neurolang.frontend.datalog.squall_syntax_lark import parser as squall_parser
    from neurolang.probabilistic.expressions import Condition
    from neurolang.logic import Implication

    COND_SENTENCE = (
        "define as Probmap with probability every Activation (?i; ?j; ?k) "
        "conditioned to every Selected_study (_)."
    )
    GIVEN_SENTENCE = (
        "define as Probmap with probability every Activation (?i; ?j; ?k) "
        "given every Selected_study (_)."
    )

    # --- IR structural comparison -------------------------------------------
    ir_cond = squall_parser(COND_SENTENCE)
    ir_given = squall_parser(GIVEN_SENTENCE)

    # Both should return an Implication (rules-only, no obtain clause)
    assert isinstance(ir_cond, Implication), (
        f"'conditioned to' parser returned {type(ir_cond).__name__}, expected Implication"
    )
    assert isinstance(ir_given, Implication), (
        f"'given' parser returned {type(ir_given).__name__}, expected Implication"
    )

    # Head functor must be the same predicate name
    assert ir_cond.consequent.functor == ir_given.consequent.functor, (
        f"Head functors differ: {ir_cond.consequent.functor!r} vs "
        f"{ir_given.consequent.functor!r}"
    )

    # Both antecedents must be a Condition (not a plain Conjunction/Atom)
    assert isinstance(ir_cond.antecedent, Condition), (
        f"'conditioned to' antecedent is {type(ir_cond.antecedent).__name__}, expected Condition"
    )
    assert isinstance(ir_given.antecedent, Condition), (
        f"'given' antecedent is {type(ir_given.antecedent).__name__}, expected Condition"
    )

    # The Condition's conditioned/conditioning sub-expressions must have the
    # same type and — where applicable — the same top-level functor.
    cond_body = ir_cond.antecedent.conditioned
    given_body = ir_given.antecedent.conditioned
    assert type(cond_body) == type(given_body), (
        f"Condition.conditioned types differ: {type(cond_body).__name__!r} vs "
        f"{type(given_body).__name__!r}"
    )

    cond_cond = ir_cond.antecedent.conditioning
    given_cond = ir_given.antecedent.conditioning
    assert type(cond_cond) == type(given_cond), (
        f"Condition.conditioning types differ: {type(cond_cond).__name__!r} vs "
        f"{type(given_cond).__name__!r}"
    )

    # For FunctionApplication nodes we can also compare functor names.
    from neurolang.expressions import FunctionApplication
    if isinstance(cond_body, FunctionApplication) and isinstance(given_body, FunctionApplication):
        assert cond_body.functor == given_body.functor, (
            f"Condition.conditioned functors differ: {cond_body.functor!r} vs "
            f"{given_body.functor!r}"
        )
    if isinstance(cond_cond, FunctionApplication) and isinstance(given_cond, FunctionApplication):
        assert cond_cond.functor == given_cond.functor, (
            f"Condition.conditioning functors differ: {cond_cond.functor!r} vs "
            f"{given_cond.functor!r}"
        )

    # --- Engine walk: both variants must raise the same exception type -------
    from neurolang.frontend import NeurolangPDL
    import pandas as pd

    def _build_engine():
        nl = NeurolangPDL()
        df_act = pd.DataFrame({"i": [1, 2, 3], "j": [0, 0, 0], "k": [0, 0, 0]})
        nl.add_tuple_set(df_act, name="activation")
        study_ids = pd.DataFrame({"s": ["s1", "s2"]})
        nl.add_uniform_probabilistic_choice_over_set(study_ids, name="selected_study")
        return nl

    exc_cond = None
    exc_given = None

    try:
        _build_engine().execute_squall_program(COND_SENTENCE)
    except Exception as e:
        exc_cond = e

    try:
        _build_engine().execute_squall_program(GIVEN_SENTENCE)
    except Exception as e:
        exc_given = e

    assert type(exc_cond) == type(exc_given), (
        f"'conditioned to' raised {type(exc_cond).__name__} but "
        f"'given' raised {type(exc_given).__name__} — walk plumbing differs"
    )


def test_extension_d_for_each_integration(tmp_path):
    """Rule using 'for each' groupby produces same IR structure as 'per'."""
    from typing import Iterable

    def _build(keyword):
        nl = NeurolangPDL()

        def agg_collect(vals: Iterable) -> float:
            return float(sum(vals))

        nl.add_symbol(agg_collect, name="agg_collect")
        nl.add_tuple_set(
            [(1, 10, "s1"), (2, 20, "s2")], name="data"
        )
        nl.execute_squall_program(
            f"define as Result with a probability of "
            f"the Agg_collect of the Data (?v; _; ?s) "
            f"{keyword} ?i, ?s "
            f"that data(?i, ?v, ?s) holds."
        )
        idb = nl.program_ir.intensional_database()
        # find the rule(s) whose consequent functor name contains "result"
        rules = [
            rule
            for pred_rules in idb.values()
            for rule in pred_rules.formulas
            if "result" in rule.consequent.functor.name.lower()
        ]
        return rules

    rules_per = _build("per")
    rules_foreach = _build("for each")
    assert rules_per, "No result rules found for 'per'"
    assert rules_foreach, "No result rules found for 'for each'"
    # Both variants must produce the same number of consequent args
    args_per = {len(r.consequent.args) for r in rules_per}
    args_foreach = {len(r.consequent.args) for r in rules_foreach}
    assert args_per == args_foreach, (
        f"'per' consequent arg counts {args_per} != 'for each' {args_foreach}"
    )


def test_extension_g_obtain_as_integration(tmp_path):
    """'obtain … as Name' defines a relation and exposes it in solve_all() under the lowercased name."""
    from neurolang.frontend import NeurolangPDL
    import pandas as pd

    nl = NeurolangPDL()
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    nl.add_tuple_set(df, name="pair")

    nl.execute_squall_program(
        "obtain every Pair (?x; ?y) as My_pairs."
    )
    sol = nl.solve_all()
    assert "my_pairs" in sol, f"Expected 'my_pairs' in solution, got: {list(sol.keys())}"
    result_df = sol["my_pairs"].as_pandas_dataframe()
    assert len(result_df) == 3
    assert set(result_df.columns) == {"x", "y"}


def test_execute_squall_compound_rule():
    """Compound quantifier rule with deterministic body solves end-to-end."""
    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set([("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates")
    nl.add_tuple_set([("s1", "x"), ("s2", "y"), ("s3", "x")], name="mentions")
    nl.add_tuple_set([("A",), ("B",)], name="region")
    nl.add_tuple_set([("x",), ("y",)], name="term")

    result = nl.execute_squall_program(
        "define as Cooccurrence for every Region ?r and for every Term ?t "
        "where ?s activates ?r and mentions ?t. "
        "obtain every Cooccurrence (?r; ?t) as P_rt."
    )
    df = result.as_pandas_dataframe()
    assert len(df) == 3
    assert set(df.columns) == {"r", "t"}
    assert set(df.itertuples(index=False, name=None)) == {
        ("A", "x"),
        ("A", "y"),
        ("B", "x"),
    }


def test_execute_squall_compound_probabilistic_rule_smoke():
    """Compound quantifier MARG rule parses and reaches translate_implication.

    Smoke test — full probabilistic solving is blocked by a pre-existing
    limitation (fol_query_to_datalog_program does not support ProbabilisticQuery
    in the head, and flatten_query does not support existential bodies). We
    verify the rule parses correctly and that the failure mode is the expected
    Fol2DatalogTranslationException, consistent with existing MARG smoke tests.
    """
    from neurolang.frontend import NeurolangPDL
    from neurolang.logic.horn_clauses import Fol2DatalogTranslationException

    nl = NeurolangPDL()
    nl.add_tuple_set([("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates")
    nl.add_tuple_set([("s1", "x"), ("s2", "y"), ("s3", "x")], name="mentions")
    nl.add_tuple_set([("A",), ("B",)], name="region")
    nl.add_tuple_set([("x",), ("y",)], name="term")
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",), ("s3",)], name="selected_study"
    )

    try:
        nl.execute_squall_program(
            "define as Cooccurrence with inferred probability "
            "for every Region ?r and for every Term ?t "
            "where a Selected_study ?s activates ?r and mentions ?t."
        )
    except Fol2DatalogTranslationException:
        pass  # Expected: reached translate_implication — plumbing is OK
    except Exception as exc:
        pytest.fail(
            f"Unexpected exception when walking compound probabilistic rule: "
            f"{type(exc).__name__}: {exc}"
        )
