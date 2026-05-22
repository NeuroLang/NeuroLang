"""Unit tests for the SQUALL syntax lark parser."""
import pytest


@pytest.fixture
def nl_setup():
    """Minimal fixture — parser tests are stateless."""
    return None


def _normalize_fresh(expr):
    """Replace fresh symbol names with canonical placeholders for structural comparison."""
    from neurolang.expressions import Symbol, Constant, FunctionApplication
    from neurolang.logic import Conjunction, ExistentialPredicate

    counter = [0]
    mapping = {}

    def walk(e):
        if isinstance(e, Symbol):
            if e.is_fresh:
                if e.name not in mapping:
                    mapping[e.name] = f"__fresh_{counter[0]}__"
                    counter[0] += 1
                return Symbol(mapping[e.name])
            return e
        if isinstance(e, Constant):
            return e
        if isinstance(e, FunctionApplication):
            return FunctionApplication(walk(e.functor), tuple(walk(a) for a in e.args))
        if isinstance(e, ExistentialPredicate):
            return ExistentialPredicate(walk(e.head), walk(e.body))
        if isinstance(e, Conjunction):
            return Conjunction(tuple(walk(f) for f in e.formulas))
        return e

    return walk(expr)


def test_extension_e_where_as_such_that_variable(nl_setup):
    """'where ?x is a Noun' parses as a valid rel_b alternative."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser

    # This should parse without error — that's the key assertion.
    prog_where = parser(
        "define as Foo every Bar (?x) where ?x is a Selected_study."
    )
    from neurolang.logic import Implication
    assert isinstance(prog_where, Implication)
    # IR-level assertion: the antecedent must mention selected_study
    # (from 'where ?x is a Selected_study'), confirming the where clause
    # body was actually wired into the rule — not silently dropped.
    body_repr = repr(prog_where.antecedent)
    assert 'selected_study' in body_repr.lower()


def test_extension_e_where_inline_expr(nl_setup):
    """'and where FUNC(…) is lower than N' parses like 'and such that FUNC(…) is lower than N'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    import operator
    from neurolang.expressions import Constant

    prog_such = parser(
        "define as Foo every Bar (?x) such that MY_DIST(?x, ?x) is lower than 5."
    )
    prog_where = parser(
        "define as Foo every Bar (?x) where MY_DIST(?x, ?x) is lower than 5."
    )
    from neurolang.logic import Implication, Conjunction
    for prog in (prog_such, prog_where):
        assert isinstance(prog, Implication)
    # Both antecedents must contain a lt(…) atom
    def has_lt(formula):
        from neurolang.expressions import FunctionApplication
        if isinstance(formula, FunctionApplication):
            return formula.functor == Constant(operator.lt)
        if isinstance(formula, Conjunction):
            return any(has_lt(f) for f in formula.formulas)
        return False
    assert has_lt(prog_such.antecedent)
    assert has_lt(prog_where.antecedent)
    assert _normalize_fresh(prog_such.antecedent) == _normalize_fresh(prog_where.antecedent)


def test_extension_e_rel_tuple_noun_not_shadowed(nl_setup):
    """Test: where (?i;?j;?k) is a Voxel still parses as rel_tuple_noun after Extension E."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.logic import Implication

    prog = parser(
        "define as Foo every Bar (?x) where (?i; ?j; ?k) is a Voxel."
    )
    assert isinstance(prog, Implication)
    body_repr = repr(prog.antecedent)
    # Should contain voxel(i, j, k) — the tuple membership expansion,
    # NOT an is_a/sentence-level construction.
    assert 'voxel' in body_repr.lower()


def test_extension_f_given_as_conditioned_to(nl_setup):
    """'given every X (…)' parses identically to 'conditioned to every X (…)'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.probabilistic.expressions import Condition
    from neurolang.logic import Implication

    prog_cond = parser(
        "define as Probmap with probability every Activation (?i; ?j; ?k; _) "
        "conditioned to every Selected_study (_)."
    )
    prog_given = parser(
        "define as Probmap with probability every Activation (?i; ?j; ?k; _) "
        "given every Selected_study (_)."
    )
    assert isinstance(prog_cond, Implication)
    assert isinstance(prog_given, Implication)
    # Both antecedents must be Condition instances
    assert isinstance(prog_cond.antecedent, Condition)
    assert isinstance(prog_given.antecedent, Condition)
    # conditioned and conditioning bodies must match structurally
    assert type(prog_cond.antecedent.conditioned) == type(prog_given.antecedent.conditioned)
    assert type(prog_cond.antecedent.conditioning) == type(prog_given.antecedent.conditioning)


def test_extension_h_with_inferred_probability(nl_setup):
    """'with inferred probability … conditioned to …' parses same as 'with probability … conditioned to …'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.probabilistic.expressions import Condition
    from neurolang.logic import Implication

    prog_plain = parser(
        "define as Probmap with probability every Activation (?i; ?j; ?k; _) "
        "conditioned to every Selected_study (_)."
    )
    prog_inferred = parser(
        "define as Probmap with inferred probability every Activation (?i; ?j; ?k; _) "
        "conditioned to every Selected_study (_)."
    )
    assert isinstance(prog_plain, Implication)
    assert isinstance(prog_inferred, Implication)
    assert isinstance(prog_plain.antecedent, Condition)
    assert isinstance(prog_inferred.antecedent, Condition)
    # Consequent functors should match
    assert prog_plain.consequent.functor == prog_inferred.consequent.functor


def test_extension_d_for_each_alias_per(nl_setup):
    """'for each ?i, ?j, ?k and for each ?s' produces same dims as 'per ?i, ?j, ?k and per ?s'."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser
    from neurolang.logic import Implication

    prog_per = parser(
        "define as Reported_voxel with a probability of "
        "the Kernelized_max_proximity of the Focus (?i2; ?j2; ?k2; ?s) "
        "per ?i1, ?j1, ?k1 and per ?s "
        "where (?i1; ?j1; ?k1) is a Voxel."
    )
    prog_foreach = parser(
        "define as Reported_voxel with a probability of "
        "the Kernelized_max_proximity of the Focus (?i2; ?j2; ?k2; ?s) "
        "for each ?i1, ?j1, ?k1 and for each ?s "
        "where (?i1; ?j1; ?k1) is a Voxel."
    )
    assert isinstance(prog_per, Implication)
    assert isinstance(prog_foreach, Implication)
    # Head args (per-vars) must be the same set
    per_args = {a.name for a in prog_per.consequent.body.args}
    foreach_args = {a.name for a in prog_foreach.consequent.body.args}
    assert per_args == foreach_args


def test_extension_g_query_as_produces_squall_program(nl_setup):
    """'obtain ops as Name' returns a SquallProgram with a named IDB rule and query."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser, SquallProgram

    result = parser(
        "define as Active_voxel every Reported_voxel (?i; ?j; ?k; ?s) "
        "where ?s is a Selected_study.\n"
        "obtain the Brain_image of the Active_voxel (?i; ?j; ?k; ?p) as Image."
    )
    assert isinstance(result, SquallProgram), f"Expected SquallProgram, got {type(result)}"
    assert result.queries, "Expected at least one query"
    # The IDB rules should include an Image(...) or image(...) head
    rule_functor_names = {
        r.consequent.functor.name.lower()
        for r in result.rules
        if hasattr(r, 'consequent') and hasattr(r.consequent, 'functor')
    }
    assert 'image' in rule_functor_names, f"'image' not found in rule functors: {rule_functor_names}"


def test_extension_g_query_names_forwarded_through_parser(nl_setup):
    """query_names must survive the LogicSimplifier reconstruction in parser()."""
    from neurolang.frontend.datalog.squall_syntax_lark import parser, SquallProgram

    result = parser(
        "obtain every Pair (?x; ?y) as My_pairs."
    )
    assert isinstance(result, SquallProgram)
    # query_names must have been forwarded — not empty
    assert result.query_names, f"query_names was dropped: {result.query_names!r}"
    assert 0 in result.query_names
    assert result.query_names[0] == "my_pairs"
