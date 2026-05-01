"""
Unit tests for the SQUALL syntax lark parser.
"""
import pytest


@pytest.fixture
def nl_setup():
    """Minimal fixture — parser tests are stateless."""
    return None


def _normalize_fresh(expr):
    """Replace fresh symbol names with canonical placeholders for structural comparison."""
    import re
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
    """where (?i;?j;?k) is a Voxel still parses as rel_tuple_noun after Extension E."""
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
