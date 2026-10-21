"""Lifted queries run through the frontend: the solver's tests are in
``neurolang/probabilistic/tests/test_dalvi_suciu_lift.py``."""

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from nl_core.relational_algebra_provenance import DisjointProjection


@pytest.mark.parametrize("rules, expected", [
    # Disjuncts sharing studies, and disjoint: P[A and B] is 0, left out
    # of the plan of the conjunction.
    ((("language",), ("speech",)), 0.5),
    ((("language",), ("pain",)), 0.75),
    # Three disjuncts: the tail is the disjunction of the two others.
    ((("language",), ("speech",), ("pain",)), 0.75),
    ((("pain",), ("speech",), ("memory",)), 0.75),
    # A conjunctive disjunct.
    ((("language", "speech"), ("pain",)), 0.5),
])
def test_union_of_rules_joining_a_choice(rules, expected):
    """
    A union of rules each joining the probabilistic choice, under PROB,
    crashed: it is lifted by the disjoint separator, one disjoint
    projection on the study.
    """
    from .. import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [(1,), (2,), (3,), (4,)], name="selected_study"
    )
    nl.add_tuple_set(
        [(1, "language"), (2, "speech"), (2, "language"), (3, "pain"),
         (4, "memory")],
        name="mentions",
    )
    program = "\n".join(
        "a(s) :- " + ", ".join(f"mentions(s, '{t}')" for t in terms)
        + ", selected_study(s)"
        for terms in rules
    ) + "\nans(p) :- PROB[a(s)] = p"
    result = nl.execute_datalog_program(program)
    assert [p for p, in result] == [pytest.approx(expected)]


LIFTED_ONLY = dict(
    probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
    probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
)


@pytest.mark.parametrize("solvers", [{}, LIFTED_ONLY])
def test_union_by_term_with_a_variable_of_a_rule_alone(solvers):
    """
    A union whose rules have a variable that their head has not
    (``s2``): the deterministic ``mentions(s2, t)`` is factored out of the
    conjunctive normal form, whose clauses then do not all have ``t``,
    and whose components are not lifted; the disjoint projection lifts
    the union as it is.
    """
    from .. import NeurolangPDL

    mentions = [
        (0, "a"), (0, "c"), (1, "a"), (1, "d"), (1, "e"), (2, "a"),
        (2, "c"), (2, "e"), (3, "a"), (3, "c"), (3, "d"), (4, "b"),
        (5, "b"), (5, "c"), (5, "d"),
    ]
    nl = NeurolangPDL(**solvers)
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(6)], name="selected_study"
    )
    nl.add_tuple_set(mentions, name="mentions")
    result = nl.execute_datalog_program(
        "a(t, s) :- mentions(s, t), selected_study(s)\n"
        "a(t, s) :- mentions(s, 'a'), mentions(s, 'c'), mentions(s2, t), "
        "selected_study(s)\n"
        "a(t, s) :- mentions(s, 'd'), mentions(s2, t), selected_study(s)\n"
        "ans(t, p) :- PROB[a(t, s)] = p"
    )
    m = set(mentions)
    expected = {
        t: sum(
            (s, t) in m or ((s, "a") in m and (s, "c") in m)
            or (s, "d") in m
            for s in range(6)
        ) / 6
        for t in "abcde"
    }
    assert dict(result) == pytest.approx(expected)


@pytest.mark.parametrize("solvers", [{}, LIFTED_ONLY])
def test_union_with_a_choice_on_a_free_and_on_a_bound_variable(solvers):
    """
    The choice is on the free variable s in one rule and on a variable of
    its own, f, in the other: the choice optimisation unified C(s) and
    C(f) across the disjuncts, as if they were a conjunction, and the
    first solver gave no answer; the disjuncts are now optimised apart,
    and C(s) is C(f') ∧ f' = s for the disjoint projection.
    """
    from .. import NeurolangPDL

    nl = NeurolangPDL(**solvers)
    nl.add_uniform_probabilistic_choice_over_set(
        [(1,), (2,), (3,), (4,)], name="selected_study"
    )
    nl.add_tuple_set([(1,), (2,), (3,), (4,)], name="study")
    nl.add_tuple_set(
        [(1, "x"), (2, "y"), (3, "x"), (4, "z")], name="mentions"
    )
    result = nl.execute_datalog_program(
        "a(s) :- selected_study(s), mentions(s, 'x')\n"
        "a(s) :- study(s), selected_study(f), mentions(f, 'y')\n"
        "ans(s, p) :- PROB[a(s)] = p"
    )
    assert dict(result) == pytest.approx({1: 0.5, 2: 0.25, 3: 0.5, 4: 0.25})


def _choice_union_engine():
    from .. import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [(1,), (2,), (3,), (4,)], name="selected_study"
    )
    nl.add_tuple_set(
        [(1, "language"), (2, "speech"), (2, "language"), (3, "pain"),
         (4, "memory")],
        name="mentions",
    )
    return nl


def _plan_of(program):
    """The plan of the lifted query of *program*, whose solving is
    stopped once the plan is made."""
    plans = []
    original = dalvi_suciu_lift.is_pure_lifted_plan

    def record(plan):
        plans.append(plan)
        return original(plan)

    dalvi_suciu_lift.is_pure_lifted_plan = record
    try:
        _choice_union_engine().execute_datalog_program(program)
    finally:
        dalvi_suciu_lift.is_pure_lifted_plan = original
    return plans[-1]


def test_union_of_a_self_join_and_a_choice_unrelated_to_the_head():
    """
    The separator declines a union with a rule whose choice is on a
    variable unrelated to the head (the deterministic atom binding the
    head is factored out of the union, leaving a disjunct without it),
    and weighted model counting solves it. The equality that the
    rewriting of the choices leaves between the two choice atoms of the
    self-join was translated to a projection overwriting the column of
    an atom, which the provenance sets reject.
    """
    from .. import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(0, "b"), (0, "c"), (1, "a"), (1, "d")], name="mentions"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(0,), (1,)], name="selected_study"
    )
    result = nl.execute_datalog_program(
        "a(t) :- mentions(s, t), selected_study(s).\n"
        "a(t) :- mentions(s, t), selected_study(s), selected_study(f), "
        "mentions(f, 'e').\n"
        "a(t) :- mentions(s, t), selected_study(f), mentions(f, 'e').\n"
        "ans(t, p) :- PROB[a(t)] = p."
    )
    assert dict(result) == pytest.approx(
        {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    )


def test_conjunctive_query_with_a_choice_and_a_negation():
    from .. import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(0, "a"), (0, "b"), (1, "b"), (1, "c"), (2, "a")], name="mentions"
    )
    nl.add_tuple_set([("a",)], name="lang")
    nl.add_uniform_probabilistic_choice_over_set(
        [(0,), (1,), (2,)], name="selected_study"
    )
    result = nl.execute_datalog_program(
        "a(t) :- selected_study(s), mentions(s, t), ~lang(t).\n"
        "ans(t, p) :- PROB[a(t)] = p."
    )
    assert dict(result) == pytest.approx({"b": 2 / 3, "c": 1 / 3})


def test_union_with_a_choice_and_a_negation():
    """
    A union whose rule negates an atom of the study chosen: the negation
    removes the tuples of the studies that have it, which a support that
    kept the probability of the choice in it (a difference of
    probabilities, not of sets) did not.
    """
    from .. import NeurolangPDL

    mentions = {
        0: "deab", 1: "da", 2: "cab", 3: "e", 4: "dcab"
    }
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(s, t) for s, terms in mentions.items() for t in terms],
        name="mentions",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in mentions], name="selected_study"
    )
    result = nl.execute_datalog_program(
        "a(t) :- mentions(s, t), selected_study(s), ~mentions(s, 'a').\n"
        "a(t) :- mentions(s, t), selected_study(s), mentions(s, 'c').\n"
        "ans(t, p) :- PROB[a(t)] = p."
    )
    assert dict(result) == pytest.approx(
        {"a": 0.4, "b": 0.4, "c": 0.4, "d": 0.2, "e": 0.2}
    )


def test_union_with_a_rule_local_variable_and_a_negation():
    """
    A rule with a variable its head has not (``s`` of ``mentions(s, t)``,
    the choice being on ``f``) in a union with a negation. The variables a
    disjunct alone has are projected in the disjunct, so that the branches
    of the union of the deterministic part have the same columns, and the
    projection is a support: it would sum the labels of the tuples it
    merges, and a label above 1 is not one of a set (the union of the
    labels is ``1 - prod(1 - l)``).
    """
    from .. import NeurolangPDL

    mentions = {0: "ab", 1: "bc", 2: "cd"}
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(s, t) for s, terms in mentions.items() for t in terms],
        name="mentions",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in mentions], name="selected_study"
    )
    result = nl.execute_datalog_program(
        "a(t) :- mentions(s, t), selected_study(s), ~mentions(s, 'a').\n"
        "a(t) :- mentions(s, t), selected_study(f), mentions(f, 'c').\n"
        "ans(t, p) :- PROB[a(t)] = p."
    )
    assert dict(result) == pytest.approx(
        {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3, "d": 2 / 3}
    )


def test_union_whose_normal_form_has_an_unsafe_clause():
    """
    The union of a rule with a negation and a rule with a choice unrelated
    to its head: factoring out the atom they share gives the clause
    ``exists s M(s, t) or not L(t)``, which is not safe range and was lifted
    apart (``ValueError``). The conjunctive normal form is not used then.
    """
    from .. import NeurolangPDL

    mentions = {0: "ab", 1: "bc", 2: "cd"}
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [(s, t) for s, terms in mentions.items() for t in terms],
        name="mentions",
    )
    nl.add_tuple_set([("a",)], name="lang")
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in mentions], name="selected_study"
    )
    result = nl.execute_datalog_program(
        "a(t) :- mentions(s, t), selected_study(s), ~lang(t).\n"
        "a(t) :- mentions(s, t), selected_study(f), mentions(f, 'c').\n"
        "ans(t, p) :- PROB[a(t)] = p."
    )
    assert dict(result) == pytest.approx(
        {"a": 2 / 3, "b": 1.0, "c": 2 / 3, "d": 2 / 3}
    )


def test_union_with_a_self_join_of_the_choice_in_a_rule():
    """
    A rule of a union joins the choice twice: the equality that makes the
    two atoms the same tuple holds in that rule only, which the unification
    of the variables took for the whole body, turning the union into a
    conjunction.
    """
    from .. import NeurolangPDL

    mentions = [
        (0, "b"), (0, "c"), (1, "a"), (1, "d"), (2, "a"), (2, "c"), (3, "a")
    ]
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in range(4)], name="selected_study"
    )
    nl.add_tuple_set(mentions, name="mentions")
    result = nl.execute_datalog_program(
        "a(t) :- selected_study(s), selected_study(f), mentions(s, t), "
        "mentions(f, 'b')\n"
        "a(t) :- selected_study(s), mentions(s, t), mentions(s, 'c')\n"
        "ans(t, p) :- PROB[a(t)] = p"
    )
    assert dict(result) == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})


def test_disjoint_separator_plans_a_union_in_one_projection():
    """The union over the choice is one disjoint projection on the study,
    of the choice joined with the support of the deterministic union."""
    plan = _plan_of(
        "a(s) :- mentions(s, 'language'), selected_study(s)\n"
        "a(s) :- mentions(s, 'speech'), selected_study(s)\n"
        "a(s) :- mentions(s, 'pain'), selected_study(s)\n"
        "ans(p) :- PROB[a(s)] = p"
    )
    assert isinstance(plan, DisjointProjection)
    assert "Σ" not in repr(plan)
