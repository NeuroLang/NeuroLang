"""
``EXPECT[f | a] = e``, the expected value of ``f`` summed over the answers of
the probabilistic query ``a``:

    E[f(g, x) | a(g, x)] = Σ_x f(g, x) P[a(g, x)]

and ``EXPECT_NORM``, that divided by Σ_x P[a(g, x)]. The expected values are
computed by enumerating the possible worlds, over engines that have only the
lifted solvers: the program the sugar is rewritten to is solved by them.
"""
import itertools
from collections import defaultdict

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from nl_expr.exceptions import NeuroLangException
from ...frontend import NeurolangPDL

# probabilistic choices: name -> {value: probability}
CHOICES = {
    "roll": {1: 0.2, 2: 0.3, 4: 0.5},
    "sel": {"s1": 0.5, "s2": 0.3, "s3": 0.2},
}
# independent facts: name -> {tuple: probability}
FACTS = {
    "item": {
        ("a", 1): 0.5, ("a", 2): 0.5, ("a", 3): 0.8,
        ("b", 1): 0.4, ("b", 4): 0.9,
    },
    "flag": {(1,): 0.6, (2,): 0.3, (4,): 0.5},
    "act": {
        ("s1", "r1"): 0.7, ("s2", "r1"): 0.4, ("s1", "r2"): 0.2,
        ("s3", "r2"): 0.9,
    },
}
# deterministic relations
LINK = [(1, "u"), (2, "u"), (2, "v"), (3, "v"), (4, "v")]
SIZE = [("s1", 10.0), ("s2", 20.0), ("s3", 40.0)]


def _engine():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    for name, values in CHOICES.items():
        nl.add_probabilistic_choice_from_tuples(
            [(p, v) for v, p in values.items()], name=name
        )
    for name, tuples in FACTS.items():
        nl.add_probabilistic_facts_from_tuples(
            [(p,) + t for t, p in tuples.items()], name=name
        )
    nl.add_tuple_set(LINK, name="link")
    nl.add_tuple_set(SIZE, name="size")
    return nl


def _rows(program, name="q"):
    nl = _engine()
    nl.execute_datalog_program(program)
    return _engine_rows(nl, name)


def _engine_rows(nl, name="q"):
    df = nl.solve_all()[name].as_pandas_dataframe()
    return [tuple(row) for row in df.itertuples(index=False)]


def _solve(program, name="q"):
    """The expectation of each group of the answers of `name`."""
    return {row[:-1]: row[-1] for row in _rows(program, name)}


def _worlds():
    """(probability, chosen, world) of each possible world: the value of
    each choice and the set of the facts present."""
    names = list(CHOICES)
    facts = [(n, t) for n, ts in FACTS.items() for t in ts]
    for values in itertools.product(*(CHOICES[n].items() for n in names)):
        p_choices = 1.
        for _, p in values:
            p_choices *= p
        chosen = {n: v for n, (v, _) in zip(names, values)}
        for present in itertools.product((False, True), repeat=len(facts)):
            p = p_choices
            for (n, t), is_present in zip(facts, present):
                q = FACTS[n][t]
                p *= q if is_present else 1 - q
            world = {f for f, is_present in zip(facts, present) if is_present}
            yield p, chosen, world


def _expectation(answers, f, knowing=None, normalised=False):
    """Σ_w P(w) Σ_{(group, x) in answers(w)} f(group, x) by group;
    conditioned on the worlds where `knowing` holds; the normalised divides
    by the Σ of the probabilities of the answers."""
    total, weight, p_known = defaultdict(float), defaultdict(float), 0.
    for p, chosen, world in _worlds():
        if knowing is not None and not knowing(chosen, world):
            continue
        p_known += p
        for group, x in answers(chosen, world):
            group = group if isinstance(group, tuple) else (group,)
            total[group] += p * f(group, x)
            weight[group] += p
    return {
        g: total[g] / (weight[g] if normalised else p_known) for g in total
    }


def _assert_close(found, expected):
    assert found.keys() == expected.keys()
    for k, v in expected.items():
        assert found[k] == pytest.approx(v)


def _items(chosen, world):
    return [t for (n, t) in world if n == "item"]


def _selected_act(chosen, world):
    """(r, s): the selected study s is active in r."""
    s = chosen["sel"]
    return [(r, s) for r in ("r1", "r2") if ("act", (s, r)) in world]


def test_a_choice_is_the_expected_value_of_the_chosen_one():
    found = _solve("q(e) :- EXPECT[x | roll(x)] = e")
    _assert_close(
        found,
        _expectation(lambda c, w: [((), c["roll"])], lambda g, x: x),
    )
    assert found[()] == pytest.approx(0.2 + 0.6 + 2.0)


def test_an_event_on_a_choice_is_not_normalised():
    found = _solve("q(e) :- EXPECT[x | roll(x), (x > 1)] = e")
    assert found[()] == pytest.approx(0.6 + 2.0)
    _assert_close(
        found,
        _expectation(
            lambda c, w: [((), c["roll"])] if c["roll"] > 1 else [],
            lambda g, x: x,
        ),
    )


def test_the_normalised_expectation_is_the_value_knowing_the_event():
    found = _solve("q(e) :- EXPECT_NORM[x | roll(x), (x > 1)] = e")
    assert found[()] == pytest.approx((0.6 + 2.0) / 0.8)


def test_normalised_counts_answers_with_the_same_value_and_weight():
    """x = 1 and x = 2 have the same f and the same p: both are summed in
    the numerator and in the denominator."""
    nl = _engine()
    nl.add_tuple_set([(1, 7.0), (2, 7.0), (3, 2.0)], name="weight")
    nl.execute_datalog_program(
        "q(e) :- EXPECT_NORM[w | item('a', x), weight(x, w)] = e"
    )
    [(found,)] = _engine_rows(nl)
    assert found == pytest.approx((.5 * 7 + .5 * 7 + .8 * 2) / (.5 + .5 + .8))


def test_independent_tuples_give_the_expected_sum_over_the_random_set():
    found = _solve("q(e) :- EXPECT[x * x | exists(g; item(g, x))] = e")
    # the probability of each x is that of ∃g item(g, x)
    expected = _expectation(
        lambda c, w: [((), x) for x in {x for _, x in _items(c, w)}],
        lambda g, x: x * x,
    )
    _assert_close(found, expected)


def test_tuples_with_the_same_weight_are_all_counted():
    """Two tuples of probability .5 and value 1: the expected number of
    answers is 1, not 0.5 (a set of the values f * p)."""
    found = _solve("q(e) :- EXPECT[1 | item('a', x), (x < 3)] = e")
    assert found[()] == pytest.approx(1.0)
    _assert_close(
        found,
        _expectation(
            lambda c, w: [((), x) for g, x in _items(c, w)
                          if g == "a" and x < 3],
            lambda g, x: 1,
        ),
    )


def test_a_variable_of_the_event_not_in_f_is_summed():
    found = _solve("q(e) :- EXPECT[2 | item(g, x)] = e")
    _assert_close(
        found,
        _expectation(lambda c, w: [((), t) for t in _items(c, w)],
                     lambda g, x: 2),
    )
    assert found[()] == pytest.approx(2 * sum(FACTS["item"].values()))


def test_grouping_gives_one_expectation_per_group():
    found = _solve("q(g, e) :- EXPECT[x * 3 | item(g, x)] = e")
    expected = _expectation(_items, lambda g, x: x * 3)
    assert set(g for g, in found) == {"a", "b"}
    _assert_close(found, expected)


def test_the_group_can_be_bound_outside_the_event():
    nl = _engine()
    nl.add_tuple_set([("a",), ("b",)], name="groups")
    nl.execute_datalog_program(
        "q(g, e) :- groups(g), EXPECT[x | item(g, x)] = e"
    )
    found = {r[:-1]: r[-1] for r in _engine_rows(nl)}
    expected = _expectation(_items, lambda g, x: x)
    _assert_close(found, expected)


def test_f_is_a_function_of_the_group_and_x():
    nl = _engine()
    nl.add_tuple_set([("a", 2.0), ("b", 5.0)], name="scale")
    # the weight is joined inside the event, deterministic
    nl.execute_datalog_program(
        "q(g, e) :- EXPECT[x * w | item(g, x), scale(g, w)] = e"
    )
    found = {r[:-1]: r[-1] for r in _engine_rows(nl)}
    scale = {"a": 2.0, "b": 5.0}
    expected = _expectation(_items, lambda g, x: x * scale[g[0]])
    _assert_close(found, expected)


def test_a_join_inside_the_event():
    """x is chosen, linked to y, flagged: Σ over the pairs (x, y)."""
    found = _solve(
        "q(e) :- EXPECT[x | roll(x), link(x, y), flag(x)] = e"
    )

    def answers(c, w):
        x = c["roll"]
        return [((), x) for (v, y) in LINK if v == x
                and ("flag", (x,)) in w]

    _assert_close(found, _expectation(answers, lambda g, x: x))


def test_a_join_of_a_choice_and_independent_tuples_by_group():
    found = _solve(
        "q(r, e) :- EXPECT[w | sel(s), act(s, r), size(s, w)] = e"
    )
    size = dict(SIZE)
    _assert_close(
        found,
        _expectation(_selected_act, lambda g, s: size[s]),
    )


def test_normalised_by_group_is_the_weighted_mean():
    found = _solve(
        "q(r, e) :- EXPECT_NORM[w | sel(s), act(s, r), size(s, w)] = e"
    )
    size = dict(SIZE)
    _assert_close(
        found,
        _expectation(_selected_act, lambda g, s: size[s], normalised=True),
    )


def test_conditioning_is_that_of_prob():
    found = _solve(
        "q(r, e) :- EXPECT[w | sel(s), act(s, r), size(s, w) "
        "// (sel(t), act(t, 'r1'))] = e"
    )
    size = dict(SIZE)

    def knowing(c, w):
        return ("act", (c["sel"], "r1")) in w

    _assert_close(
        found,
        _expectation(_selected_act, lambda g, s: size[s], knowing=knowing),
    )


def test_an_expectation_is_a_value_like_any():
    # as the probability of PROB, the expectation joins the head
    [(ten_times, mean)] = _rows(
        "q(e) :- EXPECT[x | roll(x)] = m, (e == (m * 10))"
    )
    assert (ten_times, mean) == pytest.approx((28.0, 2.8))


def test_the_variance_of_the_normalised_expectation():
    [(variance, *_)] = _rows(
        "q(v) :- EXPECT_NORM[x * x | roll(x)] = m2, "
        "EXPECT_NORM[x | roll(x)] = m1, (v == (m2 - m1 * m1))"
    )
    mean = 0.2 + 0.6 + 2.0
    second = 0.2 + 1.2 + 8.0
    assert variance == pytest.approx(second - mean * mean)


def test_f_must_use_variables_of_the_event():
    with pytest.raises(NeuroLangException) as error:
        _engine().execute_datalog_program(
            "q(e) :- EXPECT[x + z | roll(x)] = e"
        )
    assert "z" in str(error.value)


def test_the_event_needs_a_variable():
    with pytest.raises(NeuroLangException, match='has none'):
        _engine().execute_datalog_program(
            "q(e) :- EXPECT[1 | roll(2)] = e"
        )


def test_f_cannot_call_a_relation():
    """A probabilistic relation is not a deterministic function of x."""
    nl = _engine()
    nl.execute_datalog_program("q(e) :- EXPECT[item(x, 1) | roll(x)] = e")
    with pytest.raises(NeuroLangException, match="relation"):
        nl.solve_all()


def test_expect_cannot_be_inside_an_aggregation():
    with pytest.raises(NeuroLangException, match="AGGREGATE"):
        _engine().execute_datalog_program(
            "q(n) :- AGGREGATE[](EXPECT[x | roll(x)] = m @ sum(m)) = n"
        )


# The schema of the tables of the probabilistic-query benchmark
TARGET = [("d1", "g1"), ("d1", "g2"), ("d2", "g2"), ("d3", "g3")]
APPROVED = [("d1",), ("d2",)]
EVIDENCE = {
    ("e1", "g1", "PD"): 0.5, ("e2", "g1", "PD"): 0.4, ("e1", "g2", "PD"): 0.3,
    ("e1", "g3", "PD"): 0.9, ("e1", "g1", "AD"): 0.7,
}
RELIABLE = {("e1",): 0.8, ("e2",): 0.6}


def _benchmark_engine():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_tuple_set(TARGET, name="drug_target")
    nl.add_tuple_set(APPROVED, name="drug_approved")
    nl.add_probabilistic_facts_from_tuples(
        [(p,) + t for t, p in EVIDENCE.items()], name="test_evidence"
    )
    nl.add_probabilistic_facts_from_tuples(
        [(p,) + t for t, p in RELIABLE.items()], name="dataset_reliable"
    )
    return nl


def _benchmark_worlds():
    facts = [("e", t) for t in EVIDENCE] + [("r", t) for t in RELIABLE]
    prob = {**{("e", t): p for t, p in EVIDENCE.items()},
            **{("r", t): p for t, p in RELIABLE.items()}}
    for present in itertools.product((False, True), repeat=len(facts)):
        p = 1.
        for f, is_present in zip(facts, present):
            p *= prob[f] if is_present else 1 - prob[f]
        yield p, {f for f, is_present in zip(facts, present) if is_present}


def _benchmark_value(program):
    nl = _benchmark_engine()
    with nl.scope:
        [row] = [tuple(r) for r in nl.execute_datalog_program(program)]
    return row[-1]


def test_expected_approved_drugs_with_evidence_for_a_disease():
    """P7a of the benchmark: the sum over a PROB rule, and EXPECT."""
    before = _benchmark_value(
        "p(u, PROB) :- drug_approved(u) & drug_target(u, g) & "
        "test_evidence(d, g, 'PD')\n"
        "ans(s) :- AGGREGATE[](p(u, q) @ sum(q)) = s"
    )
    after = _benchmark_value(
        "ans(s) :- EXPECT[1 | drug_approved(u), exists(g, d; "
        "drug_target(u, g), test_evidence(d, g, 'PD'))] = s"
    )
    expected = sum(
        p * len({
            u for (u,) in APPROVED for (v, g) in TARGET if v == u
            for (d, h, dis) in EVIDENCE
            if h == g and dis == "PD" and ("e", (d, h, dis)) in world
        })
        for p, world in _benchmark_worlds()
    )
    assert before == pytest.approx(expected)
    assert after == pytest.approx(expected)


def test_expected_genes_with_reliable_evidence():
    """P7b: the unit is a gene, the dataset is projected."""
    after = _benchmark_value(
        "ans(s) :- EXPECT[1 | drug_target('d1', u), exists(d; "
        "test_evidence(d, u, 'PD'), dataset_reliable(d))] = s"
    )
    expected = sum(
        p * len({
            g for (v, g) in TARGET if v == "d1"
            for (d, h, dis) in EVIDENCE
            if h == g and dis == "PD" and ("e", (d, h, dis)) in world
            and ("r", (d,)) in world
        })
        for p, world in _benchmark_worlds()
    )
    assert after == pytest.approx(expected)


def test_without_exists_the_sum_is_over_all_the_variables():
    """The triples (u, g, d), not the drugs."""
    triples = _benchmark_value(
        "ans(s) :- EXPECT[1 | drug_approved(u), drug_target(u, g), "
        "test_evidence(d, g, 'PD')] = s"
    )
    assert triples == pytest.approx(
        0.5 + 0.4 + 0.3  # (d1, g1, e1) (d1, g1, e2) (d1, g2, e1)
        + 0.3            # (d2, g2, e1)
    )


def test_a_group_that_is_only_in_what_is_known():
    """r is in the head and in `knowing` only: one expectation per r."""
    found = _solve(
        "q(r, e) :- EXPECT[w | sel(s), size(s, w) // (sel(t), act(t, r))] = e"
    )
    size = dict(SIZE)
    total, p_known = defaultdict(float), defaultdict(float)
    for p, chosen, world in _worlds():
        for r in ("r1", "r2"):
            if ("act", (chosen["sel"], r)) in world:
                p_known[r] += p
                total[r] += p * size[chosen["sel"]]
    _assert_close(found, {(r,): total[r] / p_known[r] for r in total})


def test_a_disjunctive_event():
    """x is rolled or flagged: the events are not exclusive."""
    found = _solve("q(e) :- EXPECT[x | roll(x) | flag(x)] = e")
    _assert_close(
        found,
        _expectation(
            lambda c, w: [
                ((), x) for x in sorted(
                    {c["roll"]} | {t[0] for n, t in w if n == "flag"}
                )
            ],
            lambda g, x: x,
        ),
    )


def test_a_group_bound_by_an_equality_is_a_group():
    """g is an argument of the comparison, so a group: the same as 'a'."""
    assert _solve(
        "q(e) :- EXPECT[x | item(g, x)] = e, (g == 'a')"
    ) == pytest.approx(_solve("q(e) :- EXPECT[x | item('a', x)] = e"))


def test_the_expectation_in_a_query():
    """A query body cannot hold an EXPECT: ans reads a rule that does."""
    [(value,)] = _engine().execute_datalog_program(
        "ans(e) :- EXPECT[x | roll(x)] = e"
    ).as_pandas_dataframe().values.tolist()
    assert value == pytest.approx(2.8)
    # and the expectation joins the head when it is not in it
    [(ten_times, mean)] = _engine().execute_datalog_program(
        "ans(v) :- EXPECT[x | roll(x)] = m, (v == (m * 10))"
    ).as_pandas_dataframe().values.tolist()
    assert (ten_times, mean) == pytest.approx((28.0, 2.8))


def test_the_expectations_of_named_queries():
    answers = _engine().execute_datalog_program(
        "ans[mean](e) :- EXPECT[x | roll(x)] = e\n"
        "ans[above](e) :- EXPECT[x | roll(x), (x > 1)] = e\n"
        "ans[above_norm](e) :- EXPECT_NORM[x | roll(x), (x > 1)] = e"
    )
    found = {
        name: answer.as_pandas_dataframe().values.item()
        for name, answer in answers.items()
    }
    assert found == pytest.approx(
        {"mean": 2.8, "above": 2.6, "above_norm": 3.25}
    )


def test_the_parsed_rule_is_a_rule_for_the_other_consumers():
    """The rewriting happens in the engine: the parser gives the rule with
    its EXPECT, whose atoms are those of its event."""
    from nl_core.datalog.expression_processing import extract_logic_atoms

    from ..datalog.standard_syntax import parser

    [rule] = parser("q(e) :- EXPECT[x | roll(x), flag(x)] = e").formulas
    assert {a.functor.name for a in extract_logic_atoms(rule.antecedent)} == {
        "roll", "flag"
    }


@pytest.mark.parametrize("head", ("q(g, e, p)", "ans(g, e)"))
def test_an_expectation_and_a_probability_in_one_rule(head):
    """The expectation and the probability both join the head."""
    program = (
        f"{head} :- EXPECT[x | item(g, x)] = e, PROB[flag(y)] = p"
    )
    nl = _engine()
    if head.startswith("q"):
        nl.execute_datalog_program(program)
        rows = _engine_rows(nl)
    else:
        rows = [
            tuple(r) for r in nl.execute_datalog_program(program)
            .as_pandas_dataframe().itertuples(index=False)
        ]
    p_flag = 1 - (1 - 0.6) * (1 - 0.3) * (1 - 0.5)
    expected = _expectation(_items, lambda g, x: x)
    assert [row[0] for row in sorted(rows)] == [g for (g,) in sorted(expected)]
    for (_, e, p), (group, mean) in zip(sorted(rows), sorted(expected.items())):
        assert (e, p) == pytest.approx((mean, p_flag))


@pytest.mark.parametrize("program", (
    "q(e) :- ~EXPECT[x | roll(x)] = e",
    "q(e) :- EXPECT[x | roll(x)] = e | roll(e)",
    "q(e) :- exists(y; EXPECT[x | roll(x)] = e)",
    "q(p) :- PROB[EXPECT[x | roll(x)] = e] = p",
    "q(e) :- EXPECT[x | EXPECT[y | roll(y)] = x] = e",
))
def test_an_expectation_is_a_conjunct_of_a_rule_body(program):
    """Under a negation, a disjunction, a quantifier or another PROB or
    EXPECT it is refused, not read as something else."""
    with pytest.raises(NeuroLangException):
        nl = _engine()
        nl.execute_datalog_program(program)
        nl.solve_all()


def test_the_printer_shows_the_expectation():
    from nl_core.datalog.pretty_printer import DatalogPrettyPrinter

    from ..datalog.standard_syntax import parser

    [rule] = parser(
        "q(e) :- EXPECT_NORM[x | roll(x) // flag(y)] = e"
    ).formulas
    printed = DatalogPrettyPrinter().walk(rule)
    assert "EXPECT_NORM[x | " in printed and "// " in printed
    assert printed.endswith("= e")
