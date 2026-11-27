"""
``PROB[COUNT(x) θ k | a(g, x)] = p``, the probability that the event ``a``
holds for ``θ k`` values of ``x``, per group ``g``: the Poisson-binomial tail
(``>=``) of the probabilities of the values, and its complements. The
probabilities are computed by enumerating the possible worlds, over engines
that have only the lifted solvers: the program the sugar is rewritten to is
solved by them, or refused.
"""
import itertools
import operator
from collections import defaultdict

import pytest

from nl_core.probabilistic import dalvi_suciu_lift
from nl_expr.exceptions import NeuroLangException
from ...frontend import NeurolangPDL

OPERATORS = {
    ">=": operator.ge, ">": operator.gt, "==": operator.eq,
    "!=": operator.ne, "<": operator.lt, "<=": operator.le,
}
# independent facts: name -> {tuple: probability}
FACTS = {
    "seen": {
        ("g1", 1): 0.9, ("g1", 2): 0.6, ("g1", 3): 0.3,
        ("g2", 1): 0.5, ("g2", 2): 0.5,
        ("g3", 1): 0.4,
        ("g4", 1): 1.0, ("g4", 2): 1.0,
    },
    "solo": {(1,): 0.9, (2,): 0.6, (3,): 0.3},
    "flag": {("g1",): 0.5, ("g2",): 0.25, ("g3",): 0.75},
    "link": {(1, "u"): 0.6, (2, "u"): 0.5, (2, "v"): 0.8, (3, "v"): 0.3},
    "other": {("u",): 0.9, ("v",): 0.5},
}
# a probabilistic choice, and the independent facts of what is chosen
CHOICES = {"sel": {"s1": 0.5, "s2": 0.3, "s3": 0.2}}
FACTS["inside"] = {
    ("s1", 1): 0.5, ("s1", 2): 0.6, ("s2", 1): 0.4, ("s2", 2): 0.3,
    ("s3", 1): 0.8,
}
KEPT = [("g1", 1), ("g1", 3), ("g2", 1), ("g2", 2), ("g3", 1), ("g4", 2)]
GROUPS = ["g1", "g2", "g3", "g4"]
LIFTED_ONLY = dict(
    probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
    probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
)


def _engine(lifted_only=True):
    nl = NeurolangPDL(**(LIFTED_ONLY if lifted_only else {}))
    for name, tuples in FACTS.items():
        nl.add_probabilistic_facts_from_tuples(
            [(p,) + t for t, p in tuples.items()], name=name
        )
    for name, values in CHOICES.items():
        nl.add_probabilistic_choice_from_tuples(
            [(p, v) for v, p in values.items()], name=name
        )
    nl.add_tuple_set(KEPT, name="kept")
    nl.add_tuple_set([(g,) for g in GROUPS], name="groups")
    return nl


def _rows(program, lifted_only=True):
    result = _engine(lifted_only).execute_datalog_program(program)
    return [tuple(row) for row in result]


def _by_group(program, lifted_only=True):
    return {row[:-1]: row[-1] for row in _rows(program, lifted_only)}


def _worlds(names, choices=()):
    """(probability, chosen, world) of each possible world of the facts of
    `names` and the choices `choices`."""
    facts = [(n, t) for n in names for t in FACTS[n]]
    for values in itertools.product(*(CHOICES[c].items() for c in choices)):
        p_chosen = 1.
        for _, p in values:
            p_chosen *= p
        chosen = dict(zip(choices, (v for v, _ in values)))
        for present in itertools.product((False, True), repeat=len(facts)):
            p = p_chosen
            for (n, t), is_present in zip(facts, present):
                q = FACTS[n][t]
                p *= q if is_present else 1 - q
            yield p, chosen, {
                f for f, is_present in zip(facts, present) if is_present
            }


def _distribution(counts, names, choices=()):
    """P[N_g = j] by enumerating the worlds, `counts(chosen, world)` the
    count of each group, the groups with an answer being those it names."""
    result = defaultdict(lambda: defaultdict(float))
    for p, chosen, world in _worlds(names, choices):
        for group, n in counts(chosen, world).items():
            result[group][n] += p
    return result


def _expected(distribution, comparison, k):
    """P[N θ k] of each group with an answer of positive probability."""
    return {
        group: sum(p for n, p in counts.items() if OPERATORS[comparison](n, k))
        for group, counts in distribution.items()
    }


def _assert_close(found, expected, support=None):
    """`found` is `expected` on the groups of `support`, those with a tuple
    of the event of positive probability, the others having no row."""
    for group in found.keys() | expected.keys():
        if support is not None and group not in support:
            assert group not in found, group
            continue
        assert found.get(group, 0.) == pytest.approx(
            expected.get(group, 0.), abs=1e-12
        ), group


def _seen_counts(chosen, world, keep=None):
    counts = {}
    for g in GROUPS:
        values = {
            t[1] for (n, t) in world
            if n == "seen" and t[0] == g and (keep is None or t in keep)
        }
        counts[(g,)] = len(values)
    return counts


@pytest.mark.parametrize("lifted_only", [True, False])
@pytest.mark.parametrize("comparison", list(OPERATORS))
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_independent_tuples_per_group(comparison, k, lifted_only):
    """Groups with more, as many and fewer tuples than k, with certain
    tuples and with a probability of no value, with the lifted solvers
    only and with the default ones."""
    found = _by_group(
        f"ans(g, p) :- PROB[COUNT(x) {comparison} {k} | seen(g, x)] = p",
        lifted_only
    )
    distribution = _distribution(_seen_counts, ["seen"])
    _assert_close(found, _expected(distribution, comparison, k))


@pytest.mark.parametrize("comparison", list(OPERATORS))
@pytest.mark.parametrize("k", [1, 2])
def test_without_a_group_variable(comparison, k):
    [(found,)] = _rows(
        f"ans(p) :- PROB[COUNT(x) {comparison} {k} | solo(x)] = p"
    )
    distribution = _distribution(
        lambda c, w: {(): len({t[0] for (n, t) in w if n == "solo"})},
        ["solo"]
    )[()]
    expected = sum(
        p for n, p in distribution.items() if OPERATORS[comparison](n, k)
    )
    assert found == pytest.approx(expected)


@pytest.mark.parametrize("comparison, k", [(">=", 2), ("<", 3), ("==", 1)])
def test_a_constant_in_the_event(comparison, k):
    """The constant is shattered into a variable projected in each copy,
    which the plan matches up to the renaming."""
    found = _rows(
        f"ans(p) :- PROB[COUNT(x) {comparison} {k} | seen('g1', x)] = p"
    )
    distribution = _distribution(_seen_counts, ["seen"])[("g1",)]
    expected = sum(
        p for n, p in distribution.items() if OPERATORS[comparison](n, k)
    )
    assert [r[0] for r in found] == [pytest.approx(expected)]


def test_a_constant_with_more_copies_than_model_counting_bears():
    """k = 5 copies of an event with a constant: the lifted plan counts the
    11 tuples in linear time."""
    nl = NeurolangPDL(**LIFTED_ONLY)
    ps = [0.1 * (i % 9 + 1) for i in range(11)]
    nl.add_probabilistic_facts_from_tuples(
        [(p, "c", i) for i, p in enumerate(ps)], name="wide"
    )
    [(found,)] = [tuple(r) for r in nl.execute_datalog_program(
        "ans(p) :- PROB[COUNT(x) >= 5 | wide('c', x)] = p"
    )]
    expected = sum(
        _product(ps, on) for size in range(5, 12)
        for on in itertools.combinations(range(11), size)
    )
    assert found == pytest.approx(expected)


@pytest.mark.parametrize("variables", [("x", "y"), ("zz", "aa")])
@pytest.mark.parametrize("comparison, k", [(">=", 2), ("<", 3), ("==", 1)])
def test_a_variable_projected_in_the_event(variables, comparison, k):
    """x counted, y projected: the count of the values of x that some y
    makes hold, whichever variable the names sort first, the separator
    being the counted one."""
    x, y = variables
    found = _by_group(
        f"ans(p) :- PROB[COUNT({x}) {comparison} {k} | link({x}, {y})] = p"
    )
    distribution = _distribution(
        lambda c, w: {(): len({t[0] for (n, t) in w if n == "link"})},
        ["link"]
    )[()]
    expected = sum(
        p for n, p in distribution.items() if OPERATORS[comparison](n, k)
    )
    assert found.get((), 0.) == pytest.approx(expected)


def test_a_variable_projected_by_exists():
    found = _by_group(
        "ans(p) :- PROB[COUNT(x) >= 2 | exists(y; link(x, y))] = p"
    )
    distribution = _distribution(
        lambda c, w: {(): len({t[0] for (n, t) in w if n == "link"})},
        ["link"]
    )[()]
    assert found[()] == pytest.approx(distribution[2] + distribution[3])


@pytest.mark.parametrize("comparison, k", [(">=", 2), ("<", 3), ("==", 1)])
def test_a_group_bound_outside_the_event(comparison, k):
    found = _by_group(
        "ans(g, p) :- groups(g), "
        f"PROB[COUNT(x) {comparison} {k} | seen(g, x)] = p"
    )
    distribution = _distribution(_seen_counts, ["seen"])
    _assert_close(found, _expected(distribution, comparison, k))


@pytest.mark.parametrize("comparison, k", [(">=", 1), (">=", 2), ("<=", 1)])
def test_deterministic_atom_in_the_event(comparison, k):
    found = _by_group(
        f"ans(g, p) :- PROB[COUNT(x) {comparison} {k} | seen(g, x), "
        "kept(g, x)] = p"
    )
    distribution = _distribution(
        lambda c, w: _seen_counts(c, w, set(KEPT)), ["seen"]
    )
    _assert_close(found, _expected(distribution, comparison, k))


@pytest.mark.parametrize("comparison, k", [(">=", 2), ("<", 2), ("==", 0)])
def test_an_independent_conjunct_in_the_event(comparison, k):
    """flag(g) holds for every copy: N is 0 when it does not."""
    found = _by_group(
        f"ans(g, p) :- PROB[COUNT(x) {comparison} {k} | flag(g), "
        "seen(g, x)] = p"
    )

    def counts(chosen, world):
        return {
            (g,): (
                len({t[1] for (n, t) in world if n == "seen" and t[0] == g})
                if ("flag", (g,)) in world else 0
            )
            for (g,) in FACTS["flag"]
        }

    distribution = _distribution(counts, ["seen", "flag"])
    _assert_close(found, _expected(distribution, comparison, k))


@pytest.mark.parametrize("comparison, k", [(">=", 2), ("==", 1), ("<", 2)])
def test_a_threshold_comparison_in_the_event(comparison, k):
    found = _by_group(
        f"ans(g, p) :- PROB[COUNT(x) {comparison} {k} | seen(g, x), "
        "(x > 1)] = p"
    )
    distribution = _distribution(
        lambda c, w: {
            (g,): len({
                t[1] for (n, t) in w
                if n == "seen" and t[0] == g and t[1] > 1
            })
            for g in GROUPS
        },
        ["seen"]
    )
    support = {(g,) for (g, x) in FACTS["seen"] if x > 1}
    _assert_close(found, _expected(distribution, comparison, k), support)


@pytest.mark.parametrize("comparison, k", [(">=", 2), (">=", 1), ("<", 2)])
def test_a_choice_counts_at_most_one(comparison, k):
    """The values of a choice are exclusive: the count is 1."""
    expected = {">=2": 0., ">=1": 1., "<2": 1.}[f"{comparison}{k}"]
    rows = _rows(f"ans(p) :- PROB[COUNT(s) {comparison} {k} | sel(s)] = p")
    assert (rows[0][0] if rows else 0.) == pytest.approx(expected)


@pytest.mark.parametrize("lifted_only", [True, False])
@pytest.mark.parametrize("comparison, k", [(">=", 2), ("==", 1), ("<", 2)])
def test_independent_values_of_the_chosen(comparison, k, lifted_only):
    """x counted among the tuples of the chosen s: given s the values are
    independent, and the choice is a disjoint projection of the count."""
    program = (
        f"ans(p) :- PROB[COUNT(x) {comparison} {k} | sel(s), inside(s, x)] = p"
    )
    distribution = _distribution(
        lambda c, w: {(): len({
            t[1] for (n, t) in w if n == "inside" and t[0] == c["sel"]
        })},
        ["inside"], ["sel"]
    )
    expected = _expected(distribution, comparison, k)[()]
    [(found,)] = _rows(program, lifted_only)
    assert found == pytest.approx(expected)


def test_dependent_events_are_refused_by_the_lifted_solvers():
    """The events of two values of x share other(y): the tail of the
    independent events would be wrong. Model counting is right."""
    program = (
        "ans(p) :- PROB[COUNT(x) >= 2 | exists(y; link(x, y), other(y))] = p"
    )
    with pytest.raises(NeuroLangException):
        _rows(program)
    distribution = _distribution(
        lambda c, w: {(): len({
            x for x in (1, 2, 3)
            if any(("link", (x, y)) in w and ("other", (y,)) in w
                   for y in ("u", "v"))
        })},
        ["link", "other"]
    )
    [(found,)] = _rows(program, lifted_only=False)
    assert found == pytest.approx(distribution[()][2] + distribution[()][3])


def test_a_disjunctive_event_is_counted_by_model_counting():
    [(found,)] = _rows(
        "ans(p) :- PROB[COUNT(x) >= 2 | seen('g1', x) | seen('g2', x)] = p",
        lifted_only=False
    )
    distribution = _distribution(
        lambda c, w: {(): len({
            x for (n, (g, x)) in w if n == "seen" and g in ("g1", "g2")
        })},
        ["seen"]
    )
    assert found == pytest.approx(distribution[()][2] + distribution[()][3])


def test_conditioning_on_an_independent_event():
    """P[N >= 2 | flag(g)] of an independent flag is P[N >= 2]."""
    found = _by_group(
        "ans(g, p) :- PROB[COUNT(x) >= 2 | seen(g, x) // flag(g)] = p"
    )
    expected = _expected(
        _distribution(_seen_counts, ["seen"]), ">=", 2
    )
    _assert_close(
        {g: p for g, p in found.items() if g[0] in ("g1", "g2", "g3")},
        {g: p for g, p in expected.items() if g[0] in ("g1", "g2", "g3")},
    )


def test_conditioning_on_a_related_event_is_refused_then_counted():
    """P[N >= 2 | seen('g1', 1)]: the conditioning shares seen with the
    event, which the lifted solvers decline; model counting answers."""
    program = (
        "ans(p) :- PROB[COUNT(x) >= 2 | seen('g1', x) // seen('g1', 1)] = p"
    )
    with pytest.raises(NeuroLangException):
        _rows(program)
    [(found,)] = _rows(program, lifted_only=False)
    # given seen(g1, 1): at least one more of the two others
    assert found == pytest.approx(1 - (1 - 0.6) * (1 - 0.3))


def test_a_large_count_is_lifted():
    """k = 8 on 12 independent tuples, beyond the former limit of 4 and
    far from what the enumeration of copies handles."""
    probabilities = [round(0.05 * (i + 3), 2) for i in range(12)]
    nl = NeurolangPDL(**LIFTED_ONLY)
    nl.add_probabilistic_facts_from_tuples(
        [(p, i) for i, p in enumerate(probabilities)], name="many"
    )
    [(found,)] = [tuple(r) for r in nl.execute_datalog_program(
        "ans(p) :- PROB[COUNT(x) >= 8 | many(x)] = p"
    )]
    expected = sum(
        _product(probabilities, on)
        for size in range(8, 13)
        for on in itertools.combinations(range(12), size)
    )
    assert found == pytest.approx(expected)


def _product(probabilities, on):
    p = 1.
    for i, q in enumerate(probabilities):
        p *= q if i in on else 1 - q
    return p


@pytest.mark.parametrize("program, message", [
    ("ans(p) :- PROB[COUNT(x) >= 2.5 | seen(g, x)] = p", "integer"),
    ("ans(p) :- PROB[COUNT(x) >= n | seen(g, x)] = p", "integer"),
    ("ans(g, p) :- PROB[COUNT(z) >= 2 | seen(g, x)] = p", "z"),
    ("ans(g, x, p) :- PROB[COUNT(x) >= 2 | seen(g, x)] = p", "x"),
    ("ans(p) :- PROB[COUNT(x) >= 30 | seen('g1', x)] = p", "conjuncts"),
    (
        "ans(n) :- AGGREGATE[](PROB[COUNT(x) >= 2 | seen('g1', x)] = m "
        "@ sum(m)) = n", "AGGREGATE"
    ),
])
def test_what_is_refused(program, message):
    with pytest.raises(NeuroLangException, match=message):
        _rows(program)


def test_aggregating_a_probabilistic_query_without_answer():
    """A probabilistic query with no answer is not in the solution of the
    probabilistic stratum: the rules reading it, an aggregation among them,
    are stratified over an empty relation."""
    program = (
        "none(g, p) :- PROB[seen(g, x), seen(g, y), seen(g, z), seen(g, u),"
        " (x != y), (x != z), (x != u), (y != z), (y != u), (z != u)] = p\n"
        "total(g, sum(p)) :- none(g, p)\n"
        "ans(g, s) :- total(g, s)"
    )
    assert _rows(program) == []


@pytest.mark.parametrize("comparison, k", [("<", 2), ("==", 1), (">=", 2)])
def test_complements_under_conditioning(comparison, k):
    """The complements of the conditional probabilities, which are
    those of the count knowing the event."""
    found = _by_group(
        f"ans(g, p) :- PROB[COUNT(x) {comparison} {k} | seen(g, x) // "
        "flag(g)] = p"
    )
    expected = _expected(_distribution(_seen_counts, ["seen"]), comparison, k)
    _assert_close(
        {g: p for g, p in found.items() if g[0] in ("g1", "g2", "g3")},
        {g: p for g, p in expected.items() if g[0] in ("g1", "g2", "g3")},
    )


def test_a_count_at_the_limit_of_copies():
    """k = 19 of an atom, 190 conjuncts, is within what the transformations
    of the lifting bear, from the stack of a test."""
    nl = NeurolangPDL(**LIFTED_ONLY)
    ps = [0.5 + 0.02 * (i % 20) for i in range(25)]
    nl.add_probabilistic_facts_from_tuples(
        [(p, i) for i, p in enumerate(ps)], name="deep"
    )
    [(found,)] = [tuple(r) for r in nl.execute_datalog_program(
        "ans(p) :- PROB[COUNT(x) >= 19 | deep(x)] = p"
    )]
    tail = [1.] + [0.] * 19
    for p in ps:
        tail = [tail[0]] + [
            tail[j] + p * (tail[j - 1] - tail[j]) for j in range(1, 20)
        ]
    assert found == pytest.approx(tail[19])


def test_a_disjunctive_event_has_at_most_four_copies():
    with pytest.raises(NeuroLangException, match="disjunction"):
        _rows(
            "ans(p) :- PROB[COUNT(x) >= 5 | seen('g1', x) | seen('g2', x)]"
            " = p"
        )


def test_count_before_the_probability_for_the_chosen():
    """The recipe for the values of a choice: count first, then the
    probability that the chosen has enough."""
    nl = _engine()
    nl.add_tuple_set(
        [("s1", "a"), ("s1", "b"), ("s1", "c"), ("s2", "a"), ("s3", "a"),
         ("s3", "b")],
        name="mentions"
    )
    [(found,)] = [tuple(r) for r in nl.execute_datalog_program(
        "n_terms(s, n) :- AGGREGATE[s](mentions(s, t) @ count(t)) = n\n"
        "many(s) :- n_terms(s, n), (n >= 2), sel(s)\n"
        "ans(p) :- PROB[many(s)] = p"
    )]
    assert found == pytest.approx(0.5 + 0.2)


def test_the_tutorial_block():
    """The facts and the three queries of the tutorial, as written."""
    answers = NeurolangPDL().execute_datalog_program("""
        0.9 :: seen('u', 1).
        0.6 :: seen('u', 2).
        0.3 :: seen('u', 3).
        0.5 :: seen('v', 1).
        0.5 :: seen('v', 2).
        ans[two_or_more](g, p) :- PROB[COUNT(x) >= 2 | seen(g, x)] = p
        ans[exactly_one](g, p) :- PROB[COUNT(x) == 1 | seen(g, x)] = p
        ans[fewer_than_two](g, p) :- PROB[COUNT(x) < 2 | seen(g, x)] = p
    """)
    assert {
        name: {g: round(p, 3) for g, p in map(tuple, answers[name])}
        for name in ("two_or_more", "exactly_one", "fewer_than_two")
    } == {
        "two_or_more": {"u": 0.666, "v": 0.25},
        "exactly_one": {"u": 0.306, "v": 0.5},
        "fewer_than_two": {"u": 0.334, "v": 0.75},
    }
