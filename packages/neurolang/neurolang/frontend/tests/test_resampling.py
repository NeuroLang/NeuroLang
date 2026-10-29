"""Tests of the resampling of a probabilistic choice, ``RESAMPLE``."""
import numpy as np
import pytest

from nl_expr.exceptions import ForbiddenExpressionError
from neurolang.frontend import NeurolangPDL
from neurolang.frontend.resampling import resampling_ratios

STUDIES = [f"s{i:02d}" for i in range(12)]
RNG = np.random.default_rng(0)
ACTIVATES = [
    (s, r) for s in STUDIES for r in ("r1", "r2") if RNG.random() < .5
]
MENTIONS = [
    (s, t) for s in STUDIES for t in ("t1", "t2", "t3") if RNG.random() < .5
]
N = 7

PROGRAM = """
active_region(r) :- region(r), selected_study(s), activates(s, r).
mentioned_term(t) :- term(t), selected_study(s), mentions(s, t).
cooccurrence(r, t) :- region(r), term(t), selected_study(s), \
activates(s, r), mentions(s, t).
region_probability(r, p) :- PROB[active_region(r)] = p.
term_probability(t, p) :- PROB[mentioned_term(t)] = p.
joint_probability(r, t, p) :- PROB[cooccurrence(r, t)] = p.
bayes_factor(r, t, bf) :- joint_probability(r, t, p_rt), \
region_probability(r, p_r), term_probability(t, p_t), \
(bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r))).
given(r, t, p) :- PROB[activates(s, r) & selected_study(s) // \
mentions(s, t) & selected_study(s)] = p.
"""


def engine(choice):
    nl = NeurolangPDL()
    nl.add_tuple_set(ACTIVATES, name="activates")
    nl.add_tuple_set(MENTIONS, name="mentions")
    nl.add_tuple_set([("r1",), ("r2",)], name="region")
    nl.add_tuple_set([("t1",), ("t2",), ("t3",)], name="term")
    choice(nl)
    nl.execute_datalog_program(PROGRAM)
    return nl


def uniform(nl):
    nl.add_uniform_probabilistic_choice_over_set(
        [(s,) for s in STUDIES], name="selected_study"
    )


def rows(nl, query):
    return {
        row[:-1]: row[-1]
        for row in nl.execute_datalog_program(query)
        .as_pandas_dataframe().itertuples(index=False)
    }


@pytest.fixture(scope="module")
def resampled():
    nl = engine(uniform)
    nl.add_resampling("boot_studies", "selected_study", n=N)
    nl.execute_datalog_program("""
bf_rep(b, r, t, bf) :- RESAMPLE[boot_studies](bayes_factor(r, t, bf)) = b.
given_rep(b, r, t, p) :- RESAMPLE[boot_studies](given(r, t, p)) = b.
""")
    return nl


@pytest.mark.parametrize("measure", ["bayes_factor", "given"])
def test_replicates_are_the_program_on_reweighted_choices(resampled, measure):
    rep = "bf_rep" if measure == "bayes_factor" else "given_rep"
    got = rows(resampled, f"ans(b, r, t, x) :- {rep}(b, r, t, x).")
    ratios = resampling_ratios(np.full(len(STUDIES), 1 / len(STUDIES)), N)
    for b in range(N):
        replicate = engine(lambda nl: nl.add_probabilistic_choice_from_tuples(
            [(float(ratios[b, i]) / len(STUDIES), s)
             for i, s in enumerate(STUDIES)],
            name="selected_study",
        ))
        expected = rows(replicate, f"ans(r, t, x) :- {measure}(r, t, x).")
        for (r, t), x in expected.items():
            if np.isfinite(x):
                assert got[(b, r, t)] == pytest.approx(x)


def test_interval_of_the_replicates(resampled):
    ci = rows(resampled, """
ans(r, t, percentile(x, 2.5), percentile(x, 97.5)) :- bf_rep(b, r, t, x).
""")
    reps = rows(resampled, "ans(b, r, t, x) :- bf_rep(b, r, t, x).")
    for (r, t, lo), hi in ci.items():
        values = [x for (b, r_, t_), x in reps.items() if (r_, t_) == (r, t)]
        assert (lo, hi) == pytest.approx(
            tuple(np.nanpercentile(values, [2.5, 97.5]))
        )


def test_resamplings_share_their_replicates(resampled):
    joined = rows(resampled, """
ans(b, r, t, x, p) :- bf_rep(b, r, t, x), given_rep(b, r, t, p).
""")
    assert {b for b, *_ in joined} == set(range(N))


def test_a_second_resample_reuses_the_rules(resampled):
    before = set(resampled.program_ir.intensional_database())
    resampled.execute_datalog_program(
        "again(b, r, t, x) :- RESAMPLE[boot_studies](bayes_factor(r, t, x))"
        " = b."
    )
    added = set(resampled.program_ir.intensional_database()) - before
    assert {s.name for s in added} == {"again"}


def test_resampling_of_a_weighted_choice():
    probabilities = np.linspace(1, 2, len(STUDIES))
    probabilities /= probabilities.sum()

    def weighted(nl):
        nl.add_probabilistic_choice_from_tuples(
            list(zip(map(float, probabilities), STUDIES)),
            name="selected_study",
        )

    nl = engine(weighted)
    nl.add_resampling("boot", "selected_study", n=3, scheme="multinomial")
    got = rows(nl, """
rep(b, r, p) :- RESAMPLE[boot](region_probability(r, p)) = b.
ans(b, r, p) :- rep(b, r, p).
""")
    ratios = resampling_ratios(probabilities, 3, "multinomial")
    for b in range(3):
        replicate = engine(lambda nl: nl.add_probabilistic_choice_from_tuples(
            list(zip(map(float, probabilities * ratios[b]), STUDIES)),
            name="selected_study",
        ))
        for (r,), p in rows(
            replicate, "ans(r, p) :- region_probability(r, p)."
        ).items():
            assert got[(b, r)] == pytest.approx(p)


@pytest.mark.parametrize("scheme", ["bayesian", "multinomial", "poisson"])
def test_ratios_are_probabilities_of_the_replicates(scheme):
    probabilities = np.full(10, .1)
    ratios = resampling_ratios(probabilities, 50, scheme, seed=1)
    assert ratios.shape == (50, 10)
    assert (ratios >= 0).all()
    assert ratios @ probabilities == pytest.approx(np.ones(50))


def test_unknown_scheme():
    with pytest.raises(ValueError, match="scheme"):
        resampling_ratios(np.full(2, .5), 3, "jackknife")


def test_a_predicate_without_probabilities_is_refused():
    nl = engine(uniform)
    nl.add_resampling("boot", "selected_study", n=2)
    with pytest.raises(ForbiddenExpressionError, match="nothing to resample"):
        nl.execute_datalog_program(
            "rep(b, r) :- RESAMPLE[boot](region(r)) = b."
        )


def test_resampling_needs_a_choice():
    nl = engine(uniform)
    with pytest.raises(ValueError, match="not a probabilistic choice"):
        nl.add_resampling("boot", "region")


def test_resample_after_the_rules_changed():
    nl = engine(uniform)
    nl.add_resampling("boot", "selected_study", n=5)
    query = """
rep(b, r, p) :- RESAMPLE[boot](region_probability(r, p)) = b.
ans(b, r, p) :- rep(b, r, p).
"""
    before = rows(nl, query)
    nl.execute_datalog_program(
        "active_region(r) :- region(r), selected_study(s), mentions(s, 't1')."
    )
    after = rows(nl, query.replace("rep", "rep2"))
    expected = rows(nl, "ans(r, p) :- region_probability(r, p).")
    assert after != before
    assert np.mean([after[(b, "r1")] for b in range(5)]) == pytest.approx(
        expected[("r1",)], abs=.2
    )
    assert rows(nl, query.replace("rep", "rep3")) == after


def test_resample_in_a_query(resampled):
    res = rows(resampled, """
ans(b, x) :- RESAMPLE[boot_studies](bayes_factor('r1', 't1', x)) = b.
""")
    reps = rows(resampled, "ans(b, r, t, x) :- bf_rep(b, r, t, x).")
    assert res == {(b,): x for (b, r, t), x in reps.items()
                   if (r, t) == ("r1", "t1")}
