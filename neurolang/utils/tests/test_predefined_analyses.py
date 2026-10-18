"""The lift, relative risk and uniformity ratio the coordinate-based engines
define: checked against a direct computation on a tiny engine.  The term is a
constant of the query, as in the tutorial: probabilistic queries without
a constant do not go through the magic-sets rewriting."""
import itertools
import re

import numpy as np
import pandas as pd
import pytest

from neurolang.frontend import NeurolangPDL
from neurolang.utils import engine_registry

ANALYSIS_RULES = re.compile(
    r"(term_in_study|activated|p_[a-z_]+|lift|"
    r"relative_risk|uniformity|ratios)\("
)


@pytest.fixture(scope="module")
def small_neurosynth():
    """A hand-built engine with the data of the coordinate-based engines
    and the analysis rules of ``engines.yaml``."""
    nl = NeurolangPDL()
    studies = [1, 2, 3]
    nl.add_tuple_set([(s,) for s in studies], name="study")
    nl.add_tuple_set(list(itertools.product(range(5), repeat=3)), name="brain_mask")
    nl.add_tuple_set(
        [(1, 1, 1, 1), (2, 2, 2, 1), (4, 4, 4, 2), (0, 4, 0, 3)],
        name="peak_reported",
    )
    nl.add_tuple_set(
        [(1, "memory", 0.5), (2, "memory", 0.3), (3, "memory", 0.4),
         (3, "language", 0.02), (2, "language", 0.0005)],
        name="term_in_study_tfidf",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame({"study_id": studies}), name="selected_study"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame(
            list(itertools.product(range(5), repeat=3)), columns=["i", "j", "k"]
        ),
        name="selected_voxel",
    )
    rules = [
        line for line in engine_registry.get_engine_config("neurosynth")[
            "datalog_init"
        ].splitlines()
        if ANALYSIS_RULES.match(line)
    ]
    nl.execute_datalog_program("\n".join(rules))
    return nl


def _expected(nl, term):
    """Direct computation of P(A_v | T), P(A_v), P(T), P(A_V | T)."""
    brain = sorted(nl.symbols["brain_mask"].value)
    peaks = {}
    for i, j, k, s in nl.symbols["peak_reported"].value:
        peaks.setdefault(s, []).append((i, j, k))
    studies = sorted(peaks)
    tfidf = nl.symbols["term_in_study_tfidf"].value
    of_term = {s for s, t, w in tfidf if t == term and w > 0.001}
    near = lambda v, s: any(  # noqa: E731
        np.linalg.norm(np.subtract(v, p)) <= 3 for p in peaks[s]
    )
    activated = {v: {s for s in studies if near(v, s)} for v in brain}
    p_at = {v: len(a & of_term) / len(of_term) for v, a in activated.items()}
    p_a = {v: len(a) / len(studies) for v, a in activated.items()}
    p_bar = np.mean(list(p_at.values()))
    return of_term, studies, p_at, p_a, p_bar


@pytest.mark.parametrize("term", ["language", "memory"])
def test_lift_and_uniformity(small_neurosynth, term):
    nl = small_neurosynth
    of_term, studies, p_at, p_a, p_bar = _expected(nl, term)
    pt = len(of_term) / len(studies)

    lift = {
        (i, j, k): v for i, j, k, v in nl.execute_datalog_program(
            f"ans(i, j, k, l) :- lift('{term}', i, j, k, l)"
        )
    }
    uniformity = {
        (i, j, k): v for i, j, k, v in nl.execute_datalog_program(
            f"ans(i, j, k, u) :- uniformity('{term}', i, j, k, u)"
        )
    }
    voxels = {v for v, p in p_at.items() if p > 0}
    assert set(lift) == set(uniformity) == voxels
    for v in voxels:
        assert lift[v] == pytest.approx(p_at[v] / p_a[v])
        assert uniformity[v] == pytest.approx(p_at[v] / p_bar)

    both = {
        (i, j, k): (l, r, u)
        for i, j, k, l, r, u in nl.execute_datalog_program(
            f"ans(i, j, k, l, r, u) :- ratios('{term}', i, j, k, l, r, u)"
        )
    }
    assert set(both) == voxels
    for v, (l, r, u) in both.items():
        assert (l, u) == pytest.approx((lift[v], uniformity[v]))
        denominator = p_a[v] - p_at[v] * pt
        if denominator > 1e-12:
            assert r == pytest.approx(p_at[v] * (1 - pt) / denominator)


def test_selected_voxel_is_a_uniform_choice_over_the_brain(small_neurosynth):
    brain = {tuple(v) for v in small_neurosynth.symbols["brain_mask"].value}
    chosen = list(small_neurosynth.symbols["selected_voxel"].value)
    assert {tuple(row[1:]) for row in chosen} == brain
    assert all(row[0] == pytest.approx(1 / len(brain)) for row in chosen)
