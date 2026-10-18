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


LLM_RULES = re.compile(
    r"(term_in_study|activated|p_[a-z_]+|study_similarity|cutoff_similarity|"
    r"similar_study|[a-z_]+_llm)\("
)
SIMILARITY = {s: 0.2 + 0.04 * ((s * 7) % 20) for s in range(1, 21)}
SIMILARITY[20] = 0.0  # a study without text: left out


@pytest.fixture(scope="module")
def llm_engine():
    """Twenty studies, a function standing for ``query_similarity``, and the
    free-text rules of ``engines.yaml``."""
    nl = NeurolangPDL()
    studies = list(range(1, 21))
    nl.add_tuple_set([(s,) for s in studies], name="study")
    voxels = list(itertools.product(range(5), repeat=3))
    nl.add_tuple_set(voxels, name="brain_mask")
    nl.add_tuple_set(
        [((s * 3) % 5, (s * 2) % 5, s % 5, s) for s in studies]
        + [((s * 4 + 1) % 5, (s * 3 + 2) % 5, (s + 2) % 5, s) for s in studies],
        name="peak_reported",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame({"study_id": studies}), name="selected_study"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame(voxels, columns=["i", "j", "k"]), name="selected_voxel"
    )

    def query_similarity(query: str, study: int) -> float:
        return SIMILARITY[study] if query == "a text" else 0.0

    nl.add_symbol(query_similarity, name="query_similarity")
    rules = [
        line for line in engine_registry.get_engine_config("neurosynth")[
            "datalog_init"
        ].splitlines()
        if LLM_RULES.match(line)
    ]
    nl.execute_datalog_program("\n".join(rules))
    return nl


def _top_five_percent():
    values = {s: v for s, v in SIMILARITY.items() if v > 0}
    cutoff = np.percentile(list(values.values()), 95)
    return {s for s, v in values.items() if v >= cutoff}


def test_similar_study_is_the_top_five_percent_of_the_texts(llm_engine):
    similarity = {
        s: v for s, v in llm_engine.execute_datalog_program(
            "ans(s, v) :- study_similarity('a text', s, v)"
        )
    }
    assert set(similarity) == set(SIMILARITY) - {20}
    similar = {
        s for (s,) in llm_engine.execute_datalog_program(
            "ans(s) :- similar_study('a text', s)"
        )
    }
    assert similar == _top_five_percent() and len(similar) == 1


def test_uniformity_and_lift_of_a_free_text_query(llm_engine):
    nl = llm_engine
    peaks = {}
    for i, j, k, s in nl.symbols["peak_reported"].value:
        peaks.setdefault(s, []).append((i, j, k))
    brain = sorted(nl.symbols["brain_mask"].value)
    activated = {
        s: {
            v for v in brain
            if any(np.linalg.norm(np.subtract(v, p)) <= 3 for p in ps)
        } for s, ps in peaks.items()
    }
    similar = _top_five_percent()
    p_at = {v: sum(v in activated[s] for s in similar) / len(similar) for v in brain}
    p_a = {v: sum(v in a for a in activated.values()) / len(peaks) for v in brain}
    p_bar = np.mean(list(p_at.values()))

    uniformity = {
        (i, j, k): u for i, j, k, u in nl.execute_datalog_program(
            "ans(i, j, k, u) :- uniformity_llm('a text', i, j, k, u)"
        )
    }
    lift = {
        (i, j, k): u for i, j, k, u in nl.execute_datalog_program(
            "ans(i, j, k, l) :- lift_llm('a text', i, j, k, l)"
        )
    }
    voxels = {v for v, p in p_at.items() if p > 0}
    assert set(uniformity) == set(lift) == voxels
    for v in voxels:
        assert uniformity[v] == pytest.approx(p_at[v] / p_bar)
        assert lift[v] == pytest.approx(p_at[v] / p_a[v])
