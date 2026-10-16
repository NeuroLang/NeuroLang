"""Tests for ``NeurolangPDL.k_fold_assign``."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from ..probabilistic_frontend import NeurolangPDL

STUDIES = [(f"s{i:02d}",) for i in range(23)]


def folds_of(assign):
    """Return the mapping from each tuple to its fold."""
    return {row[:-1]: row[-1] for row in assign.value}


def test_each_tuple_in_exactly_one_fold_and_balanced():
    nl = NeurolangPDL()
    assign, fold_id = nl.k_fold_assign(STUDIES, k=5, seed=0, name="cv")
    assert assign.symbol_name == "cv_assign"
    assert fold_id.symbol_name == "cv_fold_id"
    assert set(fold_id.value) == {(i,) for i in range(5)}
    folds = folds_of(assign)
    assert len(assign.value) == len(STUDIES)
    assert set(folds) == set(STUDIES)
    sizes = Counter(folds.values())
    assert set(sizes) == set(range(5))
    assert max(sizes.values()) - min(sizes.values()) <= 1


def test_seed_reproducible_whatever_the_input_order():
    nl = NeurolangPDL()
    folds = folds_of(nl.k_fold_assign(STUDIES, k=4, seed=3)[0])
    reversed_folds = folds_of(
        nl.k_fold_assign(STUDIES[::-1], k=4, seed=3)[0]
    )
    set_folds = folds_of(nl.k_fold_assign(set(STUDIES), k=4, seed=3)[0])
    other_seed_folds = folds_of(nl.k_fold_assign(STUDIES, k=4, seed=4)[0])
    assert folds == reversed_folds == set_folds
    assert folds != other_seed_folds


def test_no_shuffle_is_round_robin_in_input_order():
    nl = NeurolangPDL()
    assign, _ = nl.k_fold_assign(STUDIES, k=3, shuffle=False)
    assert folds_of(assign) == {s: i % 3 for i, s in enumerate(STUDIES)}


def test_input_formats_and_repeated_tuples():
    nl = NeurolangPDL()
    df = pd.DataFrame({"study": ["a", "b", "c"], "n": [1, 2, 3]})
    assign, _ = nl.k_fold_assign(df, k=3, seed=0)
    assert set(folds_of(assign)) == {("a", 1), ("b", 2), ("c", 3)}
    assign, _ = nl.k_fold_assign(["a", "b", "a", ["c"], ("b",)], k=3)
    folds = folds_of(assign)
    assert set(folds) == {("a",), ("b",), ("c",)}
    assert sorted(folds.values()) == [0, 1, 2]


@pytest.mark.parametrize("k", [0, 1, len(STUDIES) + 1])
def test_invalid_number_of_folds(k):
    nl = NeurolangPDL()
    with pytest.raises(ValueError):
        nl.k_fold_assign(STUDIES, k=k)


def test_train_and_test_sets_in_a_query():
    nl = NeurolangPDL()
    assign, fold_id = nl.k_fold_assign(STUDIES, k=5, seed=0)
    with nl.scope as e:
        e.train[e.f, e.s] = fold_id[e.f] & assign[e.s, e.g] & (e.g != e.f)
        e.test[e.f, e.s] = assign[e.s, e.f]
        solution = nl.solve_all()
    train = set(solution["train"].as_pandas_dataframe().itertuples(
        index=False, name=None
    ))
    test = set(solution["test"].as_pandas_dataframe().itertuples(
        index=False, name=None
    ))
    for fold in range(5):
        train_fold = {s for f, s in train if f == fold}
        test_fold = {s for f, s in test if f == fold}
        assert train_fold.isdisjoint(test_fold)
        assert train_fold | test_fold == {s for s, in STUDIES}
    assert {s for _, s in test} == {s for s, in STUDIES}


def test_probabilistic_query_over_folds():
    nl = NeurolangPDL()
    probs = np.linspace(0.05, 0.5, len(STUDIES))
    nl.add_probabilistic_facts_from_tuples(
        [(float(p), s) for p, (s,) in zip(probs, STUDIES)], name="active"
    )
    assign, fold_id = nl.k_fold_assign(STUDIES, k=3, seed=1)
    folds = folds_of(assign)
    with nl.scope as e:
        e.train[e.f, e.s] = fold_id[e.f] & assign[e.s, e.g] & (e.g != e.f)
        e.any_active[e.f, e.PROB[e.f]] = e.train[e.f, e.s] & e.active[e.s]
        result = nl.query((e.f, e.p), e.any_active[e.f, e.p])
    result = dict(result.as_pandas_dataframe().itertuples(
        index=False, name=None
    ))
    for fold in range(3):
        expected = 1 - np.prod([
            1 - p for p, s in zip(probs, STUDIES) if folds[s] != fold
        ])
        assert result[fold] == pytest.approx(expected)
