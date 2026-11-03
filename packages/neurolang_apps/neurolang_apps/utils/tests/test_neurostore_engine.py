import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurolang.frontend import NeurolangPDL
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.engines.neurostore import init as neurostore_init

NS_INIT = "neurolang_apps.utils.engines.neurosynth.init"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    def offline():
        raise OSError("offline")

    monkeypatch.setattr(neurostore_init, "_fetch_releases", offline)
    monkeypatch.setattr(neurostore_init, "MIN_TERM_STUDIES", 1)
    monkeypatch.setattr(neurostore_init, "NGRAM_RANGE", (1, 1))
    release = tmp_path / "neurostore" / "2026-10"
    release.mkdir(parents=True)
    pd.DataFrame(
        {
            "study_id": ["s1", "s1", "s2", "s3"],
            "x": [0.0, 10.0, -20.0, 0.0],
            "y": [0.0, 10.0, -20.0, 0.0],
            "z": [0.0, 10.0, -20.0, 0.0],
            "space": ["MNI", "MNI", "TAL", "UNKNOWN"],
        }
    ).to_parquet(release / "coordinates.parquet")
    pd.DataFrame(
        {
            "study_id": ["s1", "s1", "s2", "s3"],
            "TaskExtractor.fMRITasks[0].Concepts": [
                json.dumps(["memory", "attention"]),
                json.dumps(["memory"]),
                None,
                json.dumps(["ignored"]),
            ],
            "TaskExtractor.fMRITasks[0].Domain": [
                json.dumps(["Learning"]), None, json.dumps(["Emotion"]), None
            ],
        }
    ).to_parquet(release / "annotations.parquet")
    pd.DataFrame(
        {
            "study_id": ["s1", "s2", "s3"],
            "name": ["Memory paper", "Emotion paper", "Other"],
            "description": [
                "Memory and attention network",
                "Emotion network",
                "Ignored study",
            ],
        }
    ).to_parquet(release / "studies.parquet")
    return tmp_path


def _build(data_dir):
    mask = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype="uint8"), np.diag([2.0, 2.0, 2.0, 1.0])
    )
    nl = NeurolangPDL()
    with patch(
        f"{NS_INIT}.fetch_cortical_atlas",
        return_value=nib.Nifti1Image(
            np.ones((4, 4, 4), dtype="int16"), np.diag([2.0, 2.0, 2.0, 1.0])
        ),
    ), patch(
        f"{NS_INIT}.fetch_cognitive_atlas_tasks",
        return_value=pd.DataFrame(
            {"task_id": ["t1"], "name": ["n-back"], "description": ["d"]}
        ),
    ):
        neurostore_init.init_engine(nl, mask, Path(data_dir))
    return nl


def test_init_engine(data_dir):
    nl = _build(data_dir)
    rel = lambda name: set(nl.symbols[name].value)  # noqa: E731

    assert {s for (s,) in rel("study")} == {"s1", "s2"}
    assert len(nl.symbols["peak_reported"].value) == 3
    assert len(nl.symbols["selected_study"].value) == 2
    assert rel("study_concept") == {("s1", "memory"), ("s1", "attention")}
    assert rel("study_domain") == {("s1", "Learning"), ("s2", "Emotion")}
    assert rel("study_abstract") == {
        ("s1", "Memory and attention network"), ("s2", "Emotion network"),
    }
    assert rel("study_title") == {
        ("s1", "Memory paper"), ("s2", "Emotion paper"),
    }
    tfidf = nl.symbols["term_in_study_tfidf"].value
    assert {(s, t) for s, t, _ in tfidf} == {
        ("s1", "memory"), ("s1", "attention"), ("s1", "network"),
        ("s2", "emotion"), ("s2", "network"),
    }
    assert all(0 < v <= 1 for _, _, v in tfidf)
    assert {t for (t,) in rel("term")} == {
        "memory", "attention", "network", "emotion"
    }
    assert rel("cognitive_task") == {("t1", "n-back", "d")}


def test_engine_declarations_match_neurosynth_names():
    ns = engine_registry.get_predicates("neurosynth")
    nstore = engine_registry.get_predicates("neurostore")
    assert set(ns) <= set(nstore)
    assert nstore["term_in_study_tfidf"]["columns"] == ns[
        "term_in_study_tfidf"
    ]["columns"]
    assert nstore["study_abstract"]["columns"] == ["study_id", "text"]


def test_embeddings_read_the_engine_texts(data_dir):
    nl = _build(data_dir)
    cfg = engine_registry.get_engine_config("neurostore")
    source = cfg["text_embeddings"]["study_abstracts"]["source"]
    assert engine_registry._predicate_texts(nl, source) == {
        "s1": ("Memory paper", "Memory and attention network"),
        "s2": ("Emotion paper", "Emotion network"),
    }
    # Same function and models as the neurosynth engine.
    entry = cfg["text_embeddings"]["study_abstracts"]
    assert entry["function"] == "query_similarity"
    assert entry["model"] == "allenai/specter2_base"


def test_select_release():
    releases = [
        {"version": "nightly", "release_type": "nightly", "built_at": "3"},
        {"version": "2026-09", "release_type": "monthly", "built_at": "1"},
        {"version": "2026-10", "release_type": "monthly", "built_at": "2"},
    ]
    select = neurostore_init._select_release
    assert select(releases, "latest")["version"] == "2026-10"
    assert select(releases, "nightly")["version"] == "nightly"
    with pytest.raises(ValueError, match="Unknown NeuroStore release"):
        select(releases, "1999-01")


def test_terms_include_word_sequences():
    abstracts = pd.Series(
        ["working memory task", "working memory network"], index=["a", "b"]
    )
    with patch.object(neurostore_init, "MIN_TERM_STUDIES", 1):
        terms = neurostore_init._tfidf_terms(abstracts)
    assert {"working memory", "memory task", "working memory network"} <= set(
        terms["term"]
    )
