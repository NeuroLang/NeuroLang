from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurolang.utils import engine_registry

NS_INIT = "neurolang.utils.engines.neurosynth.init"


@pytest.fixture
def engine():
    peaks = pd.DataFrame(
        {"id": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0],
         "space": ["MNI", "MNI"]}
    )
    terms = pd.DataFrame(
        {"id": np.array([1, 2, 2], dtype=np.int64),
         "term": ["working memory", "spatial working memory", "pain"],
         "tfidf": [0.5, 0.4, 0.3]}
    )
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    tasks = pd.DataFrame({"task_id": ["t"], "name": ["n"], "description": ["d"]})
    with patch(f"{NS_INIT}.fetch_neurosynth_peak_data", return_value=peaks), \
            patch(f"{NS_INIT}.get_ns_term_study_associations",
                  return_value=terms), \
            patch(f"{NS_INIT}.fetch_cortical_atlas", return_value=nib.Nifti1Image(
                np.ones((4, 4, 4), dtype=np.int16), np.eye(4))), \
            patch(f"{NS_INIT}.fetch_cognitive_atlas_tasks", return_value=tasks), \
            patch.object(engine_registry, "_resolve_mask", return_value=mask), \
            patch.object(engine_registry, "_phase_atlases",
                         lambda *a, **k: None):
        return engine_registry.build_engine("neurosynth", Path("unused"))


def run(nl, program):
    return set(nl.execute_datalog_program(program))


def test_labels_and_definitions(engine):
    assert run(
        engine,
        "ans(d) :- rdfs:label(c, 'Working memory'), skos:definition(p, d), "
        "rdfs:label(p, 'Memory')",
    ) == {("The retention of information.",)}


def test_classes_below_a_concept(engine):
    assert run(
        engine,
        "ans(l) :- rdfs:label(c, 'Memory'), cogat_subclass_of(x, c), "
        "rdfs:label(x, l)",
    ) == {("Working memory",), ("Spatial working memory",)}


def test_contrasts_measuring_a_concept(engine):
    assert run(
        engine,
        "ans(l) :- rdfs:label(c, 'Working memory'), cogat_measures(x, c), "
        "rdfs:label(x, l)",
    ) == {("Digit_span_task-backward",)}


def test_studies_mentioning_a_concept_or_its_subclasses(engine):
    studies = run(
        engine,
        "ans(s) :- rdfs:label(c, 'Memory'), cogat_subclass_of(x, c), "
        "cogat_term(x, t), mentions(s, t)",
    )
    assert studies == {(1,), (2,)}
    assert run(
        engine,
        "ans(s) :- rdfs:label(c, 'Working memory'), "
        "cogat_subclass_of(x, c), cogat_term(x, t), mentions(s, t)",
    ) == {(2,)}


def test_cogat_predicates_are_declared():
    predicates = engine_registry.get_predicates("neurosynth")
    assert predicates["cogat_subclass_of"]["columns"] == ["iri", "iri"]
