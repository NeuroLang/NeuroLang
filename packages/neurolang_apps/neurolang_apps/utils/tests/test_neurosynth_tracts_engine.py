"""The engine ``neurosynth_tracts`` of engines.yaml: neurosynth and tracts,
on fake data (no download), and its example, uncinate_cogat.dl of
nl_tracts."""

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurolang.frontend.neurosynth_utils import StudyID
from neurolang_apps.utils import engine_registry

from .test_engines import _COGNITIVE_ATLAS_TASKS, FAKE_METADATA
from .test_tracts_engine import FAKE_LABELS

NS_INIT = "neurolang_apps.utils.engines.neurosynth.init"
EXAMPLE = Path(__file__).parents[4] / "nl_tracts" / "examples" / (
    "uncinate_cogat.dl")


def test_the_engine_is_neurosynth_and_tracts():
    cfg = engine_registry.get_engine_config("neurosynth_tracts")
    # both inits, neurosynth's first, given the options of neurosynth
    assert cfg["python_init"] == [NS_INIT, "nl_tracts.frontend"]
    assert cfg["python_init_options"] == {"unknown_space": "mni"}
    assert cfg["solver_mixins"] == ["nl_tracts.frontend:TractsCommandsMixin"]
    # the atlases keep the studies activating them and get their points
    schaefer = cfg["atlases"]["schaefer"]
    assert schaefer["activated"]["predicate_name"] == "schaefer_activated"
    assert schaefer["points"]["predicate_name"] == "schaefer_points"
    # mni_atlases, a base of both, is merged once
    assert cfg["datalog_init"].count(
        "schaefer_label(l) :- schaefer(l, r)") == 1
    assert {"peak_reported", "schaefer_voxels"} <= set(
        cfg["hidden_predicates"])
    assert cfg["requires_mni_mask"]
    assert "study_abstracts" in cfg["text_embeddings"]
    predicates = engine_registry.get_predicates("neurosynth_tracts")
    assert {"peak_reported", "term_in_study_tfidf", "schaefer_activated",
            "schaefer_points"} <= set(predicates)


@pytest.fixture
def files(tmp_path):
    """The atlas, colour LUT and tractogram of the uncinate's tests of
    nl_tracts."""
    pytest.importorskip("nl_tracts")
    from nl_tracts.tests import uncinate_files

    uncinate_files.write_files(tmp_path)
    return uncinate_files


@pytest.fixture
def engine(tmp_path, files, monkeypatch):
    """The engine on two studies, 1001 with a peak in front of the left
    uncinate, 1002 of the right one, and atlases of the space of the
    uncinate's files whose regions are its hemispheres, 1 the left one."""
    volume = np.zeros(files.SHAPE, dtype=np.int16)
    volume[:files.SHAPE[0] // 2] = 1
    volume[files.SHAPE[0] // 2 + 1:] = 2
    path = tmp_path / "hemispheres.nii.gz"
    nib.save(nib.Nifti1Image(volume, files.AFFINE), path)
    for name, labels in FAKE_LABELS.items():
        monkeypatch.setitem(
            engine_registry._ATLAS_REGISTRY[name], "fetch",
            lambda labels=labels, **kwargs: {
                "maps": str(path), "labels": list(labels[:3])},
        )
    peaks = pd.DataFrame({
        "id": [1001, 1002], "x": [-20.0, 20.0], "y": [40.0, 40.0],
        "z": [-5.0, -5.0], "space": ["MNI", "MNI"],
    })
    terms = pd.DataFrame({
        "id": [StudyID(1001), StudyID(1002)],
        "term": ["memory", "language"], "tfidf": [0.5, 0.3],
    })
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    with patch(f"{NS_INIT}.fetch_neurosynth_peak_data", return_value=peaks), \
            patch(f"{NS_INIT}.get_ns_term_study_associations",
                  return_value=terms), \
            patch(f"{NS_INIT}.fetch_cortical_atlas",
                  return_value=nib.Nifti1Image(
                      np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))), \
            patch(f"{NS_INIT}.fetch_cognitive_atlas_tasks",
                  return_value=_COGNITIVE_ATLAS_TASKS), \
            patch(f"{NS_INIT}.fetch_study_metadata",
                  return_value=FAKE_METADATA), \
            patch.object(engine_registry, "_resolve_mask",
                         return_value=mask):
        yield engine_registry.build_engine(
            "neurosynth_tracts", tmp_path / "data")


def rows(result):
    frame = result.as_pandas_dataframe()
    return set(frame.itertuples(index=False, name=None))


def test_the_engine_has_both(engine):
    from nl_tracts.frontend import TractsCommandsMixin
    from nl_tracts.vocabulary import predicates

    assert isinstance(engine.program_ir, TractsCommandsMixin)
    symbols = set(engine.symbols)
    # neurosynth's relations and rules, the atlases' studies and points,
    # and the vocabulary of tractography
    assert {"peak_reported", "term_in_study", "similar_study",
            "schaefer_activated", "schaefer_points", "cogat_term"} <= symbols
    assert set(predicates()) <= symbols


def example(directory, side="left"):
    """uncinate_cogat.dl, reading the files of *directory*, for *side*."""
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE} not found (an installed package)")
    text = EXAMPLE.read_text()
    for name, path in (
        ('"wmparc.mgz"', directory / "labels.nii.gz"),
        ('"FreeSurferColorLUT.txt"', directory / "lut.txt"),
        ('"whole_brain_MNI.trk"', directory / "tracts.trk"),
    ):
        assert name in text
        text = text.replace(name, f'"{path}"')
    return text.replace('uncinate(x, "left")', f'uncinate(x, "{side}")')


@pytest.mark.parametrize("side, study", [("left", 1001), ("right", 1002)])
def test_the_example_runs_to_the_studies(engine, files, tmp_path, side,
                                         study):
    """The example, its texts compared by a fake similarity (SPECTER2 is
    not there). Indexed by the Schaefer atlas too, which has no terms, the
    uncinate is that of uncinate.dl; its ends are in the parcel of its
    hemisphere, which the study of that side activates."""
    def query_similarity(q: str, s: int) -> float:
        return 1.
    engine.add_symbol(query_similarity, name="query_similarity")
    program = example(tmp_path, side)
    # magic sets bind the definitions of similar_study, though the
    # vocabulary of the uncinate is not rewritten whole
    assert engine.execute_datalog_program(program).columns == ("t", "p")
    assert {x for x, in rows(engine.execute_datalog_program(
        f'ans(x) :- uncinate(x, "{side}")'))} == files.expected(side)
    parcel = engine_registry._label_name(
        FAKE_LABELS["schaefer"][1 if side == "left" else 2])
    assert rows(engine.execute_datalog_program(
        "ans(n) :- uf_parcel(n)")) == {(parcel,)}
    assert rows(engine.execute_datalog_program(
        "ans(s) :- uf_parcel(n), schaefer_activated(n, s)")) == {(study,)}


def test_the_example_parses():
    from neurolang.frontend.datalog.standard_syntax import parser

    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE} not found (an installed package)")
    program = parser(EXAMPLE.read_text())
    heads = {
        f.consequent.functor.name for f in program.formulas
        if hasattr(f, "consequent")
    }
    assert {"uncinate", "uf_parcel", "uf_study", "cogat_concept",
            "concept_definition", "about", "p_concept_given_uf"} <= heads


def test_the_example_is_a_question_of_the_assistant():
    from neurolang.frontend.datalog.standard_syntax import parser
    from neurolang_apps.utils.llm.examples import load_examples

    bank = load_examples("neurosynth_tracts")
    assert bank[:-1] == load_examples("neurosynth")
    question = bank[-1]
    assert "uncinate" in question.question
    parser(question.queries["datalog"])
