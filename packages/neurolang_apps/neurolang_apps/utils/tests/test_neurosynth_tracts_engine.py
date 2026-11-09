"""The engine ``neurosynth_tracts`` of engines.yaml: neurosynth and tracts,
on fake data (no download), and its example, uncinate_cogat.dl of
the engine tracts."""

import warnings
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurolang.atlases import label_name
from neurolang.frontend.neurosynth_utils import StudyID
from neurolang_apps.utils import engine_registry

from .test_engines import _COGNITIVE_ATLAS_TASKS, FAKE_METADATA
from .test_tracts_engine import FAKE_LABELS

NS_INIT = "neurolang_apps.utils.engines.neurosynth.init"
TRACTS = "neurolang_apps.utils.engines.tracts.frontend"
EXAMPLE = (Path(__file__).parents[1] / "engines" / "tracts" / "examples"
           / "uncinate_cogat.dl")


def test_the_engine_is_neurosynth_and_tracts():
    cfg = engine_registry.get_engine_config("neurosynth_tracts")
    # both inits, neurosynth's first, given the options of neurosynth
    assert cfg["python_init"] == [NS_INIT, TRACTS]
    assert cfg["python_init_options"] == {"unknown_space": "mni"}
    assert cfg["solver_mixins"] == [f"{TRACTS}:TractsCommandsMixin"]
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


# The frontal end of one more left uncinate, the only one near a parcel
# of its own (3 of FAKE_LABELS), and the voxel of that parcel.
LONE_END = (-20.0, 48.0, -4.0)
LONE_VOXEL = (20, 74, 18)


@pytest.fixture
def files(tmp_path):
    """The atlas and colour LUT of the uncinate's tests of the engine, and
    their streamlines with 21 more left uncinates (positions 16 to 36): 20
    copies of the designed one, shifted by less than 1 mm, and one whose
    frontal end is LONE_END."""
    pytest.importorskip("neurolang_apps.utils.engines.tracts")
    from neurolang_apps.utils.engines.tracts.tests import uncinate_files

    uncinate_files.write_files(tmp_path)
    temporal, insula, frontal = uncinate_files.DESIGNED["uncinate"][0]
    extra = [
        [temporal, insula, (frontal[0] + 0.04 * k,) + frontal[1:]]
        for k in range(20)
    ] + [[temporal, insula, LONE_END]]
    nib.streamlines.save(
        nib.streamlines.Tractogram(
            uncinate_files.streamlines()
            + [uncinate_files.polyline(w) for w in extra],
            affine_to_rasmm=np.eye(4),
        ),
        str(tmp_path / "tracts.trk"),
    )
    return uncinate_files


def expected(files, side):
    """The uncinates of a side, the designed ones and the extra ones."""
    return files.expected(side) | (
        set(range(16, 37)) if side == "left" else set())


@pytest.fixture
def engine(tmp_path, files, monkeypatch):
    """The engine on three studies, 1001 with a peak in front of the left
    uncinate, 1002 of the right one and 1003 at LONE_END, and atlases of the
    space of the uncinate's files whose regions are its hemispheres, 1 the
    left one, but for the voxel of region 3, 6 mm away from region 1."""
    volume = np.zeros(files.SHAPE, dtype=np.int16)
    volume[:files.SHAPE[0] // 2] = 1
    volume[files.SHAPE[0] // 2 + 1:] = 2
    i, j, k = LONE_VOXEL
    volume[i - 3:i + 4, j - 3:j + 4, k - 3:k + 4] = 0
    volume[LONE_VOXEL] = 3
    path = tmp_path / "hemispheres.nii.gz"
    nib.save(nib.Nifti1Image(volume, files.AFFINE), path)
    for name, labels in FAKE_LABELS.items():
        monkeypatch.setitem(
            engine_registry._ATLAS_REGISTRY[name], "fetch",
            lambda labels=labels, **kwargs: {
                "maps": str(path), "labels": list(labels[:4])},
        )
    peaks = pd.DataFrame({
        "id": [1001, 1002, 1003], "x": [-20.0, 20.0, LONE_END[0]],
        "y": [40.0, 40.0, LONE_END[1]], "z": [-5.0, -5.0, LONE_END[2]],
        "space": ["MNI", "MNI", "MNI"],
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
    from ..engines.tracts import vocabulary_predicates as predicates
    from ..engines.tracts.frontend import TractsCommandsMixin

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
    hemisphere, which the study of that side activates. On the left, the
    parcel 3 near 1 of the 46 ends, fewer than 5 %, is not one of the
    uncinate's, nor is 1003, which activates it only."""
    def query_similarity(q: str, s: int) -> float:
        return 1.
    engine.add_symbol(query_similarity, name="query_similarity")
    program = example(tmp_path, side)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # magic sets bind the definitions of similar_study, though the
        # vocabulary of the uncinate is not rewritten whole
        result = engine.execute_datalog_program(program)
    assert result.columns == ("t", "p")
    # the pairs of points within 3 mm are computed with a tree
    assert not [w for w in caught if "euclidean" in str(w.message)]
    assert {x for x, in rows(engine.execute_datalog_program(
        f'ans(x) :- uncinate(x, "{side}")'))} == expected(files, side)
    names = [label_name(n) for n in FAKE_LABELS["schaefer"]]
    parcel = names[1 if side == "left" else 2]
    ends = rows(engine.execute_datalog_program(
        "ans(n, c) :- uf_parcel_ends(n, c)"))
    assert ends == ({(parcel, 45), (names[3], 1)} if side == "left" else
                    {(parcel, 4)})
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
