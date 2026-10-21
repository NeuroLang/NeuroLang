"""Tests for engine initialization modules."""

import numbers
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from neurolang.frontend.neurosynth_utils import StudyID


class TestNeurosynthEngineStudyIdConsistency:
    """
    Regression test: study_id dtype must be consistent across all tuple sets.

    Currently ``init_engine`` loads ``peak_data`` study IDs as ``int64``
    (via ``fetch_neurosynth_peak_data`` without ``convert_study_ids``) while
    ``term_in_study_tfidf`` study IDs are ``StudyID``/``str`` (via
    ``get_ns_term_study_associations`` with ``convert_study_ids=True``).

    This test asserts that all ``study_id`` columns share the same integer
    dtype.  It is expected to **fail** until the inconsistency is fixed.
    """

    @patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cognitive_atlas_tasks")
    @patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cortical_atlas")
    @patch("neurolang_apps.utils.engines.neurosynth.init.fetch_neurosynth_peak_data")
    @patch("neurolang_apps.utils.engines.neurosynth.init.get_ns_term_study_associations")
    def test_study_id_consistency(
        self, mock_get_term, mock_get_peaks, mock_atlas, mock_tasks,
        tmp_path
    ):
        """All study_id columns should share the same integer dtype."""
        from neurolang_apps.utils.engines.neurosynth.init import init_engine

        # ---- Mock peak data: study_id is int64 -------------------------------
        mock_peaks_df = pd.DataFrame({
            "id": [1, 2, 3],
            "x": [0.0, 10.0, 20.0],
            "y": [0.0, 10.0, 20.0],
            "z": [0.0, 10.0, 20.0],
            "space": ["MNI", "MNI", "MNI"],
        })
        mock_get_peaks.return_value = mock_peaks_df

        # ---- Mock term data: study_id is StudyID (integer alias) -----------
        mock_term_df = pd.DataFrame({
            "id": [StudyID(1), StudyID(2), StudyID(3)],
            "term": ["memory", "memory", "language"],
            "tfidf": [0.5, 0.3, 0.7],
        })
        mock_get_term.return_value = mock_term_df

        # ---- Minimal nibabel mask (identity affine) --------------------------
        import nibabel as nib

        mask = nib.Nifti1Image(
            np.ones((10, 10, 10), dtype=np.uint8),
            np.eye(4),
        )
        mock_atlas.return_value = nib.Nifti1Image(
            np.zeros((10, 10, 10), dtype=np.int16), np.eye(4)
        )

        # ---- Mock NeurolangPDL to capture add_tuple_set calls ---------------
        nl = MagicMock()
        init_engine(nl, mask, data_dir=str(tmp_path / "fake"))

        # Collect the DataFrames that were passed to add_tuple_set
        captured_dfs: dict[str, pd.DataFrame] = {}
        for call in nl.add_tuple_set.call_args_list:
            args, kwargs = call
            df = args[0] if args else kwargs["iterable"]
            name = kwargs.get("name")
            if name is not None:
                captured_dfs[name] = df

        # ---- Assertions -----------------------------------------------------
        study_id_dtypes: dict[str, np.dtype] = {}
        for name in ("peak_reported", "study", "term_in_study_tfidf"):
            df = captured_dfs[name]
            study_id_dtypes[name] = df["study_id"].dtype

        # All three must be the same dtype
        dtypes_set = set(study_id_dtypes.values())
        assert len(dtypes_set) == 1, (
            f"study_id dtypes differ: {study_id_dtypes}"
        )

        # And that dtype must be a numpy integer subtype
        for name, dtype in study_id_dtypes.items():
            assert np.issubdtype(dtype, np.integer), (
                f"{name}.study_id has non-integer dtype: {dtype}"
            )


@patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cognitive_atlas_tasks")
@patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cortical_atlas")
@patch("neurolang_apps.utils.engines.neurosynth.init.fetch_neurosynth_peak_data")
@patch("neurolang_apps.utils.engines.neurosynth.init.get_ns_term_study_associations")
def test_neurosynth_cortical_mask(
    mock_get_term, mock_get_peaks, mock_atlas, mock_tasks
):
    """``cortical_mask(i, j, k)`` holds the voxels of the engine's grid
    labelled by the cortical atlas, resampled to that grid: here an atlas
    shifted by two voxels along i, and larger than the grid."""
    import nibabel as nib

    from neurolang_apps.utils.engines.neurosynth.init import init_engine

    mock_get_peaks.return_value = pd.DataFrame({
        "id": [1], "x": [0.0], "y": [0.0], "z": [0.0], "space": ["MNI"],
    })
    mock_get_term.return_value = pd.DataFrame({
        "id": [StudyID(1)], "term": ["memory"], "tfidf": [0.5],
    })
    labels = np.zeros((14, 10, 10), dtype=np.int16)
    labels[3, 4, 5] = 7
    labels[5, 0, 9] = 2
    labels[13, 0, 0] = 1  # outside the grid of the engine
    affine = np.eye(4)
    affine[0, 3] = -2
    mock_atlas.return_value = nib.Nifti1Image(labels, affine)
    mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.uint8), np.eye(4))

    nl = MagicMock()
    init_engine(nl, mask, data_dir=Path("unused"))

    sets = {
        call.kwargs["name"]: call.args[0]
        for call in nl.add_tuple_set.call_args_list
    }
    assert {tuple(v) for v in sets["cortical_mask"]} == {(1, 4, 5), (3, 0, 9)}


_COGNITIVE_ATLAS_TASKS = pd.DataFrame({
    "task_id": ["tsk_1", "trm_2"],
    "name": ["stop signal task", "n-back task"],
    "description": ["Stop when a signal appears.", ""],
})


@patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cognitive_atlas_tasks")
@patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cortical_atlas")
@patch("neurolang_apps.utils.engines.neurosynth.init.fetch_neurosynth_peak_data")
@patch("neurolang_apps.utils.engines.neurosynth.init.get_ns_term_study_associations")
def test_neurosynth_cognitive_atlas_tasks(
    mock_get_term, mock_get_peaks, mock_atlas, mock_tasks
):
    """``cognitive_task(task_id, name, description)`` holds the tasks of
    the Cognitive Atlas."""
    import nibabel as nib

    from neurolang_apps.utils.engines.neurosynth.init import init_engine

    mock_get_peaks.return_value = pd.DataFrame({
        "id": [1], "x": [0.0], "y": [0.0], "z": [0.0], "space": ["MNI"],
    })
    mock_get_term.return_value = pd.DataFrame({
        "id": [StudyID(1)], "term": ["memory"], "tfidf": [0.5],
    })
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    mock_atlas.return_value = nib.Nifti1Image(
        np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)
    )
    mock_tasks.return_value = _COGNITIVE_ATLAS_TASKS

    nl = MagicMock()
    init_engine(nl, mask, data_dir=Path("unused"))

    sets = {
        call.kwargs["name"]: call.args[0]
        for call in nl.add_tuple_set.call_args_list
    }
    pd.testing.assert_frame_equal(
        sets["cognitive_task"], _COGNITIVE_ATLAS_TASKS
    )


def test_fetch_cognitive_atlas_tasks(tmp_path):
    """The tasks of the Cognitive Atlas API, with their names and
    descriptions stripped, a missing description empty, and the tasks
    without a name left out."""
    import json

    from neurolang_apps.utils.engines.neurosynth import init

    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([
        {"id": "tsk_1", "name": " stop signal task ",
         "definition_text": "Stop when a signal appears.\n"},
        {"id": "trm_2", "name": "n-back task", "definition_text": None},
        {"id": "trm_3", "name": "", "definition_text": "None"},
    ]))
    with patch.object(init, "fetch_files", return_value=[str(tasks)]) as fetch:
        result = init.fetch_cognitive_atlas_tasks(tmp_path)

    (directory, files), _ = fetch.call_args
    assert directory == str(tmp_path / "cognitive_atlas")
    assert files[0][1] == init.COGNITIVE_ATLAS_TASKS_URL
    pd.testing.assert_frame_equal(result, _COGNITIVE_ATLAS_TASKS)


GRAPH = """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .
ex:Animal a owl:Class .
ex:rex a ex:Dog ; ex:name "Rex" .
"""


def test_ontologies_load_local_files_with_options(tmp_path):
    """``ontologies`` loads local files together, with their options."""
    from neurolang_apps.utils import engine_registry

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "vocab.ttl").write_text(GRAPH)
    (tmp_path / "data" / "more.nt").write_text(
        '<http://example.org/rex> <http://example.org/age> "3" .\n'
    )
    config = tmp_path / "engines.yaml"
    config.write_text(
        "engines:\n"
        "  pets:\n"
        "    ontologies:\n"
        "      - urls: [data/vocab.ttl, data/more.nt]\n"
        "        individuals: true\n"
        "        names: prefixed\n"
    )
    nl = engine_registry.build_engine(
        "pets", tmp_path / "cache", yaml_path=config
    )
    animals = nl.execute_datalog_program(
        "ans(n, a) :- ex:Animal(x), ex:name(x, n), ex:age(x, a)"
    )
    assert set(animals) == {("Rex", "3")}


def test_sparql_builtins(tmp_path):
    """The SPARQL-like builtins apply to strings and numbers."""
    from neurolang_apps.utils import engine_registry

    config = tmp_path / "engines.yaml"
    config.write_text(
        "engines:\n"
        "  words:\n"
        "    builtins: [lcase, strlen, str_contains, regex, round, int,"
        " strstarts]\n"
    )
    nl = engine_registry.build_engine(
        "words", tmp_path / "cache", yaml_path=config
    )
    nl.add_tuple_set(
        [("a", "Russia", "12", 2.5), ("b", "France", "7", -2.5)],
        name="country",
    )
    result = nl.execute_datalog_program(
        "ans(x, l, n, q, r) :- country(x, c, s, v), (l == lcase(c)), "
        "(n == strlen(c)), (q == int(s)), (r == round(v))"
    )
    assert set(result) == {
        ("a", "russia", 6, 12, 3.0), ("b", "france", 6, 7, -3.0)
    }
    matched = nl.execute_datalog_program(
        "ans(x) :- country(x, c, s, v), str_contains(lcase(c), 'russ'), "
        "regex(c, '(?i)^RU'), strstarts(c, 'R')"
    )
    assert set(matched) == {("a",)}


class TestNeurosynthEngineVocabulary:
    """Check the vocabulary predicates the neurosynth engine defines."""

    @patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cognitive_atlas_tasks")
    @patch("neurolang_apps.utils.engines.neurosynth.init.fetch_cortical_atlas")
    @patch("neurolang_apps.utils.engines.neurosynth.init.fetch_neurosynth_peak_data")
    @patch("neurolang_apps.utils.engines.neurosynth.init.get_ns_term_study_associations")
    def test_term_relation_holds_distinct_terms(
        self, mock_get_term, mock_get_peaks, mock_atlas, mock_tasks
    ):
        """Check that ``term`` lists every TF-IDF term once."""
        import nibabel as nib

        from neurolang_apps.utils.engines.neurosynth.init import init_engine

        mock_get_peaks.return_value = pd.DataFrame({
            "id": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0],
            "z": [0.0, 1.0], "space": ["MNI", "MNI"],
        })
        mock_get_term.return_value = pd.DataFrame({
            "id": [StudyID(1), StudyID(2), StudyID(2)],
            "term": ["memory", "memory", "language"],
            "tfidf": [0.5, 0.3, 0.7],
        })
        mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
        mock_atlas.return_value = nib.Nifti1Image(
            np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)
        )
        nl = MagicMock()

        init_engine(nl, mask, data_dir="unused")

        added = {
            call.kwargs["name"]: (call.args[0] if call.args else None)
            for call in nl.add_tuple_set.call_args_list
        }
        assert list(added["term"].columns) == ["term"]
        assert sorted(added["term"]["term"]) == ["language", "memory"]

    def test_vocabulary_predicates_are_declared(self):
        """Check the YAML declares the vocabulary predicates' columns."""
        from neurolang_apps.utils import engine_registry

        predicates = engine_registry.get_predicates("neurosynth")
        expected = {
            "term": ["term"],
            "study": ["study_id"],
            "voxel": ["i", "j", "k"],
            "mentions": ["study_id", "term"],
            "reports": ["study_id", "i", "j", "k"],
        }
        for name, columns in expected.items():
            assert predicates[name]["columns"] == columns
            assert predicates[name]["arity"] == len(columns)
            assert predicates[name]["description"]

    def test_mentions_and_reports_match_their_declared_order(self):
        """Check the datalog_init rules agree with the declared columns."""
        from neurolang_apps.utils import engine_registry

        datalog_init = engine_registry.get_engine_config("neurosynth")[
            "datalog_init"
        ]
        assert (
            "mentions(s, t) :- term_in_study_tfidf(s, t, v)" in datalog_init
        )
        assert "reports(s, i, j, k) :- peak_reported(i, j, k, s)" in datalog_init

    def test_weighted_mention_is_declared(self):
        """Check ``weighted_mention`` has the columns of the TF-IDF."""
        from neurolang_apps.utils import engine_registry

        config = engine_registry.get_engine_config("neurosynth")
        info = engine_registry.get_predicates("neurosynth")["weighted_mention"]
        assert info["columns"] == ["study_id", "term", "tfidf"]
        assert info["arity"] == 3
        assert "Weighted_mention" in info["description"]
        assert (
            "weighted_mention(s, t, w) :- term_in_study_tfidf(s, t, w)"
            in config["datalog_init"]
        )
        assert "term_in_study_tfidf" in engine_registry.get_predicates(
            "neurosynth"
        )

    def test_weighted_mention_reads_in_squall(self):
        """Check a SQUALL sentence on ``weighted_mention`` gets its rows."""
        from neurolang.frontend import NeurolangPDL
        from neurolang_apps.utils import engine_registry

        datalog_init = engine_registry.get_engine_config("neurosynth")[
            "datalog_init"
        ]
        rule = next(
            line.strip() for line in datalog_init.splitlines()
            if line.strip().startswith("weighted_mention(")
        )
        nl = NeurolangPDL()
        nl.add_tuple_set(
            [(10, "pain", 0.5), (11, "pain", 0.05), (11, "fear", 0.4)],
            name="term_in_study_tfidf",
        )
        nl.execute_datalog_program(rule)
        result = nl.execute_squall_program(
            "obtain every Weighted_mention (?s; ?t; ?w) "
            "such that ?t is 'pain' and ?w is greater than 0.1."
        )
        rows = {
            tuple(row)
            for row in result.as_pandas_dataframe()[["s", "t", "w"]].values
        }
        assert rows == {(10, "pain", 0.5)}


_FAKE_ATLASES = {
    "schaefer": "17Networks LH DefaultA PCC 1",
    "destrieux": "L G temporal middle",
    "tian": "lAMY lh",
}


def _fake_schaefer(nl, cfg, data_dir, name):
    """
    Register one-region ``schaefer``, ``destrieux`` and ``tian`` atlases
    with the studies activating them, as ``_phase_atlases``.
    """
    import nibabel as nib

    from neurolang.regions import ExplicitVBR
    from neurolang_apps.utils import engine_registry

    voxels = np.array([[1, 1, 1], [1, 2, 1]])
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[tuple(voxels.T)] = 1
    image = nib.Nifti1Image(data, np.eye(4))
    for atlas, label in _FAKE_ATLASES.items():
        region = ExplicitVBR(voxels, np.eye(4), image_dim=(4, 4, 4))
        nl.add_tuple_set([(label, region)], name=atlas)
        engine_registry._register_atlas_activation(
            nl, {1: label}, image, {"radius_mm": 3}, atlas
        )


# The predicates of a free-text query: computed for the text a query gives
# as a constant, not for every text (``solve_all`` cannot).
FREE_TEXT_PREDICATES = frozenset({
    "study_similarity", "similar_study", "lift_llm", "relative_risk_llm",
    "uniformity_llm", "ratios_llm",
})


def _without_free_text_rules(config):
    """Return the engine *config* without the rules of a free-text query."""
    names = FREE_TEXT_PREDICATES | {
        "cutoff_similarity", "p_similar", "p_a_given_similar",
        "p_any_voxel_given_similar",
    }
    rules = [
        line for line in config["datalog_init"].splitlines()
        if line.split("(")[0] not in names
    ]
    return {**config, "datalog_init": "\n".join(rules)}


@pytest.fixture(scope="module")
def small_neurosynth():
    """Build the neurosynth engine of the registry on tiny mock data."""
    return _build_small_neurosynth()


def _build_small_neurosynth(free_text=True):
    """Build the neurosynth engine on tiny data, with the free-text rules
    of its configuration or without."""
    import nibabel as nib

    from neurolang_apps.utils import engine_registry

    peaks = pd.DataFrame({
        "id": [1001, 1001, 1002], "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0],
        "space": ["MNI", "MNI", "MNI"],
    })
    terms = pd.DataFrame({
        "id": [StudyID(1001), StudyID(1002), StudyID(1002)],
        "term": ["memory", "memory", "language"],
        "tfidf": [0.5, 0.3, 0.02],
    })
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4))
    cortex = np.zeros((4, 4, 4), dtype=np.int16)
    cortex[1, 2, 3] = 1
    init = "neurolang_apps.utils.engines.neurosynth.init"
    with patch(f"{init}.fetch_neurosynth_peak_data", return_value=peaks), \
            patch(f"{init}.get_ns_term_study_associations",
                  return_value=terms), \
            patch(f"{init}.fetch_cortical_atlas",
                  return_value=nib.Nifti1Image(cortex, np.eye(4))), \
            patch(f"{init}.fetch_cognitive_atlas_tasks",
                  return_value=_COGNITIVE_ATLAS_TASKS), \
            patch.object(engine_registry, "_resolve_mask",
                         return_value=mask), \
            patch.object(engine_registry, "_phase_atlases", _fake_schaefer):
        if free_text:
            return engine_registry.build_engine("neurosynth", Path("unused"))
        config = engine_registry.get_engine_config

        def get_config(name, yaml_path=None):
            return _without_free_text_rules(config(name, yaml_path))

        with patch.object(engine_registry, "get_engine_config", get_config):
            return engine_registry.build_engine("neurosynth", Path("unused"))


_TYPES = {
    "int": numbers.Integral, "float": float, "str": str,
}


def _relation(nl, solution, name):
    """
    Return the rows of the relation *name* of *nl*, as a DataFrame.

    A stored relation is read from the engine, which keeps its regions;
    a derived one from the *solution*.
    """
    from nl_expr.expressions import Symbol

    stored = getattr(nl.program_ir.symbol_table[Symbol(name)], "value", None)
    if hasattr(stored, "unwrap"):
        return stored.unwrap().as_pandas_dataframe()
    return solution[name].as_pandas_dataframe()


def _column_type(nl, name, column):
    """Return the type of the *column* of the stored relation *name*."""
    from nl_expr.expressions import Symbol

    row_type = nl.program_ir.symbol_table[Symbol(name)].type.__args__[0]
    return row_type.__args__[column]


def _column_values(frame, start, width):
    """Return the set of rows of *width* columns of *frame* from *start*."""
    columns = frame.iloc[:, start:start + width]
    return set(columns.itertuples(index=False, name=None))


# The relation holding every entity of a sort.
_NOUNS = {
    "study": "study", "term": "term", "voxel": "voxel",
    "label": "schaefer_label",
}


def _check_argument(nl, solution, name, argument, start):
    """
    Check the *argument* of *name* from its column *start*.

    Its values have the declared type and, for a sort of which a
    relation holds every entity, are entities of that relation.
    """
    from neurolang.regions import ExplicitVBR

    width = len(argument.columns)
    values = _column_values(_relation(nl, solution, name), start, width)
    if argument.type in _TYPES:
        assert all(
            isinstance(value, _TYPES[argument.type])
            for row in values for value in row
        ), (name, argument)
    else:
        assert _column_type(nl, name, start) is ExplicitVBR, name
    noun = _NOUNS.get(argument.sort, name)
    if noun != name:
        known = _column_values(_relation(nl, solution, noun), 0, width)
        assert values <= known, (name, argument)


class TestNeurosynthDeclarations:
    """Check the declared arguments against the symbols of the engine."""

    def test_every_declaration_matches_its_symbol(self):
        """Check the column count, order and types of each predicate."""
        from neurolang_apps.utils import engine_registry

        small_neurosynth = _build_small_neurosynth(free_text=False)
        predicates = engine_registry.get_predicates("neurosynth")
        signatures = engine_registry.get_signatures("neurosynth")
        with small_neurosynth.scope:
            solution = small_neurosynth.solve_all()
        assert set(signatures) == set(predicates)
        for name, arguments in signatures.items():
            if name in FREE_TEXT_PREDICATES:
                continue
            frame = _relation(small_neurosynth, solution, name)
            probability = int(bool(predicates[name].get("probability")))
            assert frame.shape[1] == predicates[name]["arity"] + probability
            assert len(frame) > 0, name
            start = probability
            for argument in arguments:
                _check_argument(
                    small_neurosynth, solution, name, argument, start
                )
                start += len(argument.columns)

    def test_hidden_predicates_are_declared(self):
        """Check the hidden predicates are declared predicates."""
        from neurolang_apps.utils import engine_registry

        hidden = engine_registry.get_hidden_predicates("neurosynth")
        assert hidden == {"peak_reported", "schaefer_voxels"}
        assert hidden <= set(engine_registry.get_predicates("neurosynth"))

    def test_probability_declared_for_the_choice(self):
        """Check the choices over the studies and the voxels are the
        probabilistic predicates."""
        from neurolang_apps.utils import engine_registry

        assert engine_registry.get_probabilistic_predicates(
            "neurosynth"
        ) == {"selected_study", "selected_voxel"}


class TestPredicateArguments:
    """Check the parsing and validation of the declared arguments."""

    def test_columns_and_arity_come_from_the_arguments(self):
        """Check a voxel is one entity of three columns."""
        from neurolang_apps.utils import engine_registry

        info = engine_registry.get_predicates("neurosynth")["reports"]
        assert info["columns"] == ["study_id", "i", "j", "k"]
        assert info["arity"] == 4
        assert info["entities"] == 2
        reports = engine_registry.get_signatures("neurosynth")["reports"]
        assert [argument.sort for argument in reports] == ["study", "voxel"]
        assert reports[1].columns == ("i", "j", "k")
        assert reports[1].type == "int"

    def test_declared_columns_must_agree(self, tmp_path):
        """Check columns contradicting the arguments are rejected."""
        from neurolang_apps.utils import engine_registry

        path = _engine_yaml(tmp_path, "columns: [a, b]")
        with pytest.raises(ValueError, match="columns"):
            engine_registry.get_predicates("e", path)
        path = _engine_yaml(tmp_path, "arity: 3")
        with pytest.raises(ValueError, match="arity"):
            engine_registry.get_predicates("e", path)
        path = _engine_yaml(tmp_path, "columns: [s, i, j, k]")
        assert engine_registry.get_predicates("e", path)["p"]["arity"] == 4

    def test_a_sort_spans_the_same_columns_everywhere(self, tmp_path):
        """Check a sort of different widths is rejected."""
        from neurolang_apps.utils import engine_registry

        path = _engine_yaml(
            tmp_path, "",
            "      q:\n        arguments:\n"
            "          - {sort: voxel, columns: [x, y], type: int}\n",
        )
        with pytest.raises(ValueError, match="voxel"):
            engine_registry.get_signatures("e", path)

    def test_destrieux_regions(self):
        """Check the destrieux predicates relate a label and a region."""
        from neurolang_apps.utils import engine_registry

        signatures = engine_registry.get_signatures("destrieux")
        for name in ("destrieux", "left_region", "right_region"):
            assert [a.sort for a in signatures[name]] == ["label", "region"]
        assert engine_registry.get_predicates("destrieux")["destrieux"][
            "columns"
        ] == ["name", "region"]


_SORTED_ENGINE = (
    "engines:\n  e:\n"
    "    sorts:\n"
    "      study: {columns: [study_id], type: int, description: a study}\n"
    "      voxel:\n"
    "        columns: [i, j, k]\n"
    "        type: int\n"
    "        description: a voxel of the MNI template grid, in 3D\n"
    "    predicates:\n"
    "      reports:\n"
    "        arguments: [study, {sort: voxel}]\n"
    "      peak:\n"
    "        arguments:\n"
    "          - {sort: voxel, columns: [x, y, z]}\n"
    "          - {sort: study, columns: [s], type: int}\n"
)


class TestDeclaredSorts:
    """Check the sorts an engine declares once, by name."""

    def test_sorts_are_read(self, tmp_path):
        """Check each sort keeps its columns, type and description."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(_SORTED_ENGINE)
        sorts = engine_registry.get_sorts("e", path)
        assert sorts["voxel"] == engine_registry.Sort(
            "voxel", ("i", "j", "k"), "int",
            "a voxel of the MNI template grid, in 3D",
        )
        assert sorts["study"].columns == ("study_id",)

    def test_an_argument_names_its_sort(self, tmp_path):
        """Check a reference to a sort takes its columns and type."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(_SORTED_ENGINE)
        info = engine_registry.get_predicates("e", path)["reports"]
        assert info["columns"] == ["study_id", "i", "j", "k"]
        assert info["arity"] == 4 and info["entities"] == 2
        reports = engine_registry.get_signatures("e", path)["reports"]
        assert reports == (
            engine_registry.Argument("study", ("study_id",), "int"),
            engine_registry.Argument("voxel", ("i", "j", "k"), "int"),
        )

    def test_an_argument_may_rename_the_columns(self, tmp_path):
        """Check a predicate may name the columns of a sort its own way."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(_SORTED_ENGINE)
        peak = engine_registry.get_signatures("e", path)["peak"]
        assert peak[0] == engine_registry.Argument(
            "voxel", ("x", "y", "z"), "int"
        )
        assert peak[1].columns == ("s",)

    def test_a_reference_names_a_declared_sort(self, tmp_path):
        """Check a reference to an undeclared sort is rejected."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(
            _SORTED_ENGINE.replace("[study, {sort: voxel}]", "[study, place]")
        )
        with pytest.raises(ValueError, match="place"):
            engine_registry.get_predicates("e", path)

    def test_renamed_columns_keep_the_width_of_the_sort(self, tmp_path):
        """Check renamed columns must span as many columns as the sort."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(_SORTED_ENGINE.replace("[x, y, z]", "[x, y]"))
        with pytest.raises(ValueError, match="voxel"):
            engine_registry.get_signatures("e", path)

    @pytest.mark.parametrize("engine", ["neurosynth", "destrieux"])
    def test_the_engines_reference_their_sorts(self, engine):
        """Check the shipped engines declare their arguments by sort."""
        from neurolang_apps.utils import engine_registry

        config = engine_registry.get_engine_config(engine)
        sorts = engine_registry.get_sorts(engine)
        for name, info in config["predicates"].items():
            for argument in info["arguments"]:
                assert isinstance(argument, str), (name, argument)
                assert argument in sorts, (name, argument)

    def test_neurosynth_sorts(self):
        """Check the voxel grids of neurosynth are 3D sorts."""
        from neurolang_apps.utils import engine_registry

        sorts = engine_registry.get_sorts("neurosynth")
        assert sorts["voxel"].columns == ("i", "j", "k")
        assert sorts["atlas_voxel"].columns == ("i", "j", "k")
        assert "3D" in sorts["voxel"].description
        assert sorts["study"] == engine_registry.Sort(
            "study", ("study_id",), "int", sorts["study"].description
        )

    def test_cognitive_task_is_declared(self):
        """Check ``cognitive_task`` relates a task, its name and its
        description."""
        from neurolang_apps.utils import engine_registry

        predicates = engine_registry.get_predicates("neurosynth")
        assert predicates["cognitive_task"]["columns"] == [
            "task_id", "name", "description"
        ]


def _engine_yaml(tmp_path, extra: str, more: str = "") -> Path:
    """Write an engine ``e`` declaring ``p(s, i, j, k)`` with *extra*."""
    path = tmp_path / "engines.yaml"
    path.write_text(
        "engines:\n  e:\n    predicates:\n      p:\n"
        + (f"        {extra}\n" if extra else "")
        + "        arguments:\n"
        "          - {sort: study, columns: [s], type: int}\n"
        "          - {sort: voxel, columns: [i, j, k], type: int}\n"
        + more
    )
    return path


class TestSortSpaces:
    """Check the sorts declaring the space of their coordinates."""

    def test_space_is_read(self, tmp_path):
        """Check a sort keeps its declared space, empty by default."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(
            "engines:\n  e:\n    sorts:\n"
            "      voxel: {columns: [i, j, k], type: int, space: voxel}\n"
            "      mni: {columns: [x, y, z], type: float, space: mm}\n"
            "      study: {columns: [s], type: int}\n"
        )
        sorts = engine_registry.get_sorts("e", path)
        assert sorts["voxel"].space == "voxel"
        assert sorts["mni"].space == "mm"
        assert sorts["study"].space == ""

    def test_unknown_space_is_rejected(self, tmp_path):
        """Check a space is ``voxel`` or ``mm``."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(
            "engines:\n  e:\n    sorts:\n"
            "      voxel: {columns: [i, j, k], type: int, space: grid}\n"
        )
        with pytest.raises(ValueError, match="grid"):
            engine_registry.get_sorts("e", path)

    def test_neurosynth_spaces(self):
        """Check neurosynth declares its template grid and MNI millimetres."""
        from neurolang_apps.utils import engine_registry

        sorts = engine_registry.get_sorts("neurosynth")
        assert sorts["voxel"].space == "voxel"
        assert sorts["mni"] == engine_registry.Sort(
            "mni", ("x", "y", "z"), "float", sorts["mni"].description, "mm"
        )
        assert sorts["voxel"].grid == ""
        # The Schaefer atlas indexes its own grid, not the template's.
        assert sorts["atlas_voxel"].space == "voxel"
        assert sorts["atlas_voxel"].grid == "schaefer"

    @pytest.mark.parametrize("grid", ["template", "schaefer"])
    def test_grid_is_read(self, tmp_path, grid):
        """Check a sort of voxels names the grid it indexes."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(
            "engines:\n  e:\n    atlases:\n      schaefer: {n_rois: 100}\n"
            "    sorts:\n"
            "      v: {columns: [i, j, k], type: int, space: voxel, "
            f"grid: {grid}}}\n"
        )
        assert engine_registry.get_sorts("e", path)["v"].grid == grid

    def test_unknown_grid_is_rejected(self, tmp_path):
        """Check a grid is the template's or one of the engine's atlases."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(
            "engines:\n  e:\n    atlases:\n      schaefer: {n_rois: 100}\n"
            "    sorts:\n"
            "      v: {columns: [i, j, k], type: int, space: voxel, "
            "grid: nowhere}\n"
        )
        with pytest.raises(ValueError, match="nowhere") as info:
            engine_registry.get_sorts("e", path)
        assert "template, schaefer" in str(info.value)
        with pytest.raises(ValueError, match="nowhere"):
            engine_registry.get_predicates("e", path)

    def test_grid_of_a_sort_not_of_voxels_is_rejected(self, tmp_path):
        """Check only a sort of ``space: voxel`` names a grid."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text(
            "engines:\n  e:\n    sorts:\n"
            "      p: {columns: [x, y, z], type: float, space: mm, "
            "grid: template}\n"
        )
        with pytest.raises(ValueError, match="space: voxel"):
            engine_registry.get_sorts("e", path)


class TestEngineTemplate:
    """Check an engine keeps the template its voxel indices refer to."""

    def test_the_template_is_the_mask(self, small_neurosynth):
        """Check the voxel grid of neurosynth is the MNI mask's."""
        template = small_neurosynth.template
        assert template.shape == (4, 4, 4)
        assert np.array_equal(template.affine, np.eye(4))

    def test_no_template_without_a_mask(self, tmp_path):
        """Check an engine not requiring the MNI mask has no template."""
        from neurolang_apps.utils import engine_registry

        path = tmp_path / "engines.yaml"
        path.write_text("engines:\n  e:\n    description: bare\n")
        nl = engine_registry.build_engine("e", tmp_path, yaml_path=path)
        assert nl.template is None
        assert nl.grids == {}

    def test_the_template_is_a_grid(self, small_neurosynth):
        """Check the template is the engine's ``template`` grid."""
        assert small_neurosynth.grids["template"] is (
            small_neurosynth.template
        )

    def test_an_atlas_is_a_grid(self, tmp_path, monkeypatch):
        """Check loading an atlas registers its image as a grid."""
        import nibabel as nib

        from neurolang_apps.utils import engine_registry

        data = np.zeros((3, 5, 2), dtype=np.int16)
        data[2, 4, 1], data[0, 0, 0] = 1, 2
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        nib.save(nib.Nifti1Image(data, affine), tmp_path / "atlas.nii")
        monkeypatch.setitem(
            engine_registry._ATLAS_REGISTRY, "schaefer", {
                "fetch": lambda **kw: {
                    "maps": str(tmp_path / "atlas.nii"),
                    "labels": ["Background", "A", "B"],
                },
                "probabilistic": False,
            },
        )
        path = tmp_path / "engines.yaml"
        path.write_text(
            "engines:\n  e:\n    atlases:\n"
            "      schaefer: {predicate_name: parcels}\n"
        )
        nl = engine_registry.build_engine("e", tmp_path, yaml_path=path)
        grid = nl.grids["schaefer"]
        assert grid.shape == (3, 5, 2)
        assert np.array_equal(grid.affine, affine)
