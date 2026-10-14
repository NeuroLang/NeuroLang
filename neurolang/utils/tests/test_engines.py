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

    @patch("neurolang.utils.engines.neurosynth.init.fetch_neurosynth_peak_data")
    @patch("neurolang.utils.engines.neurosynth.init.get_ns_term_study_associations")
    def test_study_id_consistency(
        self, mock_get_term, mock_get_peaks, tmp_path
    ):
        """All study_id columns should share the same integer dtype."""
        from neurolang.utils.engines.neurosynth.init import init_engine

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


class TestNeurosynthEngineVocabulary:
    """Check the vocabulary predicates the neurosynth engine defines."""

    @patch("neurolang.utils.engines.neurosynth.init.fetch_neurosynth_peak_data")
    @patch("neurolang.utils.engines.neurosynth.init.get_ns_term_study_associations")
    def test_term_relation_holds_distinct_terms(
        self, mock_get_term, mock_get_peaks
    ):
        """Check that ``term`` lists every TF-IDF term once."""
        import nibabel as nib

        from neurolang.utils.engines.neurosynth.init import init_engine

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
        from neurolang.utils import engine_registry

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
        from neurolang.utils import engine_registry

        datalog_init = engine_registry.get_engine_config("neurosynth")[
            "datalog_init"
        ]
        assert (
            "mentions(s, t) :- term_in_study_tfidf(s, t, v)" in datalog_init
        )
        assert "reports(s, i, j, k) :- peak_reported(i, j, k, s)" in datalog_init

    def test_weighted_mention_is_declared(self):
        """Check ``weighted_mention`` has the columns of the TF-IDF."""
        from neurolang.utils import engine_registry

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
        from neurolang.utils import engine_registry

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


def _fake_schaefer(nl, cfg, data_dir, name):
    """Register a one-parcel ``schaefer`` atlas, as ``_phase_atlases``."""
    from neurolang.regions import ExplicitVBR

    region = ExplicitVBR(
        np.array([[1, 1, 1], [1, 2, 1]]), np.eye(4), image_dim=(4, 4, 4)
    )
    nl.add_tuple_set(
        [("17Networks LH DefaultA PCC 1", region)], name="schaefer"
    )


@pytest.fixture(scope="module")
def small_neurosynth():
    """Build the neurosynth engine of the registry on tiny mock data."""
    import nibabel as nib

    from neurolang.utils import engine_registry

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
    init = "neurolang.utils.engines.neurosynth.init"
    with patch(f"{init}.fetch_neurosynth_peak_data", return_value=peaks), \
            patch(f"{init}.get_ns_term_study_associations",
                  return_value=terms), \
            patch.object(engine_registry, "_resolve_mask",
                         return_value=mask), \
            patch.object(engine_registry, "_phase_atlases", _fake_schaefer):
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
    from neurolang.expressions import Symbol

    stored = getattr(nl.program_ir.symbol_table[Symbol(name)], "value", None)
    if hasattr(stored, "unwrap"):
        return stored.unwrap().as_pandas_dataframe()
    return solution[name].as_pandas_dataframe()


def _column_type(nl, name, column):
    """Return the type of the *column* of the stored relation *name*."""
    from neurolang.expressions import Symbol

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

    def test_every_declaration_matches_its_symbol(self, small_neurosynth):
        """Check the column count, order and types of each predicate."""
        from neurolang.utils import engine_registry

        predicates = engine_registry.get_predicates("neurosynth")
        signatures = engine_registry.get_signatures("neurosynth")
        with small_neurosynth.scope:
            solution = small_neurosynth.solve_all()
        assert set(signatures) == set(predicates)
        for name, arguments in signatures.items():
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
        from neurolang.utils import engine_registry

        hidden = engine_registry.get_hidden_predicates("neurosynth")
        assert hidden == {"peak_reported", "schaefer_voxels"}
        assert hidden <= set(engine_registry.get_predicates("neurosynth"))

    def test_probability_declared_for_the_choice(self):
        """Check ``selected_study`` is the probabilistic predicate."""
        from neurolang.utils import engine_registry

        assert engine_registry.get_probabilistic_predicates(
            "neurosynth"
        ) == {"selected_study"}


class TestPredicateArguments:
    """Check the parsing and validation of the declared arguments."""

    def test_columns_and_arity_come_from_the_arguments(self):
        """Check a voxel is one entity of three columns."""
        from neurolang.utils import engine_registry

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
        from neurolang.utils import engine_registry

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
        from neurolang.utils import engine_registry

        path = _engine_yaml(
            tmp_path, "",
            "      q:\n        arguments:\n"
            "          - {sort: voxel, columns: [x, y], type: int}\n",
        )
        with pytest.raises(ValueError, match="voxel"):
            engine_registry.get_signatures("e", path)

    def test_destrieux_regions(self):
        """Check the destrieux predicates relate a label and a region."""
        from neurolang.utils import engine_registry

        signatures = engine_registry.get_signatures("destrieux")
        for name in ("destrieux", "left_region", "right_region"):
            assert [a.sort for a in signatures[name]] == ["label", "region"]
        assert engine_registry.get_predicates("destrieux")["destrieux"][
            "columns"
        ] == ["name", "region"]


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
