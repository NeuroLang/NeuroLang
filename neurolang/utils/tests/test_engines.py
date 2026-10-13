"""Tests for engine initialization modules."""

import numpy as np
import pandas as pd
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
