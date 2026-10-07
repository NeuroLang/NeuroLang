"""Tests for engine initialization modules."""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from neurolang.frontend.neurosynth_utils import StudyID


class TestNeurosynthEngineStudyIdConsistency:
    """Regression test: study_id dtype must be consistent across all tuple sets.

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
        self, mock_get_term, mock_get_peaks
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
        init_engine(nl, mask, data_dir="/tmp/fake")

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
