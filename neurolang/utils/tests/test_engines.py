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
    from neurolang.utils import engine_registry

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
    from neurolang.utils import engine_registry

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
