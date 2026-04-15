from pathlib import Path
from typing import AbstractSet, Tuple
from unittest.mock import patch

import pandas as pd
import pytest

from .. import NeurolangDL
from ..neurosynth_utils import StudyID


@pytest.fixture
def metadata():
    return pd.DataFrame(
        [
            (
                9065511,
                "MNI",
            ),
            (
                9084599,
                "MNI",
            ),
            (
                9114263,
                "MNI",
            ),
            (
                9185551,
                "TAL",
            ),
            (
                9256495,
                "TAL",
            ),
        ],
        columns=["id", "space"],
    )


@pytest.fixture
def features():
    df = pd.DataFrame(
        [
            (9065511, 0.0, 0.0, 0.0, 0.079103, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (9084599, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (9114263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (
                9185551,
                0.0,
                0.079103,
                0.0,
                0.0,
                0.0,
                0.0,
                0.124442,
                0.0,
                0.0,
                0.0,
            ),
            (9256495, 0.0, 0.0, 0.0, 0.0, 0.219273, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
        columns=[
            "id",
            "scanner",
            "periodicals",
            "composed",
            "adhd",
            "tms",
            "space",
            "stimulus response",
            "hemisphere",
            "neurofunctional",
            "youth",
        ],
    )
    df = df.set_index("id")
    return df


@pytest.fixture
def peak_data():
    df = pd.DataFrame(
        [
            (9114263, 27003, "1.", 517253, -38.0, -38.0, 69.0, "MNI"),
            (9256495, 40779, "1", 771542, 55.0, 28.0, 9.0, "TAL"),
            (9114263, 27003, "1.", 517461, -11.0, -20.0, 62.0, "MNI"),
            (9114263, 27003, "1.", 517255, -29.0, -29.0, 64.0, "MNI"),
            (9256495, 40779, "1", 771532, -14.0, -8.0, -18.0, "TAL"),
            (9114263, 27003, "1.", 517298, -18.0, -58.0, 62.0, "MNI"),
            (9114263, 27003, "1.", 517349, -28.0, -8.0, 64.0, "MNI"),
            (9114263, 27003, "1.", 517445, -63.0, -41.0, 38.0, "MNI"),
            (9114263, 27003, "1.", 517443, -63.0, -29.0, 31.0, "MNI"),
            (9114263, 27003, "1.", 517272, -5.0, -48.0, 52.0, "MNI"),
            (9114263, 27003, "1.", 517552, -2.0, -30.0, 36.0, "MNI"),
            (9084599, 27000, "4.", 517181, 12.0, 10.0, 16.0, "MNI"),
            (9256495, 40779, "1", 771581, 14.0, -56.0, -7.0, "TAL"),
            (9256495, 40779, "1", 771588, -20.0, -64.0, 9.0, "TAL"),
            (9114263, 27003, "1.", 517416, -39.0, -34.0, 18.0, "MNI"),
        ],
        columns=[
            "id",
            "table_id",
            "table_num",
            "peak_id",
            "x",
            "y",
            "z",
            "space",
        ],
    )
    return df


@pytest.fixture
def mni_peaks(peak_data):
    df = peak_data[["id", "x", "y", "z"]]
    return df.astype({"x": int, "y": int, "z": int})


@pytest.fixture
def term_data(features):
    term_data = pd.melt(
        features.reset_index(),
        var_name="term",
        id_vars="id",
        value_name="tfidf",
    )
    return term_data


@patch("neurolang.frontend.query_resolution.fetch_study_metadata")
def test_load_neurosynth_study_ids(mock_fetch_study_metadata, metadata):
    mock_fetch_study_metadata.return_value = metadata
    data_dir = Path("mock_data_dir")

    nl = NeurolangDL()
    symbol = nl.load_neurosynth_study_ids(data_dir, name="StudyIds")
    assert symbol.symbol_name == "StudyIds"
    assert set(nl[symbol.symbol_name].value) == set(
        (id,) for id in metadata["id"].values
    )

    symbol = nl.load_neurosynth_study_ids(data_dir, convert_study_ids=True)
    assert nl[symbol.symbol_name].type is AbstractSet[Tuple[StudyID]]


@patch("neurolang.frontend.query_resolution.get_ns_term_study_associations")
def test_load_neurosynth_term_study_associations(
    mock_get_term_data, term_data
):
    mock_get_term_data.return_value = term_data
    data_dir = Path("mock_data_dir")

    nl = NeurolangDL()
    symbol = nl.load_neurosynth_term_study_associations(
        data_dir, name="TermInStudyTFIDF"
    )
    assert symbol.symbol_name == "TermInStudyTFIDF"
    assert set(nl[symbol.symbol_name].value) == set(
        tuple(e) for e in term_data.values
    )

    symbol = nl.load_neurosynth_term_study_associations(
        data_dir, convert_study_ids=True
    )
    assert (
        nl[symbol.symbol_name].type is AbstractSet[Tuple[StudyID, str, float]]
    )


@patch("neurolang.frontend.query_resolution.get_ns_mni_peaks_reported")
def test_load_neurosynth_mni_peaks_reported(mock_peaks_reported, mni_peaks):
    mock_peaks_reported.return_value = mni_peaks
    data_dir = Path("mock_data_dir")

    nl = NeurolangDL()
    symbol = nl.load_neurosynth_mni_peaks_reported(
        data_dir, name="PeakReported"
    )
    assert symbol.symbol_name == "PeakReported"
    assert set(nl[symbol.symbol_name].value) == set(
        tuple(e) for e in mni_peaks.values
    )

    symbol = nl.load_neurosynth_mni_peaks_reported(
        data_dir, convert_study_ids=True
    )
    assert (
        nl[symbol.symbol_name].type
        is AbstractSet[Tuple[StudyID, int, int, int]]
    )
