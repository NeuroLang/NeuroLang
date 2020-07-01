import numpy as np
import pytest

from .. import NeurolangDL
from ..neurosynth_utils import NeuroSynthHandler, StudyID


def mock_study_ids(*args, **kwargs):
    return {(StudyID(1),), (StudyID(2),)}


def test_load_study_ids(monkeypatch):
    monkeypatch.setattr(NeuroSynthHandler, "ns_study_ids", mock_study_ids)
    frontend = NeurolangDL()
    symbol = frontend.load_neurosynth_study_ids()
    expected = {(StudyID(1),), (StudyID(2),)}
    assert frontend[symbol.symbol_name].value == expected
