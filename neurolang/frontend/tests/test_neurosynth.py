import numpy as np
import pytest

from .. import NeurolangDL
from ..neurosynth_utils import NeuroSynthHandler


def mock_study_ids(*args, **kwargs):
    return np.array([[1, 2, 3, 4]])


def test_load_study_ids(monkeypatch):
    monkeypatch.setattr(NeuroSynthHandler, "ns_study_ids", mock_study_ids)
    frontend = NeurolangDL()
    symbol = frontend.load_neurosynth_study_ids()
    study_ids = frontend.symbol_table[symbol].value.to_numpy()
    assert study_ids == np.array([[1, 2, 3, 4]])
