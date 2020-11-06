import numpy as np
import pytest

from .. import NeurolangDL
from ..neurosynth_utils import NeuroSynthHandler


def mock_study_ids(*args, **kwargs):
    return {(1,), (2,)}


def test_load_study_ids(monkeypatch):
    monkeypatch.setattr(NeuroSynthHandler, "ns_study_ids", mock_study_ids)
    frontend = NeurolangDL()
    symbol = frontend.load_neurosynth_study_ids()
    assert frontend[symbol.symbol_name].value == mock_study_ids()


def mock_reported_activations(*args, **kwargs):
    return {
        (1, 42),
        (1, 21),
        (2, 42),
        (3, 21),
    }


def test_reported_activations(monkeypatch):
    monkeypatch.setattr(
        NeuroSynthHandler, "ns_reported_activations", mock_reported_activations
    )
    frontend = NeurolangDL()
    symbol = frontend.load_neurosynth_reported_activations()
    assert frontend[symbol.symbol_name].value == mock_reported_activations()
