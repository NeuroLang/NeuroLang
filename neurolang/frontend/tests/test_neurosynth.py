import numpy as np
import pandas as pd
import pytest

from .. import NeurolangDL
from ..neurosynth_utils import NeuroSynthHandler


class MockNeurosynthImageTable:
    def __init__(self, data):
        self.data = data


class MockNeurosynthFeatureTable:
    def __init__(self, data):
        self.data = data


class MockNeurosynthDataset:
    def __init__(self, activations, tfidfs):
        self.image_table = MockNeurosynthImageTable(activations)
        self.feature_table = MockNeurosynthFeatureTable(tfidfs)


def mock_neurosynth_handler():
    activations = pd.DataFrame(
        data=np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 1, 0, 0],
            ],
            dtype=bool,
        ),
        index=[78, 99, 45, 23],
    )
    tfidfs = pd.DataFrame(
        data=np.array(
            [
                [0.223, 0.881, 0.004],
                [0.598, 0.011, 0.172],
                [0.011, 0.991, 0.064],
                [0.555, 0.271, 0.983],
            ]
        ),
        columns=["memory", "visual", "auditory"],
        index=[78, 99, 45, 23],
    )
    mock_dataset = MockNeurosynthDataset(activations, tfidfs)
    handler = NeuroSynthHandler(ns_dataset=mock_dataset)
    return handler


def test_load_study_ids_handler_level():
    handler = mock_neurosynth_handler()
    assert set(tuple(x) for x in handler.ns_study_ids()) == {
        (78,),
        (99,),
        (45,),
        (23,),
    }


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
