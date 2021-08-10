import json
import urllib.parse
from concurrent.futures import Future
from typing import AbstractSet, Tuple
from unittest.mock import MagicMock, create_autospec
from uuid import uuid4

import pytest
import tornado.testing
import tornado.websocket
from neurolang.exceptions import NeuroLangException
from neurolang.type_system import Unknown
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)
from neurolang.utils.server.app import Application
from neurolang.utils.server.queries import NeurolangQueryManager


@pytest.fixture
def future():
    future = create_autospec(Future)
    future.cancelled.return_value = False
    future.done.return_value = False
    future.running.return_value = True
    return future


@pytest.fixture
def error():
    return NeuroLangException("Something went wrong")


@pytest.fixture
def data():
    data = [
        (5, "dog", frozenset({(1, 2), (5, 6)})),
        (10, "cat", frozenset({(5, 6), (8, 9)})),
        (-5.25, "mouse", frozenset({(8, 9), (12, 13)})),
    ]
    return data


@pytest.fixture
def result(data):
    ans = NamedRelationalAlgebraFrozenSet(("a", "b", "c"), data)
    ans.row_type = Tuple[float, str, AbstractSet[Unknown]]
    results = {"ans": ans}
    return results


@pytest.fixture
def results(data):
    ans = NamedRelationalAlgebraFrozenSet(("a", "b", "c"), data)
    ans.row_type = Tuple[float, str, AbstractSet[Unknown]]
    voxels = NamedRelationalAlgebraFrozenSet(
        ("i", "j", "k"), [[12, 15, 98], [107, 2, 33], [89, 8, 34]]
    )
    voxels.row_type = Tuple[int, int, int]
    results = {"ans": ans, "Voxel": voxels}
    return results


@pytest.fixture
def nqm():
    mock_nqm = MagicMock(spec=NeurolangQueryManager)
    return mock_nqm


class TornadoTestCase(tornado.testing.AsyncHTTPTestCase):
    """
    unittest.TestCase inheriting from tornado's testing.AsyncHTTPTestCase
    to create the app under test and benefit from the convenience methods
    to generate requests for it.

    It's a bit hackish but works for writing pytest tests using the
    tornado.testing utilities.
    """

    def __init__(self, nqm) -> None:
        super().__init__()
        self.nqm = nqm

    def get_app(self):
        return Application(self.nqm)

    def runTest(self):
        pass


@pytest.fixture
def test_case(nqm):
    test_case = TornadoTestCase(nqm)
    test_case.setUp()

    yield test_case

    test_case.tearDown()


def test_status_handler_calls_get_result_on_nqm(
    test_case, nqm, future, results
):
    uuid = uuid4()
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = results
    nqm.get_result.return_value = future

    response = test_case.fetch(f"/v1/status/{uuid}")
    assert response.code == 200
    nqm.get_result.assert_called_with(str(uuid))


def test_status_handler_passes_params_to_query_results(
    test_case, nqm, future, results
):
    uuid = uuid4()
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = results
    nqm.get_result.return_value = future

    params = {"symbol": "ans", "start": 1, "length": 1, "sort": 1, "asc": 0}

    response = test_case.fetch(
        f"/v1/status/{uuid}?" + urllib.parse.urlencode(params)
    )
    assert response.code == 200
    qr = json.loads(response.body)
    assert qr["status"] == "ok"
    data = qr["data"]
    assert data["start"] == params["start"]
    assert data["length"] == params["length"]
    assert data["sort"] == params["sort"]
    assert data["asc"] == bool(params["asc"])
    assert data["uuid"] == str(uuid)
    assert data["done"] == True
    assert len(data["results"]) == 1
    ans = data["results"]["ans"]
    assert ans["columns"] == list(results["ans"].columns)
    assert ans["size"] == 3
    assert len(ans["values"]) == 1


def test_symbols_handler_calls_get_symbols_on_nqm(
    test_case, nqm, future, results
):
    engine = "neuroquery"
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = results
    nqm.get_symbols.return_value = future

    response = test_case.fetch(f"/v1/symbol/{engine}")
    assert response.code == 200
    nqm.get_symbols.assert_called_with(engine)
