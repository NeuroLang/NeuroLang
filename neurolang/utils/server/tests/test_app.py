import json
import urllib.parse
from concurrent.futures import Future
from typing import AbstractSet, Tuple
from unittest.mock import MagicMock, create_autospec
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tornado.testing
import tornado.websocket
from neurolang.exceptions import NeuroLangException
from neurolang.type_system import Unknown
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)

from ..app import Application
from ..queries import NeurolangQueryManager


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
def figures(data):
    ans = NamedRelationalAlgebraFrozenSet(("a", "b", "c"), data)
    ans.row_type = Tuple[float, str, AbstractSet[Unknown]]
    N_points = 10000
    n_bins = 20
    x = np.random.randn(N_points)
    y = 0.4 * x + np.random.randn(N_points) + 5
    figx, ax = plt.subplots()
    ax.hist(x, bins=n_bins)
    plt.close(figx)
    figy, axy = plt.subplots()
    axy.hist(y, bins=n_bins)
    plt.close(figy)

    hists = NamedRelationalAlgebraFrozenSet(("agg_hist",), [[figx], [figy]])
    hists.row_type = Tuple[matplotlib.figure.Figure]
    figures = {"Hists": hists}
    return figures


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


def test_figure_handler_gets_figure(test_case, nqm, future, figures):
    uuid = uuid4()
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = figures
    nqm.get_result.return_value = future

    params = {"symbol": "Hists", "row": 1, "col": 0}

    response = test_case.fetch(
        f"/v1/figure/{uuid}?" + urllib.parse.urlencode(params)
    )
    assert response.code == 200
    nqm.get_result.assert_called_with(str(uuid))


# ---------------------------------------------------------------------------
# QuerySocketHandler tests
# ---------------------------------------------------------------------------


class QuerySocketTornadoTestCase(tornado.testing.AsyncHTTPTestCase):
    """
    Tornado test case with WebSocket support for QuerySocketHandler tests.
    """

    def __init__(self, nqm) -> None:
        super().__init__()
        self.nqm = nqm

    def get_app(self):
        return Application(self.nqm)

    def runTest(self):
        pass

    @tornado.testing.gen_test(timeout=10)
    async def test_query_socket_sends_values(self):
        """
        QuerySocketHandler.send_query_update() must pass get_values=True to
        QueryResults so the WebSocket response includes the 'values' field.
        """
        data = [
            (5, "dog"),
            (10, "cat"),
        ]
        from typing import Tuple

        ans = NamedRelationalAlgebraFrozenSet(("a", "b"), data)
        ans.row_type = Tuple[float, str]
        result = {"ans": ans}

        future = create_autospec(Future)
        future.cancelled.return_value = False
        future.done.return_value = True
        future.running.return_value = False
        future.exception.return_value = None
        future.result.return_value = result
        self.nqm.submit_query.return_value = future

        url = self.get_url("/v1/statementsocket").replace(
            "http://", "ws://"
        )
        conn = await tornado.websocket.websocket_connect(url)

        # Send a query message
        await conn.write_message(
            json.dumps(
                {"query": "ans(x) :- Study(x).", "engine": "neurosynth"}
            )
        )

        # Read up to 2 messages (initial + completion callback)
        messages = []
        for _ in range(2):
            msg = await conn.read_message()
            if msg is None:
                break
            messages.append(json.loads(msg))

        conn.close()

        # At least one message should be present
        assert len(messages) >= 1

        # Find the done message
        done_messages = [
            m for m in messages
            if m.get("data", {}).get("done") is True
        ]
        assert len(done_messages) >= 1, (
            f"Expected at least one 'done' message, got: {messages}"
        )

        done_data = done_messages[-1]["data"]
        assert "results" in done_data, (
            f"'results' key missing from done message data: {done_data}"
        )
        assert "ans" in done_data["results"], (
            f"'ans' symbol missing from results: {done_data['results']}"
        )
        symbol_data = done_data["results"]["ans"]
        assert "values" in symbol_data, (
            f"'values' field missing from symbol data "
            f"(get_values=True not passed): {symbol_data}"
        )
        assert isinstance(symbol_data["values"], list), (
            f"'values' should be a list, got: {type(symbol_data['values'])}"
        )
        assert len(symbol_data["values"]) == 2, (
            f"Expected 2 rows, got: {len(symbol_data['values'])}"
        )


@pytest.fixture
def ws_test_case(nqm):
    tc = QuerySocketTornadoTestCase(nqm)
    tc.setUp()
    yield tc
    tc.tearDown()


def test_query_socket_handler_sends_values(ws_test_case):
    """
    Verify that the WebSocket response includes 'values' rows when a query
    completes successfully. This confirms get_values=True is passed to
    QueryResults in QuerySocketHandler.send_query_update().
    """
    ws_test_case.test_query_socket_sends_values()
