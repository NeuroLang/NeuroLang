import json
from concurrent.futures import Future
from typing import AbstractSet, Tuple
from unittest.mock import create_autospec
from uuid import uuid4

import pytest
from neurolang.exceptions import NeuroLangException
from neurolang.type_system import Unknown
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
)
from ..responses import (
    CustomQueryResultsEncoder,
    QueryResults,
)


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


def test_query_results_has_metadata(future):
    uuid = str(uuid4())
    page = 2
    limit = 100
    qr = QueryResults(uuid, future, page, limit)
    assert qr.uuid == uuid
    assert qr.cancelled == False
    assert qr.done == False
    assert qr.running == True


def test_query_results_has_exceptions(future, error):
    future.done.return_value = True
    future.exception.return_value = error

    qr = QueryResults(str(uuid4()), future)
    assert qr.errorName == str(type(error))
    assert qr.errorDoc == error.__doc__
    assert qr.message == str(error)


def test_query_results_has_results(future, results, data):
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = results

    qr = QueryResults(str(uuid4()), future, get_values=True)
    assert qr.results is not None
    assert qr.results["ans"]["row_type"] == [
        str(t) for t in (float, str, AbstractSet[Unknown])
    ]
    assert qr.results["ans"]["columns"] == ["a", "b", "c"]
    assert qr.results["ans"]["size"] == 3
    assert qr.results["ans"]["values"] == [[a, b, str(c)] for a, b, c in data]
    assert qr.results["Voxel"]["row_type"] == [str(t) for t in (int, int, int)]
    assert qr.results["Voxel"]["columns"] == ["i", "j", "k"]
    assert qr.results["Voxel"]["size"] == 3


def test_query_results_does_not_return_values_by_default(future, result):
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = result

    qr = QueryResults(str(uuid4()), future)
    assert qr.results is not None
    assert qr.results["ans"]["row_type"] == [
        str(t) for t in (float, str, AbstractSet[Unknown])
    ]
    assert qr.results["ans"]["columns"] == ["a", "b", "c"]
    assert qr.results["ans"]["size"] == 3
    assert "values" not in qr.results["ans"]


def test_query_results_only_returns_requested_symbol(future, results, data):
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = results

    qr = QueryResults(str(uuid4()), future, "Voxel")
    assert qr.results is not None
    assert len(qr.results) == 1
    assert qr.results["Voxel"]["row_type"] == [str(t) for t in (int, int, int)]
    assert qr.results["Voxel"]["columns"] == ["i", "j", "k"]
    assert qr.results["Voxel"]["size"] == 3
    assert len(qr.results["Voxel"]["values"]) == 3

    qr = QueryResults(str(uuid4()), future, "ans")
    assert qr.results is not None
    assert len(qr.results) == 1
    assert qr.results["ans"]["row_type"] == [
        str(t) for t in (float, str, AbstractSet[Unknown])
    ]
    assert qr.results["ans"]["columns"] == ["a", "b", "c"]
    assert qr.results["ans"]["size"] == 3
    assert qr.results["ans"]["values"] == [[a, b, str(c)] for a, b, c in data]


def test_query_results_can_paginate(future, result, data):
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = result

    qr = QueryResults(str(uuid4()), future, start=0, length=1, get_values=True)
    assert qr.results is not None
    assert qr.results["ans"]["values"] == [
        [a, b, str(c)] for a, b, c in data[0:1]
    ]

    qr = QueryResults(str(uuid4()), future, start=1, length=2, get_values=True)
    assert qr.results is not None
    assert qr.results["ans"]["values"] == [
        [a, b, str(c)] for a, b, c in data[1:3]
    ]

    qr = QueryResults(str(uuid4()), future, start=2, length=2, get_values=True)
    assert qr.results is not None
    assert qr.results["ans"]["values"] == [
        [a, b, str(c)] for a, b, c in data[2:]
    ]


def test_query_results_can_sort(future, result, data):
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = result

    qr = QueryResults(
        str(uuid4()), future, start=0, length=2, sort=1, get_values=True
    )
    assert qr.results is not None
    assert qr.results["ans"]["values"] == [
        [a, b, str(c)] for a, b, c in data[1::-1]
    ]


def test_query_results_can_serialize_to_json(future, result, data):
    future.done.return_value = True
    future.exception.return_value = None
    future.result.return_value = result

    uuid = str(uuid4())
    qr = QueryResults(uuid, future, get_values=True)
    json_string = json.dumps(qr, cls=CustomQueryResultsEncoder)
    expected = {
        "uuid": uuid,
        "cancelled": False,
        "running": True,
        "done": True,
        "start": 0,
        "length": 50,
        "asc": True,
        "sort": -1,
        "results": {
            "ans": {
                "columns": ["a", "b", "c"],
                'last_parsed_symbol': False,
                'probabilistic': False,
                "row_type": [
                    str(t) for t in (float, str, AbstractSet[Unknown])
                ],
                "size": 3,
                "values": [[a, b, str(c)] for a, b, c in data],
            }
        },
    }
    assert json.loads(json_string) == expected
