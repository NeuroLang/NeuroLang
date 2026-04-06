"""
Tests for V2ExamplesHandler: GET /v2/examples/:engine
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import tornado.testing

from ..app import Application
from ..queries import NeurolangQueryManager


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

MOCK_QUERIES_YAML = {
    "neurosynth": [
        {
            "id": "neuro1",
            "title": "CBMA Single Term",
            "shortTitle": "CBMA Single Term",
            "query": "TermInStudy(t, s) :- TermInStudyTFIDF(t, tfidf, s)",
            "description": "An example CBMA query.",
        },
        {
            "id": "neuro2",
            "title": "CBMA Multiple Terms",
            "shortTitle": "CBMA Multiple Terms",
            "query": "TermInStudy(t, s) :- TermInStudyTFIDF(t, tfidf, s)",
            "description": "Multiple terms example.",
        },
    ],
    "destrieux": [
        {
            "id": "destrieux1",
            "title": "Union of Destrieux atlas regions",
            "shortTitle": "Union of atlas regions",
            "query": "LeftRegion(s, r) :- destrieux(s, r)",
            "description": "Region union example.",
        },
    ],
}


def _make_nqm_mock():
    """Create a minimal NeurolangQueryManager mock with two engine configs."""
    mock_nqm = MagicMock(spec=NeurolangQueryManager)

    config_neurosynth = MagicMock()
    config_neurosynth.key = "neurosynth"
    config_destrieux = MagicMock()
    config_destrieux.key = "destrieux"
    mock_nqm.configs = {config_neurosynth: 2, config_destrieux: 2}
    mock_nqm.engines = {}

    return mock_nqm


class TornadoExamplesTestCase(tornado.testing.AsyncHTTPTestCase):
    """Reusable Tornado test case for V2ExamplesHandler tests."""

    def __init__(self, nqm) -> None:
        super().__init__()
        self._nqm = nqm

    def get_app(self):
        return Application(self._nqm)

    def runTest(self):
        pass


@pytest.fixture
def nqm_mock():
    return _make_nqm_mock()


@pytest.fixture
def test_case(nqm_mock):
    tc = TornadoExamplesTestCase(nqm_mock)
    tc.setUp()
    yield tc
    tc.tearDown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV2ExamplesHandlerStatusCodes:
    def test_known_engine_returns_200(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        assert response.code == 200

    def test_unknown_engine_returns_404(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/nonexistent")
        assert response.code == 404

    def test_content_type_is_json(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        ct = response.headers.get("Content-Type", "")
        assert "application/json" in ct


class TestV2ExamplesHandlerResponseShape:
    def test_response_has_status_ok(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        body = json.loads(response.body)
        assert body["status"] == "ok"

    def test_response_data_is_list(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        body = json.loads(response.body)
        assert isinstance(body["data"], list)

    def test_neurosynth_returns_correct_examples(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        body = json.loads(response.body)
        examples = body["data"]
        assert len(examples) == 2
        ids = [e["id"] for e in examples]
        assert "neuro1" in ids
        assert "neuro2" in ids

    def test_destrieux_returns_correct_examples(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/destrieux")
        body = json.loads(response.body)
        examples = body["data"]
        assert len(examples) == 1
        assert examples[0]["id"] == "destrieux1"

    def test_example_has_required_fields(self, test_case):
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=MOCK_QUERIES_YAML,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        body = json.loads(response.body)
        example = body["data"][0]
        for field in ("id", "title", "shortTitle", "query", "description"):
            assert field in example, f"Missing field: {field}"

    def test_engine_with_no_examples_returns_empty_list(self, test_case):
        """Engine known to the query manager but absent from queries.yaml."""
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value={},  # empty YAML – no examples for any engine
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        body = json.loads(response.body)
        assert body["status"] == "ok"
        assert body["data"] == []


class TestV2ExamplesHandlerEdgeCases:
    def test_malformed_yaml_value_returns_empty_list(self, test_case):
        """If queries.yaml has a non-list value for an engine, return []."""
        bad_yaml = {"neurosynth": "not-a-list"}
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value=bad_yaml,
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        body = json.loads(response.body)
        assert body["status"] == "ok"
        assert body["data"] == []

    def test_queries_yaml_not_found_returns_empty_list(self, test_case):
        """If queries.yaml is not found, return empty list (not 500)."""
        with patch(
            "neurolang.utils.server.v2_handlers._load_queries_yaml",
            return_value={},
        ):
            response = test_case.fetch("/v2/examples/neurosynth")
        assert response.code == 200
        body = json.loads(response.body)
        assert body["data"] == []
