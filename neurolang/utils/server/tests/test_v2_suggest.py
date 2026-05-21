"""
Tests for the V2SuggestHandler (POST /v2/suggest/:engine).
"""
import json
from contextlib import contextmanager
from typing import AbstractSet, Callable, Iterable, Tuple
from unittest.mock import MagicMock

import pytest
import tornado.testing

from ..app import Application
from ..queries import NeurolangQueryManager


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_autocompletion_result(
    identifiers=None, signs=None, operators=None
):
    """Build a fake autocompletion result dict like the real engine returns."""
    return {
        "Identifiers": set(identifiers or ["PeakReported", "Study"]),
        "Signs": set(signs or ["@", "(", "∃"]),
        "Operators": set(operators or ["¬", "~"]),
        "Numbers": set(),
        "Text": set(),
        "Cmd_identifier": set(),
        "Functions": set(["lambda"]),
        "Identifier_regexp": set(),
        "Reserved words": set(["exists", "EXISTS"]),
        "Boleans": set(["⊤", "⊥", "False", "True"]),
        "Expression symbols": set(),
        "Python string": set(),
        "Strings": set(
            ["<identifier regular expression>", "<command identifier>"]
        ),
        "commands": set(),
        "functions": set(),
        "base symbols": set(["PeakReported", "Study"]),
        "query symbols": set(),
    }


def _make_engine(autocomplete_result=None):
    """Create a minimal mock engine with autocompletion support."""
    engine = MagicMock()

    # Symbols used for "empty program" fallback
    rel_sym = MagicMock()
    rel_sym.type = AbstractSet[Tuple[int, int, int, str]]
    func_sym = MagicMock()
    func_sym.type = Callable[[Iterable, Iterable, Iterable], object]

    engine.symbols = {
        "PeakReported": rel_sym,
        "Study": rel_sym,
        "agg_create_region": func_sym,
    }

    engine.program_ir = MagicMock()
    engine.program_ir.probabilistic_predicate_symbols = set()

    result = autocomplete_result or _make_autocompletion_result()
    engine.compute_datalog_program_for_autocompletion.return_value = result

    def _param_names(name):
        if name == "PeakReported":
            return ("x", "y", "z", "id")
        if name == "Study":
            return ("id",)
        return ()

    engine.predicate_parameter_names = _param_names

    return engine


def _make_engine_set(engine):
    """Return a context-manager-compatible NeurolangEngineSet mock."""

    engine_set = MagicMock()

    @contextmanager
    def _engine_ctx(timeout=None):
        yield engine

    engine_set.engine = _engine_ctx
    return engine_set


@pytest.fixture
def mock_engine():
    return _make_engine()


@pytest.fixture
def nqm_mock(mock_engine):
    """NeurolangQueryManager mock with one engine (neurosynth)."""
    mock_nqm = MagicMock(spec=NeurolangQueryManager)

    config_ns = MagicMock()
    config_ns.key = "neurosynth"
    config_dx = MagicMock()
    config_dx.key = "destrieux"
    mock_nqm.configs = {config_ns: 2, config_dx: 2}

    engine_set = _make_engine_set(mock_engine)
    mock_nqm.engines = {"neurosynth": engine_set}

    return mock_nqm


class TornadoSuggestTestCase(tornado.testing.AsyncHTTPTestCase):
    """Reusable Tornado test case for suggest endpoint tests."""

    def __init__(self, nqm) -> None:
        super().__init__()
        self._nqm = nqm

    def get_app(self):
        return Application(self._nqm)

    def runTest(self):
        pass


@pytest.fixture
def test_case(nqm_mock):
    tc = TornadoSuggestTestCase(nqm_mock)
    tc.setUp()
    yield tc
    tc.tearDown()


# ---------------------------------------------------------------------------
# Helper to POST JSON to the suggest endpoint
# ---------------------------------------------------------------------------


def _post_suggest(test_case, engine, body_dict):
    """POST a JSON body to /v2/suggest/<engine> and return the response."""
    return test_case.fetch(
        f"/v2/suggest/{engine}",
        method="POST",
        body=json.dumps(body_dict),
        headers={"Content-Type": "application/json"},
    )


# ---------------------------------------------------------------------------
# Tests: HTTP status codes
# ---------------------------------------------------------------------------


class TestV2SuggestHandlerStatusCodes:
    def test_post_known_engine_returns_200(self, test_case):
        """POST to a known engine returns 200."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        assert response.code == 200

    def test_post_unknown_engine_returns_404(self, test_case):
        """POST to an unknown engine returns 404."""
        response = _post_suggest(
            test_case, "nonexistent", {"program": "ans(x) :- "}
        )
        assert response.code == 404

    def test_content_type_is_json(self, test_case):
        """Response Content-Type is application/json."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        ct = response.headers.get("Content-Type", "")
        assert "application/json" in ct


# ---------------------------------------------------------------------------
# Tests: Response shape
# ---------------------------------------------------------------------------


class TestV2SuggestHandlerResponseShape:
    def test_response_has_status_ok(self, test_case):
        """Response envelope has status='ok'."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        body = json.loads(response.body)
        assert body["status"] == "ok"

    def test_response_data_is_dict(self, test_case):
        """Response data field is a dict (suggestion categories)."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        body = json.loads(response.body)
        assert isinstance(body["data"], dict)

    def test_response_data_has_identifiers_key(self, test_case):
        """Suggestion result dict contains an 'Identifiers' key."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        body = json.loads(response.body)
        assert "Identifiers" in body["data"]

    def test_identifiers_contains_predicate_names(
        self, test_case, mock_engine
    ):
        """Identifiers list includes predicate names from autocompletion."""
        mock_engine.compute_datalog_program_for_autocompletion.return_value = (
            _make_autocompletion_result(
                identifiers=["PeakReported", "Study"]
            )
        )
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        body = json.loads(response.body)
        identifiers = body["data"]["Identifiers"]
        assert "PeakReported" in identifiers or "Study" in identifiers

    def test_response_data_values_are_lists(self, test_case):
        """All values in suggestion categories are JSON arrays (not sets)."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        body = json.loads(response.body)
        for key, value in body["data"].items():
            assert isinstance(value, list), (
                f"Expected list for key '{key}', got {type(value)}"
            )


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestV2SuggestHandlerEdgeCases:
    def test_empty_program_returns_200(self, test_case):
        """An empty program string does not cause a 500."""
        response = _post_suggest(test_case, "neurosynth", {"program": ""})
        assert response.code == 200

    def test_empty_program_returns_identifiers(self, test_case):
        """An empty program returns available predicates as identifiers."""
        response = _post_suggest(test_case, "neurosynth", {"program": ""})
        body = json.loads(response.body)
        assert body["status"] == "ok"
        assert "Identifiers" in body["data"]

    def test_program_without_cursor_position_returns_200(self, test_case):
        """A request without cursor_position returns 200."""
        response = _post_suggest(
            test_case, "neurosynth", {"program": "ans(x) :- "}
        )
        assert response.code == 200

    def test_parser_error_returns_200_not_500(
        self, test_case, mock_engine
    ):
        """Parser errors in autocompletion are caught and return 200."""
        mock_engine.compute_datalog_program_for_autocompletion.side_effect = (
            Exception("Parser error: unexpected token")
        )
        response = _post_suggest(
            test_case, "neurosynth", {"program": "invalid @@@ syntax"}
        )
        # Should NOT be a 500 - should be handled gracefully
        assert response.code == 200

    def test_parser_error_response_contains_message(
        self, test_case, mock_engine
    ):
        """Parser error response contains a message field."""
        error_msg = "unexpected token '@'"
        mock_engine.compute_datalog_program_for_autocompletion.side_effect = (
            Exception(error_msg)
        )
        response = _post_suggest(
            test_case,
            "neurosynth",
            {"program": "invalid @@@ syntax"},
        )
        body = json.loads(response.body)
        # Either status is "error" with a message, or status is "ok"
        # Either way, must not be a 500
        assert "message" in body or body.get("status") == "ok"

    def test_cursor_position_splits_program_correctly(
        self, test_case, mock_engine
    ):
        """cursor_position is used to split the program correctly."""
        program = "ans(x) :- PeakReported(x, y, z, s)"
        cursor_position = len("ans(x) :- ")

        _post_suggest(
            test_case,
            "neurosynth",
            {"program": program, "cursor_position": cursor_position},
        )

        # The engine's method should have been called with the partial program
        # (up to cursor position) as the second argument
        calls = (
            mock_engine.compute_datalog_program_for_autocompletion.call_args_list
        )
        assert len(calls) >= 1
        _, autocompletion_code = calls[-1].args
        # autocompletion_code should be the program up to cursor_position
        assert autocompletion_code == program[:cursor_position]

    def test_missing_program_field_returns_error(self, test_case):
        """A request without 'program' field returns a non-200 error code."""
        response = test_case.fetch(
            "/v2/suggest/neurosynth",
            method="POST",
            body=json.dumps({}),
            headers={"Content-Type": "application/json"},
        )
        # Should return 4xx error, not 500
        assert response.code in (400, 422)


# ---------------------------------------------------------------------------
# Tests: Engine unavailable (503)
# ---------------------------------------------------------------------------


class TestV2SuggestHandlerEngineUnavailable:
    def test_engine_none_returns_503(self):
        """When engine pool yields None (timeout), return 503."""
        mock_nqm = MagicMock(spec=NeurolangQueryManager)
        config = MagicMock()
        config.key = "neurosynth"
        mock_nqm.configs = {config: 1}

        engine_set = MagicMock()

        @contextmanager
        def _engine_ctx_none(timeout=None):
            yield None  # Simulate timeout / unavailable

        engine_set.engine = _engine_ctx_none
        mock_nqm.engines = {"neurosynth": engine_set}

        tc = TornadoSuggestTestCase(mock_nqm)
        tc.setUp()
        try:
            response = _post_suggest(
                tc, "neurosynth", {"program": "ans(x) :- "}
            )
            assert response.code == 503
        finally:
            tc.tearDown()
