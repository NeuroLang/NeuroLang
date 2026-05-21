"""
Tests for v2 API endpoints: V2EnginesHandler, V2SchemaHandler,
V2AtlasHandler.
"""
import inspect
import json
from concurrent.futures import Future
from typing import AbstractSet, Callable, Iterable, Tuple
from unittest.mock import MagicMock, create_autospec, patch

import nibabel as nib
import numpy as np
import pytest
import tornado.testing

from ..app import Application
from ..queries import NeurolangQueryManager


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_engine(prob_symbols=None):
    """
    Create a minimal mock NeurolangPDL engine with enough surface to satisfy
    the v2 schema handler.
    """
    engine = MagicMock()

    # Symbols on the engine – mimic what the real engine exposes
    # relation symbol: AbstractSet
    rel_sym = MagicMock()
    rel_sym.type = AbstractSet[Tuple[int, int, int, str]]

    # function symbol: Callable
    func_sym = MagicMock()
    func_sym.type = Callable[[Iterable, Iterable, Iterable], object]
    func_sym.value = MagicMock()
    func_sym.value.__doc__ = "Aggregate function docstring."
    func_sym.value.__name__ = "agg_create_region"

    def _fake_func(x: Iterable, y: Iterable, z: Iterable):
        """Aggregate function docstring."""
        pass

    func_sym.value.__wrapped__ = _fake_func

    engine.symbols = {
        "PeakReported": rel_sym,
        "agg_create_region": func_sym,
    }

    # program_ir for probabilistic symbols
    if prob_symbols is None:
        prob_symbols = set()
    engine.program_ir = MagicMock()
    engine.program_ir.probabilistic_predicate_symbols = prob_symbols

    # predicate_parameter_names
    def _param_names(name):
        if name == "PeakReported":
            return ("x", "y", "z", "id")
        return ()

    engine.predicate_parameter_names = _param_names

    return engine


def _make_engine_set(engine):
    """Return a context-manager-compatible NeurolangEngineSet mock."""
    from contextlib import contextmanager

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
    """
    A MagicMock NeurolangQueryManager whose configs expose two engine keys
    and whose engines dict is pre-populated.
    """
    mock_nqm = MagicMock(spec=NeurolangQueryManager)

    # configs (used to list engines in V2EnginesHandler)
    config_neurosynth = MagicMock()
    config_neurosynth.key = "neurosynth"
    config_destrieux = MagicMock()
    config_destrieux.key = "destrieux"
    mock_nqm.configs = {config_neurosynth: 2, config_destrieux: 2}

    # engines dict (used by V2SchemaHandler to acquire an engine)
    engine_set = _make_engine_set(mock_engine)
    mock_nqm.engines = {"neurosynth": engine_set}

    # atlas
    atlas = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    mock_nqm.get_atlas.return_value = atlas

    return mock_nqm


class TornadoV2TestCase(tornado.testing.AsyncHTTPTestCase):
    """Reusable Tornado test case for v2 endpoint tests."""

    def __init__(self, nqm) -> None:
        super().__init__()
        self._nqm = nqm

    def get_app(self):
        return Application(self._nqm)

    def runTest(self):
        pass


@pytest.fixture
def test_case(nqm_mock):
    tc = TornadoV2TestCase(nqm_mock)
    tc.setUp()
    yield tc
    tc.tearDown()


# ---------------------------------------------------------------------------
# V2EnginesHandler tests
# ---------------------------------------------------------------------------


class TestV2EnginesHandler:
    def test_engines_returns_200(self, test_case):
        response = test_case.fetch("/v2/engines")
        assert response.code == 200

    def test_engines_returns_json_array(self, test_case):
        response = test_case.fetch("/v2/engines")
        body = json.loads(response.body)
        # The response is {"status": "ok", "data": [...]}
        assert body["status"] == "ok"
        assert isinstance(body["data"], list)

    def test_engines_contains_expected_keys(self, test_case):
        response = test_case.fetch("/v2/engines")
        body = json.loads(response.body)
        engine_keys = body["data"]
        assert "neurosynth" in engine_keys
        assert "destrieux" in engine_keys

    def test_engines_content_type_is_json(self, test_case):
        response = test_case.fetch("/v2/engines")
        ct = response.headers.get("Content-Type", "")
        assert "application/json" in ct


# ---------------------------------------------------------------------------
# V2SchemaHandler tests
# ---------------------------------------------------------------------------


class TestV2SchemaHandler:
    def test_schema_known_engine_returns_200(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        assert response.code == 200

    def test_schema_unknown_engine_returns_404(self, test_case):
        response = test_case.fetch("/v2/schema/nonexistent")
        assert response.code == 404

    def test_schema_returns_symbols_grouped_by_type(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        body = json.loads(response.body)
        data = body["data"]
        # Must contain at least "relations" and "functions" keys
        assert "relations" in data
        assert "functions" in data

    def test_schema_relation_has_name_and_params(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        body = json.loads(response.body)
        relations = body["data"]["relations"]
        names = [r["name"] for r in relations]
        assert "PeakReported" in names
        peak = next(r for r in relations if r["name"] == "PeakReported")
        assert "params" in peak
        assert list(peak["params"]) == ["x", "y", "z", "id"]

    def test_schema_relation_has_type_field(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        body = json.loads(response.body)
        relations = body["data"]["relations"]
        for rel in relations:
            assert rel.get("type") == "relation"

    def test_schema_function_has_name_type_and_params(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        body = json.loads(response.body)
        functions = body["data"]["functions"]
        names = [f["name"] for f in functions]
        assert "agg_create_region" in names
        func_entry = next(
            f for f in functions if f["name"] == "agg_create_region"
        )
        assert func_entry.get("type") == "function"
        assert "params" in func_entry

    def test_schema_function_may_have_docstring(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        body = json.loads(response.body)
        functions = body["data"]["functions"]
        func_entry = next(
            (f for f in functions if f["name"] == "agg_create_region"), None
        )
        # docstring is optional but should be present when available
        assert func_entry is not None
        # docstring field existence is OK (may be None/null)
        assert "docstring" in func_entry

    def test_schema_probabilistic_grouped_separately(self):
        """
        When an engine has probabilistic symbols, they appear under
        the 'probabilistic' key.
        """
        engine = _make_engine(prob_symbols={"SelectedStudy"})
        # add a relation for SelectedStudy
        sel_sym = MagicMock()
        sel_sym.type = AbstractSet[Tuple[str]]
        engine.symbols["SelectedStudy"] = sel_sym
        engine.predicate_parameter_names = lambda name: (
            ("id",) if name == "SelectedStudy" else ("x", "y", "z", "id")
        )

        engine_set = _make_engine_set(engine)
        mock_nqm = MagicMock(spec=NeurolangQueryManager)
        config = MagicMock()
        config.key = "neurosynth"
        mock_nqm.configs = {config: 1}
        mock_nqm.engines = {"neurosynth": engine_set}

        tc = TornadoV2TestCase(mock_nqm)
        tc.setUp()
        try:
            response = tc.fetch("/v2/schema/neurosynth")
            body = json.loads(response.body)
            data = body["data"]
            assert "probabilistic" in data
            prob_names = [s["name"] for s in data["probabilistic"]]
            assert "SelectedStudy" in prob_names
        finally:
            tc.tearDown()

    def test_schema_content_type_is_json(self, test_case):
        response = test_case.fetch("/v2/schema/neurosynth")
        ct = response.headers.get("Content-Type", "")
        assert "application/json" in ct


# ---------------------------------------------------------------------------
# V2AtlasHandler tests
# ---------------------------------------------------------------------------


class TestV2AtlasHandler:
    def test_atlas_known_engine_returns_200(self, test_case):
        response = test_case.fetch("/v2/atlas/neurosynth")
        assert response.code == 200

    def test_atlas_unknown_engine_returns_404(self, test_case, nqm_mock):
        nqm_mock.get_atlas.side_effect = IndexError("no engine")
        response = test_case.fetch("/v2/atlas/unknown")
        assert response.code == 404

    def test_atlas_returns_image_key(self, test_case):
        response = test_case.fetch("/v2/atlas/neurosynth")
        body = json.loads(response.body)
        assert "image" in body["data"]

    def test_atlas_image_is_nonempty_string(self, test_case):
        response = test_case.fetch("/v2/atlas/neurosynth")
        body = json.loads(response.body)
        image_data = body["data"]["image"]
        assert isinstance(image_data, str)
        assert len(image_data) > 0

    def test_atlas_content_type_is_json(self, test_case):
        response = test_case.fetch("/v2/atlas/neurosynth")
        ct = response.headers.get("Content-Type", "")
        assert "application/json" in ct


# ---------------------------------------------------------------------------
# Backward compat: v1 endpoints still work
# ---------------------------------------------------------------------------


class TestV1BackwardCompatibility:
    def test_v1_engines_still_works(self, test_case, nqm_mock):
        response = test_case.fetch("/v1/engines")
        # Should not return 404; the exact code depends on mock setup
        assert response.code != 404


# ---------------------------------------------------------------------------
# V2SquallHandler
# ---------------------------------------------------------------------------


def _make_squall_engine_and_nqm():
    """Create a real (toy) engine backed by simple EDB facts and an NQM mock."""
    from contextlib import contextmanager
    from neurolang.frontend.probabilistic_frontend import RegionFrontendCPLogicSolver
    from neurolang.expressions import Symbol

    engine = RegionFrontendCPLogicSolver()
    engine.add_extensional_predicate_from_tuples(
        Symbol("item"), [("a",), ("b",), ("c",), ("d",)]
    )
    engine.add_extensional_predicate_from_tuples(
        Symbol("item_count"), [("a", 0), ("a", 1), ("b", 2), ("c", 3)]
    )

    engine_set = MagicMock()

    @contextmanager
    def _ctx(timeout=None):
        yield engine

    engine_set.engine = _ctx

    mock_nqm = MagicMock(spec=NeurolangQueryManager)
    config_toy = MagicMock()
    config_toy.key = "toy"
    mock_nqm.configs = {config_toy: 1}
    mock_nqm.engines = {"toy": engine_set}
    mock_nqm.get_atlas.side_effect = KeyError("no atlas")
    return mock_nqm


@pytest.fixture
def squall_test_case():
    nqm = _make_squall_engine_and_nqm()
    tc = TornadoV2TestCase(nqm)
    tc.setUp()
    yield tc
    tc.tearDown()


class TestV2SquallHandler:
    def test_squall_parse_only_returns_200(self, squall_test_case):
        """A rule-only program (no obtain) returns 200 with 'parsed' key."""
        body = json.dumps({"program": "define as large every Item that has an item_count."})
        resp = squall_test_case.fetch(
            "/v2/squall/toy",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.code == 200
        data = json.loads(resp.body)
        assert data["status"] == "ok"
        assert "parsed" in data["data"]

    def test_squall_obtain_returns_results(self, squall_test_case):
        """An 'obtain' program executes and returns rows in 'results'."""
        body = json.dumps({"program": "obtain every Item that has an item_count."})
        resp = squall_test_case.fetch(
            "/v2/squall/toy",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.code == 200
        data = json.loads(resp.body)
        assert data["status"] == "ok", f"error: {data}"
        results = data["data"]["results"]
        assert isinstance(results, list)
        assert len(results) == 1          # one obtain clause
        rows = results[0]
        assert isinstance(rows, list)
        # Items a, b, c have item_count; d does not
        row_values = {tuple(r) for r in rows}
        assert row_values == {("a",), ("b",), ("c",)}

    def test_squall_unknown_engine_returns_404(self, squall_test_case):
        body = json.dumps({"program": "obtain every item."})
        resp = squall_test_case.fetch(
            "/v2/squall/does_not_exist",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.code == 404

    def test_squall_empty_program_returns_400(self, squall_test_case):
        body = json.dumps({"program": "   "})
        resp = squall_test_case.fetch(
            "/v2/squall/toy",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.code == 400

    def test_squall_invalid_program_returns_error_status(self, squall_test_case):
        body = json.dumps({"program": "this is not valid squall @@@@"})
        resp = squall_test_case.fetch(
            "/v2/squall/toy",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        data = json.loads(resp.body)
        assert data["status"] == "error"
