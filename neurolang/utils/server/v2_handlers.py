"""
V2 API handlers for the NeuroLang Sparklis GUI.

Provides:
    - V2EnginesHandler  : GET /v2/engines
    - V2SchemaHandler   : GET /v2/schema/:engine
    - V2AtlasHandler    : GET /v2/atlas/:engine
"""
import inspect
import logging
from typing import AbstractSet, Callable

import tornado.web

from .base_handlers import JSONRequestHandler
from .responses import base64_encode_nifti
from ...type_system import is_leq_informative

LOG = logging.getLogger(__name__)


class V2EnginesHandler(JSONRequestHandler):
    """
    Return a JSON array of available engine keys.

    GET /v2/engines
    """

    def get(self) -> None:
        """Return a list of engine keys registered in the query manager."""
        engine_keys = [
            config.key for config in self.application.nqm.configs.keys()
        ]
        self.write_json_reponse(engine_keys)


class V2SchemaHandler(JSONRequestHandler):
    """
    Return schema information (symbols) for a given engine.

    GET /v2/schema/:engine

    Response shape
    --------------
    {
        "status": "ok",
        "data": {
            "relations": [
                {
                    "name": "PeakReported",
                    "type": "relation",
                    "params": ["x", "y", "z", "id"],
                    "row_type": ["<class 'int'>", ...]  // optional
                },
                ...
            ],
            "functions": [
                {
                    "name": "agg_create_region",
                    "type": "function",
                    "params": ["x", "y", "z"],
                    "docstring": "..."  // may be null
                },
                ...
            ],
            "probabilistic": [
                {
                    "name": "SelectedStudy",
                    "type": "probabilistic",
                    "params": ["id"],
                    ...
                },
                ...
            ]
        }
    }

    Raises tornado.web.HTTPError 404 for unknown engine names.
    """

    def get(self, engine_key: str) -> None:
        """Return schema for *engine_key*.

        Parameters
        ----------
        engine_key : str
            Identifier of the engine (e.g. ``'neurosynth'``).

        Raises
        ------
        tornado.web.HTTPError
            404 when *engine_key* is not found.
        """
        LOG.debug("V2SchemaHandler: requesting schema for '%s'.", engine_key)
        nqm = self.application.nqm
        engine_set = nqm.engines.get(engine_key)
        if engine_set is None:
            raise tornado.web.HTTPError(
                status_code=404,
                log_message=f"Engine '{engine_key}' not found.",
            )

        with engine_set.engine() as engine:
            if engine is None:
                raise tornado.web.HTTPError(
                    status_code=503,
                    log_message=(
                        f"Engine '{engine_key}' is temporarily unavailable."
                    ),
                )
            schema = self._build_schema(engine)

        self.write_json_reponse(schema)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_function_params(symbol) -> list:
        """
        Extract parameter names from a callable symbol.

        Tries ``inspect.signature`` on the underlying callable first;
        falls back to an empty list if introspection fails.

        Parameters
        ----------
        symbol : neurolang Symbol (Callable type)
            Symbol whose underlying value is a Python callable.

        Returns
        -------
        list[str]
            Parameter names (excluding ``self``).
        """
        callable_obj = getattr(symbol, "value", None)
        if callable_obj is None:
            return []
        # Some symbols wrap the original function in an attribute
        for attr in ("__wrapped__", "__func__", "__call__"):
            candidate = getattr(callable_obj, attr, None)
            if candidate is not None and callable(candidate):
                callable_obj = candidate
                break
        try:
            sig = inspect.signature(callable_obj)
            return [
                name
                for name, param in sig.parameters.items()
                if param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
        except (ValueError, TypeError):
            return []

    @staticmethod
    def _get_function_docstring(symbol) -> str | None:
        """
        Return the docstring of the callable underlying a symbol.

        Parameters
        ----------
        symbol : neurolang Symbol (Callable type)

        Returns
        -------
        str | None
            Docstring, or ``None`` when unavailable.
        """
        callable_obj = getattr(symbol, "value", None)
        if callable_obj is None:
            return None
        doc = getattr(callable_obj, "__doc__", None)
        return doc

    def _build_schema(self, engine) -> dict:
        """
        Iterate ``engine.symbols`` and build the schema dict.

        Parameters
        ----------
        engine : NeurolangPDL | NeurolangDL
            An acquired engine from the pool.

        Returns
        -------
        dict
            ``{"relations": [...], "functions": [...], "probabilistic": [...]}``.
        """
        prob_symbols: frozenset = (
            engine.program_ir.probabilistic_predicate_symbols
        )

        relations = []
        functions = []
        probabilistic = []

        for name in engine.symbols:
            if name.startswith("_"):
                continue

            symbol = engine.symbols[name]

            if is_leq_informative(symbol.type, Callable):
                # ---------- function / aggregation symbol ----------
                entry = {
                    "name": name,
                    "type": "function",
                    "params": self._get_function_params(symbol),
                    "docstring": self._get_function_docstring(symbol),
                }
                functions.append(entry)

            elif is_leq_informative(symbol.type, AbstractSet):
                # ---------- relation (or probabilistic relation) ----------
                try:
                    params = list(engine.predicate_parameter_names(name))
                except Exception:
                    params = []

                entry = {
                    "name": name,
                    "params": params,
                }

                if name in prob_symbols:
                    entry["type"] = "probabilistic"
                    probabilistic.append(entry)
                else:
                    entry["type"] = "relation"
                    relations.append(entry)

        return {
            "relations": relations,
            "functions": functions,
            "probabilistic": probabilistic,
        }


class V2AtlasHandler(JSONRequestHandler):
    """
    Return the base64-encoded NIfTI atlas image for an engine.

    GET /v2/atlas/:engine

    Response shape
    --------------
    {"status": "ok", "data": {"image": "<base64-string>"}}

    Raises tornado.web.HTTPError 404 for unknown engine names.
    """

    def get(self, engine_key: str) -> None:
        """Return atlas image for *engine_key*.

        Parameters
        ----------
        engine_key : str
            Identifier of the engine (e.g. ``'neurosynth'``).

        Raises
        ------
        tornado.web.HTTPError
            404 when *engine_key* is not found or has no atlas.
        """
        LOG.debug("V2AtlasHandler: requesting atlas for '%s'.", engine_key)
        try:
            atlas = self.application.nqm.get_atlas(engine_key)
        except (IndexError, KeyError):
            raise tornado.web.HTTPError(
                status_code=404,
                log_message=f"Atlas for engine '{engine_key}' not found.",
            )

        self.write_json_reponse({"image": base64_encode_nifti(atlas)})
