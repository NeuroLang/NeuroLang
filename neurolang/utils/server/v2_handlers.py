"""
V2 API handlers for the NeuroLang Sparklis GUI.

Provides:
    - V2EnginesHandler  : GET /v2/engines
    - V2SchemaHandler   : GET /v2/schema/:engine
    - V2AtlasHandler    : GET /v2/atlas/:engine
    - V2SuggestHandler  : POST /v2/suggest/:engine
"""
import inspect
import json
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


class V2SuggestHandler(JSONRequestHandler):
    """
    Return context-aware autocompletion suggestions for a partial
    Datalog program.

    POST /v2/suggest/:engine

    Request body (JSON)
    -------------------
    {
        "program": "<datalog program text up to cursor>",
        "cursor_position": <int, optional>
    }

    Response shape
    --------------
    {
        "status": "ok",
        "data": {
            "Identifiers": ["PeakReported", "Study", ...],
            "Signs": ["@", "(", ...],
            "Operators": ["¬", "~"],
            ...
        }
    }

    On parser/autocompletion error:
    {
        "status": "error",
        "message": "<error description>"
    }

    Raises tornado.web.HTTPError 404 for unknown engine names, 503 when
    the engine pool is temporarily exhausted.
    """

    def post(self, engine_key: str) -> None:
        """Return autocompletion suggestions for *engine_key*.

        Parameters
        ----------
        engine_key : str
            Identifier of the engine (e.g. ``'neurosynth'``).

        Raises
        ------
        tornado.web.HTTPError
            404 when *engine_key* is not found.
            503 when the engine is temporarily unavailable.
        """
        LOG.debug(
            "V2SuggestHandler: requesting suggestions for '%s'.", engine_key
        )
        nqm = self.application.nqm
        engine_set = nqm.engines.get(engine_key)
        if engine_set is None:
            raise tornado.web.HTTPError(
                status_code=404,
                log_message=f"Engine '{engine_key}' not found.",
            )

        # Parse the request body
        try:
            body = json.loads(self.request.body or b"{}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise tornado.web.HTTPError(
                status_code=400,
                log_message=f"Invalid JSON request body: {exc}",
            )

        if "program" not in body:
            raise tornado.web.HTTPError(
                status_code=400,
                log_message="Request body must contain 'program' field.",
            )

        program: str = body["program"]
        cursor_position: int | None = body.get("cursor_position")

        with engine_set.engine() as engine:
            if engine is None:
                raise tornado.web.HTTPError(
                    status_code=503,
                    log_message=(
                        f"Engine '{engine_key}' is temporarily unavailable."
                    ),
                )
            suggestions = self._get_suggestions(engine, program, cursor_position)

        self.write_json_reponse(suggestions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_program_at_cursor(
        program: str, cursor_position: int | None
    ) -> tuple[str, str]:
        """Split *program* at *cursor_position* into a complete prefix
        and the partial suffix up to the cursor.

        The first return value (``complete_code``) contains all complete
        Datalog statements that appear before the cursor.  The second
        value (``partial_code``) is the program text from the start up
        to (and including) the cursor position.

        Parameters
        ----------
        program : str
            The full program text (or text up to the cursor).
        cursor_position : int | None
            Zero-based character position of the cursor.  When ``None``
            the entire program string is used as both the complete code
            and the partial code.

        Returns
        -------
        tuple[str, str]
            ``(complete_code, partial_code)``
        """
        if cursor_position is None:
            # No explicit cursor: treat the full text as the
            # autocompletion code.  The "complete" prefix is derived by
            # stripping any trailing incomplete statement (i.e., the
            # last statement without a terminating newline or period).
            partial_code = program
        else:
            partial_code = program[:cursor_position]

        # Build complete_code by keeping only full lines (terminated by
        # newline) from partial_code.  This avoids passing a broken
        # mid-statement line to the parser as "complete".
        last_newline = partial_code.rfind("\n")
        if last_newline == -1:
            # No newline found: the program is a single partial line,
            # so the "complete" portion is empty.
            complete_code = ""
        else:
            complete_code = partial_code[: last_newline + 1]

        return complete_code, partial_code

    @staticmethod
    def _fallback_identifiers(engine) -> dict:
        """Return a suggestions dict whose Identifiers contain all
        predicate/relation names available on *engine*.

        Used when the program is empty or when autocompletion fails.

        Parameters
        ----------
        engine : NeurolangPDL | NeurolangDL
            An acquired engine from the pool.

        Returns
        -------
        dict
            Suggestions dict with at least an ``Identifiers`` key.
        """
        identifiers = [
            name
            for name in engine.symbols
            if not name.startswith("_")
        ]
        return {
            "Identifiers": identifiers,
            "Signs": [],
            "Operators": [],
            "Numbers": [],
            "Text": [],
            "Cmd_identifier": [],
            "Functions": [],
            "Identifier_regexp": [],
            "Reserved words": [],
            "Boleans": [],
            "Expression symbols": [],
            "Python string": [],
            "Strings": [],
            "commands": [],
            "functions": [],
            "base symbols": identifiers,
            "query symbols": [],
        }

    @staticmethod
    def _serialise_suggestions(raw: dict) -> dict:
        """Convert a raw suggestions dict (which may contain ``set``
        values) into a JSON-serialisable dict whose values are lists.

        Parameters
        ----------
        raw : dict
            Result from ``compute_datalog_program_for_autocompletion``.

        Returns
        -------
        dict
            Same keys, values converted to sorted lists.
        """
        return {
            key: sorted(value) if isinstance(value, (set, frozenset))
            else list(value) if not isinstance(value, list)
            else value
            for key, value in raw.items()
        }

    def _get_suggestions(
        self,
        engine,
        program: str,
        cursor_position: int | None,
    ) -> dict:
        """Compute autocompletion suggestions for *program*.

        Parameters
        ----------
        engine : NeurolangPDL | NeurolangDL
            An acquired engine from the pool.
        program : str
            The Datalog program text.
        cursor_position : int | None
            Zero-based cursor position, or ``None`` to use the end of the
            program.

        Returns
        -------
        dict
            JSON-serialisable suggestions dict.
        """
        # Edge case: empty program – return available predicates.
        if not program or not program.strip():
            return self._fallback_identifiers(engine)

        complete_code, partial_code = self._split_program_at_cursor(
            program, cursor_position
        )

        try:
            raw = engine.compute_datalog_program_for_autocompletion(
                complete_code, partial_code
            )
        except Exception as exc:
            LOG.debug(
                "V2SuggestHandler: autocompletion error for program=%r: %s",
                program,
                exc,
            )
            # Return empty suggestions with an error message rather than
            # propagating a 500.
            return {
                "Identifiers": [],
                "Signs": [],
                "Operators": [],
                "Numbers": [],
                "Text": [],
                "message": str(exc),
            }

        if raw is None:
            return self._fallback_identifiers(engine)

        return self._serialise_suggestions(raw)
