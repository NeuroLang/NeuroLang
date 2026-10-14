"""
Description of an engine for the LLM: schema linking and values.

The model is told the predicates of the engine, with their columns, the
description the engine's configuration gives them and their size
(:func:`schema_summary`). The ReAct agent can then look at the rows of a
predicate (:func:`sample_rows`) and look for the constants matching a
word of the question (:func:`search_values`), so that ``'pain'`` is
written as the engine stores it.
"""

import difflib
from typing import Optional

from neurolang.expressions import Symbol
from neurolang.utils import engine_registry
from neurolang.utils.cli import result_to_dataframe
from neurolang.utils.predicate_catalog import predicate_catalog

# Order in which the kinds of predicates are listed.
KIND_ORDER = ("stored", "rule", "probabilistic", "loaded")
RULE_KINDS = frozenset({"derived", "session"})
MAX_VALUES_SCANNED = 200_000


def declared_predicates(engine_name: str) -> dict:
    """Return the predicates the configuration of *engine_name* declares."""
    if not engine_name:
        return {}
    return engine_registry.get_predicates(engine_name)


def _declared_columns(declared: dict) -> dict:
    """Return the declared columns of the predicates, by name."""
    return {
        name: list(info["columns"])
        for name, info in declared.items() if info.get("columns")
    }


def _description(info: dict) -> str:
    """Return the one-line description of a declared predicate."""
    return " ".join(str(info.get("description", "")).split())


class EngineContext:
    """
    The predicates of an engine, as the LLM is told about them.

    *engine_name* is the engine of the registry whose configuration
    gives the descriptions and columns of the predicates, and *loaded*
    the columns of the relations loaded in the REPL with ``.load``.
    """

    def __init__(self, nl, engine_name: str = "", loaded=None):
        self.nl = nl
        self.engine_name = engine_name
        self.loaded = loaded or {}
        self.declared = declared_predicates(engine_name)
        self.signatures = (
            engine_registry.get_signatures(engine_name)
            if self.declared else {}
        )

    def catalog(self) -> dict:
        """
        Return the predicates of the engine, by name.

        The rules are of kind ``rule``, whether the engine or the
        session defines them, and the columns the configuration declares
        replace the names of the variables of their head.
        """
        catalog = predicate_catalog(
            self.nl, _declared_columns(self.declared), self.loaded,
            self.signatures,
        )
        for name, info in catalog.items():
            if info.kind in RULE_KINDS:
                info = info._replace(kind="rule")
            columns = self.declared.get(name, {}).get("columns")
            if columns and (info.arity is None or info.arity == len(columns)):
                info = info._replace(
                    columns=tuple(columns), arity=len(columns)
                )
            catalog[name] = info
        return catalog

    def columns(self, name: str) -> Optional[list]:
        """Return the column names of the predicate *name*, if known."""
        info = self.catalog().get(name)
        if info is None:
            return None
        if info.columns:
            return list(info.columns)
        if info.arity is not None:
            return [f"c{i}" for i in range(info.arity)]
        return None

    def predicate_line(self, info, size=None) -> str:
        """
        Return the line describing the predicate *info*.

        Columns whose names are unknown are named ``c0``, ``c1``…, as in
        the rows of the predicate.
        """
        if info.columns is None and info.arity and info.kind != "function":
            info = info._replace(
                columns=tuple(f"c{i}" for i in range(info.arity))
            )
        line = info.signature()
        description = _description(self.declared.get(info.name, {}))
        if description:
            line += f" — {description}"
        notes = [info.kind]
        if size is not None:
            notes.append(f"{size} rows")
        return f"{line} [{', '.join(notes)}]"

    def _size(self, name: str):
        """Return the number of rows of the stored relation *name*."""
        relation = self._stored_relation(name)
        if relation is None:
            return None
        try:
            return len(relation)
        except Exception:
            return None

    def schema_summary(self) -> str:
        """
        Return the description of the engine's predicates for the LLM.

        The relations and rules come first, one per line, by kind; the
        functions, usable in comparisons and arithmetic, are listed on
        the last line.
        """
        catalog = self.catalog()
        lines = []
        for kind in KIND_ORDER:
            for name in sorted(catalog, key=str.lower):
                info = catalog[name]
                if info.kind == kind:
                    lines.append(
                        "- " + self.predicate_line(info, self._size(name))
                    )
        functions = sorted(
            name for name, info in catalog.items() if info.kind == "function"
        )
        if functions:
            lines.append("Functions: " + ", ".join(functions))
        return "\n".join(lines) or "(the engine defines no predicate)"

    def _stored_relation(self, name: str):
        """Return the relational set of the stored relation *name*, if any."""
        try:
            value = self.nl.program_ir.symbol_table[Symbol(name)]
        except KeyError:
            return None
        inner = getattr(value, "value", None)
        unwrap = getattr(inner, "unwrap", None)
        return unwrap() if unwrap is not None else None

    def frame(self, name: str):
        """
        Return the rows of the predicate *name* as a DataFrame.

        A stored relation is read as it is; a rule is evaluated with a
        query in a scope that is discarded afterwards.
        """
        columns = self.columns(name)
        relation = self._stored_relation(name)
        if relation is not None:
            frame = result_to_dataframe(relation)
            if columns and len(columns) == frame.shape[1]:
                frame.columns = columns
            return frame
        if columns is None:
            raise ValueError(f"Unknown predicate {name!r}.")
        variables = ", ".join(f"V{i}" for i in range(len(columns)))
        query = f"ans({variables}) :- {name}({variables})"
        with self.nl.scope:
            result = self.nl.execute_datalog_program(query)
        frame = result_to_dataframe(result)
        frame.columns = columns
        return frame

    def describe(self, name: str, rows: int = 5) -> str:
        """Return the description of the predicate *name* and a few rows."""
        catalog = self.catalog()
        info = catalog.get(name)
        if info is None:
            close = difflib.get_close_matches(name, catalog, n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            return f"Unknown predicate {name!r}.{hint}"
        text = self.predicate_line(info, self._size(name))
        if info.kind == "function":
            return text
        return f"{text}\n{self.sample_rows(name, rows)}"

    def sample_rows(self, name: str, rows: int = 5) -> str:
        """Return the first *rows* rows of the predicate *name* as text."""
        frame = self.frame(name)
        if frame.empty:
            return "(no rows)"
        return f"{len(frame)} rows, e.g.:\n" + frame.head(rows).to_string(
            index=False
        )

    def search_values(
        self, name: str, column: str, pattern: str, limit: int = 10
    ) -> str:
        """
        Return the values of *column* of *name* matching *pattern*.

        Values containing *pattern* (ignoring case) come first; when
        there are none, the values closest to it are given instead.
        """
        frame = self.frame(name)
        if column not in frame.columns:
            return (
                f"{name} has no column {column!r}; its columns are "
                f"{', '.join(map(str, frame.columns))}."
            )
        values = frame[column].drop_duplicates().head(MAX_VALUES_SCANNED)
        strings = values.astype(str)
        needle = pattern.lower()
        matching = list(strings[strings.str.lower().str.contains(
            needle, regex=False
        )])
        matching.sort(key=lambda value: (value.lower() != needle, len(value)))
        if matching:
            shown = matching[:limit]
            more = len(matching) - len(shown)
            text = ", ".join(repr(v) for v in shown)
            if more > 0:
                text += f" and {more} more"
            return f"Values of {name}.{column} matching {pattern!r}: {text}"
        close = difflib.get_close_matches(
            pattern, list(strings), n=limit, cutoff=0.6
        )
        if close:
            return (
                f"No value of {name}.{column} contains {pattern!r}; the "
                f"closest are: {', '.join(repr(v) for v in close)}"
            )
        return f"No value of {name}.{column} matches {pattern!r}."
