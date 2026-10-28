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
import re
from collections import Counter
from numbers import Number
from typing import Optional

from pandas.api.types import is_object_dtype, is_string_dtype

from nl_expr.expressions import Symbol
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.cli import result_to_dataframe
from neurolang_apps.utils.predicate_catalog import predicate_catalog

# Kinds of the predicates searched by search_values when none is given.
SEARCHED_KINDS = frozenset({"stored", "loaded", "probabilistic"})
MAX_FACTS_PER_COLUMN = 50
# The rows of stored relations, kept across the questions on an engine:
# id of the relation -> (relation, frame); the relation is kept so that
# its id is not reused while cached.
_STORED_FRAMES = {}
MAX_CACHED_FRAMES = 512
# Order in which the kinds of predicates are listed.
KIND_ORDER = ("stored", "rule", "probabilistic", "loaded")
RULE_KINDS = frozenset({"derived", "session"})
MAX_VALUES_SCANNED = 200_000
# A column with at most this many values lists them all when none matches.
MAX_VALUES_LISTED = 200


def declared_predicates(engine_name: str) -> dict:
    """
    Return the predicates the configuration of *engine_name* declares.

    An engine that is not in the registry declares none.
    """
    if engine_name not in engine_registry.list_engine_names():
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
        self._frames = {}
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
        line = info._replace(arguments=None).signature()
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
                if info.kind == kind and not _is_iri(name):
                    lines.append(
                        "- " + self.predicate_line(info, self._size(name))
                    )
        namespaces = Counter(
            re.sub(r"[^#/]*$", "", name) for name in catalog if _is_iri(name)
        )
        for namespace, count in sorted(namespaces.items()):
            lines.append(
                f"- {count} classes named by IRI in <{namespace}>, not "
                "listed: search_values finds their IRIs by label"
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
        query in a scope that is discarded afterwards. The rows are kept
        for the next calls.
        """
        if name not in self._frames:
            self._frames[name] = self._read_frame(name)
        return self._frames[name]

    def _read_frame(self, name: str):
        """Return the rows of the predicate *name* (see :meth:`frame`)."""
        columns = self.columns(name)
        relation = self._stored_relation(name)
        if relation is not None:
            cached = _STORED_FRAMES.get(id(relation))
            if cached is not None and cached[0] is relation:
                frame = cached[1]
            else:
                frame = result_to_dataframe(relation)
                if len(_STORED_FRAMES) >= MAX_CACHED_FRAMES:
                    _STORED_FRAMES.pop(next(iter(_STORED_FRAMES)))
                _STORED_FRAMES[id(relation)] = (relation, frame)
            if columns and len(columns) == frame.shape[1]:
                frame = frame.set_axis(columns, axis=1)
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

    def describe(self, name: str, rows: int = 3) -> str:
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
        facts = "\n".join(
            _fact(name, row) for row in frame.head(rows).values.tolist()
        )
        return f"{len(frame)} rows, e.g.:\n{facts}"

    def _column(self, frame, column):
        """
        Return the name of *column* of *frame*, given by name or number.

        Raise ValueError when *frame* has no such column.
        """
        columns = [str(c) for c in frame.columns]
        if isinstance(column, int) or str(column).isdigit():
            index = int(column)
            if 0 <= index < len(columns):
                return frame.columns[index]
        elif str(column) in columns:
            return frame.columns[columns.index(str(column))]
        raise ValueError(
            f"no column {column!r}; its columns are {', '.join(columns)}"
        )

    def _searched_predicates(self) -> list:
        """Return the stored relations, searched when none is given."""
        return [
            name for name, info in sorted(self.catalog().items())
            if info.kind in SEARCHED_KINDS
        ]

    def search_values(
        self, pattern: str, name: Optional[str] = None, column=None,
        limit: int = 12,
    ) -> str:
        """
        Return the facts whose values contain *pattern*, ignoring case.

        The facts searched are those of the predicate *name*, by default
        of all the stored relations, and the values those of its *column*
        (a name or a number), by default of all its columns of strings.
        The facts whose value is *pattern* come first. When none matches,
        the words of *pattern* are searched for, then the values closest
        to it.
        """
        pattern = str(pattern).strip()
        if not pattern:
            return "Give a pattern to search for."
        names = [name] if name else self._searched_predicates()
        if name and name not in self.catalog():
            return self.describe(name)
        try:
            found = self._matching_facts(names, pattern, column)
        except ValueError as exc:
            return f"{name} has {exc}."
        words = [w for w in pattern.split() if len(w) > 2]
        if not found and len(words) > 1:
            for word in sorted(words, key=len, reverse=True):
                found = self._matching_facts(names, word, column)
                if found:
                    pattern = word
                    break
        if found:
            shown = _diverse(found, limit)
            text = "\n".join(shown)
            if len(found) > len(shown):
                text += f"\n… and {len(found) - len(shown)} more"
            return f"Facts with values containing {pattern!r}:\n{text}"
        if name:
            return self._closest_values(name, column, pattern, limit)
        return f"No stored value contains {pattern!r}."

    def _matching_facts(self, names, pattern: str, column) -> list:
        """
        Return ``(rank, fact)`` of the facts with a value matching.

        A value matches when it contains *pattern*, ignoring case, or,
        for an IRI, *pattern* with its spaces as underscores (the IRI of
        DBpedia's Mount Everest ends with Mount_Everest).
        """
        needle = pattern.lower()
        iri_needle = "/" + needle.replace(" ", "_")
        found = []
        for name in names:
            frame = self.frame(name)
            if frame.empty:
                continue
            if column is not None and len(names) == 1:
                columns = [self._column(frame, column)]
            else:
                columns = _string_columns(frame)
            for c in columns:
                values = frame[c].astype(str)
                lowered = values.str.lower()
                mask = lowered.str.contains(needle, regex=False)
                mask |= lowered.str.contains(iri_needle, regex=False)
                for index in frame.index[mask][:MAX_FACTS_PER_COLUMN]:
                    value = lowered[index]
                    exact = value == needle or value.endswith(iri_needle)
                    rank = (not exact, len(value))
                    found.append(
                        (rank, _fact(name, frame.loc[index].tolist()))
                    )
        found.sort(key=lambda item: item[0])
        return list(dict.fromkeys(found))

    def _closest_values(self, name, column, pattern, limit) -> str:
        """Return the values of a predicate closest to *pattern*."""
        frame = self.frame(name)
        try:
            columns = (
                [self._column(frame, column)] if column is not None
                else _string_columns(frame)
            )
        except ValueError as exc:
            return f"{name} has {exc}."
        strings = sorted({
            str(v) for c in columns
            for v in frame[c].drop_duplicates().head(MAX_VALUES_SCANNED)
        })
        if len(strings) <= MAX_VALUES_LISTED:
            return (
                f"No value of {name} contains {pattern!r}; its values are: "
                f"{', '.join(repr(v) for v in strings)}"
            )
        close = difflib.get_close_matches(pattern, strings, n=limit,
                                          cutoff=0.6)
        if close:
            return (
                f"No value of {name} contains {pattern!r}; the closest "
                f"are: {', '.join(repr(v) for v in close)}"
            )
        return f"No value of {name} matches {pattern!r}."


def _diverse(found, limit: int, per_predicate: int = 3) -> list:
    """
    Return at most *limit* facts of *found*, ranked, at most
    *per_predicate* of each predicate, so that the properties of an
    entity show.
    """
    shown, counts = [], {}
    for _, fact in found:
        predicate = fact.split("(", 1)[0]
        if counts.get(predicate, 0) < per_predicate:
            counts[predicate] = counts.get(predicate, 0) + 1
            shown.append(fact)
            if len(shown) == limit:
                break
    return shown


def _string_columns(frame) -> list:
    """Return the columns of *frame* holding strings."""
    return [
        c for c in frame.columns
        if (is_object_dtype(frame[c]) or is_string_dtype(frame[c]))
        and isinstance(next(iter(frame[c].dropna()), ""), str)
    ]


def _is_iri(name: str) -> bool:
    """Whether the predicate *name* is an IRI (an ontology's class)."""
    return name.startswith(("http://", "https://"))


def _term(value) -> str:
    """Return *value* as written in Datalog: <IRI>, 'string' or number."""
    if isinstance(value, str):
        if value.startswith(("http://", "https://", "_:")):
            return f"<{value}>"
        return repr(value)
    if isinstance(value, Number):
        return str(value)
    return f"{type(value).__name__}(…)"


def _fact(name: str, values) -> str:
    """Return the fact *name*(*values*) as written in Datalog."""
    return f"{name}({', '.join(_term(v) for v in values)})"
