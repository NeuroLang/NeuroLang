"""
Submissions to the engine of the interactive REPL.

A submission is one input of the REPL: a Datalog program of one or more
rules, facts and possibly a query, or a SQUALL program of one or more
sentences. It runs in a scope of its own (:class:`Submission`), whose
symbols are committed to the session's scope only if the whole
submission succeeds, and then tells what it defined, one line per
predicate (:meth:`Submission.definitions`).
"""

from neurolang.datalog import Fact, Union as DatalogUnion
from neurolang.datalog.expressions import AdornedSymbol
from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from neurolang.expressions import Constant, Query
from neurolang.frontend.datalog.squall_syntax_lark import (
    SquallProgram,
    parser as squall_parser,
)


def is_user_symbol(symbol) -> bool:
    """
    Return whether *symbol* names a predicate the user defined.

    The symbols NeuroLang generates are left out: fresh symbols, the
    adorned symbols of magic sets and of the helpers of a conditional
    probability (``P^cond_num``), and private names such as
    ``__constraints__``.
    """
    name = getattr(symbol, "name", "")
    return not (
        isinstance(symbol, AdornedSymbol)
        or name.startswith(("_", "fresh_"))
    )


def _plural(count: int, noun: str) -> str:
    """Return *count* and *noun*, e.g. ``1 rule`` or ``2 rules``."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def relation_rows(value):
    """Return the rows of *value* if it is a relation, None otherwise."""
    rows = getattr(value, "value", None)
    return rows if isinstance(rows, WrappedRelationalAlgebraSet) else None


def predicate_extent(value):
    """
    Return the arity, size and unit of the predicate *value*, or None.

    A predicate is defined by rules, a union whose size is its number of
    rules, or by facts, a relation whose size is its number of rows.
    """
    if isinstance(value, DatalogUnion) and value.formulas:
        rules = value.formulas
        return len(rules[0].consequent.args), len(rules), "rule"
    rows = relation_rows(value)
    if rows is not None:
        return rows.arity, len(rows), "fact"
    return None


def definition_line(name: str, extent, previous) -> str:
    """
    Return the line telling that the predicate *name* was defined.

    *extent* and *previous* are the arity, size and unit of the
    predicate after and before the submission (see :func:`predicate_extent`);
    *previous* is None for a new predicate.
    """
    arity, size, unit = extent
    if previous is None:
        more = f" ({_plural(size, unit)})" if size > 1 else ""
        return f"defined {name}/{arity}{more}"
    if previous[1] == size:
        return f"{name}/{arity} unchanged ({_plural(size, unit)})"
    return f"defined {name}/{arity} (now {_plural(size, unit)})"


def _datalog_outline(nl, program: str) -> tuple:
    """Return whether the Datalog *program* has a query, and its facts."""
    parsed = nl.datalog_parser(program)
    formulas = getattr(parsed, "formulas", (parsed,))
    has_query = any(isinstance(f, Query) for f in formulas)
    facts = {f.fact.functor for f in formulas if isinstance(f, Fact)}
    return has_query, facts


def submission_outline(nl, program: str, squall: bool) -> tuple:
    """
    Return whether *program* has a query, and the predicates of its facts.

    A query is an ``ans`` rule in Datalog and an ``obtain`` in SQUALL,
    whose programs have no facts. A program that does not parse counts
    as a query, without facts: running it reports the error.
    """
    try:
        if not squall:
            return _datalog_outline(nl, program)
        parsed = squall_parser(program)
    except Exception:
        return True, set()
    return isinstance(parsed, SquallProgram) and bool(parsed.queries), set()


def _commit(scope, session) -> dict:
    """
    Move the symbols of *scope* into the enclosing scope *session*.

    A symbol already in *session* gets the value it has in *scope*: a
    rule added to a predicate gives *scope* the union of all its rules.
    Return the values the moved symbols had, looked up from *session*,
    by symbol; None for the symbols that were not defined.
    """
    previous = {}
    for symbol, value in scope._symbols.items():
        previous[symbol] = session.get(symbol)
        if session._symbols.get(symbol) is not None:
            del session[symbol]
        session[symbol] = value
    return previous


class Submission:
    """
    The scope of the program of an engine where one submission runs.

    The submission runs in a scope of its own, opened in the session's
    when the ``with`` block starts. If the block succeeds, what the
    submission defined is committed to the session's scope; if it fails,
    including on KeyboardInterrupt, it is discarded, so that the session
    is exactly as before. The session's scope is restored even when the
    failure left a scope of the engine open.
    """

    def __init__(self, nl, facts=()):
        """
        Prepare a submission to the program of the engine *nl*.

        *facts* are the symbols of the predicates the submission adds
        facts to (see :func:`submission_outline`).
        """
        self._program_ir = nl.program_ir
        self._facts = facts
        self.session = None
        self.scope = None
        self.previous = None

    def __enter__(self):
        """Open the submission's scope and return the submission."""
        self.session = self._program_ir.symbol_table
        self._program_ir.push_scope()
        self.scope = self._program_ir.symbol_table
        self._copy_relations()
        return self

    def __exit__(self, exc_type, exc, traceback_) -> bool:
        """Commit the scope if the block succeeded, discard it otherwise."""
        self._program_ir.symbol_table = self.session
        if exc_type is None:
            self.previous = _commit(self.scope, self.session)
        return False

    def _copy_relations(self) -> None:
        """
        Copy into the scope the relations the submission adds facts to.

        NeuroLang adds a fact to its relation in place: it is added to
        the copy instead, which is discarded with the scope.
        """
        for symbol in self._facts:
            value = self.session.get(symbol)
            rows = relation_rows(value)
            if rows is not None:
                self.scope[symbol] = Constant[value.type](
                    rows.copy(), auto_infer_type=False, verify_type=False
                )

    @property
    def committed(self) -> bool:
        """Return whether the submission succeeded and was committed."""
        return self.previous is not None

    @property
    def discarded(self) -> bool:
        """Return whether the submission failed after defining symbols."""
        return (
            not self.committed and self.scope is not None
            and bool(self.scope._symbols)
        )

    def added_rules(self) -> dict:
        """
        Return the rules the committed submission added, by symbol.

        They are the rules of the predicates it defined, helpers
        included, that the predicates did not have before.
        """
        added = {}
        for symbol, value in self.scope._symbols.items():
            if isinstance(value, DatalogUnion):
                before = getattr(self.previous.get(symbol), "formulas", ())
                added[symbol] = tuple(
                    rule for rule in value.formulas if rule not in before
                )
        return added

    def definitions(self) -> list:
        """
        Return a line per predicate the committed submission defined.

        See :func:`definition_line`; the symbols NeuroLang generates are
        left out (see :func:`is_user_symbol`).
        """
        lines = []
        for symbol, value in self.scope._symbols.items():
            extent = predicate_extent(value)
            if extent is not None and is_user_symbol(symbol):
                previous = predicate_extent(self.previous.get(symbol))
                lines.append(definition_line(symbol.name, extent, previous))
        return lines
