"""
Session commands of the interactive REPL.

The rules defined during a session, and the relations loaded with
``.load``, go to a scope of the engine's program opened when the REPL
starts (:func:`open_session_scope`), so that ``.idb`` lists the rules,
``.drop`` removes a rule, a predicate or a relation, and ``.clear``
removes them all. Each submission runs in a scope of its own, committed
to the session's only if the whole submission succeeds (see
:mod:`.submission`). The columns of the loaded relations, named by the
header of their file, are kept in the application's ``loaded_relations``
to name the columns of the results and of ``.sets``. The other commands
list the engines and the engine's sets and predicates, and show how the
last query is rewritten (``.rewritten``) or translated from SQUALL to
Datalog, and the Datalog rules of a predicate (``.datalog``).
"""

import re
import shlex
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from rich import box
from rich.table import Table

from neurolang.datalog import Union as DatalogUnion
from neurolang.datalog.basic_representation import UnionOfConjunctiveQueries
from neurolang.datalog.expressions import AdornedSymbol
from neurolang.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from neurolang.expressions import Query, Symbol
from neurolang.frontend.datalog.pretty_printer import DatalogPrettyPrinter
from neurolang.frontend.datalog.squall_syntax_lark import (
    SquallProgram,
    parser as squall_parser,
)
from neurolang.logic import Implication
from neurolang.utils import engine_registry
from neurolang.utils.cli import (
    _execute_program,
    _execute_squall_program,
    _list_predicates,
    _list_sets,
    _show_squall_datalog,
)

from .commands import _switch, command

def open_session_scope(nl) -> None:
    """Open the scope of the program of *nl* holding the session's rules."""
    nl.program_ir.push_scope()


def session_idb_symbols(nl) -> set:
    """
    Return the symbols of the predicates defined in the session.

    The helpers NeuroLang compiles a conditional probability into, such
    as ``P^cond_num`` and ``P^cond_den`` for ``P``, are left out: they
    are named ``P`` too.
    """
    symbol_table = nl.program_ir.symbol_table
    return {
        symbol
        for symbol in nl.program_ir.intensional_database()
        if symbol in symbol_table._symbols
        and not isinstance(symbol, AdornedSymbol)
    }


def _drop_helpers(app, name: str) -> None:
    """Remove the session's helper predicates of the predicate *name*."""
    symbol_table = app._nl.program_ir.symbol_table
    helpers = [
        symbol for symbol in symbol_table._symbols
        if isinstance(symbol, AdornedSymbol) and symbol.name == name
    ]
    for symbol in helpers:
        del symbol_table[symbol]


def _warn(app, message: str) -> None:
    """Print *message*, which tells why a command did nothing."""
    app.console.print(message, style="yellow", markup=False)


def _plural(count: int, noun: str) -> str:
    """Return *count* and *noun*, e.g. ``1 rule`` or ``2 rules``."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def rule_text(rule) -> str:
    """Return the Datalog text of *rule* on one line."""
    return re.sub(r"\s*\n\s*", " ", DatalogPrettyPrinter().walk(rule))


def _print_idb_table(app) -> None:
    """Print the session's predicates with their number of rules."""
    symbols = session_idb_symbols(app._nl)
    if not symbols:
        app.console.print("  [dim](no session-defined IDB predicates)[/dim]")
        return
    idb = app._nl.program_ir.intensional_database()
    table = Table(
        title="Session IDB predicates", box=box.ROUNDED,
        header_style="bold cyan",
    )
    for column in ("Predicate", "Rules", "Arity"):
        table.add_column(column)
    for symbol in sorted(symbols, key=lambda s: s.name):
        rules = idb[symbol].formulas
        arity = len(rules[0].consequent.args)
        table.add_row(symbol.name, str(len(rules)), str(arity))
    app.console.print(table)
    app.console.print(
        f"\n[dim]Total session IDB predicates: {len(symbols)}[/dim]"
    )


def _print_rules(app, name: str) -> None:
    """Print the rules of the IDB predicate *name*, numbered from 1."""
    symbol_table = app._nl.program_ir.symbol_table
    symbol = Symbol(name)
    value = symbol_table[symbol] if symbol in symbol_table else None
    if not isinstance(value, DatalogUnion):
        _warn(app, f"{name!r} is not an IDB predicate.")
        return
    width = len(str(len(value.formulas)))
    for number, rule in enumerate(value.formulas, 1):
        app.console.print(
            f"{number:>{width}}  {rule_text(rule)}", markup=False,
            highlight=False,
        )


@command(
    ".idb", "List the predicates defined in this session, or the rules "
    "of one", ".idb [PREDICATE]",
)
def cmd_idb(app, arg: str) -> None:
    """Print the session's predicates, or the rules of the one in *arg*."""
    name = arg.strip()
    if name:
        _print_rules(app, name)
    else:
        _print_idb_table(app)


def _session_value(app, name: str):
    """
    Return the value of *name* in the session's scope.

    If *name* is not defined in this session, print why and return None.
    """
    symbol = Symbol(name)
    symbol_table = app._nl.program_ir.symbol_table
    if symbol in symbol_table._symbols:
        return symbol_table[symbol]
    if symbol in symbol_table:
        _warn(
            app, f"Predicate {name!r} is part of the engine; only the "
            "session's predicates and relations can be removed.",
        )
    else:
        _warn(app, f"Predicate {name!r} is not defined.")
    return None


def _engine_rules(app, name: str) -> tuple:
    """Return the rules of *name* defined outside the session's scope."""
    enclosing = app._nl.program_ir.symbol_table.enclosing_scope
    if enclosing is None or Symbol(name) not in enclosing:
        return ()
    value = enclosing[Symbol(name)]
    return value.formulas if isinstance(value, DatalogUnion) else ()


def _drop_all_rules(app, name: str, rules: tuple) -> None:
    """Remove the session's rules of the predicate *name*."""
    del app._nl.program_ir.symbol_table[Symbol(name)]
    _drop_helpers(app, name)
    app.squall_rules.pop(name, None)
    kept = len(_engine_rules(app, name))
    message = f"removed {name} ({_plural(len(rules) - kept, 'rule')})"
    if kept:
        message += f"; its {_plural(kept, 'engine rule')} remain"
    app.console.print(message, style="green", markup=False)


def _drop_rule(app, name: str, rules: tuple, number: int) -> None:
    """Remove the rule *number*, counted from 1, of the predicate *name*."""
    if not 1 <= number <= len(rules):
        _warn(
            app, f"{name} has {_plural(len(rules), 'rule')}; there is no "
            f"rule {number}. See .idb {name}.",
        )
        return
    if rules[number - 1] in _engine_rules(app, name):
        _warn(app, f"Rule {number} of {name} is part of the engine.")
        return
    remaining = rules[:number - 1] + rules[number:]
    # its rules as compiled are not numbered as those SQUALL translated
    app.squall_rules.pop(name, None)
    symbol_table = app._nl.program_ir.symbol_table
    if remaining:
        symbol = Symbol(name).cast(UnionOfConjunctiveQueries)
        symbol_table[symbol] = DatalogUnion(remaining)
        left = f"{_plural(len(remaining), 'rule')} left"
    else:
        del symbol_table[Symbol(name)]
        _drop_helpers(app, name)
        left = f"no rules left: {name} removed"
    app.console.print(
        f"removed rule {number} of {name} ({left})", style="green",
        markup=False,
    )


def drop(app, name: str, number=None) -> None:
    """
    Remove the session predicate or relation *name*.

    With a rule *number*, counted from 1 as ``.idb NAME`` shows them,
    remove only that rule of the predicate *name*.
    """
    value = _session_value(app, name)
    if value is None:
        return
    if isinstance(value, DatalogUnion):
        if number is None:
            _drop_all_rules(app, name, value.formulas)
        else:
            _drop_rule(app, name, value.formulas, number)
    elif not isinstance(
        getattr(value, "value", None), WrappedRelationalAlgebraSet
    ):
        _warn(app, f"{name!r} is neither an IDB predicate nor a relation.")
    elif number is not None:
        _warn(app, f"{name!r} is a relation, it has no rules.")
    else:
        del app._nl.program_ir.symbol_table[Symbol(name)]
        app.loaded_relations.pop(name, None)
        app.console.print(
            f"removed relation {name}", style="green", markup=False
        )


def _rule_number(text: str):
    """Return the rule number *text*, or None if it is not one."""
    return int(text) if text.isdigit() and int(text) > 0 else None


@command(
    ".drop", "Remove rule N of a session predicate, all its rules, or a "
    "loaded relation", ".drop PREDICATE [N]",
)
def cmd_drop(app, arg: str) -> None:
    """Remove the session predicate, rule or relation named in *arg*."""
    words = arg.split()
    if not 1 <= len(words) <= 2:
        _warn(app, "Usage: .drop PREDICATE [N]")
        return
    if len(words) == 1:
        drop(app, words[0])
        return
    number = _rule_number(words[1])
    if number is None:
        _warn(
            app, f"The rule number must be a positive integer, "
            f"not {words[1]!r}.",
        )
        return
    drop(app, words[0], number)


@command(
    ".clear", "Clear the predicates and relations of this session, or one",
    ".clear [PREDICATE]",
)
def cmd_clear(app, arg: str) -> None:
    """Remove the session's predicates and relations, or the one in *arg*."""
    name = arg.strip()
    if name:
        drop(app, name)
        return
    app._nl.program_ir.pop_scope()
    open_session_scope(app._nl)
    app.loaded_relations.clear()
    app.squall_rules.clear()
    app.console.print(
        "[green]All session-defined IDB predicates and loaded relations "
        "cleared.[/green]"
    )


def default_relation_name(path) -> str:
    """
    Return the default name of the relation loaded from *path*.

    It is the file name without its ``.csv``/``.tsv`` and ``.gz``
    extensions, in lower case, with the characters other than letters
    and digits replaced by ``_`` and ``rel_`` prepended if it would start
    with a digit.
    """
    stem = re.sub(r"(\.(csv|tsv))?(\.gz)?$", "", Path(path).name.lower())
    name = re.sub(r"[^a-z0-9]", "_", stem)
    return f"rel_{name}" if not name or name[0].isdigit() else name


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _name_problem(app, name: str, explicit: bool):
    """
    Return why *name* cannot name a loaded relation, or None.

    Only a relation loaded in this session can be replaced, and only
    when its name is given *explicit*-ly.
    """
    if not _IDENTIFIER.match(name):
        return (
            f"Invalid relation name {name!r}: use letters, digits and "
            "'_', not starting with a digit."
        )
    if Symbol(name) not in app._nl.program_ir.symbol_table:
        return None
    if name not in app.loaded_relations:
        return (
            f"{name!r} is already defined; give another name: "
            ".load PATH NAME"
        )
    if not explicit:
        return (
            f"Relation {name!r} is already loaded; to replace it, give "
            f"its name: .load PATH {name}"
        )
    return None


def _read_rows(path: Path) -> pd.DataFrame:
    """Return the rows of the relation file *path*, which must have some."""
    df = engine_registry.read_relation_file(path)
    if df.empty:
        raise pd.errors.EmptyDataError("no rows")
    return df


def _load_error_message(path: Path, exc: Exception) -> str:
    """Return the one-line message of the error *exc* loading *path*."""
    if isinstance(exc, FileNotFoundError):
        return f"No such file: {path}"
    if isinstance(exc, engine_registry.UnsupportedRelationFile):
        return str(exc)
    if isinstance(exc, pd.errors.EmptyDataError):
        return f"Cannot load {path}: the file is empty (no rows)."
    lines = str(exc).strip().splitlines() or [type(exc).__name__]
    return f"Cannot load {path}: {lines[0]}"


def _load_failed(app, message: str, exc=None) -> None:
    """Report that ``.load`` failed with *message* and the error *exc*."""
    app.failed = True
    if exc is not None and app.settings.debug:
        traceback.print_exception(
            type(exc), exc, exc.__traceback__, file=app.err_console.file
        )
    app.err_console.print(message, style="red", markup=False)


def _add_relation(app, name: str, df: pd.DataFrame, path: Path) -> None:
    """Add the rows *df* read from *path* as the session relation *name*."""
    replaced = name in app.loaded_relations
    if replaced:
        del app._nl.program_ir.symbol_table[Symbol(name)]
    app._nl.add_tuple_set(df, name=name)
    columns = [str(column) for column in df.columns]
    app.loaded_relations[name] = columns
    message = (
        f"loaded {name}({', '.join(columns)}): "
        f"{_plural(len(df), 'row')} from {path}"
    )
    if replaced:
        message += f" (replaces the previous {name})"
    app.console.print(message, style="green", markup=False)


def _load_arguments(arg: str):
    """Return the words ``PATH [NAME]`` of *arg*, or None if malformed."""
    try:
        words = shlex.split(arg)
    except ValueError:
        return None
    return words if 1 <= len(words) <= 2 else None


@command(
    ".load", "Load a CSV or TSV file (optionally gzipped) as a relation",
    ".load PATH [NAME]",
)
def cmd_load(app, arg: str) -> None:
    """Load the file named in *arg* as a relation of the session."""
    words = _load_arguments(arg)
    if words is None:
        _load_failed(app, "Usage: .load PATH [NAME]")
        return
    path = Path(words[0]).expanduser()
    name = words[1] if len(words) == 2 else default_relation_name(path)
    try:
        df = _read_rows(path)
    except Exception as exc:
        _load_failed(app, _load_error_message(path, exc), exc)
        return
    problem = _name_problem(app, name, explicit=len(words) == 2)
    if problem:
        _load_failed(app, problem)
        return
    try:
        _add_relation(app, name, df, path)
    except Exception as exc:
        _load_failed(app, _load_error_message(path, exc), exc)


@command(".sets", "List the engine's sets (EDB relations) and the loaded ones")
def cmd_sets(app, arg: str) -> None:
    """Print the relational algebra sets of the engine."""
    _list_sets(app._nl, app.loaded_relations)


@command(".predicates", "List the predicates the engine declares")
def cmd_predicates(app, arg: str) -> None:
    """Print the predicates declared in the engine's configuration."""
    if app.settings.engine_name:
        _list_predicates(app.settings.engine_name)


@command(".engines", "List the available engines")
def cmd_engines(app, arg: str) -> None:
    """Print the engines of the registry."""
    engine_registry.show_engines()


def _no_last_query(app) -> bool:
    """Print a message and return True if no query has been run yet."""
    if app.last_query is None:
        app.console.print("[yellow]No query has been run yet.[/yellow]")
        return True
    return False


def _replace(symbol_table, symbol, value) -> None:
    """Set *symbol* to *value* in *symbol_table*, or remove it if None."""
    if symbol in symbol_table._symbols:
        del symbol_table[symbol]
    if value is not None:
        symbol_table[symbol] = value


@contextmanager
def last_query_scope(app):
    """
    Open a scope of the program where the last query can run again.

    The rules that its submission added are taken out of the session
    while the block runs: running the query again adds them again, and
    SQUALL gives them new variables, so that they would be there twice.
    """
    session = app._nl.program_ir.symbol_table
    saved = {}
    for symbol, added in app.last_added_rules.items():
        value = session._symbols.get(symbol)
        if isinstance(value, DatalogUnion):
            saved[symbol] = value
            rules = tuple(r for r in value.formulas if r not in added)
            _replace(session, symbol, DatalogUnion(rules) if rules else None)
    try:
        with app._nl.scope:
            yield
    finally:
        for symbol, value in saved.items():
            _replace(session, symbol, value)


def rerun_last_query(app, **flags) -> None:
    """
    Run the last query again, in a scope, with the execution *flags*.

    The flags are those of :func:`~neurolang.utils.cli._execute_program`,
    such as ``dry_run``; errors are reported as when it first ran.
    """
    if _no_last_query(app):
        return
    program, squall = app.last_query
    execute = _execute_squall_program if squall else _execute_program
    try:
        with last_query_scope(app):
            execute(app._nl, program, **flags)
    except Exception as exc:
        app._report_error(exc, program)


@command(
    ".rewritten",
    "Show the rewritten program of the last query, or of every query",
    ".rewritten [on|off]",
)
def cmd_rewritten(app, arg: str) -> None:
    """Print the magic-sets rewriting of the last query, or toggle it."""
    if arg.strip():
        _switch(app, arg, "show_rewritten", "Rewritten program")
        return
    rerun_last_query(app, dry_run=True)


def _parsed_rules(parsed) -> tuple:
    """Return the rules of the parsed SQUALL program *parsed*."""
    if isinstance(parsed, SquallProgram):
        formulas = parsed.rules
    else:
        formulas = getattr(parsed, "formulas", None) or (parsed,)
    return tuple(f for f in formulas if isinstance(f, Implication))


def record_squall_rules(app, program: str) -> None:
    """
    Keep the Datalog rules of the SQUALL *program*, which just ran.

    They are kept by predicate name in the application's
    ``squall_rules``, as SQUALL translated them, for ``.datalog
    PREDICATE``: NeuroLang compiles some of them further, for instance a
    conditional probability into the predicate and two helpers.
    """
    if "define" not in program.lower():
        return
    for rule in _parsed_rules(squall_parser(program)):
        name = rule.consequent.functor.name
        app.squall_rules.setdefault(name, []).append(rule)


def _print_datalog_of(app, name: str) -> None:
    """
    Print the Datalog rules of the predicate *name*.

    A predicate defined in SQUALL during the session has its rules as
    SQUALL translated them; any other has its rules as the engine holds
    them.
    """
    rules = app.squall_rules.get(name)
    if rules is None:
        symbol_table = app._nl.program_ir.symbol_table
        value = symbol_table.get(Symbol(name))
        if not isinstance(value, DatalogUnion):
            _warn(app, f"{name!r} is not an IDB predicate.")
            return
        rules = value.formulas
    printer = DatalogPrettyPrinter()
    app.console.print(
        "\n\n".join(printer.walk(rule) for rule in rules), markup=False,
        highlight=False, soft_wrap=True,
    )


def _print_datalog_program(app, program: str) -> None:
    """Print the rules and the query of the Datalog *program*."""
    printer = DatalogPrettyPrinter()
    parsed = app._nl.datalog_parser(program)
    for formula in getattr(parsed, "formulas", (parsed,)):
        label = "query" if isinstance(formula, Query) else "rule"
        app.console.print(
            f"── {label} ──\n{printer.walk(formula)}\n", markup=False,
            highlight=False, soft_wrap=True,
        )


@command(
    ".datalog", "Show the Datalog program of the last query, translated "
    "from SQUALL, or the Datalog rules of a predicate",
    ".datalog [PREDICATE]",
)
def cmd_datalog(app, arg: str) -> None:
    """
    Print the Datalog rules of the predicate in *arg*.

    Without a predicate, print the Datalog program of the last query,
    as the SQUALL of a SQUALL query translates to.
    """
    if arg.strip():
        _print_datalog_of(app, arg.strip())
        return
    if _no_last_query(app):
        return
    program, squall = app.last_query
    try:
        if squall:
            _show_squall_datalog(program)
        else:
            _print_datalog_program(app, program)
    except Exception as exc:
        app._report_error(exc, program)
