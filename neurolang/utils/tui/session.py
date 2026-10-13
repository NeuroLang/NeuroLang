"""
Session commands of the interactive REPL.

The rules defined during a session go to a scope of the engine's program
opened when the REPL starts (:func:`open_session_scope`), so that
``.idb`` lists them and ``.clear`` removes them. The other commands list
the engines and the engine's sets and predicates, and show how the last
query is rewritten (``.rewritten``) or translated from SQUALL to Datalog
(``.datalog``).
"""

from rich import box
from rich.table import Table

from neurolang.datalog import Union as DatalogUnion
from neurolang.expressions import Symbol
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
    # TODO(B4): a program that fails part-way leaves the rules walked
    # before the failure in this scope; they should be committed only
    # when the whole program succeeds, or rolled back on failure.
    nl.program_ir.push_scope()


def session_idb_symbols(nl) -> set:
    """Return the symbols of the predicates defined in the session."""
    symbol_table = nl.program_ir.symbol_table
    return {
        symbol
        for symbol in nl.program_ir.intensional_database()
        if symbol in symbol_table._symbols
    }


@command(".idb", "List the predicates defined in this session")
def cmd_idb(app, arg: str) -> None:
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


def _clear_predicate(app, name: str) -> None:
    """Remove the session predicate *name*, if it is one."""
    symbol = Symbol(name)
    symbol_table = app._nl.program_ir.symbol_table
    if symbol not in symbol_table:
        message = f"Predicate {name!r} is not defined."
    elif symbol not in symbol_table._symbols:
        message = (
            f"Predicate {name!r} is part of the engine; only "
            "session-defined IDB predicates can be cleared."
        )
    elif not isinstance(symbol_table[symbol], DatalogUnion):
        message = f"Predicate {name!r} is not an IDB predicate."
    else:
        del symbol_table[symbol]
        app.console.print(f"[green]IDB predicate {name!r} cleared.[/green]")
        return
    app.console.print(message, style="yellow", markup=False)


@command(
    ".clear", "Clear the predicates defined in this session, or one",
    ".clear [PREDICATE]",
)
def cmd_clear(app, arg: str) -> None:
    """Remove the session's predicates, or only the one named in *arg*."""
    name = arg.strip()
    if name:
        _clear_predicate(app, name)
        return
    app._nl.program_ir.pop_scope()
    open_session_scope(app._nl)
    app.console.print(
        "[green]All session-defined IDB predicates cleared.[/green]"
    )


@command(".sets", "List the engine's sets (EDB relations)")
def cmd_sets(app, arg: str) -> None:
    """Print the relational algebra sets of the engine."""
    _list_sets(app._nl)


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
    if _no_last_query(app):
        return
    program, squall = app.last_query
    execute = _execute_squall_program if squall else _execute_program
    try:
        with app._nl.scope:
            execute(app._nl, program, dry_run=True)
    except Exception as exc:
        app._report_error(exc, program)


@command(".datalog", "Show the Datalog translation of the last SQUALL query")
def cmd_datalog(app, arg: str) -> None:
    """Print the Datalog program the last SQUALL query compiles to."""
    if _no_last_query(app):
        return
    program, squall = app.last_query
    if not squall:
        app.console.print(
            "[yellow].datalog shows the Datalog translation of a SQUALL "
            "query; use .rewritten for a Datalog query.[/yellow]"
        )
        return
    try:
        _show_squall_datalog(program)
    except Exception as exc:
        app._report_error(exc, program)
