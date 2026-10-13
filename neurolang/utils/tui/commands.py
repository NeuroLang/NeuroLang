"""
Dot-commands of the interactive REPL.

Commands are registered in :data:`COMMANDS` with the :func:`command`
decorator, which records the handler together with its usage and help
text. A handler is called with the REPL application and the text that
follows the command name.
"""

import difflib
from typing import Callable, NamedTuple

from rich.panel import Panel

from neurolang.utils import engine_registry
from neurolang.utils.cli import _parse_sort_spec, sort_dataframe

from .results import ResultBlock, print_all


class QuitRepl(Exception):
    """Raised by a command to end the REPL."""


class Command(NamedTuple):
    """A dot-command: its handler, usage line and one-line help."""

    handler: Callable
    usage: str
    help: str


COMMANDS = {}


def command(name: str, description: str, usage: str = ""):
    """
    Register the decorated function as the dot-command *name*.

    *usage* is the command line shown in the help (default: *name*) and
    *description* its one-line description.
    """
    def register(handler):
        COMMANDS[name] = Command(handler, usage or name, description)
        return handler
    return register


def unknown_command_message(name: str) -> str:
    """Return the error message for the unknown command *name*."""
    message = f"Unknown command {name}."
    close = difflib.get_close_matches(name, COMMANDS, n=1)
    if close:
        message += f" Did you mean {close[0]}?"
    return message + " Type .help for the list of commands."


def run_command(app, line: str) -> None:
    """Run the dot-command on *line*, reporting unknown commands."""
    name, _, arg = line.strip().partition(" ")
    cmd = COMMANDS.get(name.lower())
    if cmd is None:
        app.failed = True
        app.console.print(
            unknown_command_message(name), style="yellow", markup=False
        )
        return
    cmd.handler(app, arg)


def engine_relations(engine_name: str) -> list:
    """
    Return ``name(col, ...)`` for each relation declared by the engine.

    Relations whose columns are not declared are shown by name only.
    """
    if not engine_name:
        return []
    return [
        f"{name}({', '.join(info['columns'])})" if info.get("columns")
        else name
        for name, info in engine_registry.get_predicates(engine_name).items()
    ]


def engine_examples(engine_name: str) -> dict:
    """
    Return an example query on the engine for each language.

    The examples obtain the first relation declared by the engine, as
    a dict from ``"datalog"`` and ``"squall"`` to the query text, or an
    empty dict when the engine declares no relation.
    """
    if not engine_name:
        return {}
    predicates = engine_registry.get_predicates(engine_name)
    for name, info in predicates.items():
        columns = ", ".join(info.get("columns", []))
        if columns:
            return {
                "datalog": f"ans({columns}) :- {name}({columns})",
                "squall": f"obtain every {name}.",
            }
    return {}


KEYS_HELP = """
[bold cyan]Keys:[/bold cyan]
  Enter                run the statement; an unfinished one (open bracket,
                       trailing ',' or ':-', SQUALL without its final '.')
                       continues on a new line, an empty line runs it
  Alt+Enter            insert a newline
  Ctrl+C               discard the input, or interrupt a running query
  Ctrl+D               exit
"""


def help_text(engine_name: str = "") -> str:
    """Return the help text: commands, keys and examples on the engine."""
    lines = [
        "[bold yellow]Interactive NeuroLang Query REPL[/bold yellow]",
        "",
        "Type a Datalog (or SQUALL) query and press Enter to run it.",
        "",
        "[bold cyan]Dot commands:[/bold cyan]",
    ]
    lines += [
        f"  [green]{cmd.usage:<20}[/green] {cmd.help}"
        for cmd in COMMANDS.values()
    ]
    examples = engine_examples(engine_name)
    if examples:
        lines += [
            KEYS_HELP, f"[bold cyan]Examples ({engine_name}):[/bold cyan]",
            f"  {examples['datalog']}", "  .more", "  .mode squall",
            f"  {examples['squall']}",
        ]
    else:
        lines.append(KEYS_HELP)
    return "\n".join(lines)


@command(".help", "Show this help message")
def cmd_help(app, arg: str) -> None:
    """Print the help text."""
    app.console.print(
        Panel(help_text(app.settings.engine_name), border_style="cyan")
    )


@command(".mode", "Switch the query language", ".mode datalog|squall")
def cmd_mode(app, arg: str) -> None:
    """Switch between Datalog and SQUALL input."""
    mode = arg.strip().lower()
    if mode == "datalog":
        app.settings.squall = False
        app.console.print("[green]Mode: Datalog[/green]")
    elif mode == "squall":
        app.settings.squall = True
        app.console.print("[green]Mode: SQUALL[/green]")
    else:
        app.console.print(
            f"[yellow]Unknown mode {mode!r}. "
            "Use 'datalog' or 'squall'.[/yellow]"
        )


@command(
    ".sort", "Show or set the sort of the results (none: unsorted)",
    ".sort [COL[:asc|desc] ...|none]",
)
def cmd_sort(app, arg: str) -> None:
    """Set the sort order of the results and re-sort the last one."""
    specs = arg.split()
    if not specs:
        sort_by = app.settings.sort_by
        current = ", ".join(
            f"{col}:{'asc' if asc else 'desc'}" for col, asc in sort_by
        )
        app.console.print(f"Sort: {current or 'none'}", markup=False)
        return
    if specs == ["none"]:
        app.settings.sort_by = []
        app.console.print("Sort: none")
        return
    app.settings.sort_by = _parse_sort_spec(specs)
    app.results = [
        ResultBlock(sort_dataframe(block.frame, app.settings.sort_by),
                    block.title)
        for block in app.results
    ]
    for block in app.results:
        app.print_block(block)


@command(".more", "Show the next page of the last result")
def cmd_more(app, arg: str) -> None:
    """Print the next page of each block of the last result."""
    blocks = [block for block in app.results if block.remaining > 0]
    if not blocks:
        app.console.print("[dim]No more rows.[/dim]")
    for block in blocks:
        app.print_block(block)


@command(
    ".head", "Set the page size to N and show the first page again",
    ".head [N]",
)
def cmd_head(app, arg: str) -> None:
    """Set the page size and print the first page of the last result."""
    if arg.strip():
        try:
            size = int(arg)
        except ValueError:
            size = 0
        if size <= 0:
            app.console.print(
                f"[yellow]Page size must be a positive integer, "
                f"not {arg.strip()!r}.[/yellow]"
            )
            return
        app.settings.page_size = size
    for block in app.results:
        block.shown = 0
        app.print_block(block)


@command(".all", "Show all the rows of the last result (in a pager)")
def cmd_all(app, arg: str) -> None:
    """Print all the rows of the last result."""
    for block in app.results:
        print_all(app.console, block)


FORMATS = ("table", "csv", "json")


@command(
    ".format", "Show or set the output format of the results",
    ".format [table|csv|json]",
)
def cmd_format(app, arg: str) -> None:
    """Set the output format of the results."""
    fmt = arg.strip().lower()
    if not fmt:
        app.console.print(f"Format: {app.settings.output_format}")
    elif fmt in FORMATS:
        app.settings.output_format = fmt
        app.console.print(f"[green]Format: {fmt}[/green]")
    else:
        app.console.print(
            f"[yellow]Unknown format {fmt!r}. "
            f"Use one of: {', '.join(FORMATS)}.[/yellow]"
        )


def _switch(app, arg: str, attribute: str, label: str) -> None:
    """Show, or set with ``on``/``off``, the boolean setting *attribute*."""
    value = arg.strip().lower()
    if value in ("on", "off"):
        setattr(app.settings, attribute, value == "on")
    elif value:
        app.console.print(
            f"[yellow]Use 'on' or 'off', not {value!r}.[/yellow]"
        )
        return
    state = "on" if getattr(app.settings, attribute) else "off"
    app.console.print(f"{label}: {state}")


@command(
    ".ra", "Print the relational algebra plans instead of running queries",
    ".ra [on|off]",
)
def cmd_ra(app, arg: str) -> None:
    """Toggle the display of the relational algebra plans."""
    _switch(app, arg, "show_ra", "RA plans")


@command(
    ".rewritten", "Print the magic-sets rewritten program of each query",
    ".rewritten [on|off]",
)
def cmd_rewritten(app, arg: str) -> None:
    """Toggle the display of the rewritten program."""
    _switch(app, arg, "show_rewritten", "Rewritten program")


@command(".quit", "Exit the REPL (also Ctrl+D)")
def cmd_quit(app, arg: str) -> None:
    """Exit the REPL."""
    raise QuitRepl()


@command(".exit", "Exit the REPL")
def cmd_exit(app, arg: str) -> None:
    """Exit the REPL."""
    raise QuitRepl()
