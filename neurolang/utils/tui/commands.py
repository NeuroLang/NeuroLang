"""
Dot-commands of the interactive REPL.

Commands are registered in :data:`COMMANDS` with the :func:`command`
decorator, which records the handler together with its usage and help
text. A handler is called with the REPL application and the text that
follows the command name.
"""

from typing import Callable, NamedTuple

from rich.panel import Panel

from neurolang.utils.cli import _parse_sort_spec, sort_dataframe


class QuitRepl(Exception):
    """Raised by a command to end the REPL."""


class Command(NamedTuple):
    """A dot-command: its handler, usage line and one-line help."""

    handler: Callable
    usage: str
    help: str


COMMANDS = {}


def command(name: str, help: str, usage: str = ""):
    """
    Register the decorated function as the dot-command *name*.

    *usage* is the command line shown in the help (default: *name*) and
    *help* its one-line description.
    """
    def register(handler):
        COMMANDS[name] = Command(handler, usage or name, help)
        return handler
    return register


HELP_EXAMPLES = """
[bold cyan]Examples:[/bold cyan]
  ans(t) :- term_in_study_tfidf(t, w, s)
  .mode squall
  obtain every peak_reported.
"""


def help_text() -> str:
    """Return the help text listing every registered command."""
    lines = [
        "[bold yellow]Interactive NeuroLang Query REPL[/bold yellow]",
        "",
        "Type a Datalog (or SQUALL) query and press Enter to execute.",
        "Use [bold]Alt+Enter[/bold] or [bold]Meta+Enter[/bold] for "
        "multi-line input.",
        "",
        "[bold cyan]Dot commands:[/bold cyan]",
    ]
    lines += [
        f"  [green]{cmd.usage:<20}[/green] {cmd.help}"
        for cmd in COMMANDS.values()
    ]
    return "\n".join(lines) + "\n" + HELP_EXAMPLES


@command(".help", "Show this help message")
def cmd_help(app, arg: str) -> None:
    """Print the help text."""
    app.console.print(Panel(help_text(), border_style="cyan"))


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
    if app._last_df is not None:
        app._last_df = sort_dataframe(app._last_df, app.settings.sort_by)
        app.print_frame(app._last_df)


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
