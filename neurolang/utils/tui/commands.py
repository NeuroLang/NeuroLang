"""
Dot-commands of the interactive REPL.

Commands are registered in :data:`COMMANDS` with the :func:`command`
decorator, which records the handler together with its usage and help
text. A handler is called with the REPL application and the text that
follows the command name.
"""

from typing import Callable, NamedTuple

from rich.panel import Panel


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


@command(".quit", "Exit the REPL (also Ctrl+D)")
def cmd_quit(app, arg: str) -> None:
    """Exit the REPL."""
    raise QuitRepl()


@command(".exit", "Exit the REPL")
def cmd_exit(app, arg: str) -> None:
    """Exit the REPL."""
    raise QuitRepl()
