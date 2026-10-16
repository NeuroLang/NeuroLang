"""
Dot-commands of the interactive REPL.

Commands are registered in :data:`COMMANDS` with the :func:`command`
decorator, which records the handler together with its usage and help
text. A handler is called with the REPL application and the text that
follows the command name.
"""

import difflib
from typing import Callable, NamedTuple

from rich.markup import escape
from rich.panel import Panel

from neurolang.utils import engine_registry
from neurolang.utils.cli import _parse_sort_spec, sort_dataframe

from .results import ResultBlock, print_all
from .table_view import browse, can_browse


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
    Return example queries on the engine for each language.

    The result is a dict from ``"datalog"`` and ``"squall"`` to a list
    of queries, the first being the one to try first. They are those of
    the engine's ``examples`` in its configuration, or else the query
    obtaining the first relation it declares. It is empty when the
    engine declares neither.
    """
    if not engine_name:
        return {}
    config = engine_registry.get_engine_config(engine_name)
    declared = config.get("examples") or {}
    examples = {
        language: list(declared[language])
        for language in ("datalog", "squall")
        if declared.get(language)
    }
    return {**_relation_examples(engine_name), **examples}


def _relation_examples(engine_name: str) -> dict:
    """Return the queries obtaining the first relation the engine declares."""
    predicates = engine_registry.get_predicates(engine_name)
    for name, info in predicates.items():
        columns = ", ".join(info.get("columns", []))
        if columns:
            return {
                "datalog": [f"ans({columns}) :- {name}({columns})"],
                "squall": [f"obtain every {name}."],
            }
    return {}


KEYS_HELP = """
[bold cyan]Keys:[/bold cyan]
  Enter                run the statement; an unfinished one (open bracket,
                       trailing ',' or ':-', SQUALL without its final '.')
                       continues on a new line, an empty line runs it
  Alt+Enter, Ctrl+J    insert a newline (write a multi-line statement)
  Tab                  complete commands, predicates, their columns and
                       the syntax; in SQUALL, the nouns and the verbs
                       that fit; Shift+Tab goes back
  Right arrow          accept the suggestion from the history (grey text)
  Up/Down              previous/next line or history entry
  Ctrl+R               search the history backwards (Ctrl+S forwards)
  Ctrl+X Ctrl+E        edit the input in $EDITOR, then run it (v in vi
                       normal mode)
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
        f"  [green]{escape(f'{cmd.usage:<20}')}[/green] {escape(cmd.help)}"
        for cmd in COMMANDS.values()
    ]
    lines.append(KEYS_HELP)
    lines += _example_lines(engine_name)
    return "\n".join(lines)


def _example_lines(engine_name: str) -> list:
    """Return the lines of the help showing the engine's examples."""
    examples = engine_examples(engine_name)
    if not examples:
        return []
    datalog = examples.get("datalog", [])
    squall = examples.get("squall", [])
    return [
        f"[bold cyan]Examples ({engine_name}):[/bold cyan]",
        *(f"  {escape(query)}" for query in datalog),
        "  .more", "  .mode squall",
        *(f"  {escape(query)}" for query in squall),
    ]


SQUALL_PRIMER = """[bold yellow]SQUALL primer[/bold yellow]
  obtain every Study that mentions 'pain'.     a query; a sentence ends with .
  define as Pain_study every Study that mentions 'pain'.
                         a rule, then: obtain every Pain_study.
  [cyan]Nouns[/cyan]      capitalised predicates, folded to lowercase: Study is study
  [cyan]Verbs[/cyan]      relate two things, subject first: a Study mentions a Term
  [cyan]Variables[/cyan]  ?s names a column: every Study ?s that mentions 'pain';
             a tuple names several: every Weighted_mention (?s; ?t; ?w)
  [cyan]Strings[/cyan]    'pain' or "working memory"
  [cyan]in ND[/cyan]      a noun of N columns: every Voxel in 3D, or (?i; ?j; ?k)
  [cyan]Results[/cyan]    obtain the Count of the Study that mentions 'pain'.
             obtain every Study that mentions 'pain' as Pain.
  [cyan]Probability[/cyan]
    define as Pain_selected with inferred probability every Study that is
      a Selected_study and that mentions 'pain'.   adds a probability
    define as Activation with probability every Voxel in 3D that a
      Selected_study reports conditioned to every Selected_study that
      mentions 'pain'."""


def squall_primer(engine_name: str = "") -> str:
    """
    Return the SQUALL primer, with the engine's sorts and examples.

    The sorts shown are those of several columns, which nouns read
    ``in ND``; the examples are the engine's SQUALL examples.
    """
    lines = [SQUALL_PRIMER]
    sorts = _sorts_in_nd(engine_name)
    if sorts:
        lines.append(f"[bold cyan]Entities in ND ({engine_name}):[/bold cyan]")
        lines += [_sort_line(sort) for sort in sorts]
    examples = engine_examples(engine_name).get("squall", [])
    if examples:
        lines.append(f"[bold cyan]Examples ({engine_name}):[/bold cyan]")
        lines += [f"  {escape(query)}" for query in examples]
    return "\n".join(lines)


def _sorts_in_nd(engine_name: str) -> list:
    """Return the sorts of several columns the engine declares."""
    if not engine_name:
        return []
    return [
        sort for sort in engine_registry.get_sorts(engine_name).values()
        if len(sort.columns) > 1
    ]


def _sort_line(sort) -> str:
    """Return the primer's line on the *sort* of several columns."""
    signature = f"{sort.name}({', '.join(sort.columns)})"
    dimension = f"in {len(sort.columns)}D"
    description = sort.description or dimension
    if dimension not in description:
        description = f"{dimension} · {description}"
    return escape(f"  {signature:<22} {description}")


def _print_panel(app, text: str) -> None:
    """Print *text* in the help's panel."""
    app.console.print(Panel(text, border_style="cyan"))


# Help on a topic, ``.help TOPIC``: the text of each topic, by name.
HELP_TOPICS = {}


@command(
    ".help", "Show this help, a primer on SQUALL, or help on a topic",
    ".help [squall|TOPIC]",
)
def cmd_help(app, arg: str) -> None:
    """
    Print the help text, and the SQUALL primer in SQUALL mode.

    ``.help squall`` prints the primer alone, and ``.help TOPIC`` the
    help on a topic of :data:`HELP_TOPICS`.
    """
    topic = arg.strip().lower()
    engine = app.settings.engine_name
    if topic in HELP_TOPICS:
        _print_panel(app, HELP_TOPICS[topic])
        return
    if topic not in ("", "squall"):
        topics = " or ".join(f".help {t}" for t in ("squall", *HELP_TOPICS))
        app.console.print(
            f"No help on {topic!r}: use .help, {topics}.",
            style="yellow", markup=False,
        )
        return
    if not topic:
        _print_panel(app, help_text(engine))
    if topic or app.settings.squall:
        _print_panel(app, squall_primer(engine))


def _language(squall: bool):
    """Return the ``.mode`` setter of the SQUALL or Datalog language."""
    def set_mode(app) -> str:
        app.settings.squall = squall
        return "SQUALL" if squall else "Datalog"
    return set_mode


MODES = {"datalog": _language(False), "squall": _language(True)}


@command(
    ".mode", "Switch the query language or the editing mode",
    ".mode datalog|squall|vi|emacs",
)
def cmd_mode(app, arg: str) -> None:
    """Switch between Datalog and SQUALL input, or vi and emacs editing."""
    mode = arg.strip().lower()
    set_mode = MODES.get(mode)
    if set_mode is None:
        app.console.print(
            f"Unknown mode {mode!r}. Use one of: {', '.join(MODES)}.",
            style="yellow", markup=False,
        )
        return
    app.console.print(f"[green]Mode: {set_mode(app)}[/green]")


def sort_spec_text(sort_by, separator: str = ", ") -> str:
    """Return the ``COL:asc|desc`` specs of the sort order *sort_by*."""
    return separator.join(
        f"{col}:{'asc' if asc else 'desc'}" for col, asc in sort_by
    )


@command(
    ".sort", "Show or set the sort of the results (none: unsorted)",
    ".sort [COL[:asc|desc] ...|none]",
)
def cmd_sort(app, arg: str) -> None:
    """Set the sort order of the results and re-sort the last one."""
    specs = arg.split()
    if not specs:
        current = sort_spec_text(app.settings.sort_by)
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


def _block_to_browse(app, arg: str):
    """
    Return the block of the last result chosen by *arg*, or None.

    *arg* is the 1-based number of the block, default the first one;
    a missing result or a wrong number is reported.
    """
    if not app.results:
        app.console.print("[dim]No result to browse.[/dim]")
        return None
    try:
        number = int(arg) if arg.strip() else 1
    except ValueError:
        number = 0
    if not 1 <= number <= len(app.results):
        app.console.print(
            f"[yellow]Give a result number from 1 to {len(app.results)}, "
            f"not {arg.strip()!r}.[/yellow]"
        )
        return None
    return app.results[number - 1]


@command(
    ".browse", "Browse the last result: scroll, sort (s), search (/)",
    ".browse [N]",
)
def cmd_browse(app, arg: str) -> None:
    """
    Open the last result in the full-screen table viewer.

    *arg* chooses the result of a SQUALL program with several ``obtain``.
    Without a terminal, all the rows are printed instead.
    """
    block = _block_to_browse(app, arg)
    if block is None:
        return
    if not can_browse(app.console):
        app.console.print(
            "[dim].browse needs a terminal; printing all the rows.[/dim]"
        )
        print_all(app.console, block)
        return
    browse(block.frame, block.title)


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


@command(".quit", "Exit the REPL (also Ctrl+D)")
def cmd_quit(app, arg: str) -> None:
    """Exit the REPL."""
    raise QuitRepl()


@command(".exit", "Exit the REPL")
def cmd_exit(app, arg: str) -> None:
    """Exit the REPL."""
    raise QuitRepl()
