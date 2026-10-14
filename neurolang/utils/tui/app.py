"""The interactive REPL loop and the execution of the queries it reads."""

import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import HasSearch
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from neurolang.frontend.show_ra import ShowRAOptions
from neurolang.utils import engine_registry
from neurolang.utils.cli import (
    _emit,
    _execute_program,
    _execute_squall_program,
    _known_symbol_names,
    warning_note,
    report_error,
    result_to_dataframe,
    sort_dataframe,
    squall_column_names,
)

from .assistant import AssistantState
from .commands import (
    QuitRepl,
    engine_examples,
    engine_relations,
    run_command,
)
from .editing import session_options
from .interrupt import call_interruptibly
from .results import DEFAULT_PAGE_SIZE, ResultBlock, print_next_page
from .session import open_session_scope
from .statements import is_complete, split_statements
from .transcript import Transcript, is_transcript_command


def _input_bindings(is_squall) -> KeyBindings:
    """
    Return the key bindings of the prompt.

    Enter runs the input when it is a complete statement (see
    :func:`~.statements.is_complete`, *is_squall* tells the current
    language) or when the current line is empty, and otherwise starts a
    new line; Alt+Enter and Ctrl+J always insert a newline.
    """
    bindings = KeyBindings()

    @bindings.add("enter", filter=~HasSearch())
    def _(event):
        buffer = event.current_buffer
        if (
            is_complete(buffer.text, is_squall())
            or not buffer.document.current_line.strip()
        ):
            buffer.validate_and_handle()
        else:
            buffer.insert_text("\n")

    @bindings.add("escape", "enter")
    @bindings.add("c-j", filter=~HasSearch())
    def _(event):
        event.current_buffer.insert_text("\n")

    return bindings


@dataclass
class TuiSettings:
    """
    Settings of a REPL session.

    They are initialised from the command line options and some of them
    can be changed with dot-commands during the session. The data
    directory and the command line describe the session in its
    transcript. *llm_config* and *llm_provider* are the configuration
    file and the provider of the LLM assistant (``.ask``), by default
    those the configuration gives.
    """

    engine_name: str = ""
    squall: bool = False
    output_format: str = "table"
    sort_by: list = field(default_factory=list)
    show_ra: bool = False
    ra_options: ShowRAOptions = field(default_factory=ShowRAOptions)
    show_rewritten: bool = False
    page_size: int = DEFAULT_PAGE_SIZE
    vi_mode: bool = False
    debug: bool = False
    history_path: Optional[str] = None
    data_dir: str = ""
    command_line: str = ""
    llm_config: Optional[str] = None
    llm_provider: Optional[str] = None


class InteractiveTuiApp:
    """Interactive NeuroLang REPL rendering query results as tables."""

    def __init__(
        self,
        nl,
        settings: Optional[TuiSettings] = None,
        console: Optional[Console] = None,
        err_console: Optional[Console] = None,
    ):
        """
        Set up a REPL on the engine *nl*.

        Results are printed on *console* (default: stdout) and progress
        messages, notes and errors on *err_console* (default: stderr).
        What they print is recorded in the session's transcript.
        """
        self._nl = nl
        self.settings = settings or TuiSettings()
        self.transcript = Transcript(self.settings)
        self.console = console or Console()
        self.err_console = err_console or Console(stderr=True)
        self._tee_consoles(console, err_console)
        self.results: list = []
        self.session = None
        self.completer = None
        self.last_query = None
        self.failed = False
        # The input put in the next prompt, e.g. the query of .ask.
        self.pending_input = ""
        self.assistant = AssistantState(
            self.settings.llm_config, self.settings.llm_provider
        )
        # The columns of the relations loaded with .load, by name.
        self.loaded_relations: dict = {}
        open_session_scope(nl)
        if self.settings.history_path:
            self._history = FileHistory(self.settings.history_path)
        else:
            self._history = InMemoryHistory()

    def _tee_consoles(self, *consoles) -> None:
        """
        Tee the files of the given *consoles* into the transcript.

        The default consoles write to ``sys.stdout`` and ``sys.stderr``,
        which the transcript tees while it records an entry.
        """
        teed = []
        for console in consoles:
            if console is not None and console not in teed:
                console.file = self.transcript.tee(console.file)
                teed.append(console)

    def _execute(self, program: str):
        """Run *program* in the current language and return its result."""
        execute = (
            _execute_squall_program if self.settings.squall
            else _execute_program
        )
        return execute(
            self._nl, program,
            show_rewritten=self.settings.show_rewritten,
            show_ra=self.settings.show_ra and self.settings.ra_options,
        )

    def _execute_noting_warnings(self, program: str):
        """Run *program* and print the warnings it raises as short notes."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                return call_interruptibly(
                    lambda: self._execute(program), self._print_stuck_hint
                )
            finally:
                self._print_notes(caught)

    def _print_stuck_hint(self) -> None:
        """Tell that the interrupted query is busy in native code."""
        self.err_console.print(
            "Interrupting... the query is busy in a native computation; "
            "press Ctrl+C again to quit NeuroLang.", style="yellow",
        )

    def _print_notes(self, caught) -> None:
        """Print one note per distinct warning in *caught*."""
        known = _known_symbol_names(self._nl)
        notes = dict.fromkeys(
            warning_note(str(w.message), known) for w in caught
        )
        for note in notes:
            self.err_console.print(note, style="yellow", markup=False)

    def _report_error(self, exc: Exception, program: str) -> None:
        """Print the error *exc* of *program* as the CLI does."""
        self.failed = True
        report_error(
            exc, program, squall=self.settings.squall,
            debug=self.settings.debug, nl=self._nl,
            file=self.err_console.file,
        )

    def _result_block(self, result, title: str, column_names) -> ResultBlock:
        """Return the sorted rows of a query *result* as a block."""
        df = result_to_dataframe(result, column_names)
        if not df.empty:
            df = sort_dataframe(df, self.settings.sort_by)
        return ResultBlock(df, title)

    def print_block(self, block: ResultBlock) -> None:
        """
        Print the next page of *block* in the output format of the session.

        The block's title, if any, is printed first. The csv and json
        formats print all the rows at once.
        """
        if block.title:
            self.console.print(
                f"── {block.title} ──", style="bold magenta", markup=False
            )
        fmt = self.settings.output_format
        if fmt == "table":
            print_next_page(self.console, block, self.settings.page_size)
        elif block.frame.empty:
            self.console.print("[dim](empty)[/dim]")
        else:
            self.console.out(_emit(block.frame, fmt), highlight=False)
            block.shown = len(block.frame)

    def _show_result(self, result, column_names: dict) -> None:
        """
        Print a query *result*, keeping its blocks for the commands.

        A dict result, from a SQUALL program with several ``obtain``,
        gives a block per ``obtain``. *column_names* maps result keys to
        their column names, as
        :func:`~neurolang.utils.cli.squall_column_names` returns them.
        """
        self.results = []
        if result is None:
            self.console.print("[dim](no result)[/dim]")
        elif isinstance(result, bool):
            self.console.print(
                "[green]true[/green]" if result else "[red]false[/red]"
            )
        elif isinstance(result, dict):
            self.results = [
                self._result_block(value, key, column_names.get(key))
                for key, value in result.items()
            ]
        else:
            names = next(iter(column_names.values()), None)
            self.results = [self._result_block(result, "", names)]
        for block in self.results:
            self.print_block(block)

    def _column_names(self, program: str) -> dict:
        """
        Return the declared column names of the results of *program*.

        They are those the engine declares and those of the header of
        the files loaded with ``.load``.
        """
        if not self.settings.squall:
            return {}
        if not (self.settings.engine_name or self.loaded_relations):
            return {}
        return squall_column_names(
            self.settings.engine_name, program, self.loaded_relations
        )

    def _execute_and_display(self, program: str) -> None:
        """Run a Datalog or SQUALL query and print its result."""
        self.last_query = (program, self.settings.squall)
        self.err_console.print("[dim]Running query...[/dim]")
        t0 = time.perf_counter()
        try:
            result = self._execute_noting_warnings(program)
            if not self.settings.show_ra:
                self._show_result(result, self._column_names(program))
        except KeyboardInterrupt:
            self.failed = True
            self.err_console.print("[yellow]Query interrupted[/yellow]")
            return
        except Exception as exc:
            self._report_error(exc, program)
            return
        elapsed = time.perf_counter() - t0
        self.err_console.print(
            f"[dim]Query completed in {elapsed:.2f} s[/dim]"
        )

    def _handle_dot_command(self, line: str) -> None:
        """Run the dot-command on *line*."""
        run_command(self, line)

    def _get_prompt(self) -> str:
        """Return the prompt showing the engine and the language."""
        mode = "SQ" if self.settings.squall else "DL"
        return f"nl({self.settings.engine_name}:{mode})> "

    def _print_banner(self) -> None:
        """Print the engine, the mode, its relations and an example query."""
        name = self.settings.engine_name
        mode = "SQUALL" if self.settings.squall else "Datalog"
        self.console.print(
            "[bold yellow]NeuroLang interactive query REPL[/bold yellow]"
        )
        if name:
            description = " ".join(
                engine_registry.get_engine_config(name)
                .get("description", "").split()
            )
            self.console.print(f"Engine: [bold]{name}[/bold] — {description}")
        self.console.print(
            f"Mode: {mode} · [green].help[/green] for help · "
            "[green]Tab[/green] completes · "
            "[green]Ctrl+D[/green] or [green].quit[/green] to exit"
        )
        relations = engine_relations(name)
        if relations:
            self.console.print("Relations:")
            for relation in relations:
                self.console.print(f"  {relation}", markup=False)
        examples = engine_examples(name).get(mode.lower())
        if examples:
            self.console.print(f"Try: {examples[0]}", markup=False)

    def _make_session(self, **kwargs) -> PromptSession:
        """
        Return the prompt session reading the REPL input.

        *kwargs* are passed to :class:`~prompt_toolkit.PromptSession`,
        e.g. the ``input`` and ``output`` to use.
        """
        return PromptSession(
            **kwargs,
            **session_options(self),
            history=self._history,
            key_bindings=_input_bindings(lambda: self.settings.squall),
            multiline=True,
            prompt_continuation=lambda width, *_: "." * (width - 1) + " ",
        )

    @contextmanager
    def _recording(self, text: str, kind: str):
        """Record the entry *text* of *kind* run in the block."""
        failed = self.failed
        self.failed = False
        with self.transcript.record(text, kind, self.settings.squall) as entry:
            try:
                yield
            finally:
                entry.failed = self.failed
                self.failed = failed or self.failed

    def _run_line(self, line: str) -> None:
        """
        Run the dot-command or the query on *line*.

        It is recorded in the transcript, unless it is ``.transcript``.
        """
        text = line.strip()
        if not text:
            return
        if is_transcript_command(text):
            self._handle_dot_command(text)
        elif text.startswith("."):
            with self._recording(text, "command"):
                self._handle_dot_command(text)
        else:
            with self._recording(text, "statement"):
                self._execute_and_display(text)

    def _read_and_run(self, session) -> bool:
        """
        Read one input from *session* and run it.

        Ctrl+C at the prompt discards the input and Ctrl+C while running
        interrupts it; both keep the session. Return False when the REPL
        must end, on Ctrl+D or ``.quit``.
        """
        options = {}
        if self.pending_input:
            options["default"] = self.pending_input
            self.pending_input = ""
        try:
            line = session.prompt(self._get_prompt(), **options)
        except KeyboardInterrupt:
            return True
        except EOFError:
            return False
        try:
            self._run_line(line)
        except QuitRepl:
            return False
        except KeyboardInterrupt:
            self.console.print("[yellow]Interrupted[/yellow]")
        return True

    def _run_initial_program(self, program: str) -> None:
        """Run the query *program* given on the command line, if any."""
        if program.strip():
            with self._recording(program.strip(), "statement"):
                self._execute_and_display(program.strip())

    def run(self, session=None, initial_program: str = "") -> None:
        """
        Start the interactive REPL, reading from *session* if given.

        A non-empty *initial_program* is run as a query before the first
        prompt.
        """
        self._print_banner()
        self._run_initial_program(initial_program)
        session = session or self._make_session()
        self.session = session
        while self._read_and_run(session):
            pass
        self.console.print("[dim]Goodbye.[/dim]")

    def run_batch(self, lines, initial_program: str = "") -> int:
        """
        Run the statements of *lines* without prompting, as a script.

        Statements are split as at the prompt; a non-empty
        *initial_program* is run first. Return the exit status: 1 if a
        statement failed, 0 otherwise.
        """
        self._run_initial_program(initial_program)
        for statement in split_statements(lines, lambda: self.settings.squall):
            try:
                self._run_line(statement)
            except QuitRepl:
                break
        return 1 if self.failed else 0


def main(nl, settings: TuiSettings, initial_program: str = "") -> int:
    """
    Run the REPL on the engine *nl*, as the CLI does, and return its status.

    When the standard input is not a terminal, its statements are run as
    a batch (see :meth:`InteractiveTuiApp.run_batch`).
    """
    app = InteractiveTuiApp(nl, settings)
    if not sys.stdin.isatty():
        return app.run_batch(sys.stdin, initial_program)
    app.run(initial_program=initial_program)
    return 0
