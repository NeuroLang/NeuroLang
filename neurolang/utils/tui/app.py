"""The interactive REPL loop and the execution of the queries it reads."""

import time
from dataclasses import dataclass, field
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import HasSearch
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel

from neurolang.frontend.show_ra import ShowRAOptions
from neurolang.utils import engine_registry
from neurolang.utils.cli import (
    _execute_program,
    _execute_squall_program,
    result_to_dataframe,
    sort_dataframe,
)

from .commands import COMMANDS, QuitRepl
from .results import _df_to_rich_table


def _newline_bindings() -> KeyBindings:
    """Return key bindings inserting a newline on Alt+Enter and Ctrl+J."""
    bindings = KeyBindings()

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
    can be changed with dot-commands during the session.
    """

    engine_name: str = ""
    squall: bool = False
    sort_by: list = field(default_factory=list)
    show_ra: Optional[ShowRAOptions] = None
    show_rewritten: bool = False
    history_path: Optional[str] = None


class InteractiveTuiApp:
    """Interactive NeuroLang REPL rendering query results as tables."""

    def __init__(
        self,
        nl,
        settings: Optional[TuiSettings] = None,
        console: Optional[Console] = None,
    ):
        self._nl = nl
        self.settings = settings or TuiSettings()
        self.console = console or Console()
        self._last_df = None
        if self.settings.history_path:
            self._history = FileHistory(self.settings.history_path)
        else:
            self._history = InMemoryHistory()

    def _execute(self, program: str):
        """Run *program* in the current language and return its result."""
        execute = (
            _execute_squall_program if self.settings.squall
            else _execute_program
        )
        return execute(
            self._nl, program,
            show_rewritten=self.settings.show_rewritten,
            show_ra=self.settings.show_ra or False,
        )

    def _show_frame(self, result, title: str = "") -> None:
        """Sort and print the rows of a query *result*."""
        df = sort_dataframe(result_to_dataframe(result), self.settings.sort_by)
        if df.empty:
            self.console.print("[dim](empty)[/dim]")
        else:
            self.console.print(_df_to_rich_table(df, title=title))
        self._last_df = df

    def _show_result(self, result) -> None:
        """Print a query *result*, keeping the last table shown."""
        if result is None:
            self.console.print("[dim](no result)[/dim]")
            self._last_df = None
            return
        if isinstance(result, bool):
            self.console.print(
                "[green]true[/green]" if result else "[red]false[/red]"
            )
            self._last_df = None
            return
        if isinstance(result, dict):
            for key, sub_result in result.items():
                self.console.print(
                    f"\n[bold magenta]── {key} ──[/bold magenta]"
                )
                self._show_frame(sub_result, title=key)
            return
        self._show_frame(result)

    def _execute_and_display(self, program: str) -> None:
        """Run a Datalog or SQUALL query and print its result."""
        self.console.print("[dim]Running query...[/dim]")
        t0 = time.perf_counter()
        try:
            result = self._execute(program)
        except KeyboardInterrupt:
            self.console.print("[yellow]Query interrupted[/yellow]")
            return
        except Exception as exc:
            self.console.print(f"[red]Query error:[/red] {exc}")
            return
        elapsed = time.perf_counter() - t0
        if self.settings.show_ra is None:
            self._show_result(result)
        self.console.print(f"\n[dim]Query completed in {elapsed:.2f} s[/dim]")

    def _handle_dot_command(self, line: str) -> None:
        """Run the dot-command on *line*."""
        verb, _, rest = line.strip().partition(" ")
        cmd = COMMANDS.get(verb.lower())
        if cmd is None:
            self.console.print(
                f"[yellow]Unknown command: {verb}. Try .help[/yellow]"
            )
            return
        cmd.handler(self, rest)

    def _get_prompt(self) -> str:
        """Return the prompt showing the engine and the language."""
        mode = "SQ" if self.settings.squall else "DL"
        return f"nl({self.settings.engine_name}:{mode})> "

    def _print_banner(self) -> None:
        """Print the welcome banner and the engine description."""
        self.console.print(Panel.fit(
            "[bold yellow]NeuroLang Interactive Query REPL[/bold yellow]\n"
            "Type [green].help[/green] for commands.  "
            "Ctrl+D or [green].quit[/green] to exit.",
            border_style="cyan",
        ))
        name = self.settings.engine_name
        cfg = engine_registry.get_engine_config(name) if name else {}
        self.console.print(
            f"[dim]Engine: {name} — "
            f"{cfg.get('description', '')}[/dim]"
        )
        mode = "SQUALL" if self.settings.squall else "Datalog"
        self.console.print(f"[dim]Mode: {mode}[/dim]")

    def _make_session(self) -> PromptSession:
        """Return the prompt session reading the REPL input."""
        return PromptSession(
            history=self._history, key_bindings=_newline_bindings(),
        )

    def _run_line(self, line: str) -> None:
        """Run the dot-command or the query on *line*."""
        if not line.strip():
            return
        if line.lstrip().startswith("."):
            self._handle_dot_command(line)
        else:
            self._execute_and_display(line.strip())

    def _read_and_run(self, session) -> bool:
        """
        Read one input from *session* and run it.

        Ctrl+C at the prompt discards the input and Ctrl+C while running
        interrupts it; both keep the session. Return False when the REPL
        must end, on Ctrl+D or ``.quit``.
        """
        try:
            line = session.prompt(self._get_prompt())
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

    def run(self, session=None, initial_program: str = "") -> None:
        """
        Start the interactive REPL, reading from *session* if given.

        A non-empty *initial_program* is run as a query before the first
        prompt.
        """
        self._print_banner()
        if initial_program.strip():
            self._execute_and_display(initial_program)
        session = session or self._make_session()
        while self._read_and_run(session):
            pass
        self.console.print("[dim]Goodbye.[/dim]")


def main(nl, settings: TuiSettings, initial_program: str = "") -> None:
    """Run the interactive REPL on the engine *nl*, as the CLI does."""
    InteractiveTuiApp(nl, settings).run(initial_program=initial_program)
