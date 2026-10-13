"""The interactive REPL loop and the execution of the queries it reads."""

import time
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import HasSearch
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel

from neurolang.utils import engine_registry
from neurolang.utils.cli import (
    _execute_program,
    _execute_squall_program,
    _parse_sort_spec,
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


class InteractiveTuiApp:
    """Interactive NeuroLang REPL rendering query results as tables."""

    def __init__(
        self,
        engine_name: str = "neurosynth",
        data_dir: str = "neurolang_data",
        resolution: Optional[float] = None,
        history_path: Optional[str] = None,
        squall_mode: bool = False,
        sort_specs: Optional[list] = None,
        console: Optional[Console] = None,
    ):
        self.engine_name = engine_name
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.squall_mode = squall_mode
        self.sort_by = _parse_sort_spec(sort_specs or [])
        self.console = console or Console()
        self._nl = None
        self._last_df = None
        history_path = history_path or str(
            Path.home() / ".neurolang_query_history"
        )
        self._history = FileHistory(history_path)

    def _ensure_engine(self) -> None:
        """Build the engine on first use."""
        if self._nl is not None:
            return
        self.console.print(
            f"[dim]Building engine {self.engine_name!r} ...[/dim]"
        )
        self._nl = engine_registry.build_engine(
            self.engine_name, self.data_dir, self.resolution
        )
        self.console.print("[green]Engine ready.[/green]")

    def _execute(self, program: str):
        """Run *program* in the current language and return its result."""
        execute = (
            _execute_squall_program if self.squall_mode else _execute_program
        )
        return execute(self._nl, program)

    def _show_frame(self, result, title: str = "") -> None:
        """Sort and print the rows of a query *result*."""
        df = sort_dataframe(result_to_dataframe(result), self.sort_by)
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
        self._ensure_engine()
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
        mode = "SQ" if self.squall_mode else "DL"
        return f"nl({self.engine_name}:{mode})> "

    def _print_banner(self) -> None:
        """Print the welcome banner and the engine description."""
        self.console.print(Panel.fit(
            "[bold yellow]NeuroLang Interactive Query REPL[/bold yellow]\n"
            "Type [green].help[/green] for commands.  "
            "Ctrl+D or [green].quit[/green] to exit.",
            border_style="cyan",
        ))
        cfg = engine_registry.get_engine_config(self.engine_name)
        self.console.print(
            f"[dim]Engine: {self.engine_name} — "
            f"{cfg.get('description', '')}[/dim]"
        )
        mode = "SQUALL" if self.squall_mode else "Datalog"
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

    def run(self, session=None) -> None:
        """Start the interactive REPL, reading from *session* if given."""
        self._print_banner()
        self._ensure_engine()
        session = session or self._make_session()
        while self._read_and_run(session):
            pass
        self.console.print("[dim]Goodbye.[/dim]")

def main(
    engine_name: str = "neurosynth",
    data_dir: str = "neurolang_data",
    resolution: Optional[float] = None,
    squall_mode: bool = False,
    sort_specs: Optional[list] = None,
) -> None:
    """Entry point for the interactive TUI launched from the CLI."""
    InteractiveTuiApp(
        engine_name=engine_name,
        data_dir=data_dir,
        resolution=resolution,
        squall_mode=squall_mode,
        sort_specs=sort_specs,
    ).run()
