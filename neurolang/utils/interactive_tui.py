"""
Interactive TUI (Terminal User Interface) for NeuroLang query execution.

Provides a prompt_toolkit-based REPL with:
- Lark grammar autocomplete
- Rich table rendering of query results
- Multi-line editing (Alt+Enter or Meta+Enter to insert newline)
- Command history (persistent via prompt_toolkit's FileHistory)
- NIfTI save / visualise support via nibabel and nilearn
- Dot-commands: .help, .engines, .sets, .predicates, .save, .view, .mode, .quit

Usage
-----
::

    neurolang-query --interactive
    neurolang-query -i -e neurosynth
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from neurolang.datalog import WrappedRelationalAlgebraSet, Union as DatalogUnion
from neurolang.frontend import NeurolangPDL
from neurolang.frontend.datalog.standard_syntax import COMPILED_GRAMMAR
from neurolang.frontend.datalog.pretty_printer import DatalogPrettyPrinter
from neurolang.frontend.datalog.squall_syntax_lark import (
    parser as squall_parser, SquallProgram,
)
from neurolang.logic import Union as LogicUnion
from neurolang.utils import engine_registry
from neurolang.utils.interactive_parsing import LarkCompleter, TERMINALS_TO_CATEGORIES, CATEGORIES

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.filters import HasSearch

import rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOT_COMMANDS = {
    ".help": "Show this help message",
    ".engines": "List available dataset engines",
    ".sets": "List loaded RA sets (EDB relations)",
    ".predicates": "List available predicates for the current engine",
    ".save": "Save last result to a file\n  .save <path>        — save as CSV\n  .save <path>.nii    — save result voxel data as NIfTI",
    ".view": "Visualise a NIfTI file (uses nilearn plotting)\n  .view <path>.nii",
    ".mode": "Toggle between Datalog and SQUALL mode\n  .mode datalog\n  .mode squall",
    ".quit": "Exit the REPL (also Ctrl+D / Ctrl+C)",
}

SLASH_COMMANDS = {
    "/rewritten": "Re-run last query showing magic-sets rewritten Datalog",
    "/datalog": "Show Datalog IR of the last SQUALL query (--show-datalog)",
    "/help": "Show this help message",
}

HELP_TEXT = """
[bold yellow]Interactive NeuroLang Query REPL[/bold yellow]

Type a Datalog (or SQUALL) query and press Enter to execute.
Use [bold]Alt+Enter[/bold] or [bold]Meta+Enter[/bold] for multi-line input.
The tab key triggers grammar-aware autocompletion.

[bold cyan]Dot commands:[/bold cyan]
""" + "\n".join(f"  [green]{cmd:<20}[/green] {desc}" for cmd, desc in DOT_COMMANDS.items()) + """

[bold cyan]Slash commands:[/bold cyan]
""" + "\n".join(f"  [green]{cmd:<20}[/green] {desc}" for cmd, desc in SLASH_COMMANDS.items()) + """

[bold cyan]Examples:[/bold cyan]
  ans(t) :- term_in_study_tfidf(t, w, s)
  .mode squall
  obtain every peak_reported.
  /datalog
  /rewritten
  .save results/query_output.nii
  .view results/query_output.nii
"""

# ---------------------------------------------------------------------------
# SQUALL keywords used for keyword-based completion (Earley parser does not
# support parse_interactive(), so we fall back to plain keyword matching).
# ---------------------------------------------------------------------------

SQUALL_KEYWORDS = [
    "define", "as", "obtain", "every", "a", "an", "the",
    "in", "where", "that", "with", "without", "if", "then",
    "and", "or", "not", "for", "of", "from", "to", "by",
    "at", "on", "is", "are", "has", "have", "there",
    "probability", "conditioned", "inferred", "choose",
    "peaks", "voxels", "studies", "terms", "experiments",
    "reported", "in", "that", "which", "atlas",
    "all", "each", "some", "no", "any",
    "most", "few", "both", "neither",
]

# ---------------------------------------------------------------------------
# Custom prompt_toolkit Completer — multi-source: grammar + predicates + keywords
# ---------------------------------------------------------------------------


class NeuroLangReplCompleter(Completer):
    """prompt_toolkit Completer combining grammar completions + engine predicates + dot commands."""

    def __init__(
        self,
        lark_completer: Optional[LarkCompleter] = None,
        engine_predicates: Optional[set[str]] = None,
        squall_keywords: Optional[list[str]] = None,
    ):
        self._lark = lark_completer
        self._predicates = engine_predicates or set()
        self._squall_keywords = squall_keywords or []

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # 1. Dot commands
        if text.startswith("."):
            for cmd in DOT_COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return

        # 2. Slash commands
        if text.startswith("/"):
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return

        # 2. Grammar completions (Datalog LALR mode)
        if self._lark is not None:
            try:
                result = self._lark.complete(text)
                seen = set()
                for category_key, container in result.token_options.items():
                    for value in container.get("values", set()):
                        if not value or value in seen:
                            continue
                        seen.add(value)
                        display = value
                        if value.startswith("<") and value.endswith(">"):
                            display = value[1:-1]
                        yield Completion(
                            value,
                            start_position=-len(result.prefix),
                            display=display,
                            display_meta=category_key,
                        )
            except Exception:
                pass

        # 3. Engine predicate completions (both modes)
        word_start = text.rfind(" ") + 1 if " " in text else 0
        prefix = text[word_start:]
        # Only suggest predicates when it looks like we're typing an identifier
        if prefix and (prefix[0].isalpha() or prefix[0] == "_"):
            for pred in sorted(self._predicates):
                if pred.startswith(prefix):
                    yield Completion(
                        pred,
                        start_position=-len(prefix),
                        display=pred,
                        display_meta="predicate",
                    )

        # 4. SQUALL keywords (when no lark completer — squall mode)
        if self._lark is None:
            for kw in self._squall_keywords:
                if kw.startswith(prefix):
                    yield Completion(
                        kw,
                        start_position=-len(prefix),
                        display=kw,
                        display_meta="keyword",
                    )


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------

_console = Console()


def _df_to_rich_table(df: pd.DataFrame, title: str = "") -> Table:
    """Render a pandas DataFrame as a rich Table."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
        show_edge=True,
    )
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for row in df.itertuples(index=False):
        table.add_row(*[str(v) if v is not None else "" for v in row])
    return table


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------


def _result_to_nifti(
    df: pd.DataFrame, template_path: Optional[str] = None
) -> "nibabel.Nifti1Image":
    """Convert a DataFrame with voxel coordinates into a NIfTI image.

    Expects columns named (i, j, k) or similar 3D coordinate triple and
    an optional value/label column.  If *template_path* is provided the
    output image uses that template's affine and shape; otherwise an
    identity-affine 256×256×256 volume is created.
    """
    # Try to detect coordinate columns
    col_lower = {c.lower(): c for c in df.columns}
    i_col = col_lower.get("i")
    j_col = col_lower.get("j")
    k_col = col_lower.get("k")

    if i_col is None or j_col is None or k_col is None:
        # Fallback: use first three numeric columns
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 3:
            i_col, j_col, k_col = num_cols[:3]

    if i_col is None:
        raise ValueError(
            "Cannot determine voxel coordinates: no (i, j, k) or numeric columns found."
        )

    # Determine value column (anything that isn't a coordinate)
    coord_cols = {i_col, j_col, k_col}
    val_cols = [c for c in df.columns if c not in coord_cols]

    # Load or create template volume
    if template_path:
        ref = nib.load(template_path)
        shape = ref.shape[:3]
        affine = ref.affine
    else:
        shape = (256, 256, 256)
        affine = np.eye(4)

    data = np.zeros(shape, dtype=np.float32)

    coords = df[[i_col, j_col, k_col]].values.astype(int)
    if val_cols:
        values = df[val_cols[0]].values.astype(float)
    else:
        values = np.ones(len(df), dtype=np.float32)

    # Clip to volume bounds
    for idx in range(len(coords)):
        ci, cj, ck = coords[idx]
        if 0 <= ci < shape[0] and 0 <= cj < shape[1] and 0 <= ck < shape[2]:
            data[ci, cj, ck] = values[idx]

    return nib.Nifti1Image(data, affine)


def _view_nifti(path: str) -> None:
    """Open a NIfTI image using nilearn plotting (saves a PNG to temp dir)."""
    try:
        from nilearn import plotting
    except ImportError:
        _console.print("[red]nilearn is required for visualisation.[/red]")
        return

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        _console.print(f"[red]File not found: {path}[/red]")
        return

    tmp_png = tempfile.mktemp(suffix=".png", prefix="neurolang_view_")
    try:
        display = plotting.plot_img(
            path,
            display_mode="ortho",
            cut_coords=None,
            black_bg=True,
            cmap="cold_hot",
        )
        display.savefig(tmp_png, dpi=150)
        display.close()
        _console.print(f"[green]Saved visualisation to: {tmp_png}[/green]")
        # Try to open with system viewer
        subprocess.Popen(["open", tmp_png] if sys.platform == "darwin" else ["xdg-open", tmp_png])
        _console.print(f"[dim]Viewing: {tmp_png}[/dim]")
    except Exception as exc:
        _console.print(f"[red]Visualisation failed: {exc}[/red]")


# ---------------------------------------------------------------------------
# Result formatting (reuses cli.py helpers where possible)
# ---------------------------------------------------------------------------


def _rename_unnamed_columns(df, column_names):
    """Replace auto-generated column names with *column_names* when safe."""
    if column_names is not None and len(column_names) == len(df.columns):
        unnamed = all(
            isinstance(c, int)
            or (isinstance(c, str) and c.startswith("c"))
            for c in df.columns
        )
        if unnamed:
            df.columns = column_names
    return df


def _format_as_df(result, column_names=None):
    """Convert a query result to a pandas DataFrame, or None on failure."""
    if result is None:
        return None
    if isinstance(result, bool):
        # Boolean result: wrap in a tiny DataFrame
        return pd.DataFrame({"result": ["true" if result else "false"]})

    try:
        # Unwrap Constant -> concrete set
        if hasattr(result, "value") and hasattr(result.value, "unwrap"):
            inner = result.value.unwrap()
            if hasattr(inner, "as_pandas_dataframe"):
                return _rename_unnamed_columns(inner.as_pandas_dataframe(), column_names)
            result = inner

        if hasattr(result, "columns") and len(result.columns) > 0:
            return _rename_unnamed_columns(pd.DataFrame(iter(result), columns=result.columns), column_names)
        rows = list(iter(result))
        if not rows:
            return pd.DataFrame()
        arity = result.arity if hasattr(result, "arity") else len(rows[0])
        return _rename_unnamed_columns(pd.DataFrame(rows, columns=[f"c{i}" for i in range(arity)]), column_names)
    except Exception as exc:
        _console.print(f"[yellow]Warning: result formatting failed: {exc}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# The Interactive TUI App
# ---------------------------------------------------------------------------


class InteractiveTuiApp:
    """Interactive NeuroLang REPL with autocomplete, rich rendering, NIfTI support."""

    def __init__(
        self,
        engine_name: str = "neurosynth",
        data_dir: str = "neurolang_data",
        resolution: Optional[float] = None,
        history_path: Optional[str] = None,
        squall_mode: bool = False,
        sort_specs: Optional[list[str]] = None,
    ):
        self.engine_name = engine_name
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.squall_mode = squall_mode
        self.sort_specs = sort_specs or []

        # State
        self._nl: Optional[NeurolangPDL] = None
        self._engine_built = False
        self._last_program: Optional[str] = None
        self._last_df: Optional[pd.DataFrame] = None
        self._last_result_type: Optional[str] = None  # 'single' or 'dict'
        self._engine_predicates: set[str] = set()
        self._squall_keywords = SQUALL_KEYWORDS

        # History
        history_path = history_path or str(Path.home() / ".neurolang_query_history")
        self._history = FileHistory(history_path)

        # Key bindings
        self._kb = KeyBindings()

        @self._kb.add("escape", "enter")  # Alt+Enter / Meta+Enter
        @self._kb.add("c-j", filter=~HasSearch())
        def _(event):
            """Insert newline instead of submitting."""
            event.current_buffer.insert_text("\n")

        # Completer (initial: Datalog LALR mode)
        try:
            self._completer = NeuroLangReplCompleter(
                lark_completer=LarkCompleter(COMPILED_GRAMMAR),
                engine_predicates=self._engine_predicates,
                squall_keywords=self._squall_keywords,
            )
        except Exception as exc:
            _console.print(f"[yellow]Warning: autocomplete init failed: {exc}[/yellow]")
            self._completer = None

        # Session
        self._session = PromptSession(
            history=self._history,
            key_bindings=self._kb,
            completer=self._completer,
            complete_while_typing=True,
            multiline=False,
            enable_open_in_editor=True,
            vi_mode=False,
        )

    # -----------------------------------------------------------------------
    # Engine lazy-build
    # -----------------------------------------------------------------------

    def _ensure_engine(self) -> None:
        if self._engine_built:
            return
        if self.engine_name not in engine_registry.list_engine_names():
            available = ", ".join(engine_registry.list_engine_names())
            _console.print(
                f"[red]Unknown engine {self.engine_name!r}.[/red] "
                f"Available: {available}"
            )
            raise SystemExit(1)
        _console.print(f"[dim]Building engine {self.engine_name!r} ...[/dim]")
        self._nl = engine_registry.build_engine(
            self.engine_name, self.data_dir, self.resolution
        )
        self._engine_built = True
        _console.print("[green]Engine ready.[/green]")

        # Collect predicates from the engine's symbol table and YAML metadata
        self._engine_predicates = self._collect_predicates()
        self._rebuild_completer()

    def _collect_predicates(self) -> set[str]:
        """Collect predicate names from the engine's symbol table and YAML config."""
        preds: set[str] = set()
        if self._nl is None:
            return preds
        st = self._nl.program_ir.symbol_table
        for sym_name, sym_val in st.items():
            val = getattr(sym_val, "value", sym_val)
            if isinstance(val, WrappedRelationalAlgebraSet):
                name = getattr(sym_name, "name", str(sym_name))
                preds.add(name)
        # Also add YAML-declared predicates from engine_registry
        try:
            yaml_preds = engine_registry.get_predicates(self.engine_name)
            preds.update(yaml_preds.keys())
        except Exception:
            pass
        return preds

    def _rebuild_completer(self) -> None:
        """Rebuild the prompt_toolkit completer for the current mode."""
        if self.squall_mode:
            lark = None
        else:
            try:
                lark = LarkCompleter(COMPILED_GRAMMAR)
            except Exception:
                lark = None

        self._completer = NeuroLangReplCompleter(
            lark_completer=lark,
            engine_predicates=self._engine_predicates,
            squall_keywords=self._squall_keywords,
        )
        # Update the session's completer
        self._session.completer = self._completer

    # -----------------------------------------------------------------------
    # Query execution
    # -----------------------------------------------------------------------

    def _execute_and_display(self, program: str) -> None:
        """Parse, execute, and render a Datalog / SQUALL query."""
        self._ensure_engine()
        self._last_program = program

        sort_by = []
        for spec in self.sort_specs:
            parts = spec.split(":", maxsplit=1)
            col = parts[0]
            ascending = True
            if len(parts) == 2 and parts[1].lower() == "desc":
                ascending = False
            sort_by.append((col, ascending))

        t0 = time.perf_counter()
        try:
            if self.squall_mode:
                result = self._nl.execute_squall_program(program)
            else:
                result = self._nl.execute_datalog_program(program)
        except Exception as exc:
            _console.print(f"[red]Query error:[/red] {exc}")
            return

        elapsed = time.perf_counter() - t0

        if result is None:
            _console.print("[dim](no result)[/dim]")
            self._last_df = None
            self._last_result_type = None
            return

        if isinstance(result, bool):
            _console.print("[green]true[/green]" if result else "[red]false[/red]")
            self._last_df = _format_as_df(result)
            self._last_result_type = "single"
            return

        if isinstance(result, dict):
            self._last_result_type = "dict"
            first = True
            for key, sub_result in result.items():
                df = _format_as_df(sub_result)
                if df is not None and not df.empty:
                    if first:
                        _console.print(f"\n[bold magenta]── {key} ──[/bold magenta]")
                        first = False
                    else:
                        _console.print(f"\n[bold magenta]── {key} ──[/bold magenta]")
                    if sort_by:
                        valid = [(c, a) for c, a in sort_by if c in df.columns]
                        if valid:
                            cols, asc = zip(*valid)
                            df = df.sort_values(by=list(cols), ascending=list(asc))
                    _console.print(_df_to_rich_table(df, title=key))
                    self._last_df = df
        else:
            self._last_result_type = "single"
            df = _format_as_df(result)
            if df is not None:
                if sort_by:
                    valid = [(c, a) for c, a in sort_by if c in df.columns]
                    if valid:
                        cols, asc = zip(*valid)
                        df = df.sort_values(by=list(cols), ascending=list(asc))
                if df.empty:
                    _console.print("[dim](empty)[/dim]")
                else:
                    _console.print(_df_to_rich_table(df))
                self._last_df = df
            else:
                _console.print(str(result))

        _console.print(f"\n[dim]Query completed in {elapsed:.2f} s[/dim]")

    # -----------------------------------------------------------------------
    # Dot-command handlers
    # -----------------------------------------------------------------------

    def _cmd_help(self) -> None:
        _console.print(Panel(HELP_TEXT, border_style="cyan"))

    def _cmd_engines(self) -> None:
        engine_registry.show_engines()

    def _cmd_sets(self) -> None:
        self._ensure_engine()
        st = self._nl.program_ir.symbol_table
        sets_found = []
        for sym_name, sym_val in st.items():
            val = getattr(sym_val, "value", sym_val)
            if not isinstance(val, WrappedRelationalAlgebraSet):
                continue
            ra_set = val.unwrap()
            arity = ra_set.arity
            columns = list(ra_set.columns)
            try:
                size = len(ra_set)
            except Exception:
                size = "?"
            clean_name = getattr(sym_name, "name", str(sym_name))
            sets_found.append((clean_name, columns, size, arity))
        if not sets_found:
            _console.print("  [dim](no RA sets registered)[/dim]")
            return
        table = Table(title="RA Sets", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Set")
        table.add_column("Columns")
        table.add_column("Rows")
        table.add_column("Arity")
        for name, cols, size, arity in sorted(sets_found, key=lambda x: x[0]):
            table.add_row(str(name), str(cols), str(size), str(arity))
        _console.print(table)
        _console.print(f"\n[dim]Total RA sets: {len(sets_found)}[/dim]")

    def _cmd_predicates(self) -> None:
        cfg = engine_registry.get_engine_config(self.engine_name)
        _console.print(f"\n[bold]Engine:[/bold] {self.engine_name}")
        _console.print(f"  {cfg.get('description', '')}")
        predicates = engine_registry.get_predicates(self.engine_name)
        if not predicates:
            _console.print("\n  [dim](no predicate metadata declared)[/dim]")
            return
        table = Table(title="Available Predicates", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Predicate")
        table.add_column("Columns")
        table.add_column("Description")
        for pname, info in predicates.items():
            cols = ", ".join(info.get("columns", []))
            desc = info.get("description", "")
            table.add_row(f"{pname}({cols})", cols, desc)
        _console.print(table)

    def _cmd_save(self, args: str) -> None:
        if self._last_df is None:
            _console.print("[yellow]Nothing to save — run a query first.[/yellow]")
            return

        path = args.strip()
        if not path:
            _console.print("[yellow]Usage: .save <path>  or  .save <path>.nii[/yellow]")
            return

        path = os.path.expanduser(path)
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

        if path.endswith(".nii"):
            try:
                img = _result_to_nifti(self._last_df)
                nib.save(img, path)
                _console.print(f"[green]NIfTI saved to: {path}[/green]")
            except Exception as exc:
                _console.print(f"[red]NIfTI save failed: {exc}[/red]")
        else:
            try:
                self._last_df.to_csv(path, index=False)
                _console.print(f"[green]CSV saved to: {path}[/green]")
            except Exception as exc:
                _console.print(f"[red]Save failed: {exc}[/red]")

    def _cmd_view(self, args: str) -> None:
        path = args.strip()
        if not path:
            _console.print("[yellow]Usage: .view <path>.nii[/yellow]")
            return
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            _console.print(f"[red]File not found: {path}[/red]")
            return
        _view_nifti(path)

    def _cmd_mode(self, args: str) -> None:
        mode = args.strip().lower()
        if mode == "datalog":
            self.squall_mode = False
            self._rebuild_completer()
            _console.print("[green]Mode: Datalog[/green]")
        elif mode == "squall":
            self.squall_mode = True
            self._rebuild_completer()
            _console.print("[green]Mode: SQUALL[/green]")
        else:
            _console.print(f"[yellow]Unknown mode {mode!r}. Use 'datalog' or 'squall'.[/yellow]")

    # -----------------------------------------------------------------------
    # Slash-command handlers
    # -----------------------------------------------------------------------

    def _cmd_datalog(self) -> None:
        """Show Datalog IR of the last SQUALL query (/datalog)."""
        if self._last_program is None:
            _console.print("[yellow]No query has been executed yet.[/yellow]")
            return
        if not self.squall_mode:
            _console.print("[yellow]/datalog only shows the Datalog IR of a SQUALL query. "
                           "Use /rewritten to see magic-sets rewriting of a Datalog query.[/yellow]")
            return
        self._ensure_engine()
        try:
            parsed = squall_parser(self._last_program)
            printer = DatalogPrettyPrinter()

            if isinstance(parsed, SquallProgram):
                if parsed.queries:
                    for i, q in enumerate(parsed.queries):
                        label = parsed.query_names.get(i, f"obtain_{i}")
                        _console.print(f"\n[bold magenta]── query ({label}) ──[/bold magenta]")
                        _console.print(printer.walk(q))
                for rule in parsed.rules:
                    _console.print(f"\n[bold magenta]── rule ──[/bold magenta]")
                    _console.print(printer.walk(rule))
            elif isinstance(parsed, (DatalogUnion, LogicUnion)):
                for f in parsed.formulas:
                    _console.print(printer.walk(f))
            else:
                _console.print(printer.walk(parsed))
        except Exception as exc:
            _console.print(f"[red]Failed to show Datalog IR: {exc}[/red]")

    def _cmd_rewritten(self) -> None:
        """Re-run last query showing magic-sets rewritten Datalog (/rewritten)."""
        if self._last_program is None:
            _console.print("[yellow]No query has been executed yet.[/yellow]")
            return
        self._ensure_engine()
        try:
            _console.print("[bold cyan]Magic-sets rewritten Datalog:[/bold cyan]")
            if self.squall_mode:
                result = self._nl.execute_squall_program(
                    self._last_program, dry_run=True,
                )
            else:
                result = self._nl.execute_datalog_program(
                    self._last_program, dry_run=True,
                )
        except Exception as exc:
            _console.print(f"[red]Failed to show rewritten query: {exc}[/red]")

    def _handle_dot_command(self, line: str) -> bool:
        """Handle a dot-command.  Returns True if the session should continue."""
        cmd = line.strip()
        parts = cmd.split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        handlers = {
            ".help": lambda: self._cmd_help(),
            ".engines": lambda: self._cmd_engines(),
            ".sets": lambda: self._cmd_sets(),
            ".predicates": lambda: self._cmd_predicates(),
            ".save": lambda: self._cmd_save(rest),
            ".view": lambda: self._cmd_view(rest),
            ".mode": lambda: self._cmd_mode(rest),
            ".quit": lambda: sys.exit(0),
            ".exit": lambda: sys.exit(0),
        }
        handler = handlers.get(verb)
        if handler:
            handler()
        else:
            _console.print(f"[yellow]Unknown command: {verb}. Try .help[/yellow]")
        return True

    def _handle_slash_command(self, line: str) -> bool:
        """Handle a slash-command.  Returns True if the session should continue."""
        cmd = line.strip()
        parts = cmd.split(None, 1)
        verb = parts[0].lower() if parts else ""

        handlers = {
            "/rewritten": lambda: self._cmd_rewritten(),
            "/datalog": lambda: self._cmd_datalog(),
            "/help": lambda: self._cmd_help(),
        }
        handler = handlers.get(verb)
        if handler:
            handler()
        else:
            _console.print(f"[yellow]Unknown slash command: {verb}. Try /help[/yellow]")
        return True

    # -----------------------------------------------------------------------
    # Prompt style
    # -----------------------------------------------------------------------

    def _get_prompt(self) -> str:
        mode = "SQ" if self.squall_mode else "DL"
        return f"nl({self.engine_name}:{mode})> "

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL."""
        _console.print()
        _console.print(Panel.fit(
            "[bold yellow]NeuroLang Interactive Query REPL[/bold yellow]\n"
            "Type [green].help[/green] for commands.  "
            "Ctrl+D or [green].quit[/green] to exit.",
            border_style="cyan",
        ))
        _console.print()

        # Show default engine info
        try:
            self._ensure_engine()
            cfg = engine_registry.get_engine_config(self.engine_name)
            _console.print(f"[dim]Engine: {self.engine_name} — {cfg.get('description', '')}[/dim]")
            _console.print(f"[dim]Mode: {'SQUALL' if self.squall_mode else 'Datalog'}[/dim]")
        except Exception as exc:
            _console.print(f"[yellow]Engine pre-load skipped: {exc}[/yellow]")

        while True:
            try:
                with patch_stdout():
                    line = self._session.prompt(self._get_prompt())
            except (EOFError, KeyboardInterrupt):
                _console.print("\n[dim]Goodbye.[/dim]")
                break

            if not line or not line.strip():
                continue

            # Dot commands
            if line.startswith("."):
                self._handle_dot_command(line)
                continue

            # Slash commands
            if line.startswith("/"):
                self._handle_slash_command(line)
                continue

            # Execute as query
            program = line.strip()
            self._execute_and_display(program)


def main(
    engine_name: str = "neurosynth",
    data_dir: str = "neurolang_data",
    resolution: Optional[float] = None,
    squall_mode: bool = False,
    sort_specs: Optional[list[str]] = None,
) -> None:
    """Entry point for the interactive TUI launched from the CLI."""
    app = InteractiveTuiApp(
        engine_name=engine_name,
        data_dir=data_dir,
        resolution=resolution,
        squall_mode=squall_mode,
        sort_specs=sort_specs,
    )
    app.run()
