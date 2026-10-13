"""
Command-line interface for NeuroLang query execution.

Provides a ``neurolang-query`` command that initializes a NeuroLang engine
from the :mod:`engine registry <neurolang.utils.engine_registry>` and
executes Datalog queries from arguments, files, or standard input.

Usage
-----
::

    # List available engines
    neurolang-query --list-engines

    # Print the SQUALL grammar and parser metadata as JSON
    neurolang-query --grammar-json

    # List predicates for an engine
    neurolang-query --engine neurosynth --list-predicates

    # List RA sets for an engine
    neurolang-query --engine neurosynth --list-sets

    # Inline query
    neurolang-query "ans(t) :- term_in_study_tfidf(t, w, s)"

    # Query from file
    neurolang-query -f query.dl

    # Query from stdin
    echo "ans(t) :- term_in_study_tfidf(t, f, s)" | neurolang-query

    # Destrieux atlas engine
    neurolang-query --engine destrieux "ans(name) :- destrieux(name, region)"

    # Squall (controlled English) query
    neurolang-query --squall "obtain every peak_reported."
    neurolang-query --squall -f query.squall

    # Show the Datalog IR for a SQUALL query (no execution)
    neurolang-query --squall --show-datalog "obtain every Voxel in 3D that a Study reported."

    # Show the relational algebra plans of a query (no execution)
    neurolang-query --show-ra "ans(x) :- P(x)"

    # ... annotated with the number of rows each plan node produces
    neurolang-query --show-ra --ra-analyze "ans(x) :- P(x)"

    # Interactive REPL (needs ``pip install "neurolang[tui]"``)
    neurolang-query -i --engine neurosynth
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from lark.exceptions import LarkError

from neurolang.datalog import WrappedRelationalAlgebraSet
from neurolang.exceptions import NeuroLangException
from neurolang.frontend import NeurolangPDL
from neurolang.frontend.datalog import (
    squall_grammar_metadata,
)
from neurolang.frontend.datalog.pretty_printer import DatalogPrettyPrinter
from neurolang.frontend.error_formatting import print_formatted_error
from neurolang.frontend.show_ra import ShowRAOptions
from neurolang.utils import engine_registry


def _parse_sort_spec(specs: list[str]) -> list[tuple[str, bool]]:
    """Parse --sort flag values into (column, ascending) pairs.

    Parameters
    ----------
    specs :
        Raw strings from ``--sort``, each in ``column`` or ``column:dir``
        format where ``dir`` is ``asc`` or ``desc``.

    Returns
    -------
    list[tuple[str, bool]]
        ``(column_name, ascending)`` tuples.  Ascending is ``True``.
    """
    sort_by: list[tuple[str, bool]] = []
    for spec in specs:
        parts = spec.split(":", maxsplit=1)
        col = parts[0]
        ascending = True
        if len(parts) == 2:
            direction = parts[1].lower()
            if direction == "asc":
                ascending = True
            elif direction == "desc":
                ascending = False
            else:
                print(
                    f"Warning: invalid sort direction {parts[1]!r} "
                    f"for column {col!r}, defaulting to ascending.",
                    file=sys.stderr,
                )
        sort_by.append((col, ascending))
    return sort_by


def _read_query(args) -> str:
    """Obtain the Datalog program from the CLI arguments."""
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.query:
        return args.query
    if not sys.stdin.isatty():
        return sys.stdin.read()
    parser = _build_parser()
    parser.print_help()
    sys.exit(1)


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


def sort_dataframe(df, sort_by):
    """
    Return *df* sorted by the ``(column, ascending)`` pairs of *sort_by*.

    Columns that *df* lacks are ignored with a warning on stderr.
    """
    if not sort_by:
        return df
    for col, _ in sort_by:
        if col not in df.columns:
            print(
                f"Warning: sort column {col!r} not found in result, "
                f"ignoring.",
                file=sys.stderr,
            )
    valid = [(c, a) for c, a in sort_by if c in df.columns]
    if not valid:
        return df
    cols, asc = zip(*valid)
    return df.sort_values(by=list(cols), ascending=list(asc))


def _emit(df, fmt, sort_by=None):
    """Format a DataFrame in *fmt* mode (table, csv, or json)."""
    if df.empty:
        return "(empty)"
    df = sort_dataframe(df, sort_by)
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "json":
        return df.to_json(orient="records", indent=2)
    return df.to_string(index=False)


def result_to_dataframe(result, column_names=None) -> pd.DataFrame:
    """
    Return the rows of a query *result* as a DataFrame.

    *result* is a relational algebra set, possibly wrapped in a
    ``Constant``. Auto-generated column names are replaced by
    *column_names* when given.
    """
    if hasattr(result, "value") and hasattr(result.value, "unwrap"):
        result = result.value.unwrap()
        if hasattr(result, "as_pandas_dataframe"):
            return _rename_unnamed_columns(
                result.as_pandas_dataframe(), column_names
            )
    if hasattr(result, "columns") and len(result.columns) > 0:
        df = pd.DataFrame(iter(result), columns=result.columns)
    else:
        rows = list(iter(result))
        arity = getattr(result, "arity", len(rows[0]) if rows else 0)
        df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(arity)])
    return _rename_unnamed_columns(df, column_names)


def _format_result(
    result, fmt: str = "table", column_names: Optional[list[str]] = None,
    sort_by: Optional[list[tuple[str, bool]]] = None,
) -> str:
    """Format a query result for display."""
    if result is None:
        return ""
    if isinstance(result, bool):
        return "true" if result else "false"
    try:
        df = result_to_dataframe(result, column_names)
        return _emit(df, fmt, sort_by=sort_by)
    except Exception as exc:
        print(f"Warning: result formatting failed: {exc}", file=sys.stderr)
        return str(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurolang-query",
        description="Run a Datalog query against a NeuroLang dataset engine.",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Datalog program string (e.g. 'ans(x) :- P(x)'). "
        "If omitted and stdin is a pipe, read from stdin.",
    )
    source.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        default=None,
        help="Read the Datalog program from a file.",
    )

    parser.add_argument(
        "--engine",
        "-e",
        default="neurosynth",
        help="Dataset engine to use (default: neurosynth).  "
        "Use --list-engines to see all available engines.",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        metavar="DIR",
        default="neurolang_data",
        help="Directory for cached data downloads (default: neurolang_data).",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=None,
        metavar="MM",
        help="Isotropic MNI resolution in mm (default: native).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--list-predicates",
        "-l",
        action="store_true",
        help="Instead of running a query, list the available predicates "
        "for the engine and exit.",
    )
    parser.add_argument(
        "--list-sets",
        action="store_true",
        help="Instead of running a query, list the available Relational "
        "Algebra sets (EDB relations) for the engine and exit.",
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available dataset engines and exit.",
    )
    parser.add_argument(
        "--grammar-json",
        action="store_true",
        help="Print the SQUALL Lark grammar and parser metadata as JSON "
        "and exit (no engine needed).",
    )
    parser.add_argument(
        "--squall",
        "-s",
        action="store_true",
        help="Interpret the input as a SQUALL (controlled English) program "
        "instead of classical Datalog syntax.  Supports ``define as`` "
        "rule definitions and ``obtain`` queries.",
    )
    parser.add_argument(
        "--show-datalog",
        "-D",
        action="store_true",
        help="When used with --squall, print the Datalog IR (rules and "
        "queries) that the SQUALL program compiles to, then exit.  "
        "Useful for debugging and understanding the translation.",
    )
    parser.add_argument(
        "--show-rewritten",
        "-R",
        action="store_true",
        help="Build the requested engine, print the Datalog program "
        "after magic-sets rewriting, then exit without running the "
        "chase. Useful for inspecting how a SQUALL or Datalog query "
        "is rewritten before execution.",
    )
    parser.add_argument(
        "--show-ra",
        "-Q",
        action="store_true",
        help="Print the Relational Algebra plan that the Datalog "
        "program compiles to, then exit without running the chase. "
        "Useful for debugging and understanding the query compilation.",
    )
    parser.add_argument(
        "--ra-verbose",
        action="store_true",
        help="When used with --show-ra, print the plans in full, without "
        "collapsing trivial renames and projections of base relations.",
    )
    parser.add_argument(
        "--ra-analyze",
        action="store_true",
        help="When used with --show-ra, run the query and annotate every "
        "node of the plans with the number of rows it produced (like "
        "SQL's EXPLAIN ANALYZE).",
    )
    parser.add_argument(
        "--sort",
        "-S",
        action="append",
        default=[],
        metavar="COL[:dir]",
        dest="sort",
        help="Sort output by COL (ascending by default). "
        "Append ':desc' for descending, ':asc' for ascending. "
        "Repeatable for multi-key sorts (first key has highest priority).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        dest="interactive",
        help="Start an interactive REPL on the engine, rendering the "
        "query results as tables. A query given as argument or with "
        "--file is run first; --show-ra and --show-rewritten apply to "
        "every query of the session.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="On errors, also print the Python traceback (same as setting "
        "the NEUROLANG_DEBUG=1 environment variable).",
    )

    return parser


def _execute_program(
    nl: NeurolangPDL, program_text: str, show_rewritten: bool = False,
    show_ra=False, dry_run: bool = False,
):
    """
    Execute a Datalog program and return the result if a query is present.

    Delegates to :meth:`NeurolangPDL.execute_datalog_program` which handles
    the full solver stack (deterministic chase + probabilistic solver + magic
    sets rewriting) through the frontend's query resolution API.

    Parameters
    ----------
    nl :
        Initialised engine (``NeurolangPDL`` or ``NeurolangDL``).
    program_text :
        Datalog program, possibly containing one ``Query`` rule.
    show_rewritten :
        If True, print the Datalog program after magic-sets rewriting.
    dry_run :
        If True, print the rewritten program and return None without
        running the chase.
    show_ra :
        If True, print the named relational algebra expression
        of each rule instead of evaluating it. A
        :class:`~neurolang.frontend.show_ra.ShowRAOptions` selects how
        the plans are displayed.

    Returns
    -------
    ``None``
        When no query is present or when *dry_run* is True.
    ``bool``
        For boolean (headless) queries.
    ``RelationalAlgebraFrozenSet``
        The query result set (may have a ``.columns`` attribute).

    """
    return nl.execute_datalog_program(
        program_text, show_rewritten=show_rewritten, dry_run=dry_run,
        show_ra=show_ra,
    )


def _execute_squall_program(
    nl: NeurolangPDL, program_text: str, show_rewritten: bool = False,
    show_ra=False, dry_run: bool = False,
):
    """
    Execute a SQUALL (controlled English) program.

    Delegates to :meth:`NeurolangPDL.execute_squall_program`, which
    handles ``define as …`` rules and ``obtain …`` queries.

    Parameters
    ----------
    nl :
        Initialised engine.
    program_text :
        SQUALL program text.
    show_rewritten :
        If True, print the Datalog program after magic-sets rewriting.
    dry_run :
        If True, print the rewritten program and return None without
        running the chase.
    show_ra :
        If True, print the named relational algebra expression
        of each rule instead of evaluating it. A
        :class:`~neurolang.frontend.show_ra.ShowRAOptions` selects how
        the plans are displayed.

    Returns
    -------
    None
        When there are no ``obtain`` queries or when *dry_run* is True.
    NamedRelationalAlgebraFrozenSet
        When there is exactly one ``obtain`` query.
    Dict[str, NamedRelationalAlgebraFrozenSet]
        When there are multiple ``obtain`` queries.

    """
    return nl.execute_squall_program(
        program_text, show_rewritten=show_rewritten, dry_run=dry_run,
        show_ra=show_ra,
    )


def _format_ir(expr, fresh_map=None, _counter=None):
    """Backward-compat wrapper around DatalogPrettyPrinter.

    Parameters
    ----------
    expr : Expression
        The IR node to format.
    fresh_map : dict, optional
        Mapping from original fresh names to their short aliases.
    _counter : list, optional
        Mutable counter [int] for generating fresh-variable aliases.
    """
    return DatalogPrettyPrinter(fresh_map, _counter).walk(expr)


def _show_squall_datalog(program_text: str) -> None:
    """Parse a SQUALL program and print the Datalog IR to stdout.

    This is the backend for ``neurolang-query --squall --show-datalog``.
    It parses the SQUALL text, then prints every rule and query using
    :class:`DatalogPrettyPrinter` — a PatternWalker-based formatter that
    strips type annotations, uses ``:-`` notation for rules/queries, and
    shortens fresh variables to ``s₀``, ``s₁``, etc.

    Parameters
    ----------
    program_text :
        SQUALL program text.
    """
    from neurolang.datalog import Union as DatalogUnion
    from neurolang.frontend.datalog.squall_syntax_lark import (
        parser as squall_parser,
        SquallProgram,
    )
    from neurolang.logic import Union as LogicUnion

    parsed = squall_parser(program_text)
    printer = DatalogPrettyPrinter()

    if isinstance(parsed, SquallProgram):
        if parsed.queries:
            for i, q in enumerate(parsed.queries):
                label = parsed.query_names.get(i, f"obtain_{i}")
                print(f"\u2500\u2500 query ({label}) \u2500\u2500")
                print(printer.walk(q))
                print()
        for rule in parsed.rules:
            print("\u2500\u2500 rule \u2500\u2500")
            print(printer.walk(rule))
            print()
    elif isinstance(parsed, (DatalogUnion, LogicUnion)):
        for f in parsed.formulas:
            print(printer.walk(f))
            print()
    else:
        print(printer.walk(parsed))


def _list_predicates(engine_name: str) -> None:
    """Print predicate metadata from the YAML engine config."""
    cfg = engine_registry.get_engine_config(engine_name)
    print(f"\nEngine: {engine_name}")
    print(f"  {cfg.get('description', '')}")
    predicates = engine_registry.get_predicates(engine_name)
    if not predicates:
        print("\n  (no predicate metadata declared)")
        return
    print("\nAvailable predicates:")
    for pname, info in predicates.items():
        cols = ", ".join(info.get("columns", []))
        desc = info.get("description", "")
        print(f"\n  {pname}({cols})")
        if desc:
            print(f"    {desc}")


def _list_sets(nl: NeurolangPDL) -> None:
    """
    Print the actual Relational Algebra sets registered in the engine.

    Iterates over the engine's symbol table and displays every entry backed
    by a :class:`~neurolang.datalog.WrappedRelationalAlgebraSet` — i.e.,
    concrete extensional (EDB) relations loaded from CSV/TSV files or added
    via the Python API.

    """
    print()
    st = nl.program_ir.symbol_table
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
        print("  (no RA sets registered)")
        return

    name_w = max(len(s) for s, _, _, _ in sets_found) + 2
    cols_w = max(len(str(c)) for _, c, _, _ in sets_found) + 2
    size_w = max(len(str(s)) for _, _, s, _ in sets_found) + 2

    header = (
        f"  {'Set':<{name_w}}  {'Columns':<{cols_w}}  {'Rows':>{size_w}}  Arity"
    )
    print(header)
    print(f"  {'-' * name_w}  {'-' * cols_w}  {'-' * size_w}  -----")
    for name, columns, size, arity in sorted(sets_found, key=lambda x: x[0]):
        cols_str = str(columns)
        print(
            f"  {name:<{name_w}}  {cols_str:<{cols_w}}  {str(size):>{size_w}}  {arity}"
        )
    print(f"\n  Total RA sets: {len(sets_found)}")




_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _known_symbol_names(nl: Optional[NeurolangPDL]) -> list:
    """Return the identifier-like symbol names defined in *nl*."""
    if nl is None:
        return []
    names = {getattr(s, "name", str(s)) for s in nl.symbol_table}
    return sorted(n for n in names if _IDENTIFIER.match(n))


def _debug_enabled(args) -> bool:
    """Return True if ``--debug`` or ``NEUROLANG_DEBUG`` is set."""
    return args.debug or os.environ.get("NEUROLANG_DEBUG", "0") not in (
        "", "0"
    )


def report_error(
    exc: Exception, program: str, *, squall: bool, debug: bool, nl=None,
    file=None,
) -> None:
    """
    Print the query error *exc* as a formatted message.

    The message shows the location of the error in *program*, hints and
    the symbols of the engine *nl* close to an unknown one. In *debug*
    mode the traceback is printed first; otherwise unexpected errors get
    a hint to use ``--debug``. *file* defaults to ``sys.stderr``.
    """
    stream = sys.stderr if file is None else file
    if debug:
        traceback.print_exception(
            type(exc), exc, exc.__traceback__, file=stream
        )
    print_formatted_error(
        exc, program,
        language="SQUALL" if squall else "Datalog",
        known_symbols=_known_symbol_names(nl),
        file=stream,
    )
    if not (debug or isinstance(exc, (NeuroLangException, LarkError))):
        print(
            "Run with --debug (or NEUROLANG_DEBUG=1) to see the "
            "traceback.", file=stream,
        )


def _run_reporting_errors(action, program: str, args, nl=None):
    """
    Run *action* and report any failure as a formatted error.

    On error, prints a readable message (source context, location and
    hints) to stderr and exits with status 1. In debug mode
    (``--debug`` or ``NEUROLANG_DEBUG=1``) the traceback is printed first.
    """
    try:
        return action()
    except Exception as exc:
        report_error(
            exc, program, squall=args.squall, debug=_debug_enabled(args),
            nl=nl,
        )
        sys.exit(1)


def _dry_run_with_engine(args, **flags) -> None:
    """
    Build the requested engine and dry-run the query on it.

    The query is read from ``args`` and run as SQUALL or Datalog with
    ``dry_run=True`` plus the extra keyword ``flags``.  Exits with status
    1 when no query is provided.
    """
    program = _read_query(args)
    if not program or not program.strip():
        print("Error: no query provided.", file=sys.stderr)
        sys.exit(1)
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution
    )
    execute = _execute_squall_program if args.squall else _execute_program
    _run_reporting_errors(
        lambda: execute(nl, program, dry_run=True, **flags),
        program, args, nl,
    )


def _show_ra_options(args):
    """Return the :class:`ShowRAOptions` requested by the CLI ``args``."""
    return ShowRAOptions(
        collapse=not args.ra_verbose, analyze=args.ra_analyze
    )


def _check_flag_dependencies(args) -> None:
    """
    Exit with an error when a flag is used without the flag it needs.

    ``--show-datalog`` needs ``--squall``, and ``--ra-verbose`` and
    ``--ra-analyze`` need ``--show-ra``.
    """
    if args.show_datalog and not args.squall:
        print("Error: --show-datalog requires --squall.", file=sys.stderr)
        sys.exit(1)
    for flag, used in (
        ("--ra-verbose", args.ra_verbose),
        ("--ra-analyze", args.ra_analyze),
    ):
        if used and not args.show_ra:
            print(f"Error: {flag} requires --show-ra.", file=sys.stderr)
            sys.exit(1)


def _read_query_or_exit(args) -> str:
    """Return the query given in ``args``, exiting when there is none."""
    program = _read_query(args)
    if not program or not program.strip():
        print("Error: no query provided.", file=sys.stderr)
        sys.exit(1)
    return program


def _run_engineless_command(args) -> bool:
    """
    Run the commands that need no engine, returning whether one ran.

    These are ``--list-engines``, ``--grammar-json`` and
    ``--show-datalog``, which are handled before the (expensive) engine
    build.
    """
    if args.list_engines:
        engine_registry.show_engines()
        return True
    # --grammar-json only needs the parser, not an engine.
    if args.grammar_json:
        print(json.dumps(squall_grammar_metadata(), indent=2))
        return True
    # --show-datalog only needs the parser, not an engine.
    if args.show_datalog:
        program = _read_query_or_exit(args)
        _check_flag_dependencies(args)
        _run_reporting_errors(
            lambda: _show_squall_datalog(program), program, args
        )
        return True
    return False


def _check_engine_name(args) -> None:
    """Exit with an error when ``args`` names an unknown engine."""
    if args.engine not in engine_registry.list_engine_names():
        available = ", ".join(engine_registry.list_engine_names())
        print(
            f"Error: unknown engine {args.engine!r}. "
            f"Available engines: {available}",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_dry_run_command(args) -> bool:
    """
    Run ``--show-rewritten`` or ``--show-ra``, returning whether one ran.

    Both require the requested engine (probabilistic SQUALL programs
    depend on engine data and the CP-Logic solver): the engine is built,
    the rewritten program or the RA plans are printed, and the CLI exits
    without running the query. With both flags, the rewritten program is
    printed before the RA plans.
    """
    if args.show_ra:
        _dry_run_with_engine(
            args, show_ra=_show_ra_options(args),
            show_rewritten=args.show_rewritten,
        )
        return True
    if args.show_rewritten:
        _dry_run_with_engine(args)
        return True
    return False


def _print_result(result, args, sort_by) -> None:
    """Print a query ``result``, one block per ``obtain`` when a dict."""
    if isinstance(result, dict):
        for key, sub_result in result.items():
            output = _format_result(
                sub_result, fmt=args.format, column_names=None,
                sort_by=sort_by,
            )
            if output:
                print(f"── {key} ──")
                print(output)
                print()
        return
    output = _format_result(
        result, fmt=args.format, column_names=None, sort_by=sort_by,
    )
    if output:
        print(output)


def _run_query(args) -> None:
    """Build the engine and run the query of ``args``, printing its result."""
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution
    )

    if args.list_sets:
        _list_sets(nl)
        return

    program = _read_query_or_exit(args)

    t0 = time.perf_counter()
    sort_by = _parse_sort_spec(args.sort)
    execute = _execute_squall_program if args.squall else _execute_program
    result = _run_reporting_errors(
        lambda: execute(nl, program), program, args, nl
    )
    _print_result(result, args, sort_by)

    elapsed = time.perf_counter() - t0
    print(f"Query completed in {elapsed:.2f} s", file=sys.stderr, flush=True)


_NOT_INTERACTIVE_FLAGS = (
    ("--list-engines", "list_engines"),
    ("--grammar-json", "grammar_json"),
    ("--show-datalog", "show_datalog"),
    ("--list-predicates", "list_predicates"),
    ("--list-sets", "list_sets"),
)


def _check_interactive_flags(args) -> None:
    """Exit with an error when ``args`` combines -i with a listing flag."""
    for flag, dest in _NOT_INTERACTIVE_FLAGS:
        if getattr(args, dest):
            print(
                f"Error: {flag} cannot be used with --interactive.",
                file=sys.stderr,
            )
            sys.exit(1)


def _tui_settings(args, tui):
    """Return the REPL settings ``tui.TuiSettings`` requested by ``args``."""
    return tui.TuiSettings(
        engine_name=args.engine,
        squall=args.squall,
        sort_by=_parse_sort_spec(args.sort),
        show_ra=_show_ra_options(args) if args.show_ra else None,
        show_rewritten=args.show_rewritten,
        debug=_debug_enabled(args),
        history_path=str(Path.home() / ".neurolang_query_history"),
    )


def _program_argument(args) -> str:
    """Return the query given as argument or with ``--file``, or ''."""
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return args.query or ""


def _run_interactive(args) -> None:
    """
    Build the engine once and start the interactive REPL on it.

    A query given as argument or with ``--file`` is run in the session
    before the first prompt. ``--show-ra`` and ``--show-rewritten``
    apply to every query of the session.
    """
    _check_interactive_flags(args)
    _check_flag_dependencies(args)
    _check_engine_name(args)
    from neurolang.utils import tui

    program = _program_argument(args)
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution
    )
    tui.main(nl, _tui_settings(args, tui), initial_program=program)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: parse arguments, build engine, run query."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.interactive:
        _run_interactive(args)
        return
    if _run_engineless_command(args):
        return
    _check_flag_dependencies(args)
    _check_engine_name(args)
    if _run_dry_run_command(args):
        return

    # Predicate metadata comes from the YAML config: no need to build
    # (and download data for) the engine.
    if args.list_predicates:
        _list_predicates(args.engine)
        return

    _run_query(args)


if __name__ == "__main__":
    main()
