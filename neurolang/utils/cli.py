"""
Command-line interface for NeuroLang query execution.

Provides a ``neurolang-query`` command that initializes a NeuroLang engine
from the :ref:`engine registry <neurolang.utils.engine_registry>` and
executes Datalog queries from arguments, files, or standard input.

Usage
-----
::

    # List available engines
    neurolang-query --list-engines

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
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from neurolang import expressions as ir
from neurolang.datalog.chase import Chase
from neurolang.datalog.expressions import Implication
from neurolang.datalog import WrappedRelationalAlgebraSet
from neurolang.expressions import (
    Constant, Expression, FunctionApplication, Lambda, Query, Symbol,
)
from neurolang.frontend import NeurolangPDL
from neurolang.frontend.datalog.pretty_printer import DatalogPrettyPrinter
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


def _emit(df, fmt, sort_by=None):
    """Format a DataFrame in *fmt* mode (table, csv, or json)."""
    if df.empty:
        return "(empty)"
    if sort_by:
        columns, ascending = zip(*sort_by)
        valid = [(c, a) for c, a in sort_by if c in df.columns]
        invalid = [c for c, _ in sort_by if c not in df.columns]
        for col in invalid:
            print(
                f"Warning: sort column {col!r} not found in result, "
                f"ignoring.",
                file=sys.stderr,
            )
        if valid:
            cols, asc = zip(*valid)
            df = df.sort_values(by=list(cols), ascending=list(asc))
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "json":
        return df.to_json(orient="records", indent=2)
    return df.to_string(index=False)


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
        # Unwrap Constant -> concrete set when chase produced wrapped result
        if hasattr(result, "value") and hasattr(result.value, "unwrap"):
            inner = result.value.unwrap()
            if hasattr(inner, "as_pandas_dataframe"):
                return _emit(
                    _rename_unnamed_columns(
                        inner.as_pandas_dataframe(), column_names
                    ),
                    fmt,
                    sort_by=sort_by,
                )
            result = inner

        # Get a DataFrame from the bare set (named or unnamed columns)
        if hasattr(result, "columns") and len(result.columns) > 0:
            df = pd.DataFrame(iter(result), columns=result.columns)
        else:
            rows = list(iter(result))
            if not rows:
                return "(empty)"
            arity = result.arity if hasattr(result, "arity") else len(rows[0])
            df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(arity)])

        return _emit(
            _rename_unnamed_columns(df, column_names), fmt, sort_by=sort_by
        )
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

    return parser


def _execute_program(nl: NeurolangPDL, program_text: str):
    """
    Execute a Datalog program using direct chase evaluation.

    This bypasses :meth:`NeurolangPDL._execute_query` which does not support
    query bodies that are conjunctions, comparisons, or EDB-only predicates.
    Instead it converts every ``Query`` into an ``Implication`` rule, adds it
    to the program's IDB, and runs the deterministic chase — the same strategy
    used by the base :class:`~neurolang.frontend.QueryBuilderDatalog`.

    Parameters
    ----------
    nl :
        Initialised engine (``NeurolangPDL`` or ``NeurolangDL``).
    program_text :
        Datalog program, possibly containing one ``Query`` rule.

    Returns
    -------
    ``None``, ``bool``, or a relational algebra set.

    """
    from neurolang.frontend.datalog.standard_syntax import (
        parser as datalog_parser,
    )

    ir_prog = datalog_parser(program_text)

    formulas = list(ir_prog.formulas)
    queries = [f for f in formulas if isinstance(f, ir.Query)]
    others = [f for f in formulas if not isinstance(f, ir.Query)]

    for f in others:
        nl.program_ir.walk(f)

    if len(queries) == 0:
        return None

    if len(queries) > 1:
        raise ValueError("Only a single query per program is supported.")

    q = queries[0]

    # Extract both the head predicate name and the argument variable names
    # from the parsed query head.  The head is always a FunctionApplication
    # whose functor is a Symbol (simple) or a Lambda wrapping a Symbol (parser
    # sugar), and whose args are Symbol nodes whose .name carries the Datalog
    # variable name.
    if isinstance(q.head, ir.FunctionApplication):
        column_names = [a.name for a in q.head.args]
        functor = (
            q.head.functor.body
            if isinstance(q.head.functor, ir.Lambda)
            else q.head.functor
        )
    else:
        column_names = None
        functor = q.head

    pred_name = (
        functor.name if isinstance(functor, ir.Symbol) else str(functor)
    )

    # Query → Implication so the DatalogProgram walker registers it as IDB
    rule = Implication(q.head, q.body)
    nl.program_ir.walk(rule)

    chase = Chase(nl.program_ir)
    solution = chase.build_chase_solution()

    for sym, val in solution.items():
        if sym.name == pred_name:
            return val, column_names

    return None


def _execute_squall_program(
    nl: NeurolangPDL, program_text: str
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

    Returns
    -------
    None
        When there are no ``obtain`` queries.
    NamedRelationalAlgebraFrozenSet
        When there is exactly one ``obtain`` query.
    Dict[str, NamedRelationalAlgebraFrozenSet]
        When there are multiple ``obtain`` queries.

    """
    return nl.execute_squall_program(program_text)


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


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: parse arguments, build engine, run query."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_engines:
        engine_registry.show_engines()
        return

    # --show-datalog only needs the parser, not an engine.  Handle it
    # early to avoid the expensive engine build (which downloads data).
    if args.show_datalog:
        program = _read_query(args)
        if not program or not program.strip():
            print("Error: no query provided.", file=sys.stderr)
            sys.exit(1)
        if not args.squall:
            print(
                "Error: --show-datalog requires --squall.", file=sys.stderr
            )
            sys.exit(1)
        _show_squall_datalog(program)
        return

    if args.engine not in engine_registry.list_engine_names():
        available = ", ".join(engine_registry.list_engine_names())
        print(
            f"Error: unknown engine {args.engine!r}. "
            f"Available engines: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution
    )

    if args.list_predicates:
        _list_predicates(args.engine)
        return

    if args.list_sets:
        _list_sets(nl)
        return

    program = _read_query(args)
    if not program or not program.strip():
        print("Error: no query provided.", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    sort_by = _parse_sort_spec(args.sort)

    if args.squall:
        result = _execute_squall_program(nl, program)
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
        else:
            output = _format_result(
                result, fmt=args.format, column_names=None,
                sort_by=sort_by,
            )
            if output:
                print(output)
    else:
        result = _execute_program(nl, program)
        if isinstance(result, tuple):
            result, column_names = result
        else:
            column_names = None
        output = _format_result(
            result, fmt=args.format, column_names=column_names,
            sort_by=sort_by,
        )
        if output:
            print(output)

    elapsed = time.perf_counter() - t0
    print(f"Query completed in {elapsed:.2f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
