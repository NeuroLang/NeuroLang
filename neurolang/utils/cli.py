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
from neurolang.expression_pattern_matching import add_match
from neurolang.expression_walker import PatternWalker
from neurolang.expressions import (
    Constant, Expression, ExpressionBlock,
    FunctionApplication, Lambda, Query, Symbol,
)
from neurolang.frontend import NeurolangPDL
from neurolang.logic import (
    Conjunction, Disjunction, Negation,
    ExistentialPredicate, UniversalPredicate,
)
from neurolang.probabilistic.expressions import Condition
from neurolang.utils import engine_registry


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


def _emit(df, fmt):
    """Format a DataFrame in *fmt* mode (table, csv, or json)."""
    if df.empty:
        return "(empty)"
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "json":
        return df.to_json(orient="records", indent=2)
    return df.to_string(index=False)


def _format_result(
    result, fmt: str = "table", column_names: Optional[list[str]] = None
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
                    _rename_unnamed_columns(inner.as_pandas_dataframe(), column_names),
                    fmt,
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

        return _emit(_rename_unnamed_columns(df, column_names), fmt)
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


class DatalogPrettyPrinter(PatternWalker):
    """Pretty-print NeuroLang IR expressions as clean, human-readable text.

    Uses the PatternWalker pattern-matching framework to recursively format
    each expression type. Strips type annotations, uses ``:-`` notation for
    rules/queries, and renames fresh variables to short names (``s₀``, ``s₁``,
    …). Handles probabilistic ``Condition`` objects (``[A | B]``),
    ``ExpressionBlock``, ``Union``, and all standard FOL/Datalog constructs.

    Parameters
    ----------
    fresh_map : dict, optional
        Pre-populated mapping from original fresh names to short aliases.
    counter : list, optional
        Mutable one-element list ``[int]`` for generating fresh-variable
        aliases.

    Examples
    --------
    >>> printer = DatalogPrettyPrinter()
    >>> printer.walk(some_implication)
    'ans(X) :-\\n    Pred(X)'
    """

    def __init__(self, fresh_map=None, counter=None):
        super().__init__()
        self._fresh_map = fresh_map if fresh_map is not None else {}
        self._counter = counter if counter is not None else [0]

    def _sym_name(self, sym: Symbol) -> str:
        if sym.is_fresh:
            if sym.name not in self._fresh_map:
                n = self._counter[0]
                if n < 10:
                    self._fresh_map[sym.name] = f"s{chr(0x2080 + n)}"
                else:
                    self._fresh_map[sym.name] = f"s{n}"
                self._counter[0] += 1
            return self._fresh_map[sym.name]
        return sym.name

    def _indent_body(self, body: str) -> str:
        return "\n".join("    " + line for line in body.split("\n"))

    @add_match(Symbol)
    def format_symbol(self, expr: Symbol) -> str:
        return self._sym_name(expr)

    @add_match(Constant)
    def format_constant(self, expr: Constant) -> str:
        v = expr.value
        if callable(v) and hasattr(v, "__qualname__"):
            return f"\u27e8{v.__qualname__}\u27e9"
        return repr(v)

    @add_match(FunctionApplication)
    def format_fa(self, expr: FunctionApplication) -> str:
        args = ", ".join(self.walk(a) for a in expr.args)
        return f"{self.walk(expr.functor)}({args})"

    @add_match(Lambda)
    def format_lambda(self, expr: Lambda) -> str:
        body_s = self.walk(expr.function_expression)
        args_s = ", ".join(
            self._sym_name(a) if isinstance(a, Symbol) else self.walk(a)
            for a in expr.args
        )
        return f"\u03bb({args_s}). {body_s}"

    @add_match(Conjunction)
    def format_conjunction(self, expr: Conjunction) -> str:
        return " \u2227 ".join(self.walk(f) for f in expr.formulas)

    @add_match(Disjunction)
    def format_disjunction(self, expr: Disjunction) -> str:
        return " \u2228 ".join(self.walk(f) for f in expr.formulas)

    @add_match(Negation)
    def format_negation(self, expr: Negation) -> str:
        return f"\u00ac({self.walk(expr.formula)})"

    @add_match(ExistentialPredicate)
    def format_exists(self, expr: ExistentialPredicate) -> str:
        return (
            f"\u2203{self._sym_name(expr.head)}."
            f" ({self.walk(expr.body)})"
        )

    @add_match(UniversalPredicate)
    def format_forall(self, expr: UniversalPredicate) -> str:
        return (
            f"\u2200{self._sym_name(expr.head)}."
            f" ({self.walk(expr.body)})"
        )

    @add_match(Condition)
    def format_condition(self, expr: Condition) -> str:
        cond_s = self.walk(expr.conditioned)
        conding_s = self.walk(expr.conditioning)
        return f"[{cond_s} | {conding_s}]"

    @add_match(ExpressionBlock)
    def format_block(self, expr: ExpressionBlock) -> str:
        return "\n\n".join(self.walk(e) for e in expr.expressions)

    @add_match(Implication)
    def format_implication(self, imp: Implication) -> str:
        cons_s = self.walk(imp.consequent)
        ante_s = self._format_body(imp.antecedent)
        return f"{cons_s} :-\n{self._indent_body(ante_s)}"

    @add_match(Query)
    def format_query(self, expr: Query) -> str:
        head_walked = self.walk(expr.head)
        if isinstance(head_walked, tuple):
            head_s = "({})".format(", ".join(head_walked))
        else:
            head_s = head_walked
        body_s = self._format_body(expr.body)
        return f"{head_s} :-\n{self._indent_body(body_s)}"

    @add_match(Expression)
    def format_default(self, expr: Expression) -> str:
        return repr(expr)

    # ── body formatting ─────────────────────────────────────

    def _format_body(self, e):
        """Format an Implication/Query body.

        Conjunctions are displayed as a comma-separated list (the standard
        Datalog convention).  Everything else is dispatched through the
        regular walker.
        """
        if isinstance(e, Conjunction):
            return ",\n".join(self.walk(f) for f in e.formulas)
        return self.walk(e)


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

    if args.squall:
        result = _execute_squall_program(nl, program)
        if isinstance(result, dict):
            for key, sub_result in result.items():
                output = _format_result(
                    sub_result, fmt=args.format, column_names=None
                )
                if output:
                    print(f"── {key} ──")
                    print(output)
                    print()
        else:
            output = _format_result(
                result, fmt=args.format, column_names=None
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
            result, fmt=args.format, column_names=column_names
        )
        if output:
            print(output)

    elapsed = time.perf_counter() - t0
    print(f"Query completed in {elapsed:.2f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
