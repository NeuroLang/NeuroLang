"""
Command-line interface for NeuroLang query execution.

Provides a ``neurolang-query`` command that initializes a NeuroLang engine
from the :mod:`engine registry <neurolang_apps.utils.engine_registry>` and
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
    neurolang-query "ans(t) :- term_in_study_tfidf(s, t, w)"

    # Query from file
    neurolang-query -f query.dl

    # Query from stdin
    echo "ans(t) :- term_in_study_tfidf(s, t, w)" | neurolang-query

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

    # ... or run REPL statements and dot-commands from a pipe
    echo 'ans(x) :- study(x)' | neurolang-query -i

    # Write a query with an LLM (needs ``pip install "neurolang[llm]"``)
    neurolang-query --squall --ask "Which studies mention pain?" --run
"""

import argparse
import dataclasses
import difflib
import json
import os
import re
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from lark.exceptions import LarkError

from nl_core.datalog import WrappedRelationalAlgebraSet
from nl_expr.exceptions import NeuroLangException
from nl_expr.expressions import Constant, Symbol
from neurolang.frontend import NeurolangPDL
from neurolang.frontend.datalog import (
    squall_grammar_metadata,
)
from nl_core.datalog.pretty_printer import DatalogPrettyPrinter
from neurolang.frontend.error_formatting import print_formatted_error
from neurolang.frontend.show_ra import ShowRAOptions
from nl_expr.logic import ExistentialPredicate
from nl_core.probabilistic.expressions import Condition
from neurolang_apps.utils import engine_registry, query_highlight
from neurolang_apps.utils.sort_hints import sort_notes


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


# Column names of positions (``c0``, ``0``) and of fresh variables.
_GENERATED_COLUMN = re.compile(r"c\d+|fresh_\w+|\d+")


def _is_generated_column(column) -> bool:
    """
    Return whether *column* is an auto-generated column name.

    These are positions, ``c0``-like names, and the names of fresh
    variables, such as the one SQUALL gives the column of
    ``obtain every R.`` for a unary ``R``.
    """
    if isinstance(column, int):
        return True
    return isinstance(column, str) and (
        _GENERATED_COLUMN.fullmatch(column) is not None
    )


def _rename_unnamed_columns(df, column_names):
    """
    Replace auto-generated column names with *column_names* when safe.

    Only the generated columns are renamed, each after the name of
    *column_names* at its position, unless that name is None or another
    column's.
    """
    if column_names is None or len(column_names) != len(df.columns):
        return df
    taken = set(df.columns)
    df.columns = [
        new if _is_generated_column(old) and new not in taken | {None}
        else old
        for old, new in zip(df.columns, column_names)
    ]
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


def _declared_columns(engine_name: str) -> dict:
    """Return the columns the engine *engine_name* declares, by relation."""
    if not engine_name:
        return {}
    return {
        name: list(info["columns"])
        for name, info in engine_registry.get_predicates(engine_name).items()
        if info.get("columns")
    }


def squall_column_names(
    engine_name: str, program: str, relations: Optional[dict] = None,
    rules=(),
) -> dict:
    """
    Return column names for the results of a SQUALL program's queries.

    SQUALL names the columns of most results after fresh variables. The
    result is a dict from the key of each ``obtain`` (as in the results
    of
    :meth:`~neurolang.frontend.QueryBuilderDatalog.execute_squall_program`)
    to the names of its columns:

    * ``obtain every R.`` takes the columns of ``R``: those the engine
      configuration declares, those *relations* (a dict from relation
      names to their columns) gives, or those of the rules defining
      ``R``, in the program or among *rules*, the rules defined before;
    * otherwise, a column is named after the user's variable (``?s``),
      the aggregate (``count``), the ``as`` name of a single column, the
      probability, or the column of the first relation of the body its
      variable is in (``i, j, k`` for ``every Voxel in 3D``); an
      undeclared unary relation names its column after itself.
    """
    from neurolang.frontend.datalog.squall_syntax_lark import (
        SquallProgram,
        parser as squall_parser,
    )

    try:
        parsed = squall_parser(program)
    except Exception:
        return {}
    if not isinstance(parsed, SquallProgram):
        return {}
    columns = {**_declared_columns(engine_name), **(relations or {})}
    for rule in (*rules, *parsed.rules):
        _add_rule_columns(rule, columns)
    names = {}
    for i, query in enumerate(parsed.queries):
        query_columns = _query_columns(query, columns)
        if query_columns and any(query_columns):
            names[parsed.query_names.get(i, f"obtain_{i}")] = query_columns
    return names


def _add_rule_columns(rule, columns: dict) -> None:
    """Add the columns of the head of *rule* to *columns*, if unknown."""
    name = getattr(getattr(rule.consequent, "functor", None), "name", None)
    if isinstance(name, str) and name not in columns:
        names = _head_names(rule.consequent.args, rule.antecedent, columns)
        if all(names):
            columns[name] = names


def _query_columns(query, columns: dict):
    """Return the column names of *query*'s result, or None if unknown."""
    head = query.head
    named = isinstance(getattr(head, "functor", None), Symbol)
    terms = head.args if named else head
    terms = terms if isinstance(terms, tuple) else (terms,)
    whole = _whole_relation(query.body, terms, columns)
    if named and len(terms) == 1 and len(whole or ()) < 2:
        return [head.functor.name]
    if whole is not None:
        return whole
    return _head_names(terms, query.body, columns)


def _whole_relation(body, terms: tuple, columns: dict):
    """
    Return the columns of ``R`` if *body* is ``R(x)`` and *terms* ``(x,)``.

    That is the body of ``obtain every R.``, whatever the arity of R.
    """
    functor = getattr(body, "functor", None)
    if isinstance(functor, Symbol) and body.args == terms == terms[:1]:
        return list(columns[functor.name]) if functor.name in columns else None
    return None


def _head_names(head, body, columns: dict) -> list:
    """Return the names of the *head* terms of a rule of *body*."""
    atoms = list(_body_atoms(body))
    names = [_term_name(term, atoms, columns) for term in head]
    return _unique(names)


def _term_name(term, atoms, columns: dict):
    """Return the name of a column of the head *term*, or None."""
    if isinstance(term, Symbol):
        if not _is_generated_column(term.name):
            return term.name
        return _variable_column(term, atoms, columns)
    functor = getattr(term, "functor", None)
    if isinstance(functor, Symbol) and functor.name == "PROB":
        return "probability"
    if isinstance(functor, Constant):
        return _aggregate_name(functor.value)
    return None


def _aggregate_name(function) -> str:
    """Return the SQUALL name of the aggregate *function*."""
    from neurolang.frontend.datalog.squall_syntax_lark import _AGG_FUNC_MAP

    for name, constant in _AGG_FUNC_MAP.items():
        if constant.value is function:
            return name
    return getattr(function, "__name__", None)


def _body_atoms(formula):
    """
    Yield the predicates of *formula*, in order, but negated ones.

    They are those of its conjunctions, existentials and conditions.
    """
    functor = getattr(formula, "functor", None)
    if isinstance(functor, Symbol):
        yield formula
    for sub in _subformulas(formula):
        yield from _body_atoms(sub)


def _subformulas(formula) -> tuple:
    """Return the formulas of a conjunction, existential or condition."""
    if isinstance(formula, ExistentialPredicate):
        return (formula.body,)
    if isinstance(formula, Condition):
        return (formula.conditioned, formula.conditioning)
    return tuple(getattr(formula, "formulas", ()))


def _variable_column(variable, atoms, columns: dict):
    """
    Return the column of the first of *atoms* that *variable* is in.

    An atom of an undeclared unary relation names it after the relation.
    """
    for atom in atoms:
        if variable not in atom.args:
            continue
        known = columns.get(atom.functor.name)
        if known is not None and len(known) == len(atom.args):
            return known[atom.args.index(variable)]
        if known is None and len(atom.args) == 1:
            return atom.functor.name
    return None


def _unique(names: list) -> list:
    """Return *names*, suffixing the repeated ones with their position."""
    return [
        f"{name}_{n}" if name is not None and names.count(name) > 1 else name
        for n, name in enumerate(names)
    ]


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


def _set_frame(ra_set, columns: list) -> pd.DataFrame:
    """
    Return the rows of *ra_set*, whose columns are *columns*, as a DataFrame.

    The backend's own DataFrame is used, when it has these columns,
    rather than copying the rows one by one.
    """
    if hasattr(ra_set, "as_pandas_dataframe"):
        df = ra_set.as_pandas_dataframe()
        if list(df.columns) == columns:
            return df.copy(deep=False)
    return pd.DataFrame(iter(ra_set), columns=columns)


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
                result.as_pandas_dataframe().copy(deep=False), column_names
            )
    columns = list(getattr(result, "columns", ()))
    if columns:
        df = _set_frame(result, columns)
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
        "--no-engine-cache",
        action="store_true",
        help="Find the tables the engine's build takes long to find, as the "
        "studies activating the regions of its atlases, instead of reading "
        "those kept in the data directory (and write none there).",
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
        "--save",
        metavar="PATH",
        default=None,
        help="Also save the result in PATH, as the REPL's .save does: a "
        "table in CSV or TSV (.csv, .tsv, optionally .gz), or a NIfTI "
        "brain image (.nii, .nii.gz) when it has voxel columns, on the "
        "engine's grid, with at most one value column, e.g. a "
        "probability. A result of several obtain statements gives a file "
        "per statement. Needs the tui extra.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="At the end, show the brain image of the result in the "
        "viewer of the REPL's .view, an HTML page opened in the browser "
        "(its path is printed). Needs the tui extra.",
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
        "every query of the session. If stdin is not a terminal, its "
        "statements and dot-commands are run as a batch (exit status 1 "
        "if one fails).",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        default=None,
        help="Write a query answering QUESTION with an LLM and print it "
        "(SQUALL with --squall, else Datalog, unless the configuration "
        "sets the language). Needs the llm extra: pip install "
        "'neurolang[llm]'.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="With --ask, also run the query and print its result.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="With --ask, show how the query is found, on stderr: -v the "
        "reasoning, tool calls and results of each step, -vv every "
        "message sent to the LLM (the system prompt first) and its "
        "replies.",
    )
    parser.add_argument(
        "--strategy",
        choices=("react", "icl"),
        default=None,
        help="With --ask, the strategy of the LLM assistant: react (an "
        "agent exploring the engine with tools) or icl (few-shot prompt, "
        "then validation and repair). Default: the configuration's.",
    )
    parser.add_argument(
        "--steps",
        metavar="N",
        type=_positive_int,
        default=None,
        help="With --ask, the most steps the ReAct agent takes, tool "
        "calls until its final answer (default: the configuration's "
        "max_steps, 8).",
    )
    parser.add_argument(
        "--llm-config",
        metavar="PATH",
        default=None,
        help="Configuration file of the LLM providers (default: "
        "$NEUROLANG_LLM_CONFIG, then ~/.config/neurolang/llm.yaml, then "
        "the example shipped with NeuroLang).",
    )
    parser.add_argument(
        "--llm-provider",
        metavar="NAME",
        default=None,
        help="LLM provider of the configuration to use with --ask and "
        ".ask (default: its default_provider).",
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
    from nl_core.datalog import Union as DatalogUnion
    from neurolang.frontend.datalog.squall_syntax_lark import (
        parser as squall_parser,
        SquallProgram,
    )
    from nl_expr.logic import Union as LogicUnion

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


def _ra_sets(nl: NeurolangPDL, column_names: dict) -> list:
    """
    Return ``(name, columns, size, arity)`` for each RA set of *nl*.

    *column_names* maps set names to the columns shown instead of the
    set's own column names. A set redefined in an inner scope of the
    program is listed once, as the inner scope defines it.
    """
    sets_found = []
    symbol_table = nl.program_ir.symbol_table
    for sym_name in dict.fromkeys(symbol_table):
        sym_val = symbol_table[sym_name]
        val = getattr(sym_val, "value", sym_val)
        if not isinstance(val, WrappedRelationalAlgebraSet):
            continue
        ra_set = val.unwrap()
        clean_name = getattr(sym_name, "name", str(sym_name))
        columns = column_names.get(clean_name, list(ra_set.columns))
        try:
            size = len(ra_set)
        except Exception:
            size = "?"
        sets_found.append((clean_name, columns, size, ra_set.arity))
    return sets_found


def _list_sets(nl: NeurolangPDL, column_names: Optional[dict] = None) -> None:
    """
    Print the actual Relational Algebra sets registered in the engine.

    Iterates over the engine's symbol table and displays every entry backed
    by a :class:`~neurolang.datalog.WrappedRelationalAlgebraSet` — i.e.,
    concrete extensional (EDB) relations loaded from CSV/TSV files or added
    via the Python API. *column_names* maps set names to the columns
    shown for them instead of the set's own column names.

    """
    print()
    sets_found = _ra_sets(nl, column_names or {})

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


_UNDEFINED_PREDICATE = re.compile(
    r"Symbol (\S+) used in rule body has no entry in the symbol table"
)


def warning_note(message: str, known_symbols) -> str:
    """
    Return a one-line note for a warning *message* raised by a query.

    A predicate used without facts or rules is reported with the closest
    of the *known_symbols*, if any.
    """
    match = _UNDEFINED_PREDICATE.match(message)
    if match is None:
        return f"note: {message.strip().splitlines()[0]}"
    name = match.group(1)
    note = f"note: predicate {name} has no facts or rules"
    close = difflib.get_close_matches(name, known_symbols, n=1)
    if close:
        note += f" — did you mean {close[0]}?"
    return note


def _debug_enabled(args) -> bool:
    """Return True if ``--debug`` or ``NEUROLANG_DEBUG`` is set."""
    return args.debug or os.environ.get("NEUROLANG_DEBUG", "0") not in (
        "", "0"
    )


def report_error(
    exc: Exception, program: str, *, squall: bool, debug: bool, nl=None,
    file=None, engine_name: str = "",
) -> None:
    """
    Print the query error *exc* as a formatted message.

    The message shows the location of the error in *program*, hints and
    the symbols of the engine *nl* close to an unknown one. When the
    error concerns a predicate of the engine *engine_name* that spans
    entities of several columns, a note tells how to write it (see
    :func:`~neurolang.utils.sort_hints.sort_notes`). In *debug* mode the
    traceback is printed first; otherwise unexpected errors get a hint
    to use ``--debug``. *file* defaults to ``sys.stderr``.
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
    for note in sort_notes(exc, program, engine_name, squall):
        print(note, file=stream)
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
            nl=nl, engine_name=getattr(args, "engine", ""),
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
        args.engine, Path(args.data_dir), args.resolution,
        use_cache=not args.no_engine_cache,
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


def _result_column_names(args, program: str) -> dict:
    """Return the column names of the results of ``program``, by key."""
    if not args.squall:
        return {}
    return squall_column_names(args.engine, program)


def _print_result(result, args, sort_by, column_names=None) -> None:
    """
    Print a query ``result``, one block per ``obtain`` when a dict.

    ``column_names`` maps result keys to the names of their columns, as
    returned by :func:`squall_column_names`.
    """
    column_names = column_names or {}
    if isinstance(result, dict):
        for key, sub_result in result.items():
            output = _format_result(
                sub_result, fmt=args.format,
                column_names=column_names.get(key), sort_by=sort_by,
            )
            if output:
                print(f"── {key} ──")
                print(output)
                print()
        return
    names = next(iter(column_names.values()), None)
    output = _format_result(
        result, fmt=args.format, column_names=names, sort_by=sort_by,
    )
    if output:
        print(output)


def _export_session(args, nl, result, column_names, program):
    """
    Return the headless session of the REPL's ``.save`` and ``.view``
    commands (:mod:`neurolang_apps.utils.tui.nifti`) holding the *result* of
    *program*: its engine, its result blocks with the sorts of their
    columns, and consoles on the standard error.
    """
    from types import SimpleNamespace

    from rich.console import Console

    from neurolang_apps.utils.tui.provenance import column_sorts
    from neurolang_apps.utils.tui.results import ResultBlock

    sorts = column_sorts(nl, args.engine, program, bool(args.squall))
    sort_by = _parse_sort_spec(args.sort)
    column_names = column_names or {}
    keyed = result if isinstance(result, dict) else {"": result}
    blocks = []
    for key, value in keyed.items():
        if value is None or isinstance(value, bool):
            continue
        names = column_names.get(key) if isinstance(result, dict) else (
            next(iter(column_names.values()), None)
        )
        df = result_to_dataframe(value, names)
        if not df.empty:
            df = sort_dataframe(df, sort_by)
        block_sorts = sorts.get(key) if isinstance(result, dict) else (
            next(iter(sorts.values()), None)
        )
        if block_sorts is not None and len(block_sorts) != len(df.columns):
            block_sorts = None
        blocks.append(ResultBlock(df, key, sorts=block_sorts))
    return SimpleNamespace(
        settings=SimpleNamespace(engine_name=args.engine), _nl=nl,
        results=blocks, failed=False,
        console=Console(stderr=True), err_console=Console(stderr=True),
    )


def _save_and_view(result, args, column_names, nl, program) -> None:
    """
    Save the *result* of *program* in ``--save`` and show its image with
    ``--view``, as the REPL's ``.save`` and ``.view`` do for its last
    result: a table as CSV or TSV (``.csv``, ``.tsv``, optionally
    ``.gz``), or a NIfTI image on the engine's grid (``.nii``,
    ``.nii.gz``), and the image in the viewer's HTML page, opened in a
    browser when stderr is a terminal. They need the ``tui`` extra.

    Problems are reported on stderr and exit with status 1, after the
    result was printed.
    """
    if not (args.save or args.view):
        return
    # The result on stdout comes before the messages on stderr.
    sys.stdout.flush()
    try:
        from neurolang_apps.utils.tui import nifti
    except ImportError as exc:
        _exit_missing(
            exc, TUI_DEPENDENCIES,
            "--save and --view need the optional dependencies of the "
            "interactive mode", "tui",
        )
    session = _export_session(args, nl, result, column_names, program)
    if not session.results:
        print("Error: the query gave no result to save.", file=sys.stderr)
        sys.exit(1)
    if args.save:
        suffix = nifti.split_suffix(Path(args.save))[1].lower()
        if suffix not in (
            ".csv", ".csv.gz", ".tsv", ".tsv.gz", ".nii", ".nii.gz"
        ):
            print(
                f"Error: cannot tell the format of {args.save}: use .csv, "
                ".tsv (optionally .gz) for a table, or .nii, .nii.gz for "
                "an image.", file=sys.stderr,
            )
            sys.exit(1)
        nifti.cmd_save(session, shlex.quote(args.save))
    if args.view and not session.failed:
        nifti.cmd_view(session, "")
    if session.failed:
        sys.exit(1)


def _run_query(args) -> None:
    """Build the engine and run the query of ``args``, printing its result."""
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution,
        use_cache=not args.no_engine_cache,
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
    column_names = _result_column_names(args, program)
    _print_result(result, args, sort_by, column_names)
    _save_and_view(result, args, column_names, nl, program)

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


def _tui_settings(args, tui, argv=()):
    """
    Return the REPL settings ``tui.TuiSettings`` requested by ``args``.

    ``argv`` are the command-line arguments, shown in the transcript.
    """
    return tui.TuiSettings(
        engine_name=args.engine,
        squall=args.squall,
        output_format=args.format,
        sort_by=_parse_sort_spec(args.sort),
        show_ra=args.show_ra,
        ra_options=_show_ra_options(args),
        show_rewritten=args.show_rewritten,
        debug=_debug_enabled(args),
        history_path=str(Path.home() / ".neurolang_query_history"),
        data_dir=str(Path(args.data_dir).expanduser().resolve()),
        command_line=shlex.join(["neurolang-query", *argv]),
        llm_config=args.llm_config,
        llm_provider=args.llm_provider,
    )


TUI_DEPENDENCIES = ("prompt_toolkit", "rich")


def _exit_missing(exc, dependencies, needs, extra):
    """Raise the ImportError *exc*, unless it is of one of the optional
    *dependencies*: exit with status 1 and a message telling how to
    install the *extra* which has them."""
    if (exc.name or "").split(".")[0] not in dependencies:
        raise exc
    print(
        f"Error: {needs}. Install {'them' if len(dependencies) > 1 else 'it'}"
        f' with:\n    pip install "neurolang[{extra}]"', file=sys.stderr,
    )
    sys.exit(1)


def _import_tui():
    """
    Return the :mod:`neurolang_apps.utils.tui` module of the interactive REPL.

    If its optional dependencies are missing, exit with status 1 and a
    message telling how to install them.
    """
    try:
        from neurolang_apps.utils import tui
    except ImportError as exc:
        _exit_missing(
            exc, TUI_DEPENDENCIES,
            "the interactive mode needs the optional dependencies "
            f"{' and '.join(TUI_DEPENDENCIES)}", "tui",
        )
    return tui


def _program_argument(args) -> str:
    """Return the query given as argument or with ``--file``, or ''."""
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return args.query or ""


def _run_interactive(args, argv=()) -> None:
    """
    Build the engine once and start the interactive REPL on it.

    A query given as argument or with ``--file`` is run in the session
    before the first prompt. ``--show-ra`` and ``--show-rewritten``
    apply to every query of the session. When the standard input is not
    a terminal, its statements are run as a batch and the CLI exits with
    status 1 if one of them failed. ``argv`` are the command-line
    arguments, shown in the session's transcript.
    """
    _check_interactive_flags(args)
    _check_flag_dependencies(args)
    _check_engine_name(args)
    tui = _import_tui()

    program = _program_argument(args)
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution,
        use_cache=not args.no_engine_cache,
    )
    settings = _tui_settings(args, tui, argv)
    status = tui.main(nl, settings, initial_program=program)
    if status:
        sys.exit(status)


def _ask_language(args, config) -> str:
    """Return the language of the query of ``--ask``."""
    if args.squall:
        return "squall"
    if config.assistant.language != "auto":
        return config.assistant.language
    return "datalog"


# The lines of an observation that ``--ask -v`` prints.
VERBOSE_OBSERVATION_LINES = 12


def _print_step(step) -> None:
    """Print a step of the LLM assistant on stderr, in one line."""
    if step.action and step.action != "answer":
        observation = step.observation.strip().splitlines()
        print(
            f"  → {step.action}: {observation[0] if observation else ''}",
            file=sys.stderr, flush=True,
        )


def _verbose_step_printer(language: str):
    """
    Return the function printing a step of the LLM assistant on stderr:
    its thought, the tool called with its arguments, its query on its
    lines and highlighted as *language*, and the start of what it
    returned.
    """
    def print_step(step) -> None:
        _print_step_verbose(step, language)
    return print_step


def _print_step_verbose(step, language: str = "datalog") -> None:
    """Print a step of the LLM assistant: see :func:`_verbose_step_printer`."""
    from neurolang_apps.utils.llm.trace import query_lines

    if not step.action:
        return
    lines = []
    if step.thought.strip():
        lines.append(f"  thought: {' '.join(step.thought.split())}")
    arguments = ", ".join(
        f"{name}={value!r}" for name, value in step.arguments.items()
        if name != "query"
    )
    lines.append(f"  → {step.action}({arguments})")
    query = step.arguments.get("query")
    if isinstance(query, str) and query.strip():
        lines += query_lines(
            query, language, query_highlight.use_color(sys.stderr), "      "
        )
    observation = step.observation.strip().splitlines()
    shown = observation[:VERBOSE_OBSERVATION_LINES]
    lines += [f"    {line}" for line in shown]
    if len(observation) > len(shown):
        lines.append(f"    … ({len(observation) - len(shown)} more lines)")
    print("\n".join(lines), file=sys.stderr, flush=True)


def _ask_reporting(args, provider, config, language):
    """
    Return the chat client and the step callback of ``--ask``, as
    ``--verbose`` asks: one line per step, the steps in full (-v), or
    the messages to and from the model (-vv).
    """
    if args.verbose >= 2:
        from neurolang_apps.utils.llm import make_client
        from neurolang_apps.utils.llm.trace import Printer, TracingClient

        client = TracingClient(
            make_client(provider, config.assistant.timeout),
            Printer(full=True, language=language).show,
        )
        return client, None
    if args.verbose == 1:
        return None, _verbose_step_printer(language)
    return None, _print_step


def _positive_int(text: str) -> int:
    """Return the integer *text*, at least 1, for argparse."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, not {text!r}"
        )
    return value


def _run_ask(args) -> None:
    """
    Write the query answering ``--ask`` with an LLM and print it.

    With ``--run`` the query is also run and its result printed. Exits
    with status 1 when no valid query is found.
    """
    try:
        from neurolang_apps.utils.llm import (
            LLMConfigError,
            LLMError,
            load_config,
            make_agent,
        )
        from neurolang_apps.utils.llm.feedback import feedback_client
    except ImportError as exc:
        _exit_missing(
            exc, ("httpx",),
            "--ask needs the optional dependency httpx", "llm",
        )
    try:
        config = load_config(args.llm_config)
        provider = config.provider(args.llm_provider)
    except LLMConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    language = _ask_language(args, config)
    nl = engine_registry.build_engine(
        args.engine, Path(args.data_dir), args.resolution,
        use_cache=not args.no_engine_cache,
    )
    client, on_step = _ask_reporting(args, provider, config, language)
    assistant = config.assistant
    if args.steps:
        assistant = dataclasses.replace(assistant, max_steps=args.steps)
    agent = make_agent(
        nl, provider, assistant, language, args.strategy,
        engine_name=args.engine, client=client, on_step=on_step,
    )
    try:
        result = agent.run(args.ask)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    feedback = feedback_client(
        config, lambda text: print(text, file=sys.stderr)
    )
    answer = feedback and feedback.share(
        args.ask, result, provider, agent.strategy, args.engine
    )
    if answer:
        print(f"Shared as {answer}: rate it at {feedback.url}",
              file=sys.stderr)
    if not result.valid:
        print(f"Error: no valid query: {result.error}", file=sys.stderr)
        if result.query:
            query = query_highlight.highlight(
                result.query, language, sys.stderr
            )
            print(f"Last attempt:\n{query}", file=sys.stderr)
        sys.exit(1)
    commented = result.commented and language == "datalog"
    print(query_highlight.highlight(
        result.commented if commented else result.query, language,
        sys.stdout,
    ))
    if result.commented and not commented:
        # SQUALL has no comments: its rules explained go to stderr.
        print(query_highlight.highlight(
            result.commented, language, sys.stderr
        ), file=sys.stderr)
    if result.explanation:
        print(f"-- {result.explanation}", file=sys.stderr)
    if result.back_translation:
        print(f"-- reads back as: {result.back_translation}", file=sys.stderr)
    if result.verified is False:
        print(
            "Warning: the check found that the query may not answer the "
            f"question: {result.verification}", file=sys.stderr,
        )
    if args.run or args.save or args.view:
        args.squall = language == "squall"
        execute = _execute_squall_program if args.squall else _execute_program
        output = _run_reporting_errors(
            lambda: execute(nl, result.query), result.query, args, nl
        )
        sort_by = _parse_sort_spec(args.sort)
        column_names = _result_column_names(args, result.query)
        _print_result(output, args, sort_by, column_names)
        _save_and_view(output, args, column_names, nl, result.query)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: parse arguments, build engine, run query."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.interactive:
        _run_interactive(args, sys.argv[1:] if argv is None else argv)
        return
    if args.ask:
        _check_engine_name(args)
        _run_ask(args)
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
