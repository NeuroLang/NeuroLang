"""
The sorts of the columns of a query's result, from the query itself.

A column of a result holds the values of a variable of the head of the
query. When the variable is an argument of a predicate declaring its
``arguments`` in the engine's configuration, in the body of the query,
the column holds values of the sort of that argument, whatever its name:
``ans(x, y, z) :- voxel(x, y, z)`` holds voxel indices, not
millimetres. ``.save`` and ``.view`` locate the voxels of a result
from these sorts (see :func:`~.nifti.point_columns`).

The sorts of the predicates follow from the engine's declarations (see
:func:`~neurolang_apps.utils.engine_registry.get_signatures`) and the
``%%`` documentation of the predicates of the program, which wins, then
from the rules defining the others, as
:func:`~neurolang_apps.utils.cli.squall_column_names` follows their names.
"""

import operator
from typing import NamedTuple, Optional

from nl_core.datalog import Implication
from nl_expr.expressions import Constant, FunctionApplication, Query, Symbol
from neurolang.frontend.datalog.standard_syntax import query_name
from neurolang.frontend.datalog.squall_syntax_lark import (
    SquallProgram,
    parser as squall_parser,
)
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.predicate_catalog import program_documentation
from neurolang_apps.utils.cli import (
    _body_atoms, _subformulas, _whole_relation,
)

# The sort of a column that a rule computes, ``b == p / p0``: a measure,
# whatever its name, which may be that of a column of a sort.
COMPUTED = "computed"


class ColumnSort(NamedTuple):
    """
    The sort of the values of a column, and which column of the sort.

    A sort spanning several columns, such as a voxel ``(i, j, k)``,
    gives each its *position* among them: 0 for ``i``.
    """

    sort: str
    position: int


def declared_column_sorts(engine_name: str) -> dict:
    """
    Return the sort of each column of the declared predicates, by name.

    Only the predicates of the engine *engine_name* declaring their
    ``arguments`` have one, a tuple of a :class:`ColumnSort` per column.
    """
    if not engine_name:
        return {}
    return {
        name: tuple(
            ColumnSort(argument.sort, position)
            for argument in arguments
            for position in range(len(argument.columns))
        )
        for name, arguments in engine_registry.get_signatures(
            engine_name
        ).items()
    }


def documented_column_sorts(nl) -> dict:
    """
    Return the sort of each column of the predicates the program of *nl*
    documents with sorts, by name (see :func:`declared_column_sorts`):
    ``%% uncinate(x: streamline, h: hemisphere)``. A column of no sort
    has None.
    """
    return {
        name: tuple(
            ColumnSort(sort, 0) if sort else None
            for _, sort in documentation.arguments
        )
        for name, documentation in program_documentation(
            nl.program_ir
        ).items()
        if any(sort for _, sort in documentation.arguments)
    }


def _term_sort(
    term, atoms, known: dict, computed=()
) -> Optional[ColumnSort]:
    """
    Return the sort of the head *term*, from the first of *atoms* with it.

    Only the atoms of the predicates of *known* sorts count; a term that
    is not a variable, or in none of them, has no known sort, unless it is
    one of the *computed* variables: that of :data:`COMPUTED`.
    """
    if not isinstance(term, Symbol):
        return None
    for atom in atoms:
        sorts = known.get(atom.functor.name)
        if term in atom.args and sorts and len(sorts) == len(atom.args):
            return sorts[atom.args.index(term)]
    return ColumnSort(COMPUTED, 0) if term in computed else None


def _computed_terms(formula) -> set:
    """Return the variables *formula* defines as ``variable == expression``."""
    defined = set()
    functor = getattr(formula, "functor", None)
    if (
        isinstance(formula, FunctionApplication)
        and isinstance(functor, Constant) and functor.value is operator.eq
        and isinstance(formula.args[0], Symbol)
    ):
        defined.add(formula.args[0])
    for sub in _subformulas(formula):
        defined |= _computed_terms(sub)
    return defined


def head_sorts(terms, body, known: dict) -> tuple:
    """
    Return the sorts of the head *terms* of a rule or query of *body*.

    *known* holds the column sorts of the predicates (see
    :func:`declared_column_sorts`); a term of unknown sort has None.
    """
    atoms = list(_body_atoms(body))
    computed = _computed_terms(body)
    return tuple(_term_sort(term, atoms, known, computed) for term in terms)


def _add_rule_sorts(rule, known: dict) -> None:
    """Add the column sorts of the head of *rule* to *known*, if unknown."""
    head = getattr(rule, "consequent", None)
    name = getattr(getattr(head, "functor", None), "name", None)
    if isinstance(name, str) and name not in known:
        sorts = head_sorts(head.args, rule.antecedent, known)
        if any(sorts):
            known[name] = sorts


def _query_sorts(head, body, known: dict) -> tuple:
    """
    Return the sorts of the columns of the result of a query.

    *head* is the head of a Datalog query, ``ans(x, y)``, or of a SQUALL
    ``obtain``, a term or a tuple of terms; ``obtain every R`` gives the
    columns of ``R``.
    """
    named = isinstance(getattr(head, "functor", None), Symbol)
    terms = head.args if named else head
    terms = terms if isinstance(terms, tuple) else (terms,)
    whole = _whole_relation(body, terms, known)
    return tuple(whole) if whole is not None else head_sorts(
        terms, body, known
    )


def _datalog_queries(nl, program: str) -> tuple:
    """
    Return the rules of the Datalog *program* and its queries, by key.

    The key of the query is ``""``, that of the one result of Datalog; a
    program with several queries has a result by the name of each, that
    of ``ans[name](...)``, or ``ans``.
    """
    parsed = nl.datalog_parser(program)
    formulas = getattr(parsed, "formulas", (parsed,))
    rules = [f for f in formulas if isinstance(f, Implication)]
    found = [f for f in formulas if isinstance(f, Query)]
    queries = {
        (query_name(f.head) if len(found) > 1 else ""): (f.head, f.body)
        for f in found
    }
    return rules, queries


def _squall_queries(program: str) -> tuple:
    """
    Return the rules of the SQUALL *program* and its queries, by key.

    The key of a query is that of its result (see
    :func:`~neurolang_apps.utils.cli.squall_column_names`).
    """
    parsed = squall_parser(program)
    if not isinstance(parsed, SquallProgram):
        return [], {}
    queries = {
        parsed.query_names.get(i, f"obtain_{i}"): (query.head, query.body)
        for i, query in enumerate(parsed.queries)
    }
    return list(parsed.rules), queries


def column_sorts(
    nl, engine_name: str, program: str, squall: bool, rules=()
) -> dict:
    """
    Return the sorts of the columns of the results of *program*'s queries.

    The result is a dict from the key of each result, ``""`` for a
    Datalog query and that of an ``obtain`` in SQUALL, to a tuple of the
    :class:`ColumnSort` of each column, None where unknown. The sorts
    follow from those the engine *engine_name* declares, then from the
    *rules* defined before and those of the *program*. A program that
    does not parse has none, nor one whose sorts cannot be told.
    """
    try:
        return _column_sorts(nl, engine_name, program, squall, rules)
    except Exception:
        return {}


def _column_sorts(nl, engine_name, program, squall, rules) -> dict:
    """Return the column sorts of :func:`column_sorts`, or raise."""
    program_rules, queries = (
        _squall_queries(program) if squall else _datalog_queries(nl, program)
    )
    known = {
        **declared_column_sorts(engine_name), **documented_column_sorts(nl)
    }
    for rule in (*rules, *program_rules):
        _add_rule_sorts(rule, known)
    return {
        key: _query_sorts(head, body, known)
        for key, (head, body) in queries.items()
    }
