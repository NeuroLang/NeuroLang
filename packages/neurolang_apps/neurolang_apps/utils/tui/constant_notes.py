"""
Notes on the constants of a query that match no value of their sort.

A string constant in an argument of a declared sort, ``'Pain'`` or
``'pian'`` in ``mentions(s, 'Pain')``, that no relation of the engine has
for that sort matches nothing: the query gives no rows, which reads as
"no studies". A variable used once whose name is a value of its sort,
``pain`` in ``mentions(s, pain)``, is likely a constant whose quotes were
forgotten: it matches every term. :func:`constant_notes` tells of both,
with the closest values. The sorts are those the engine declares (see
:func:`~neurolang_apps.utils.tui.provenance.declared_column_sorts`) and
their values those of the relations of the engine declaring them.
"""

import difflib
from collections import Counter

from nl_expr.expression_walker import expression_iterator
from nl_expr.expressions import Constant, Symbol
from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.cli import _body_atoms
from .provenance import (
    _add_rule_sorts,
    _datalog_queries,
    _squall_queries,
    declared_column_sorts,
)
from .submission import relation_rows

# The values of a column of a relation, by (id of the relation, column),
# with the relation, so that the id is not reused while cached.
_COLUMN_VALUES = {}


def _column_values(rows, position: int) -> frozenset:
    """Return the strings of the column *position* of the relation *rows*."""
    cached = _COLUMN_VALUES.get((id(rows), position))
    if cached is None or cached[0] is not rows:
        values = rows.as_pandas_dataframe().iloc[:, position].unique()
        cached = rows, frozenset(v for v in values if isinstance(v, str))
        _COLUMN_VALUES[id(rows), position] = cached
    return cached[1]


def _sort_values(nl, declared: dict, sort: str) -> frozenset:
    """Return the values of *sort* in the relations of *nl* declared."""
    symbol_table = nl.program_ir.symbol_table
    values = frozenset()
    for name, sorts in declared.items():
        rows = relation_rows(symbol_table.get(Symbol(name)))
        if rows is None or rows.arity != len(sorts):
            continue
        for position, column_sort in enumerate(sorts):
            if column_sort.sort == sort:
                values |= _column_values(rows, position)
    return values


def _closest(value: str, values) -> list:
    """Return the *values* close to *value*, those equal but case first."""
    same = [v for v in values if v.lower() == value.lower()]
    return same or difflib.get_close_matches(value, values, n=3)


def _argument_note(atom, argument, sort, values, uses) -> str:
    """Return the note on *argument* of *atom*, of *sort*, or ""."""
    name = atom.functor.name
    if isinstance(argument, Constant) and isinstance(argument.value, str):
        if argument.value in values:
            return ""
        note = f"{argument.value!r} is not a {sort} of the engine: " \
            f"{name}(...) matches no row with it"
        close = _closest(argument.value, values)
        if close:
            return f"{note}; did you mean {' or '.join(map(repr, close))}?"
        return f"{note}."
    if (
        isinstance(argument, Symbol) and uses[argument] == 1
        and argument.name in values
    ):
        return (
            f"{argument.name} in {name}(...) is a variable, matching every "
            f"{sort}: write {argument.name!r}, quoted, for the {sort}."
        )
    return ""


def constant_notes(nl, engine_name: str, program: str, squall: bool) -> list:
    """
    Return the notes on the constants of *program* matching nothing.

    The sorts follow from those the engine *engine_name* declares, then
    from the rules of *nl* and those of *program*. A program that does
    not parse has none.
    """
    try:
        return _constant_notes(nl, engine_name, program, squall)
    except Exception:
        return []


def _constant_notes(nl, engine_name, program, squall) -> list:
    """Return the notes of :func:`constant_notes`, or raise."""
    declared = declared_column_sorts(engine_name)
    if not declared:
        return []
    strings = {
        sort.name for sort in engine_registry.get_sorts(engine_name).values()
        if sort.type == "str"
    }
    program_rules, queries = (
        _squall_queries(program) if squall else _datalog_queries(nl, program)
    )
    known = dict(declared)
    rules = [
        rule for union in nl.program_ir.intensional_database().values()
        for rule in getattr(union, "formulas", ())
    ]
    for rule in (*rules, *program_rules):
        _add_rule_sorts(rule, known)
    values = {}
    notes = []
    for head, body in [
        *((r.consequent, r.antecedent) for r in program_rules),
        *queries.values(),
    ]:
        uses = Counter(
            e for _, e in expression_iterator((head, body))
            if isinstance(e, Symbol)
        )
        for atom in _body_atoms(body):
            sorts = known.get(atom.functor.name)
            if not sorts or len(sorts) != len(atom.args):
                continue
            for argument, column_sort in zip(atom.args, sorts):
                sort = column_sort and column_sort.sort
                if sort not in strings:
                    continue
                if sort not in values:
                    values[sort] = _sort_values(nl, declared, sort)
                if values[sort]:
                    notes.append(_argument_note(
                        atom, argument, sort, values[sort], uses
                    ))
    return list(dict.fromkeys(n for n in notes if n))
