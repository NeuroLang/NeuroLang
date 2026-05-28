"""
Type resolution mixin for RegionFrontendDatalogSolver.

Propagates type information across Datalog rule predicates and variables
using Unknown as a marker for unprocessed types.  Extracts types from
extensional database (EDB) predicates which carry concrete
AbstractSet[Tuple[...]] types and from builtin functions which carry
Callable types and propagates them to variable symbols.

Matches Implication nodes whose head FunctionApplication still has an
Unknown type, resolves body-predicate types, marks the head as processed
by setting its function type, and re-enters the walk so downstream
handlers (e.g. DatalogProgramMixin.statement_intensional) can register
the rule.
"""

from typing_inspect import is_callable_type

from ..datalog.expression_processing import extract_logic_atoms
from ..datalog.expressions import Implication
from ..datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ..expression_walker import PatternWalker, add_match
from ..expressions import (
    Constant,
    Expression,
    FunctionApplication,
    Symbol,
)
from ..logic import Conjunction
from ..type_system import (
    NeuroLangTypeException,
    Unknown,
    get_args,
    is_leq_informative,
    unify_types,
)


class TypeResolutionMixin(PatternWalker):
    """
    PatternWalker mixin that infers and propagates types in Datalog rules.

    Matches Implication nodes where the head FunctionApplication type is
    still Unknown.  The handler extracts atoms from the rule body, looks
    up their functors in the symbol table, and propagates column/parameter
    types to argument Symbols.  After resolution, the head
    FunctionApplication type is set to bool -- this prevents re-matching
    when self.walk re-enters the processing cycle.
    """

    @add_match(
        Implication(FunctionApplication[Unknown](Symbol, ...), Expression),
        lambda e: isinstance(e.antecedent, (FunctionApplication, Conjunction)),
    )
    def resolve_types(self, expression):
        head = expression.consequent
        body = expression.antecedent

        self._resolve_body_types(body)

        # Change the head FA class to FunctionApplication[bool] so the
        # FunctionApplication[Unknown] pattern no longer matches on
        # re-entry via self.walk().
        head.change_type(bool)

        return self.walk(expression)

    def _resolve_body_types(self, body):
        predicates = list(extract_logic_atoms(body))
        for pred in predicates:
            if (
                not isinstance(pred, FunctionApplication)
                or not isinstance(pred.functor, Symbol)
            ):
                continue
            entry = self.symbol_table.get(pred.functor)
            if entry is None or not isinstance(entry, Constant):
                continue
            col_types = self._column_types_from_entry(entry)
            if col_types is None:
                continue
            for i, arg in enumerate(pred.args):
                if not isinstance(arg, Symbol) or i >= len(col_types):
                    continue
                inferred = col_types[i]
                if inferred is Unknown:
                    continue
                if arg.type is Unknown:
                    arg.__dict__["type"] = inferred
                elif not is_leq_informative(arg.type, inferred):
                    try:
                        unified = unify_types(arg.type, inferred)
                    except NeuroLangTypeException:
                        continue
                    if unified is not arg.type:
                        arg.__dict__["type"] = unified

    def _column_types_from_entry(self, entry):
        if isinstance(entry.value, WrappedRelationalAlgebraSet):
            row_type = entry.value.row_type
            if row_type is None:
                return None
            return get_args(row_type)

        if is_callable_type(entry.type):
            args, _return = get_args(entry.type)
            if args is Ellipsis or args is ...:
                return None
            return tuple(args)

        return None
