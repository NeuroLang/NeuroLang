"""
Type resolution mixin for RegionFrontendDatalogSolver.

Propagates type information across Datalog rule predicates and variables
using Unknown as a marker for unprocessed types.  Extracts types from
extensional database (EDB) predicates which carry concrete
AbstractSet[Tuple[...]] types and from builtin functions which carry
Callable types and propagates them to variable symbols.

Matches Implication nodes with a FunctionApplication or Conjunction
body and resolves variable types before the expression continues through
the MRO to handlers such as DatalogProgramMixin.statement_intensional.

Uses a guard to prevent re-processing already-resolved rules.
"""

from typing import AbstractSet

from typing_inspect import is_callable_type

from ..datalog.expression_processing import extract_logic_atoms
from ..datalog.expressions import Implication
from ..datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ..expression_walker import PatternWalker, add_match
from ..expressions import Constant, FunctionApplication, Symbol
from ..type_system import (
    NeuroLangTypeException,
    Unknown,
    get_args,
    is_leq_informative,
    unify_types,
)


def _implication_has_untyped_body_predicates(expression):
    """
    Guard that allows matching when the expression still needs type
    resolution.

    Returns True when the expression has not yet been processed and any
    body-predicate argument symbol has Unknown type. The handler sets
    _nl_type_resolved on the expression after the first pass, so
    subsequent match attempts through self.walk() skip this pattern and
    fall through to downstream handlers.
    """
    if getattr(expression, '_nl_type_resolved', False):
        return False
    body = expression.antecedent
    predicates = list(extract_logic_atoms(body))
    for pred in predicates:
        if not isinstance(pred, FunctionApplication):
            continue
        if any(isinstance(arg, Symbol) and arg.type is Unknown
               for arg in pred.args):
            return True
    return False


class TypeResolutionMixin(PatternWalker):
    """
    PatternWalker mixin that infers and propagates types in Datalog rules.

    Intercepts Implication nodes to resolve variable types from body-predicate
    functor lookups in the symbol table.  Returns self.walk(expression) to
    re-enter the processing cycle so downstream handlers
    (e.g. statement_intensional) receive the typed rule.
    """

    @add_match(
        Implication,
        _implication_has_untyped_body_predicates,
    )
    def resolve_types(self, expression):
        self._resolve_types_from_body(expression)
        expression._nl_type_resolved = True
        return self.walk(expression)

    def _resolve_types_from_body(self, expression):
        predicates = list(extract_logic_atoms(expression.antecedent))
        for pred in predicates:
            if not isinstance(pred, FunctionApplication):
                continue
            functor = pred.functor
            if not isinstance(functor, Symbol):
                continue
            entry = self.symbol_table.get(functor)
            if entry is None or not isinstance(entry, Constant):
                continue
            col_types = self._column_types_from_entry(entry)
            if col_types is None:
                continue
            self._apply_column_types(pred, col_types)

    def _apply_column_types(self, predicate, col_types):
        for i, arg in enumerate(predicate.args):
            if not isinstance(arg, Symbol) or i >= len(col_types):
                continue
            inferred = col_types[i]
            if inferred is Unknown:
                continue
            if arg.type is Unknown:
                arg.__dict__['type'] = inferred
            elif not is_leq_informative(arg.type, inferred):
                try:
                    unified = unify_types(arg.type, inferred)
                except NeuroLangTypeException:
                    continue
                if unified is not arg.type:
                    arg.__dict__['type'] = unified

    def _column_types_from_entry(self, entry):
        if isinstance(entry.value, WrappedRelationalAlgebraSet):
            if entry.type is Unknown:
                return None
            if not is_leq_informative(entry.type, AbstractSet):
                return None
            row_type = get_args(entry.type)[0]
            return get_args(row_type)

        if is_callable_type(entry.type):
            args, _return = get_args(entry.type)
            if args is Ellipsis or args is ...:
                return None
            return tuple(args)

        return None
