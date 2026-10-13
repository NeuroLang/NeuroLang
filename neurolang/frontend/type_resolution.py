"""
Type resolution mixin for RegionFrontendDatalogSolver.

Propagates type information across Datalog rule predicates and variables
using Unknown as a marker for unprocessed types.  Extracts types from
extensional database (EDB) predicates which carry concrete
AbstractSet[Tuple[...]] types, from builtin functions which carry
Callable types, and from intensional database (IDB) predicates whose
resolved rules already carry type information on their head argument
symbols.  Propagates these types to body variable symbols.

Matches Implication nodes whose head FunctionApplication still has an
Unknown type, resolves body-predicate types, marks the head as processed
by setting its function type, and re-enters the walk so downstream
handlers (e.g. DatalogProgramMixin.statement_intensional) can register
the rule.
"""

import warnings

from typing_inspect import is_callable_type

from ..datalog.expression_processing import extract_logic_atoms
from ..datalog.expressions import AdornedSymbol, Implication, predicate_identity
from ..datalog.negation import is_conjunctive_negation
from ..datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ..expression_walker import PatternWalker, add_match
from ..expressions import (
    Constant,
    FunctionApplication,
    Symbol,
)
from ..logic import Union
from ..type_system import (
    Any,
    NeuroLangTypeException,
    Unknown,
    get_args,
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
        Implication(FunctionApplication[Unknown](Symbol, ...), ...),
        lambda e: is_conjunctive_negation(e.antecedent),
    )
    def resolve_types(self, expression):
        head = expression.consequent
        body = expression.antecedent

        # During magic-sets rewriting the adorned-datalog walker
        # creates rules whose head functor is an AdornedSymbol.
        # The original rules were already resolved — skip here.
        if isinstance(head.functor, AdornedSymbol):
            head.change_type(bool)
            return self.walk(expression)

        self._resolve_body_types(body)

        # Propagate resolved body Symbol types to head args by name.
        # The frontend creates distinct Symbol objects for head vs body
        # — without this, _column_types_from_idb_entry would see only
        # Unknown on the head args and IDB propagation would fail.
        self._propagate_types_to_head(head, body)

        # Change the head FA class to FunctionApplication[bool] so the
        # FunctionApplication[Unknown] pattern no longer matches on
        # re-entry via self.walk().
        head.change_type(bool)

        return self.walk(expression)

    def _resolve_body_types(self, body):
        predicates = list(extract_logic_atoms(body))

        # During magic-sets rewriting the adorned-datalog walker
        # processes adorned rules — skip entirely (the original
        # rules were already resolved by the main solver).  Detect
        # this by checking whether any body functor is an
        # AdornedSymbol (only the adorned walker produces these).
        if any(
            isinstance(p, FunctionApplication)
            and isinstance(p.functor, AdornedSymbol)
            for p in predicates
        ):
            return

        for pred in predicates:
            if (
                not isinstance(pred, FunctionApplication)
                or not isinstance(pred.functor, Symbol)
            ):
                continue

            entry = self.symbol_table.get(pred.functor)
            if entry is None:
                warnings.warn(
                    f"Symbol {predicate_identity(pred.functor)} used in rule body "
                    f"has no entry in the symbol table. Types cannot "
                    f"be inferred for this predicate.",
                    UserWarning,
                )
                continue
            if isinstance(entry, Constant):
                col_types = self._column_types_from_entry(entry)
            elif isinstance(entry, Union):
                col_types = self._column_types_from_idb_entry(entry)
            else:
                continue
            if col_types is None:
                continue
            for i, arg in enumerate(pred.args):
                if not isinstance(arg, Symbol) or i >= len(col_types):
                    continue
                if col_types[i] is Any:
                    continue
                self._set_or_unify_type(arg, col_types[i])

    def _set_or_unify_type(self, symbol, inferred):
        """Set a Symbol's type to inferred, unifying with its existing
        type if needed.  Delegates to unify_types which already
        handles Unknown correctly.
        """
        if inferred is Unknown:
            return
        try:
            unified = unify_types(symbol.type, inferred)
        except NeuroLangTypeException:
            return
        if unified is not symbol.type:
            symbol.type = unified

    def _propagate_types_to_head(self, head, body):
        """Copy type information from body Symbols to matching head
        Symbols by name, so the stored Implication's head args carry
        resolved types for downstream IDB propagation.

        The frontend creates distinct Symbol objects for head vs body
        even when they share the same variable name.  Without this
        step, _column_types_from_idb_entry would read Unknown from
        the head args and IDB type chains would be broken.
        """
        head_args = {
            arg.name: arg
            for arg in head.args
            if isinstance(arg, Symbol)
        }
        if not head_args:
            return

        for pred in extract_logic_atoms(body):
            if not isinstance(pred, FunctionApplication):
                continue
            for arg in pred.args:
                if not isinstance(arg, Symbol):
                    continue
                head_arg = head_args.get(arg.name)
                if head_arg is None:
                    continue
                self._set_or_unify_type(head_arg, arg.type)

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

    def _column_types_from_idb_entry(self, entry):
        """Extract column types from an IDB predicate's resolved rules.

        After type resolution, the head argument symbols of stored
        implications have their .type attribute set.  This method
        collects those types by position, unifying across multiple
        rules that define the same predicate.

        Returns a tuple of types, or None if no type information is
        available (no formulas or all head args are Unknown).
        """
        if not isinstance(entry, Union):
            return None
        col_types = None
        for formula in entry.formulas:
            if not isinstance(formula, Implication):
                continue
            head = formula.consequent
            if not isinstance(head, FunctionApplication):
                continue
            arg_types = tuple(
                arg.type if isinstance(arg, Symbol) else Unknown
                for arg in head.args
            )
            if col_types is None:
                col_types = arg_types
            else:
                unified = []
                for t1, t2 in zip(col_types, arg_types):
                    try:
                        unified.append(unify_types(t1, t2))
                    except NeuroLangTypeException:
                        unified.append(t1)
                col_types = tuple(unified)
        return col_types
