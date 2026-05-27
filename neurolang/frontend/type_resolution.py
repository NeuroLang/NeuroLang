"""Type resolution mixin for RegionFrontendDatalogSolver.

Propagates type information across Datalog rule predicates and variables
using Unknown as a marker for unprocessed types.  Extracts types from
extensional database (EDB) predicates (which carry concrete
AbstractSet[Tuple[...]] types) and from builtin functions (which carry
Callable types) and propagates them to variable symbols.

Matches individual FunctionApplication nodes (predicates) during the
recursive tree walk, leaving Implication processing to downstream MRO
handlers such as DatalogProgramMixin.statement_intensional.
"""

from typing import AbstractSet

from typing_inspect import is_callable_type

from ..datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ..expression_walker import ExpressionWalker, add_match
from ..expressions import Constant, FunctionApplication, Symbol
from ..type_system import (
    NeuroLangTypeException,
    Unknown,
    get_args,
    infer_type,
    is_leq_informative,
    unify_types,
)


class TypeResolutionMixin(ExpressionWalker):
    """Walker mixin that infers and propagates types in Datalog rules.

    When walking an Expression tree, each FunctionApplication whose
    functor is a Symbol is checked against the symbol table.  If the
    functor maps to an EDB relation (Constant[AbstractSet[Tuple[...]]]),
    the element types are extracted per-column and applied to the
    corresponding argument symbols.  For builtin functions with Callable
    types, parameter types are propagated similarly.
    """

    @add_match(
        FunctionApplication(Symbol, ...),
    )
    def resolve_predicate_types(self, expression):
        functor = expression.functor
        entry = self.symbol_table.get(functor)
        if entry is None or not isinstance(entry, Constant):
            return expression

        col_types = self._column_types_from_entry(entry)
        if col_types is None:
            return expression

        for i, arg in enumerate(expression.args):
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

        return expression

    def _column_types_from_entry(self, entry):
        """Extract per-column types from a symbol-table entry.

        For EDB predicates (Constant[AbstractSet[Tuple[T1, T2, ...]]])
        returns (T1, T2, ...).  For builtin functions
        (Constant[Callable[[P1, P2, ...], R]]) returns (P1, P2, ...).
        Returns None when no column-level type info is available.
        """
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
