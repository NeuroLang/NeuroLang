"""
Type resolution mixin for RegionFrontendDatalogSolver.

Propagates type information across Datalog rule predicates and variables
using Unknown as a marker for unprocessed types.  Extracts types from
extensional database (EDB) predicates which carry concrete
AbstractSet[Tuple[...]] types and from builtin functions which carry
Callable types and propagates them to variable symbols.

Matches FunctionApplication nodes during the walker's recursive traversal,
leaving Implication handling to downstream MRO handlers such as
DatalogProgramMixin.statement_intensional.  A catch-all rule walks children
for all non-Implication expressions so the mixin works with PatternWalker's
non-recursive walk.
"""

from typing_inspect import is_callable_type

from ..datalog.wrapped_collections import WrappedRelationalAlgebraSet
from ..expression_walker import PatternWalker, add_match
from ..datalog.expressions import Implication
from ..expressions import (
    Constant,
    Expression,
    FunctionApplication,
    Symbol,
)
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

    Each FunctionApplication whose functor is a Symbol is checked against
    the symbol table.  If the functor maps to an EDB relation, column
    types are propagated to argument symbols.  For builtin functions with
    Callable types, parameter types are propagated similarly.

    A catch-all rule recursively walks children of all expressions except
    Implication, which is left to downstream handlers such as
    statement_intensional.
    """

    @add_match(FunctionApplication(Symbol, ...))
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

    @add_match(Expression, lambda e: not isinstance(e, Implication))
    def walk_children(self, expression):
        """
        Recursively walk children for non-Implication expressions.

        Downstream MRO handlers such as statement_intensional handle
        Implication separately, so this catch-all skips it.
        """
        args = expression.unapply()
        new_args = tuple()
        changed = False
        for arg in args:
            if isinstance(arg, Expression):
                new_arg = self.walk(arg)
                changed |= new_arg is not arg
            elif isinstance(arg, tuple) and len(arg) > 0 and isinstance(arg[0], Expression):
                new_arg, arg_changed = self._walk_tuple(arg)
                changed |= arg_changed
            else:
                new_arg = arg
            new_args += (new_arg,)

        if changed:
            return self.walk(expression.apply(*new_args))
        return expression

    def _walk_tuple(self, arg):
        changed = False
        result = list()
        for sub in arg:
            walked = self.walk(sub)
            result.append(walked)
            changed |= walked is not sub
        return type(arg)(result), changed

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
