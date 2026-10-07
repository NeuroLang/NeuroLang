"""Pretty-printer for named Relational Algebra expressions."""

import collections.abc
import re

from ..expressions import Constant, FunctionApplication, Symbol
from ..utils import NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet
from .relational_algebra import (
    ColumnInt,
    ColumnStr,
    ConcatenateConstantColumn,
    Destroy,
    Difference,
    EquiJoin,
    ExtendedProjection,
    FullOuterNaturalJoin,
    FunctionApplicationListMember,
    GroupByAggregation,
    Intersection,
    LeftNaturalJoin,
    NameColumns,
    NaturalJoin,
    NaryNaturalJoin,
    NumberColumns,
    Product,
    Projection,
    RenameColumn,
    RenameColumns,
    ReplaceNull,
    Selection,
    Union,
    RelationalAlgebraOperation,
)
from .relational_algebra import _get_columns_for_RA_or_constant


def _column_name(col):
    """Extract the column name as a string from any column representation.

    Handles Symbol (Projection attributes), Constant[ColumnStr] (table
    columns), Constant[ColumnInt] (numbered columns), and plain strings
    so that cross-product detection works across mixed node types.
    """
    if isinstance(col, Symbol):
        return col.name
    if isinstance(col, Constant) and isinstance(col.value, (str, int)):
        return str(col.value)
    return str(col)


def _columns_as_str_set(expr):
    """Return column names as a set of strings for cross-product detection.

    Normalises Symbol, ColumnStr, ColumnInt and any other column
    representation to plain strings so that shared-column comparison
    works across different node types (e.g. Projection with Symbol
    attributes vs a Constant table with ColumnStr columns).
    """
    if isinstance(expr, RelationalAlgebraOperation):
        cols = expr.columns()
    else:
        cols = _get_columns_for_RA_or_constant(expr)
    return {_column_name(c) for c in cols}


def _is_cross_product(expr):
    """Check if a NaturalJoin has no shared columns — i.e. it is a cross product."""
    if not isinstance(expr, NaturalJoin):
        return False
    try:
        lc = _columns_as_str_set(expr.relation_left)
        rc = _columns_as_str_set(expr.relation_right)
        return len(lc & rc) == 0
    except Exception:
        return False


class PrettyPrinter:
    """Recursive pretty-printer for named relational algebra expressions.

    Parameters
    ----------
    name_map : dict, optional
        Mapping from `Constant` functor objects to display name strings.
        When a leaf `Constant` wrapping a relation is found, its value is
        looked up in this map by identity; if found the display name is used.
    indent : int, optional
        Number of spaces to indent per nesting level (default 2).
    width : int, optional
        Target line width, reserved for future wrapping support (default 80).
    """

    def __init__(self, name_map=None, indent=2, width=80, fresh_map=None, counter=None):
        if name_map is None:
            name_map = {}
        self.name_map = name_map
        self.indent = indent
        self.width = width
        self._fresh_map = fresh_map if fresh_map is not None else {}
        self._counter = counter if counter is not None else [0]

    def format(self, expression, level=0):
        """Return a multi-line pretty representation of an RA expression."""
        prefix = " " * (self.indent * level)
        child_prefix = " " * (self.indent * (level + 1))

        if isinstance(expression, Selection):
            return (
                f"{prefix}σ[{self._format_formula(expression.formula)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Projection):
            return (
                f"{prefix}π[{self._format_columns(expression.attributes)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, ExtendedProjection):
            proj = self._format_projection_list(expression.projection_list)
            return (
                f"{prefix}π_ext[{proj}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, NaturalJoin):
            op = "×" if _is_cross_product(expression) else "⋈"
            return (
                f"{prefix}(\n"
                f"{self.format(expression.relation_left, level + 1)}\n"
                f"{child_prefix}{op}\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, NaryNaturalJoin):
            parts = [
                f"{self.format(r, level + 1)}"
                for r in expression.relations
            ]
            op = " ⋈ "
            indent = "\n" + child_prefix
            return f"{prefix}({indent}{op.join(parts)}\n{prefix})"

        if isinstance(expression, EquiJoin):
            return (
                f"{prefix}⋈["
                f"{self._format_columns(expression.columns_left)}="
                f"{self._format_columns(expression.columns_right)}](\n"
                f"{self.format(expression.relation_left, level + 1)},\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, LeftNaturalJoin):
            return (
                f"{prefix}(\n"
                f"{self.format(expression.relation_left, level + 1)}\n"
                f"{child_prefix}⟕\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, FullOuterNaturalJoin):
            return (
                f"{prefix}(\n"
                f"{self.format(expression.relation_left, level + 1)}\n"
                f"{child_prefix}⟗\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Union):
            return (
                f"{prefix}(\n"
                f"{self.format(expression.relation_left, level + 1)}\n"
                f"{child_prefix}∪\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Intersection):
            return (
                f"{prefix}(\n"
                f"{self.format(expression.relation_left, level + 1)}\n"
                f"{child_prefix}∩\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Difference):
            return (
                f"{prefix}(\n"
                f"{self.format(expression.relation_left, level + 1)}\n"
                f"{child_prefix}−\n"
                f"{self.format(expression.relation_right, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Product):
            lines = [f"{prefix}×("]
            for i, relation in enumerate(expression.relations):
                lines.append(self.format(relation, level + 1))
                if i < len(expression.relations) - 1:
                    lines[-1] += ","
            lines.append(f"{prefix})")
            return "\n".join(lines)

        if isinstance(expression, NameColumns):
            names = self._format_columns(expression.column_names)
            return (
                f"{prefix}ρ[{names}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, NumberColumns):
            names = self._format_columns(expression.column_names)
            return (
                f"{prefix}ν[{names}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, RenameColumn):
            return (
                f"{prefix}ρ[{self._format_column(expression.src)}→"
                f"{self._format_column(expression.dst)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, RenameColumns):
            return (
                f"{prefix}ρ[{self._format_renames(expression.renames)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, GroupByAggregation):
            groupby = self._format_columns(expression.groupby)
            aggs = self._format_aggregate_functions(
                expression.aggregate_functions
            )
            return (
                f"{prefix}γ[groupby=({groupby}), "
                f"agg=({aggs})](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, ReplaceNull):
            return (
                f"{prefix}κ[{self._format_column(expression.column)}, "
                f"NULL→{self._format_value(expression.value)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, ConcatenateConstantColumn):
            return (
                f"{prefix}⊕[{self._format_column(expression.column_name)}="
                f"{self._format_value(expression.column_value)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Destroy):
            return (
                f"{prefix}δ[{self._format_column(expression.src_column)}→"
                f"{self._format_column(expression.dst_column)}](\n"
                f"{self.format(expression.relation, level + 1)}\n"
                f"{prefix})"
            )

        if isinstance(expression, Constant):
            return f"{prefix}{self._format_constant(expression)}"

        if isinstance(expression, Symbol):
            name = self._lookup_name(expression)
            if name is not None:
                return f"{prefix}{name}"
            return f"{prefix}{expression.name}"

        return f"{prefix}{repr(expression)}"

    def _shorten_fresh(self, name):
        """Shorten fresh variable names like fresh_00000007 to s₀, s₁, etc."""
        m = re.match(r'^fresh_(\d+)$', name)
        if not m:
            return name
        if name not in self._fresh_map:
            n = self._counter[0]
            if n < 10:
                self._fresh_map[name] = f"s{chr(0x2080 + n)}"
            else:
                self._fresh_map[name] = f"s{n}"
            self._counter[0] += 1
        return self._fresh_map[name]

    def _lookup_name(self, expression):
        """Look up a display name for a Constant or Symbol leaf.

        The map may use the object itself (identity) or its ``id()`` as key,
        which is useful when the wrapped value is unhashable.
        """
        for key, name in self.name_map.items():
            if expression is key:
                return name
            if isinstance(expression, Constant) and expression.value is key:
                return name
            if isinstance(key, int) and id(expression) == key:
                return name
            if isinstance(key, int) and isinstance(expression, Constant) and id(expression.value) == key:
                return name
        # Fallback: match NamedRelationalAlgebraFrozenSet by columns.
        # The CP-Logic program builder creates new instances, so identity
        # checks above may fail even though the columns are the same.
        if isinstance(expression, Constant) and isinstance(
            expression.value, NamedRelationalAlgebraFrozenSet
        ):
            expr_cols = expression.value.columns
            for key, name in self.name_map.items():
                if isinstance(key, Constant) and isinstance(
                    key.value, NamedRelationalAlgebraFrozenSet
                ):
                    if key.value.columns == expr_cols:
                        return name
        # Fallback: match Symbol by name.
        if isinstance(expression, Symbol):
            for key, name in self.name_map.items():
                if isinstance(key, Symbol) and key.name == expression.name:
                    return name
        return None

    def _format_constant(self, expression):
        """Format a Constant leaf node as a short table or value summary."""
        name = self._lookup_name(expression)
        if name is not None:
            return name

        value = expression.value

        if isinstance(value, NamedRelationalAlgebraFrozenSet):
            return self._format_named_set(value)

        if isinstance(value, RelationalAlgebraFrozenSet):
            return self._format_unnamed_set(value)

        if callable(value) and hasattr(value, "__qualname__"):
            return value.__qualname__

        return repr(value)

    def _format_named_set(self, value):
        try:
            columns = [self._format_column_name(c) for c in value.columns]
        except Exception:
            columns = []
        try:
            rows = len(value)
        except Exception:
            rows = "?"
        if rows == 0:
            rows = "unknown"
        return f"Table(columns={columns}, rows={rows})"

    def _format_unnamed_set(self, value):
        try:
            rows = len(value)
        except Exception:
            rows = "?"
        return f"Table(rows={rows})"

    def _format_column_name(self, col):
        if isinstance(col, str):
            return col
        if isinstance(col, ColumnStr):
            return str(col)
        if isinstance(col, ColumnInt):
            return f"#{col}"
        return repr(col)

    def _format_formula(self, formula):
        """Format a FunctionApplication or leaf expression for operators."""
        if isinstance(formula, FunctionApplication):
            op = self._format_functor(formula.functor)
            if op in OPERATOR_SYMBOLS.values() and len(formula.args) == 2:
                left = self._format_formula(formula.args[0])
                right = self._format_formula(formula.args[1])
                return f"{left} {op} {right}"
            if op == "not" and len(formula.args) == 1:
                return f"not {self._format_formula(formula.args[0])}"
            args = ", ".join(
                self._format_formula(arg) for arg in formula.args
            )
            return f"{op}({args})"
        if isinstance(formula, Constant):
            return self._format_value(formula)
        if isinstance(formula, Symbol):
            return self._shorten_fresh(formula.name)
        return repr(formula)

    def _format_functor(self, functor):
        if isinstance(functor, Constant):
            value = functor.value
            if value in OPERATOR_SYMBOLS:
                return OPERATOR_SYMBOLS[value]
            if callable(value) and hasattr(value, "__qualname__"):
                return value.__qualname__
            return repr(value)
        if isinstance(functor, Symbol):
            return functor.name
        return repr(functor)

    def _format_value(self, value):
        if isinstance(value, Constant):
            value = value.value
        if isinstance(value, ColumnStr):
            return self._shorten_fresh(value)
        if isinstance(value, ColumnInt):
            return f"#{value}"
        if isinstance(value, Symbol):
            return self._shorten_fresh(value.name)
        return repr(value)

    def _format_column(self, col):
        if isinstance(col, Constant):
            col = col.value
        if isinstance(col, ColumnStr):
            return self._shorten_fresh(col)
        if isinstance(col, ColumnInt):
            return f"#{col}"
        if isinstance(col, Symbol):
            return self._shorten_fresh(col.name)
        return repr(col)

    def _format_columns(self, cols):
        """Format a tuple/list of column references."""
        if cols is None:
            return ""
        return ", ".join(self._format_column(c) for c in cols)

    def _format_projection_list(self, proj_list):
        return ", ".join(self._format_projection_member(p) for p in proj_list)

    def _format_projection_member(self, member):
        if isinstance(member, FunctionApplicationListMember):
            fun = self._format_formula(member.fun_exp)
            dst = self._format_column(member.dst_column)
            return f"{fun} -> {dst}"
        return repr(member)

    def _format_aggregate_functions(self, agg_funcs):
        return ", ".join(
            self._format_projection_member(a) for a in agg_funcs
        )

    def _format_renames(self, renames):
        return ", ".join(
            f"{self._format_column(src)}→{self._format_column(dst)}"
            for src, dst in renames
        )


OPERATOR_SYMBOLS = {
    "__builtins__": None,
}


def _populate_operator_symbols():
    import operator as op
    mapping = {
        op.add: "+",
        op.sub: "-",
        op.mul: "*",
        op.truediv: "/",
        op.eq: "==",
        op.ne: "!=",
        op.gt: ">",
        op.lt: "<",
        op.ge: ">=",
        op.le: "<=",
        op.pow: "**",
        op.and_: "and",
        op.or_: "or",
        op.not_: "not",
    }
    OPERATOR_SYMBOLS.update(mapping)


_populate_operator_symbols()


def build_name_map_from_conjunction(conjunction, symbol_table=None):
    """Build a name map from a Datalog Conjunction.

    Walks the conjunction recursively to find all `FunctionApplication`
    predicates and maps their functor (a `Constant` or `Symbol`) to a display
    name string. If a `symbol_table` is provided, EDB predicate names are
    also included in the resulting map.

    Parameters
    ----------
    conjunction : Conjunction
        A Datalog conjunction whose predicates should be named.
    symbol_table : TypedSymbolTable, optional
        Optional symbol table from which to extract EDB predicate names.

    Returns
    -------
    dict
        Mapping from functor objects to display name strings.
    """
    name_map = {}

    def collect(expression):
        if isinstance(expression, FunctionApplication):
            functor = expression.functor
            if isinstance(functor, Constant):
                name_map[functor.value] = str(functor.value)
            elif isinstance(functor, Symbol):
                name_map[functor] = functor.name
            for arg in expression.args:
                collect(arg)
        elif isinstance(expression, (tuple, list)):
            for item in expression:
                collect(item)

    if hasattr(conjunction, "formulas"):
        for formula in conjunction.formulas:
            collect(formula)
    elif hasattr(conjunction, "args"):
        for formula in conjunction.args:
            collect(formula)

    if isinstance(symbol_table, collections.abc.Mapping):
        for key in symbol_table:
            try:
                name = key.name
            except AttributeError:
                continue
            value = symbol_table[key]
            if isinstance(value, Constant) and isinstance(
                value.value,
                (NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet),
            ):
                name_map[value] = name

    return name_map


def pretty_repr(expression, name_map=None, indent=2, width=80, fresh_map=None, counter=None):
    """Return a pretty representation of an RA expression.

    Parameters
    ----------
    expression : Expression
        Relational algebra expression to format.
    name_map : dict, optional
        Mapping from Constant values/objects to display name strings.
    indent : int, optional
        Indentation width per level (default 2).
    width : int, optional
        Target line width (default 80).
    fresh_map : dict, optional
        Mapping from fresh variable names to short display names.
    counter : list, optional
        Mutable counter (list with one int) for generating short names.

    Returns
    -------
    str
        Pretty formatted expression.
    """
    printer = PrettyPrinter(
        name_map=name_map, indent=indent, width=width,
        fresh_map=fresh_map, counter=counter,
    )
    return printer.format(expression)
