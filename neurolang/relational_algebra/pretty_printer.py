"""Pretty-printer for named Relational Algebra expressions."""

import collections.abc
import operator
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
    """
    Extract the column name as a string from any column representation.

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
    """
    Return column names as a set of strings for cross-product detection.

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


_FRESH_NAME = re.compile(r"^fresh_\d+$")


def _is_column_int(value):
    """Return whether ``value`` is a constant column index."""
    return isinstance(value, Constant) and value.type is ColumnInt


def _column_index(value):
    """Return the index of a constant column index, None for others."""
    if _is_column_int(value) or (
        isinstance(value, Constant) and type(value.value) in (int, ColumnInt)
        and not isinstance(value.value, bool)
    ):
        return int(value.value)
    return None


def _atom_pattern(expression):
    """
    Match a rename of (a projection and selection of) a base relation.

    Return ``(leaf, names, columns, selections)`` for
    ``ρ[names](π[columns](σ…(leaf)))`` or ``ρ[names](leaf)``, where
    the selections are equalities of a column with a constant or another
    column, or None when ``expression`` has another form.
    """
    if not isinstance(expression, NameColumns):
        return None
    names = tuple(expression.column_names)
    relation = expression.relation
    if isinstance(relation, (Constant, Symbol)):
        return relation, names, tuple(range(len(names))), ()
    if not isinstance(relation, Projection):
        return None
    columns = tuple(_column_index(c) for c in relation.attributes)
    if None in columns or len(columns) != len(names):
        return None
    selections = []
    relation = relation.relation
    while isinstance(relation, Selection):
        selection = _equality_selection(relation.formula)
        if selection is None:
            return None
        selections.append(selection)
        relation = relation.relation
    if not isinstance(relation, (Constant, Symbol)):
        return None
    return relation, names, columns, tuple(selections)


def _equality_selection(formula):
    """
    Match ``#i == value`` where ``value`` is a constant or a column index.

    Return ``(i, value)`` with ``value`` the constant, or the index of
    the other column; None for other formulas.
    """
    if not (
        isinstance(formula, FunctionApplication)
        and isinstance(formula.functor, Constant)
        and formula.functor.value is operator.eq
        and len(formula.args) == 2
    ):
        return None
    left, right = formula.args
    if not _is_column_int(left):
        return None
    if _is_column_int(right):
        return int(left.value), int(right.value)
    if isinstance(right, Constant):
        return int(left.value), right
    return None


def _is_trivial_projection(expression):
    """Return whether a projection keeps all the columns of its operand."""
    try:
        attributes = _columns_as_str_set_of(expression.attributes)
        return attributes == _columns_as_str_set(expression.relation)
    except Exception:
        return False


def _columns_as_str_set_of(columns):
    """Return a set of column references as strings."""
    return {_column_name(c) for c in columns}


def subscript(number):
    """Return ``number`` written with subscript digits."""
    return "".join(chr(0x2080 + int(digit)) for digit in str(number))


def _relation_size(value):
    """Return the number of rows of a relation, or None if unknown."""
    try:
        return len(value)
    except Exception:
        return None


class UpperBound(int):
    """A number of rows known to be an upper bound only."""


def rows_label(rows):
    """
    Return the ``‹N rows›`` annotation of a relation of ``rows`` rows.

    An :class:`UpperBound` is printed ``‹≤N rows›``.
    """
    if rows is None:
        return "‹? rows›"
    bound = "≤" if isinstance(rows, UpperBound) else ""
    return f"‹{bound}{rows:,} row{'' if rows == 1 else 's'}›"


def _is_nary_cross_product(expression):
    """Check if no two operands of an NaryNaturalJoin share a column."""
    try:
        columns = [_columns_as_str_set(r) for r in expression.relations]
    except Exception:
        return False
    seen = set()
    for relation_columns in columns:
        if seen & relation_columns:
            return False
        seen |= relation_columns
    return True


class PrettyPrinter:
    """
    Recursive pretty-printer for named relational algebra expressions.

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
    fresh_map : dict, optional
        Mapping from fresh variable names to their display names, e.g.
        the ``s₀``, ``s₁``, … the rule the plan evaluates is printed with.
        Other fresh column names are displayed as ``c₀``, ``c₁``, ….
    counter : list, optional
        Mutable counter (list with one int) numbering the ``c₀``, ``c₁``,
        … columns.
    leaf_sizes : dict, optional
        Mapping from `Symbol` leaves to the number of rows of the relation
        they stand for. Leaves found in it, and named `Constant` leaves,
        are printed with their size, e.g. ``edge ‹3 rows›``.
    column_roles : dict, optional
        Mapping from column names to the label they are displayed with,
        e.g. ``p`` for a probability column.
    collapse : bool, optional
        If True, trivial wrappers are collapsed: a rename of (a selection
        and projection of) a base relation is printed as a Datalog atom,
        e.g. ``edge(x, y)`` for ``ρ[x, y](π[#0, #1](edge))``, projections
        on all the columns of their operand are skipped and identity
        members of extended projections are printed as their column.
        Default False.
    leaf_arities : dict, optional
        Mapping from `Symbol` leaves to the number of columns of their
        relation, used to print the atoms of collapsed plans.
    row_counts : dict, optional
        Mapping from the ``id`` of nodes of the expression to the number
        of rows they produced, printed next to them (e.g. ``⋈ ‹12 rows›``).

    """

    def __init__(
        self, name_map=None, indent=2, width=80, fresh_map=None,
        counter=None, leaf_sizes=None, column_roles=None, collapse=False,
        leaf_arities=None, row_counts=None,
    ):
        """Initialise the printer; see the class docstring for parameters."""
        if name_map is None:
            name_map = {}
        self.name_map = name_map
        self.leaf_sizes = leaf_sizes if leaf_sizes is not None else {}
        self.column_roles = column_roles if column_roles is not None else {}
        self.collapse = collapse
        self.leaf_arities = leaf_arities if leaf_arities is not None else {}
        self.row_counts = row_counts if row_counts is not None else {}
        self.indent = indent
        self.width = width
        self._fresh_map = fresh_map if fresh_map is not None else {}
        self._column_map = {}
        self._counter = counter if counter is not None else [0]

    def format(self, expression, level=0):
        """
        Return a multi-line pretty representation of an RA expression.

        The expression is drawn as a tree: each operator is printed on
        its own line, followed by its operands, one per line, attached
        with ``├─`` / ``└─`` branches. Every line is indented by
        ``level`` indentation steps.
        """
        prefix = " " * (self.indent * level)
        return "\n".join(prefix + line for line in self._render(expression))

    def _render(self, expression):
        """Return the lines of the tree drawing of ``expression``."""
        header, children = self._node(expression)
        rows = self.row_counts.get(id(expression))
        if rows is not None and "‹" not in header:
            header = f"{header} {rows_label(rows)}"
        lines = [header]
        for position, child in enumerate(children):
            last = position == len(children) - 1
            child_lines = self._render(child)
            lines.append(("└─ " if last else "├─ ") + child_lines[0])
            padding = "   " if last else "│  "
            lines.extend(padding + line for line in child_lines[1:])
        return lines

    def _node(self, expression):
        """Return the header line and the operands of ``expression``."""
        if self.collapse:
            collapsed = self._collapsed_node(expression)
            if collapsed is not None:
                return collapsed
        for expression_type, formatter_name in self._FORMATTERS:
            if isinstance(expression, expression_type):
                return getattr(self, formatter_name)(expression)
        return repr(expression), ()

    def _collapsed_node(self, expression):
        """
        Return the collapsed form of ``expression``, None if it has none.

        A rename of (a projection and selection of) a base relation is
        written as a Datalog atom, e.g. ``edge(x, y)``, and a projection
        on all the columns of its operand is skipped.
        """
        atom = _atom_pattern(expression)
        if atom is not None:
            rows = self.row_counts.get(id(expression))
            return self._format_atom(*atom, rows=rows), ()
        if isinstance(expression, Projection) and _is_trivial_projection(
            expression
        ):
            return self._node(expression.relation)
        return None

    def _format_atom(self, leaf, names, columns, selections, rows=None):
        """
        Format a base relation ``leaf`` filtered and named as an atom.

        ``names`` are the names given to its ``columns`` (indices), and
        ``selections`` the ``(column, value)`` equalities it is filtered
        with, ``value`` being a constant or another column index. The
        atom produced ``rows`` rows, when known.
        """
        arguments = self._atom_arguments(
            names, columns, selections, self._leaf_arity(leaf)
        )
        name, size = self._leaf_name_and_size(leaf)
        text = f"{name}({', '.join(arguments)})"
        pure = not selections and list(columns) == list(range(len(arguments)))
        if pure and size is not None:
            return f"{text} {rows_label(size)}"
        if rows is not None and size is not None:
            return f"{text} {rows_label(rows)[:-1]} from {size:,}›"
        if rows is not None:
            return f"{text} {rows_label(rows)}"
        if size is not None:
            return f"{text} ‹from {rows_label(size)[1:]}"
        return text

    def _atom_arguments(self, names, columns, selections, arity):
        """Return the displayed arguments of an atom, ``_`` if unnamed."""
        used = set(columns)
        for column, value in selections:
            used.add(column)
            if isinstance(value, int):
                used.add(value)
        # A plan may use more columns than the relation has, e.g. for a
        # predicate used with the wrong arity: evaluating it fails then.
        arity = max(arity or 0, 1 + max(used, default=-1))
        arguments = ["_"] * arity
        for name, column in zip(names, columns):
            arguments[column] = self._format_column(name)
        for column, value in selections:
            if isinstance(value, Constant):
                arguments[column] = self._format_value(value)
            elif arguments[column] == "_":
                arguments[column] = arguments[value]
            elif arguments[value] == "_":
                arguments[value] = arguments[column]
        return arguments

    def _leaf_name_and_size(self, leaf):
        """Return the display name of a leaf and its size (or None)."""
        name = self._lookup_name(leaf)
        if isinstance(leaf, Constant):
            if name is None:
                return self._format_constant(leaf), None
            return name, _relation_size(leaf.value)
        if name is None:
            name = leaf.name
        return name, self.leaf_sizes.get(leaf)

    def _leaf_arity(self, leaf):
        """Return the number of columns of a leaf's relation, or None."""
        if isinstance(leaf, Constant):
            try:
                return leaf.value.arity
            except Exception:
                return None
        return self.leaf_arities.get(leaf)

    def _format_selection(self, expression):
        """Format a Selection node."""
        header = f"σ[{self._format_formula(expression.formula)}]"
        return header, (expression.relation,)

    def _format_projection(self, expression):
        """Format a Projection node."""
        header = f"π[{self._format_columns(expression.attributes)}]"
        return header, (expression.relation,)

    def _format_extended_projection(self, expression):
        """Format an ExtendedProjection node."""
        proj = self._format_projection_list(expression.projection_list)
        return f"π_ext[{proj}]", (expression.relation,)

    @staticmethod
    def _binary(operator, expression):
        """Format a binary operator with its two operands."""
        return operator, (expression.relation_left, expression.relation_right)

    def _format_natural_join(self, expression):
        """Format a NaturalJoin node, marking cross products with ×."""
        op = "×" if _is_cross_product(expression) else "⋈"
        return self._binary(op, expression)

    def _format_nary_natural_join(self, expression):
        """Format an NaryNaturalJoin node, marking cross products with ×."""
        op = "×" if _is_nary_cross_product(expression) else "⋈"
        return op, tuple(expression.relations)

    def _format_equijoin(self, expression):
        """Format an EquiJoin node."""
        header = (
            f"⋈[{self._format_columns(expression.columns_left)}="
            f"{self._format_columns(expression.columns_right)}]"
        )
        return self._binary(header, expression)

    def _format_left_natural_join(self, expression):
        """Format a LeftNaturalJoin node."""
        return self._binary("⟕", expression)

    def _format_full_outer_natural_join(self, expression):
        """Format a FullOuterNaturalJoin node."""
        return self._binary("⟗", expression)

    def _format_union(self, expression):
        """Format a Union node."""
        return self._binary("∪", expression)

    def _format_intersection(self, expression):
        """Format an Intersection node."""
        return self._binary("∩", expression)

    def _format_difference(self, expression):
        """Format a Difference node."""
        return self._binary("−", expression)

    def _format_product(self, expression):
        """Format a Product node."""
        return "×", tuple(expression.relations)

    def _format_name_columns(self, expression):
        """Format a NameColumns node."""
        names = self._format_columns(expression.column_names)
        return f"ρ[{names}]", (expression.relation,)

    def _format_number_columns(self, expression):
        """Format a NumberColumns node."""
        names = self._format_columns(expression.column_names)
        return f"ν[{names}]", (expression.relation,)

    def _format_rename_column(self, expression):
        """Format a RenameColumn node."""
        header = (
            f"ρ[{self._format_column(expression.src)}→"
            f"{self._format_column(expression.dst)}]"
        )
        return header, (expression.relation,)

    def _format_rename_columns(self, expression):
        """Format a RenameColumns node."""
        header = f"ρ[{self._format_renames(expression.renames)}]"
        return header, (expression.relation,)

    def _format_group_by_aggregation(self, expression):
        """Format a GroupByAggregation node as ``γ[by=(…), dst = agg, …]``."""
        groupby = self._format_columns(expression.groupby)
        aggregates = ", ".join(
            self._format_aggregate(member)
            for member in expression.aggregate_functions
        )
        header = f"γ[by=({groupby}), {aggregates}]"
        return header, (expression.relation,)

    def _format_aggregate(self, member):
        """Format an aggregate ``dst = fun(…)``, or ``fun(…)`` if alike."""
        if not isinstance(member, FunctionApplicationListMember):
            return repr(member)
        fun = self._format_formula(member.fun_exp)
        dst = self._format_column(member.dst_column)
        if dst == fun:
            return fun
        return f"{dst} = {fun}"

    def _format_replace_null(self, expression):
        """Format a ReplaceNull node."""
        header = (
            f"κ[{self._format_column(expression.column)}, "
            f"NULL→{self._format_value(expression.value)}]"
        )
        return header, (expression.relation,)

    def _format_concatenate_constant_column(self, expression):
        """Format a ConcatenateConstantColumn node."""
        header = (
            f"⊕[{self._format_column(expression.column_name)}="
            f"{self._format_value(expression.column_value)}]"
        )
        return header, (expression.relation,)

    def _format_destroy(self, expression):
        """Format a Destroy node."""
        header = (
            f"δ[{self._format_column(expression.src_column)}→"
            f"{self._format_column(expression.dst_column)}]"
        )
        return header, (expression.relation,)

    def _format_constant_leaf(self, expression):
        """Format a Constant leaf, with its size when it is named."""
        name = self._lookup_name(expression)
        if name is None:
            return self._format_constant(expression), ()
        return f"{name} {rows_label(_relation_size(expression.value))}", ()

    def _format_symbol_leaf(self, expression):
        """Format a Symbol leaf, using its display name when one is mapped."""
        name = self._lookup_name(expression)
        if name is None:
            name = expression.name
        if expression in self.leaf_sizes:
            name = f"{name} {rows_label(self.leaf_sizes[expression])}"
        return name, ()

    # Ordered dispatch table: the first matching type wins, as with the
    # original chain of ``isinstance`` checks.
    _FORMATTERS = (
        (Selection, "_format_selection"),
        (Projection, "_format_projection"),
        (ExtendedProjection, "_format_extended_projection"),
        (NaturalJoin, "_format_natural_join"),
        (NaryNaturalJoin, "_format_nary_natural_join"),
        (EquiJoin, "_format_equijoin"),
        (LeftNaturalJoin, "_format_left_natural_join"),
        (FullOuterNaturalJoin, "_format_full_outer_natural_join"),
        (Union, "_format_union"),
        (Intersection, "_format_intersection"),
        (Difference, "_format_difference"),
        (Product, "_format_product"),
        (NameColumns, "_format_name_columns"),
        (NumberColumns, "_format_number_columns"),
        (RenameColumn, "_format_rename_column"),
        (RenameColumns, "_format_rename_columns"),
        (GroupByAggregation, "_format_group_by_aggregation"),
        (ReplaceNull, "_format_replace_null"),
        (ConcatenateConstantColumn, "_format_concatenate_constant_column"),
        (Destroy, "_format_destroy"),
        (Constant, "_format_constant_leaf"),
        (Symbol, "_format_symbol_leaf"),
    )

    def _shorten_fresh(self, name):
        """
        Return the display name of a column or variable named ``name``.

        Names found in ``column_roles`` take the label mapped there.
        Fresh names (``fresh_00000007``) found in ``fresh_map`` take the
        name mapped there (that of the rule variable they are); the other
        fresh names are numbered ``c₀``, ``c₁``, …, a prefix and digits
        no variable name can have.
        """
        if name in self.column_roles:
            return self.column_roles[name]
        if not _FRESH_NAME.match(name):
            return name
        if name in self._fresh_map:
            return self._fresh_map[name]
        if name not in self._column_map:
            self._column_map[name] = "c" + subscript(self._counter[0])
            self._counter[0] += 1
        return self._column_map[name]

    def _lookup_name(self, expression):
        """
        Look up a display name for a Constant or Symbol leaf.

        The map may use the object itself (identity) or its ``id()`` as key,
        which is useful when the wrapped value is unhashable. Symbols are
        also matched by name.
        """
        candidates = (expression,)
        if isinstance(expression, Constant):
            candidates += (expression.value,)
        for key, name in self.name_map.items():
            if any(
                candidate is key
                or (isinstance(key, int) and id(candidate) == key)
                for candidate in candidates
            ):
                return name
        if isinstance(expression, Symbol):
            return self._lookup_symbol_name(expression)
        return None

    def _lookup_symbol_name(self, symbol):
        """Look up the display name of a symbol with the same name."""
        for key, name in self.name_map.items():
            if isinstance(key, Symbol) and key.name == symbol.name:
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
        """Format an unnamed-by-the-map named set with its columns, size."""
        try:
            columns = [self._format_column_name(c) for c in value.columns]
        except Exception:
            columns = []
        return f"Table(columns={columns}) {rows_label(_relation_size(value))}"

    def _format_unnamed_set(self, value):
        """Format a set not named by the map with its size."""
        return f"Table {rows_label(_relation_size(value))}"

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
        if _is_column_int(value):
            return f"#{value.value}"
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
        if _is_column_int(col):
            return f"#{col.value}"
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
            if self.collapse and fun == dst:
                return dst
            return f"{fun} -> {dst}"
        return repr(member)

    def _format_renames(self, renames):
        return ", ".join(
            f"{self._format_column(src)}→{self._format_column(dst)}"
            for src, dst in renames
        )


OPERATOR_SYMBOLS = {
    "__builtins__": None,
}


def _populate_operator_symbols():
    mapping = {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
        operator.eq: "==",
        operator.ne: "!=",
        operator.gt: ">",
        operator.lt: "<",
        operator.ge: ">=",
        operator.le: "<=",
        operator.pow: "**",
        operator.and_: "and",
        operator.or_: "or",
        operator.not_: "not",
    }
    OPERATOR_SYMBOLS.update(mapping)


_populate_operator_symbols()


def build_name_map_from_conjunction(conjunction, symbol_table=None):
    """
    Build a name map from a Datalog Conjunction.

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
    for formula in _top_level_formulas(conjunction):
        _collect_predicate_names(formula, name_map)
    if isinstance(symbol_table, collections.abc.Mapping):
        _collect_edb_names(symbol_table, name_map)
    return name_map


def _top_level_formulas(conjunction):
    """Return the top-level formulas of a conjunction-like expression."""
    if hasattr(conjunction, "formulas"):
        return conjunction.formulas
    if hasattr(conjunction, "args"):
        return conjunction.args
    return ()


def _collect_predicate_names(expression, name_map):
    """Recursively add the functors of predicates in expression to name_map."""
    if isinstance(expression, FunctionApplication):
        functor = expression.functor
        if isinstance(functor, Constant):
            name_map[functor.value] = str(functor.value)
        elif isinstance(functor, Symbol):
            name_map[functor] = functor.name
        for arg in expression.args:
            _collect_predicate_names(arg, name_map)
    elif isinstance(expression, (tuple, list)):
        for item in expression:
            _collect_predicate_names(item, name_map)


def _collect_edb_names(symbol_table, name_map):
    """Map every relation constant in symbol_table to its symbol's name."""
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
            name_map[id(value)] = name


def pretty_repr(
    expression, name_map=None, indent=2, width=80, fresh_map=None,
    counter=None, leaf_sizes=None, column_roles=None, collapse=False,
    leaf_arities=None, row_counts=None,
):
    """
    Return a pretty representation of an RA expression.

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
        Mapping from fresh variable names to their display names, e.g.
        the ``s₀``, ``s₁``, … the rule the plan evaluates is printed with.
        Other fresh column names are displayed as ``c₀``, ``c₁``, ….
    counter : list, optional
        Mutable counter (list with one int) numbering the ``c₀``, ``c₁``,
        … columns.
    leaf_sizes : dict, optional
        Mapping from `Symbol` leaves to the number of rows of their
        relation, printed next to them.
    column_roles : dict, optional
        Mapping from column names to the label they are displayed with.
    collapse : bool, optional
        If True, collapse trivial wrappers (see `PrettyPrinter`).
    leaf_arities : dict, optional
        Mapping from `Symbol` leaves to the number of columns of their
        relation.
    row_counts : dict, optional
        Mapping from the ``id`` of nodes to the number of rows they
        produced.

    Returns
    -------
    str
        Pretty formatted expression.

    """
    printer = PrettyPrinter(
        name_map=name_map, indent=indent, width=width,
        fresh_map=fresh_map, counter=counter, leaf_sizes=leaf_sizes,
        column_roles=column_roles, collapse=collapse,
        leaf_arities=leaf_arities, row_counts=row_counts,
    )
    return printer.format(expression)
