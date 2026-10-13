"""Optimisers flattening, simplifying and reordering joins and products."""

from functools import reduce
from itertools import combinations

from ... import expression_walker as ew
from ...expressions import Constant
from ..relational_algebra import (
    NameColumns,
    NaturalJoin,
    NaryNaturalJoin,
    NumberColumns,
    Product,
    Projection,
    RenameColumn,
    RenameColumns,
    Selection
)
from .columns import _project_onto_operand, _ra_columns


class ProductSimplification(ew.ExpressionWalker):
    @ew.add_match(Product, lambda x: len(x.relations) == 1)
    def single_product(self, product):
        return self.walk(product.relations[0])


def _flatten_joins(expr, with_products=False):
    """
    Collect all join operands into a flat list.

    Recurse through NaturalJoin (binary) and NaryNaturalJoin (n-ary) so
    that nested join trees are fully unwound in one pass. When
    ``with_products`` is set, also recurse through Product and through
    Projection wrappers sitting between join nodes, pushing each such
    projection down onto the individual operands. This lets a mixed tree
    such as ``NaturalJoin(Projection(Product(A, B)), C)`` be reordered
    globally when the operands share column names.
    """
    if with_products and isinstance(expr, Projection):
        inner = _flatten_joins(expr.relation, with_products)
        if len(inner) <= 1:
            return [expr]
        return [_project_onto_operand(op, expr.attributes) for op in inner]
    flattened_types = (NaturalJoin, NaryNaturalJoin)
    if with_products:
        flattened_types += (Product,)
    if not isinstance(expr, flattened_types):
        return [expr]
    if isinstance(expr, NaturalJoin):
        children = (expr.relation_left, expr.relation_right)
    else:
        children = expr.relations
    return [
        op for child in children for op in _flatten_joins(child, with_products)
    ]


def _overlap_degrees(col_sets):
    """Return, for each column set, its total overlap with all the others."""
    return [
        sum(len(cols & other) for j, other in enumerate(col_sets) if j != i)
        for i, cols in enumerate(col_sets)
    ]


_ROW_BOUNDED_OPERATIONS = (
    NameColumns, NumberColumns, Projection, RenameColumn, RenameColumns,
    Selection,
)


def _estimated_size(expression, resolve=None):
    """
    Cheaply estimate an upper bound of the number of rows of an RA leaf.

    Sizes are only known for constant relations, possibly wrapped in
    renames, projections or selections, whose size is estimated as the one
    of their input. The estimate comes from the set's
    ``row_count_estimate`` method, which never deduplicates the set.

    Parameters
    ----------
    expression : Expression
        Relational algebra expression to estimate.
    resolve : callable, optional
        Called on any other expression, it returns an expression standing
        for the same relation (for instance the value of a symbol in a
        symbol table) or ``None``.

    Returns
    -------
    int or None
        Estimated number of rows, ``None`` when unknown.

    """
    for _ in range(64):
        if isinstance(expression, _ROW_BOUNDED_OPERATIONS):
            expression = expression.relation
        elif isinstance(expression, Constant):
            estimate = getattr(expression.value, "row_count_estimate", None)
            return estimate() if callable(estimate) else None
        elif resolve is not None and expression is not None:
            expression = resolve(expression)
        else:
            return None
    return None


def _sizes_known(sizes):
    """Whether the sizes of at least two operands are known."""
    return sum(size is not None for size in sizes) >= 2


def _smallest_connected(indices, sizes, overlap):
    """
    Pick the smallest connected operand of known size.

    Among ``indices`` of known size with a positive ``overlap``, return the
    one of smallest size and then of largest overlap, ``None`` if there is
    none.
    """
    candidates = [i for i in indices if sizes[i] is not None and overlap(i)]
    return min(
        candidates, key=lambda i: (sizes[i], -overlap(i)), default=None
    )


def _greedy_chain(col_sets, sizes, start, cumulative):
    """
    Order indices of ``col_sets`` by nearest-neighbour expansion.

    When at least two ``sizes`` are known, the chain starts at the smallest
    operand sharing columns with the others and, at each step, appends the
    smallest operand sharing columns with the placed ones: cross products
    are avoided first, then small operands are joined first.

    Otherwise, or when no such operand has a known size, the chain starts
    at the operand chosen by ``start`` (``max`` or ``min``) on the overlap
    degrees, and appends the operand whose columns overlap most with the
    reference column set: the union of all placed column sets when
    ``cumulative`` is true, the column set of the last placed one otherwise.
    """
    if not _sizes_known(sizes):
        sizes = [None] * len(col_sets)
    degrees = _overlap_degrees(col_sets)
    first = _smallest_connected(
        range(len(col_sets)), sizes, degrees.__getitem__
    )
    if first is None:
        first = start(range(len(col_sets)), key=degrees.__getitem__)
    remaining = set(range(len(col_sets)))
    remaining.remove(first)
    order = [first]
    reference = set(col_sets[first])
    placed = set(col_sets[first])
    while remaining:
        best = _smallest_connected(
            sorted(remaining), sizes, lambda j: len(col_sets[j] & placed)
        )
        if best is None:
            best = max(remaining, key=lambda j: len(col_sets[j] & reference))
        order.append(best)
        remaining.remove(best)
        placed |= col_sets[best]
        if cumulative:
            reference |= col_sets[best]
        else:
            reference = set(col_sets[best])
    return order


def _sort_join_operands(operands, sizes=()):
    """
    Greedy sort to minimise cross products.

    Repeatedly pick the operand with the most column overlap with the
    cumulative column set of already-placed operands. This maximises
    shared columns at each join boundary, eliminating cross products
    unless the join graph is truly disconnected. Estimated operand
    ``sizes`` are used as described in :func:`_greedy_chain`.
    """
    if len(operands) <= 2:
        return operands
    col_sets = [_ra_columns(op) for op in operands]
    return [operands[i] for i in _greedy_chain(col_sets, sizes, max, True)]


def _connected_components(col_sets):
    """
    Group indices of ``col_sets`` into components of the overlap graph.

    Two indices are adjacent when their column sets intersect. Components
    are sorted by their smallest index and each holds its indices in
    increasing order.
    """
    n = len(col_sets)
    assigned = set()
    components = []
    for start in range(n):
        if start in assigned:
            continue
        component, frontier = {start}, [start]
        while frontier:
            i = frontier.pop()
            new = {
                j for j in range(n)
                if j not in component and col_sets[i] & col_sets[j]
            }
            component |= new
            frontier.extend(new)
        assigned |= component
        components.append(sorted(component))
    return components


def _group_operands_by_overlap(operands, sizes):
    """
    Greedy operator ordering (GOO) by pairwise column-name overlap.

    Operands are grouped into connected components based on shared column
    names, in order of first appearance. Each component is ordered with
    :func:`_order_component_greedily` so that adjacent operands overlap.

    The cost model uses column names: two sub-plans overlap when their
    column sets intersect. Estimated operand ``sizes`` are used as
    described in :func:`_greedy_chain`.
    """
    col_sets = [set(_ra_columns(op)) for op in operands]
    return [
        _order_component_greedily(
            [operands[i] for i in component],
            [col_sets[i] for i in component],
            [sizes[i] for i in component],
        )
        for component in _connected_components(col_sets)
    ]


def _order_component_greedily(operands, col_sets, sizes):
    """
    Greedily order operands in one connected component.

    Builds a path through the overlap graph using nearest-neighbour
    expansion from a low-degree endpoint. This places bridge operands
    between otherwise disconnected sub-graphs, avoiding adjacent operands
    that share no columns. Estimated operand ``sizes`` are used as
    described in :func:`_greedy_chain`.
    """
    if len(operands) <= 2:
        return list(operands)
    return [operands[i] for i in _greedy_chain(col_sets, sizes, min, False)]


class FlattenAndReorderJoins(ew.PatternWalker):
    """
    Flatten joins into an NaryNaturalJoin and reorder its operands.

    Flatten NaturalJoins and NaryNaturalJoins into a flat
    NaryNaturalJoin, then greedily reorder to minimise cross products.

    1. **Flatten**: Collect all join operands by recursing through
       NaturalJoin and NaryNaturalJoin structure.
    2. **Sort**: Greedy nearest-neighbour ordering maximising column
       overlap at each join boundary (avoids cross products by keeping
       related tables adjacent).
    3. **Wrap**: Produce an NaryNaturalJoin holding all operands.
    """

    @ew.add_match(NaturalJoin)
    def flatten_and_reorder(self, expression):
        # Walk children first so PushProjectionsDown can process inner
        # Projection(NaturalJoin, ...) patterns before we flatten.
        left = self.walk(expression.relation_left)
        right = self.walk(expression.relation_right)
        walked = NaturalJoin(left, right)
        operands = _flatten_joins(walked)
        if len(operands) <= 1:
            return operands[0] if operands else walked
        return NaryNaturalJoin(_sort_join_operands(
            operands, [_estimated_size(op) for op in operands]
        ))

    @ew.add_match(NaryNaturalJoin)
    def reorder_nary(self, expression):
        operands = expression.relations
        if len(operands) <= 1:
            return operands[0] if operands else expression
        flat_ops = [
            relation for op in operands
            for relation in (
                op.relations if isinstance(op, NaryNaturalJoin) else (op,)
            )
        ]
        sorted_ops = _sort_join_operands(
            flat_ops, [_estimated_size(op) for op in flat_ops]
        )
        if any(a is not b for a, b in zip(flat_ops, sorted_ops)):
            return NaryNaturalJoin(sorted_ops)
        # Walk operands so inner Projection patterns are reachable
        new_ops = tuple(self.walk(op) for op in flat_ops)
        if any(a is not b for a, b in zip(new_ops, flat_ops)):
            return NaryNaturalJoin(new_ops)
        return expression


def _tree_contains_product(expr):
    """
    Check whether an RA expression tree contains a Product node.

    Does NOT recurse into Selection nodes — a Product inside a Selection
    is already handled by RewriteSelections (eq-selection/product → EquiJoin)
    and must not be reordered by GOO.
    """
    if isinstance(expr, Product):
        return True
    if isinstance(expr, Selection) or not isinstance(expr, ew.Expression):
        return False
    return any(_tree_contains_product(child) for child in expr.unapply())


def _has_eliminable_cross_product(operands):
    """
    Check whether reordering can eliminate a cross product.

    Check whether a flat operand list has a cross product that can be
    eliminated by reordering: i.e., there exist at least two operands
    that share columns AND at least one pair that doesn't.

    If all operands share columns with each other, FlattenAndReorderJoins
    already handles it. If no operands share columns, reordering won't help.
    """
    col_sets = [set(_ra_columns(op)) for op in operands]
    overlaps = [bool(a & b) for a, b in combinations(col_sets, 2)]
    return any(overlaps) and not all(overlaps)


def _is_left_deep_join_of(expression, operands):
    """Check whether ``expression`` is the left-deep join of ``operands``."""
    for operand in reversed(operands[1:]):
        if (
            not isinstance(expression, NaturalJoin)
            or expression.relation_right is not operand
        ):
            return False
        expression = expression.relation_left
    return expression is operands[0]


def _degenerate_natural_join(expression):
    """Guard: True if NaturalJoin(A, B) where A and B share no columns."""
    left_cols = _ra_columns(expression.relation_left)
    right_cols = _ra_columns(expression.relation_right)
    return bool(left_cols) and bool(right_cols) and not (left_cols & right_cols)


class DegenerateNaturalJoinToProduct(ew.PatternWalker):
    """
    Convert NaturalJoin(A, B) to Product(A, B) when disjoint.

    Convert NaturalJoin(A, B) to Product(A, B) when A and B share no
    column names. A degenerate NaturalJoin is semantically a cross product;
    making it explicit lets the GOO pass detect and reorder it.
    """

    @ew.add_match(NaturalJoin, _degenerate_natural_join)
    def degenerate_join_to_product(self, expression):
        left = self.walk(expression.relation_left)
        right = self.walk(expression.relation_right)
        return Product((left, right))


class GreedyJoinOrdering(ew.PatternWalker):
    """
    Reorder mixed Product/NaturalJoin chains with greedy ordering.

    Flatten mixed Product/NaturalJoin chains and reorder by pairwise
    ColumnStr overlap using a greedy operator ordering (GOO).

    Only triggers when the expression is a NaturalJoin that contains a
    Product, or a Product whose operands have an eliminable cross product.
    This prevents infinite re-entry: once the cross product is eliminated,
    the guard no longer matches.

    Emits nested NaturalJoin trees (not NaryNaturalJoin) for connected
    groups and Product for disconnected groups.

    Operand sizes, estimated with :func:`_estimated_size`, drive the order
    within connected groups when known: the smallest operands sharing
    columns with the already joined ones are joined first. A NaturalJoin
    chain without Product is also reordered this way when at least two of
    its operands have a known size. Subclasses can resolve leaves which
    are not constants, such as symbols, by overriding
    :meth:`resolve_relation`.
    """

    @ew.add_match(NaturalJoin, _tree_contains_product)
    def reorder_join(self, expression):
        left = self.walk(expression.relation_left)
        right = self.walk(expression.relation_right)
        expression = NaturalJoin(left, right)
        operands = _flatten_joins(expression, with_products=True)
        if not _has_eliminable_cross_product(operands):
            return expression
        return self._build_reordered(operands)

    @ew.add_match(
        Product, lambda e: _has_eliminable_cross_product(e.relations)
    )
    def reorder_product(self, expression):
        new_rels = tuple(self.walk(r) for r in expression.relations)
        operands = _flatten_joins(Product(new_rels), with_products=True)
        return self._build_reordered(operands)

    @ew.add_match(NaturalJoin, lambda e: not _tree_contains_product(e))
    def reorder_join_by_size(self, expression):
        """
        Reorder a NaturalJoin chain using estimated operand sizes.

        A chain of more than two operands, at least two of known size, is
        rebuilt as a left-deep NaturalJoin tree ordered by
        :func:`_sort_join_operands`. Otherwise, only the children are walked.
        """
        operands = _flatten_joins(expression)
        if len(operands) > 2:
            sizes = self._sizes(operands)
            if _sizes_known(sizes):
                ordered = [
                    self.walk(op)
                    for op in _sort_join_operands(operands, sizes)
                ]
                if _is_left_deep_join_of(expression, ordered):
                    return expression
                return reduce(NaturalJoin, ordered)
        return self.fallthrough(expression)

    @ew.add_match(ew.Expression)
    def fallthrough(self, expression):
        """
        Recursively walk children of non-matching expressions.

        Recursively walk children for expressions that don't match
        the reorder patterns. This replaces the old walk() override.
        """
        args = expression.unapply()
        new_args = tuple()
        changed = False
        for arg in args:
            if isinstance(arg, ew.Expression):
                new_arg = self.walk(arg)
                changed |= new_arg is not arg
            elif isinstance(arg, tuple) and len(arg) > 0 and isinstance(arg[0], ew.Expression):
                new_arg = tuple(self.walk(sub) for sub in arg)
                changed |= any(a is not b for a, b in zip(new_arg, arg))
            else:
                new_arg = arg
            new_args += (new_arg,)
        if changed:
            return expression.apply(*new_args)
        return expression

    def resolve_relation(self, expression):
        """
        Resolve a non-constant leaf into an expression of the same relation.

        Used to estimate operand sizes. The default implementation resolves
        nothing, so only constant relations have a known size.

        Parameters
        ----------
        expression : Expression
            Leaf of a relational algebra expression.

        Returns
        -------
        Expression or None
            Expression for the same relation, or ``None``.

        """
        return None

    def _sizes(self, operands):
        """Estimate the sizes of ``operands`` with :func:`_estimated_size`."""
        return [_estimated_size(op, self.resolve_relation) for op in operands]

    def _build_reordered(self, operands):
        groups = _group_operands_by_overlap(operands, self._sizes(operands))
        join_groups = [reduce(NaturalJoin, g) for g in groups if len(g) > 1]
        singletons = [g[0] for g in groups if len(g) == 1]
        if len(join_groups) == 1 and not singletons:
            return join_groups[0]
        return Product(tuple(join_groups + singletons))
