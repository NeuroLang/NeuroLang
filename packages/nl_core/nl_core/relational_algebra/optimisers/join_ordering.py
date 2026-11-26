"""Optimisers flattening, simplifying and reordering joins and products."""

from functools import reduce
from itertools import combinations, permutations

from nl_expr import expression_walker as ew
from nl_expr.expressions import Constant
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


class ProductSimplification(ew.PatternWalker):
    @ew.add_match(Product, lambda x: len(x.relations) == 1)
    def single_product(self, product):
        return self.walk(product.relations[0])


def _projection_distributes(operands, attributes):
    """
    Check that a projection of the join of ``operands`` distributes.

    Projecting the join on ``attributes`` equals joining the operands
    projected on ``attributes`` when the projection keeps the columns
    shared by operands and at least one column of each operand. The
    projected operands then hold no dropped column which could join them
    with other operands.
    """
    attributes = set(attributes)
    col_sets = [set(_ra_columns(op)) for op in operands]
    return all(cols & attributes for cols in col_sets) and all(
        a & b <= attributes for a, b in combinations(col_sets, 2)
    )


def _distributed_projection_operands(projection):
    """
    Return the join operands a projection is pushed onto when flattening.

    Return the operands of the joins and products below ``projection``
    when there are several and the projection distributes over them,
    ``None`` otherwise.
    """
    inner = _flatten_joins(projection.relation, with_products=True)
    if len(inner) <= 1 or not _projection_distributes(
        inner, projection.attributes
    ):
        return None
    return inner


def _flatten_joins(expr, with_products=False):
    """
    Collect all join operands into a flat list.

    Recurse through NaturalJoin (binary) and NaryNaturalJoin (n-ary) so
    that nested join trees are fully unwound in one pass. When
    ``with_products`` is set, also recurse through Product and through
    Projection wrappers sitting between join nodes, pushing each such
    projection down onto the individual operands when it distributes
    over them (see :func:`_projection_distributes`). This lets a mixed tree
    such as ``NaturalJoin(Projection(Product(A, B)), C)`` be reordered
    globally when the operands share column names.
    """
    if with_products and isinstance(expr, Projection):
        inner = _distributed_projection_operands(expr)
        if inner is None:
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


# Operands joined in a left-deep chain; above it, in chunks of this size.
JOIN_CHUNK = 16


def join_in_chunks(
    operands, join=NaturalJoin, chunk=JOIN_CHUNK, nary=False
):
    """
    Join ``operands`` in order, without a tree as deep as they are many.

    The join of at most ``chunk`` operands is the left-deep chain
    ``join(join(o1, o2), o3)...``. Beyond it, a run of operands each
    sharing columns with the one before it is joined in chunks of
    ``chunk`` consecutive operands, each chunk a left-deep chain, and so
    on with the results: a tree of depth ``chunk`` times the logarithm in
    base ``chunk`` of their number, whose joins are all through shared
    columns, as those of the chain are.

    A run ends at an operand sharing no column with the one before it, or
    whose columns are not known. Such an operand is joined to the others
    through the operands before it, and a chunk of operands not sharing
    columns would be a cross product. When the joins of the runs are
    more than ``chunk``, they are joined in a :class:`NaryNaturalJoin`,
    in the order of the chain, if ``nary`` (no binary tree of them is
    free of cross products in general: a centre joined to leaves that
    share columns with it only); else in chunks if the runs share no
    column, their join being the product of independent operands whatever
    the tree; else in a chain. Joins being associative and commutative,
    the result is that of the chain.

    A walker recurses once per level of the tree: a chain of 250 joins
    or more exceeds the limit of the recursion.
    """
    operands = list(operands)
    if len(operands) <= chunk:
        return reduce(join, operands)
    columns = list(map(_columns_or_none, operands))
    runs, run_columns = [[operands[0]]], [columns[0]]
    for i, operand in enumerate(operands[1:]):
        if _share_columns(columns[i], columns[i + 1]):
            runs[-1].append(operand)
            run_columns[-1] = _union_or_none(run_columns[-1], columns[i + 1])
        else:
            runs.append([operand])
            run_columns.append(columns[i + 1])
    joins = [_join_run(run, join, chunk) for run in runs]
    if len(joins) <= chunk:
        return reduce(join, joins)
    if nary:
        return NaryNaturalJoin(tuple(joins))
    if _are_independent(run_columns):
        return _join_run(joins, join, chunk)
    return reduce(join, joins)


def _share_columns(columns, other):
    """Whether two known sets of columns share one."""
    return columns is not None and other is not None and bool(columns & other)


def _union_or_none(columns, other):
    return None if columns is None or other is None else columns | other


def _are_independent(column_sets):
    """Whether the column sets are all known and share no column."""
    return None not in column_sets and sum(map(len, column_sets)) == len(
        set().union(*column_sets)
    )


def _columns_or_none(operand):
    """The columns of ``operand``, none when they are not known."""
    try:
        return set(_ra_columns(operand))
    except NotImplementedError:
        return None


def _join_run(run, join, chunk):
    while len(run) > chunk:
        run = [
            reduce(join, run[i:i + chunk]) for i in range(0, len(run), chunk)
        ]
    return reduce(join, run)


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


def _join_size_bound(left_size, left_cols, right_size, right_cols):
    """
    Bound the number of rows of the join of two sets.

    A set whose columns are all columns of the other matches each row of
    the other at most once: the join has at most the rows of the other.
    Otherwise it has at most the product of their rows.
    """
    bounds = []
    if right_cols <= left_cols:
        bounds.append(left_size)
    if left_cols <= right_cols:
        bounds.append(right_size)
    return min(bounds, default=left_size * right_size)


def _cheapest_connected(indices, sizes, col_sets, placed, placed_size):
    """
    Pick the operand of known size whose join with the placed ones is the
    smallest.

    Among ``indices`` of known size sharing columns with ``placed``, of
    ``placed_size`` rows (``None`` if unknown), return the one of the
    smallest :func:`_join_size_bound`, then of the smallest size and of the
    largest overlap; ``None`` if there is none.
    """
    def cost(i):
        bound = 0 if placed_size is None else _join_size_bound(
            placed_size, placed, sizes[i], col_sets[i]
        )
        return bound, sizes[i], -len(col_sets[i] & placed)

    candidates = [
        i for i in indices if sizes[i] is not None and col_sets[i] & placed
    ]
    return min(candidates, key=cost, default=None)


def _cheapest_start(col_sets, sizes):
    """
    Pick the operand of known size starting the cheapest join of two.

    The operand of a pair of operands of known size sharing columns of the
    smallest :func:`_join_size_bound`, the smaller of the two; ``None`` if
    there is no such pair.
    """
    pairs = [
        (i, j) for i, j in permutations(range(len(col_sets)), 2)
        if sizes[i] is not None and sizes[j] is not None
        and col_sets[i] & col_sets[j]
    ]

    def cost(pair):
        i, j = pair
        bound = _join_size_bound(sizes[i], col_sets[i], sizes[j], col_sets[j])
        return bound, sizes[i]

    best = min(pairs, key=cost, default=None)
    return None if best is None else best[0]


def _greedy_chain(col_sets, sizes, start, cumulative):
    """
    Order indices of ``col_sets`` by nearest-neighbour expansion.

    When at least two ``sizes`` are known, the chain starts at the cheapest
    join of two operands sharing columns and, at each step, appends the
    operand sharing columns with the placed ones whose join with them is
    the cheapest, as bounded by :func:`_join_size_bound`: cross products
    are avoided first, then the joins which cannot grow the rows, such as
    those of an operand all of whose columns are joined, are made first.

    Otherwise, or when no such operand has a known size, the chain starts
    at the operand chosen by ``start`` (``max`` or ``min``) on the overlap
    degrees, and appends the operand whose columns overlap most with the
    reference column set: the union of all placed column sets when
    ``cumulative`` is true, the column set of the last placed one otherwise.
    """
    if not _sizes_known(sizes):
        sizes = [None] * len(col_sets)
    degrees = _overlap_degrees(col_sets)
    first = _cheapest_start(col_sets, sizes)
    if first is None:
        first = start(range(len(col_sets)), key=degrees.__getitem__)
    remaining = set(range(len(col_sets)))
    remaining.remove(first)
    order = [first]
    reference = set(col_sets[first])
    placed = set(col_sets[first])
    placed_size = sizes[first]
    while remaining:
        best = _cheapest_connected(
            sorted(remaining), sizes, col_sets, placed, placed_size
        )
        if best is None:
            best = max(remaining, key=lambda j: len(col_sets[j] & reference))
            placed_size = None
        elif placed_size is not None:
            placed_size = _join_size_bound(
                placed_size, placed, sizes[best], col_sets[best]
            )
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


def _is_join(expression):
    """Whether ``expression`` is a binary or an n-ary natural join."""
    return isinstance(expression, (NaturalJoin, NaryNaturalJoin))


def _has_join_operand(expression):
    """Guard: True if a NaturalJoin has a join operand to flatten."""
    return (
        _is_join(expression.relation_left)
        or _is_join(expression.relation_right)
    )


def _has_single_operand(expression):
    """Guard: True if an NaryNaturalJoin has a single operand."""
    return len(expression.relations) == 1


def _has_nested_join(expression):
    """Guard: True if an NaryNaturalJoin has a join operand to flatten."""
    return any(_is_join(op) for op in expression.relations)


def _sorted_by_estimated_size(operands):
    """Sort join operands as :class:`FlattenAndReorderJoins` does."""
    return _sort_join_operands(
        operands, [_estimated_size(op) for op in operands]
    )


def _operands_out_of_order(expression):
    """
    Guard: True if the operands of an NaryNaturalJoin are not sorted.

    The operands are compared with their order by
    :func:`_sort_join_operands`, which is a fixpoint of the sort.
    """
    operands = expression.relations
    return any(
        a is not b
        for a, b in zip(operands, _sorted_by_estimated_size(operands))
    )


class FlattenAndReorderJoins(ew.PatternWalker):
    """
    Flatten nested joins into an NaryNaturalJoin and reorder its operands.

    Only joins which need it are rewritten: a NaturalJoin with a join
    operand, or an NaryNaturalJoin with a join operand, a single operand
    or operands out of order. A join of two relations is left untouched.

    1. **Flatten**: Collect all join operands by recursing through
       NaturalJoin and NaryNaturalJoin structure.
    2. **Sort**: Greedy nearest-neighbour ordering maximising column
       overlap at each join boundary (avoids cross products by keeping
       related tables adjacent).
    3. **Wrap**: Produce an NaryNaturalJoin holding all operands.

    The rewritten join is walked again, so that its operands are
    optimised: a walker does not descend into the result of a rule.
    """

    # Changes the expression: a NaturalJoin becomes an NaryNaturalJoin.
    @ew.add_match(NaturalJoin, _has_join_operand)
    def flatten_join(self, expression):
        return self._flatten_and_sort(expression)

    # Changes the expression: the join is replaced by its operand.
    @ew.add_match(NaryNaturalJoin, _has_single_operand)
    def single_operand_join(self, expression):
        return self.walk(expression.relations[0])

    # Changes the expression: a join operand is replaced by its operands.
    @ew.add_match(NaryNaturalJoin, _has_nested_join)
    def flatten_nary_join(self, expression):
        return self._flatten_and_sort(expression)

    # Changes the expression: the operands are reordered.
    @ew.add_match(NaryNaturalJoin, _operands_out_of_order)
    def reorder_nary_join(self, expression):
        return self.walk(NaryNaturalJoin(
            tuple(_sorted_by_estimated_size(expression.relations))
        ))

    def _flatten_and_sort(self, expression):
        operands = _sorted_by_estimated_size(_flatten_joins(expression))
        return self.walk(NaryNaturalJoin(tuple(operands)))


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


def _is_join_of(expression, operands):
    """Check whether ``expression`` is the join of ``operands`` as
    :func:`join_in_chunks` makes it."""
    return _same_join_tree(expression, join_in_chunks(operands))


def _same_join_tree(expression, other):
    if isinstance(expression, NaturalJoin) and isinstance(other, NaturalJoin):
        return _same_join_tree(
            expression.relation_left, other.relation_left
        ) and _same_join_tree(expression.relation_right, other.relation_right)
    return expression is other


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


def _join_spine_has_product(expr):
    """
    Check whether a Product is flattened by ``_flatten_joins``.

    Follow the nodes that :func:`_flatten_joins` recurses through when
    ``with_products`` is set, that is joins, products, and projections
    distributing over several operands, and look for a Product among
    them.
    """
    if isinstance(expr, Product):
        return True
    if isinstance(expr, Projection):
        return (
            _distributed_projection_operands(expr) is not None
            and _join_spine_has_product(expr.relation)
        )
    if isinstance(expr, NaturalJoin):
        children = (expr.relation_left, expr.relation_right)
    elif isinstance(expr, NaryNaturalJoin):
        children = expr.relations
    else:
        return False
    return any(_join_spine_has_product(child) for child in children)


def _join_with_eliminable_product(expression):
    """
    Guard: True if a NaturalJoin has a cross product to eliminate.

    The operands flattened through joins, products and projections
    include a Product, which reordering removes, and some of them share
    columns while others do not.
    """
    return _join_spine_has_product(expression) and (
        _has_eliminable_cross_product(
            _flatten_joins(expression, with_products=True)
        )
    )


def _is_product_free_join_chain(expression):
    """Guard: True if a NaturalJoin chains 3+ operands and no Product."""
    return (
        not _tree_contains_product(expression)
        and len(_flatten_joins(expression)) > 2
    )


def _walk_join_chain_operands(walker, expression):
    """
    Walk the operands of a join chain, keeping the shape of the chain.

    Return ``expression`` itself when no operand changes.
    """
    if isinstance(expression, NaturalJoin):
        left = _walk_join_chain_operands(walker, expression.relation_left)
        right = _walk_join_chain_operands(walker, expression.relation_right)
        if (
            left is expression.relation_left
            and right is expression.relation_right
        ):
            return expression
        return NaturalJoin(left, right)
    if isinstance(expression, NaryNaturalJoin):
        relations = tuple(
            _walk_join_chain_operands(walker, relation)
            for relation in expression.relations
        )
        if all(a is b for a, b in zip(relations, expression.relations)):
            return expression
        return NaryNaturalJoin(relations)
    return walker.walk(expression)


class GreedyJoinOrdering(ew.PatternWalker):
    """
    Reorder mixed Product/NaturalJoin chains with greedy ordering.

    Flatten mixed Product/NaturalJoin chains and reorder by pairwise
    ColumnStr overlap using a greedy operator ordering (GOO).

    Only triggers when the expression is a NaturalJoin whose flattened
    operands include a Product, or a Product, with an eliminable cross
    product. This prevents infinite re-entry: once the cross product is
    eliminated, the guards no longer match.

    Emits nested NaturalJoin trees (not NaryNaturalJoin) for connected
    groups, left-deep chains of the operands in the order chosen, in chunks
    when they are more than a chunk (see :func:`join_in_chunks`), and
    Product for disconnected groups.

    Operand sizes, estimated with :func:`_estimated_size`, drive the order
    within connected groups when known: the smallest operands sharing
    columns with the already joined ones are joined first. A NaturalJoin
    chain without Product is also reordered this way when at least two of
    its operands have a known size. Subclasses can resolve leaves which
    are not constants, such as symbols, by overriding
    :meth:`resolve_relation`.

    The rules walk the operands they keep, and leave walking any other
    expression to the walker it is composed with, such as
    :class:`~neurolang.expression_walker.ExpressionWalker`.
    """

    # Changes the expression: the Product found by the guard is removed.
    @ew.add_match(NaturalJoin, _join_with_eliminable_product)
    def reorder_join(self, expression):
        return self._build_reordered(
            _flatten_joins(expression, with_products=True)
        )

    # Changes the expression: operands sharing columns are joined.
    @ew.add_match(
        Product, lambda e: _has_eliminable_cross_product(e.relations)
    )
    def reorder_product(self, expression):
        return self._build_reordered(
            _flatten_joins(expression, with_products=True)
        )

    # Sizes are resolved by the walker, out of reach of the guard. When
    # they do not change the order, the chain is returned with its walked
    # operands, itself if they are unchanged, which ends the rewriting.
    @ew.add_match(NaturalJoin, _is_product_free_join_chain)
    def reorder_join_by_size(self, expression):
        """
        Reorder a NaturalJoin chain using estimated operand sizes.

        A chain of more than two operands, at least two of known size, is
        rebuilt as a NaturalJoin tree of the operands ordered by
        :func:`_sort_join_operands`, a left-deep chain of them, in chunks
        when they are many (:func:`join_in_chunks`). Otherwise, only its operands are
        walked.
        """
        operands = _flatten_joins(expression)
        sizes = self._sizes(operands)
        if not _sizes_known(sizes):
            return _walk_join_chain_operands(self, expression)
        ordered = [
            self.walk(op) for op in _sort_join_operands(operands, sizes)
        ]
        if _is_join_of(expression, ordered):
            return expression
        return join_in_chunks(ordered)

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
        operands = [self.walk(op) for op in operands]
        groups = _group_operands_by_overlap(operands, self._sizes(operands))
        join_groups = [join_in_chunks(g) for g in groups if len(g) > 1]
        singletons = [g[0] for g in groups if len(g) == 1]
        if len(join_groups) == 1 and not singletons:
            return join_groups[0]
        return Product(tuple(join_groups + singletons))
