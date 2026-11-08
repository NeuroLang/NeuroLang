"""
Joins of points within a Euclidean distance, computed with numpy.

This is the numeric kernel of the Euclidean spatial-bound syntactic sugar
(see :mod:`neurolang.frontend.datalog.sugar.spatial`), which only detects
the bound and rewrites the rule. :func:`euclidean_distance` is the
``euclidean`` built-in function. :func:`euclidean_pairs` returns every
pair of points, one from each of two sets, whose distance is below (or,
non-strictly, at most) a radius; :func:`euclidean_join` joins two tables
on the distance between their points in the same way, and projects the
result on some of its columns. The distances are those
:func:`euclidean_distance` computes, so that the pairs are exactly the
ones the ``euclidean`` built-in would select.

When both sets have integer coordinates, the candidate neighbours of the
smaller set are generated from an integer offset stencil and looked up in
the larger one, which avoids building a k-d tree on the larger set.
Floats lying on a grid (see :data:`GRID_STEPS`) are numbered by their
position in it and joined in the same way, the pairs at an offset too
close to the radius to be told from rounding errors being checked on the
floats. Otherwise a k-d tree is built on the points of the larger set that lie
within the radius of the bounding box of the smaller set, and searched in
chunks of the smaller set, to hold few pairs at once. The stencil is used
whenever the larger set is dense enough, as one of its look-ups costs much
less than one of the pairs a k-d tree returns.
"""

import logging
import math
import numbers
import operator

import numpy
import pandas
import scipy.spatial

from nl_expr.utils.various import log_performance

LOG = logging.getLogger(__name__)

#: Maximum number of cells of the cube enclosing the offset stencil.
STENCIL_MAX_CELLS = 1_000_000
#: Number of stencil look-ups always allowed before using a k-d tree.
STENCIL_MIN_WORK = 20_000_000
#: Number of stencil look-ups allowed per point of the larger set.
STENCIL_WORK_PER_POINT = 32
#: A stencil, whose look-ups are far cheaper than the pairs a k-d tree
#: returns, is always used when the occupancy grid has at most this many
#: cells per point of the larger set.
STENCIL_GRID_CELLS_PER_POINT = 40
#: Occupancy grids are used up to this many cells per point...
GRID_CELLS_PER_POINT = 4
#: ... or this many cells; sorted keys are searched otherwise.
GRID_MIN_CELLS = 1 << 24
#: Coordinate keys must be smaller than this to fit an int64.
MAX_KEY = 1 << 62
#: Steps of the grids that float coordinates can lie on, coarsest first.
GRID_STEPS = (1, 0.5, 0.25, 0.1, 0.05, 0.01)
#: Largest coordinate, in steps, of a grid whose points are joined as integers.
GRID_MAX_COORDINATE = 1e4
#: Distance, in steps, within which a stencil offset is checked on the floats.
GRID_TOLERANCE = 1e-4
#: Number of pairs a k-d tree search is expected to return at a time.
KDTREE_CHUNK_PAIRS = 4_000_000
#: Number of points whose neighbours are counted to size those searches.
KDTREE_SAMPLE = 1_000


def euclidean(*coordinates):
    """
    Return the Euclidean distance between two points.

    Parameters
    ----------
    *coordinates : numbers or array-likes
        ``2N`` coordinates, the ``N`` of the first point followed by the
        ``N`` of the second one. Array-likes are processed element-wise.

    Returns
    -------
    float or array-like
        The distance, or the distances when arrays are given.

    Raises
    ------
    ValueError
        If the number of coordinates is not even and positive.
    """
    n_dims, remainder = divmod(len(coordinates), 2)
    if n_dims == 0 or remainder:
        raise ValueError(
            "euclidean takes the coordinates of two points of the same "
            f"dimension, got {len(coordinates)} coordinates"
        )
    differences = (
        a - b for a, b in zip(coordinates[:n_dims], coordinates[n_dims:])
    )
    result = numpy.sqrt(sum(diff * diff for diff in differences))
    if isinstance(result, (numbers.Number, numpy.generic)):
        return float(result)
    return result


#: Alias of :func:`euclidean`, the name under which programs know it.
euclidean_distance = euclidean


def euclidean_pairs(first, second, radius, strict=True):
    """
    Return the pairs of points of two sets within a Euclidean distance.

    Parameters
    ----------
    first, second : array-like of shape (n, N) and (m, N)
        Point sets, expected without repeated points.
    radius : float
        Distance bound.
    strict : bool, optional
        Whether the distance must be lower than *radius* (the default)
        rather than lower or equal.

    Returns
    -------
    tuple of numpy.ndarray
        The index in *first* of the point of each pair, its index in
        *second* and their distance.
    """
    first, second = _as_points(first), _as_points(second)
    compare = operator.lt if strict else operator.le
    swap = len(first) > len(second)
    small, large = (second, first) if swap else (first, second)
    with log_performance(
        LOG, "euclidean pairs of %s x %s points within %s",
        (len(small), len(large), radius),
    ):
        small_index, large_index, distances = _pairs(
            small, large, radius, compare
        )
    LOG.info("%s pairs found", len(distances))
    if swap:
        return large_index, small_index, distances
    return small_index, large_index, distances


def euclidean_join(
    first, second, radius, strict=True, columns=None, distance=None
):
    """
    Return the rows of two tables whose points are within a distance.

    Each row of *first* is paired with each row of *second* whose point is
    at a distance lower than *radius* (or, non-strictly, at most
    *radius*) from its own. The result is projected on *columns*, without
    repeated rows.

    Parameters
    ----------
    first, second : tuple
        Each table, a pandas.DataFrame without repeated rows, and the names
        of its columns holding the coordinates of its point. The names of
        the columns of the two tables are distinct.
    radius : float
        Distance bound.
    strict : bool, optional
        Whether the distance must be lower than *radius* (the default)
        rather than lower or equal.
    columns : sequence of str, optional
        Columns of the result, among those of the tables and *distance*.
        All of them by default.
    distance : str, optional
        Name of the column of the distance between the points.

    Returns
    -------
    pandas.DataFrame
        The distinct rows of the join, with the *columns*.
    """
    if columns is None:
        columns = [*first[0].columns, *second[0].columns]
        columns += [] if distance is None else [distance]
    small, large = sorted(
        (_restrict(*first, columns), _restrict(*second, columns)),
        key=lambda table: len(table[0]),
    )
    compare = operator.lt if strict else operator.le
    with log_performance(
        LOG, "euclidean join of %s x %s rows within %s",
        (len(small[0]), len(large[0]), radius),
    ):
        indices = _row_pairs(
            small, large, radius, compare, distance in columns
        )
        if not set(small[1]).union(large[1]) <= set(columns):
            indices = _distinct_pairs(
                small, large, indices, columns, distance
            )
        result = _join_columns(small, large, indices, columns, distance)
    LOG.info("%s rows joined", len(result))
    return result


def _restrict(frame, point, columns):
    """
    Return a table restricted to its point and its result columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows of the table.
    point : sequence of str
        Names of the coordinate columns.
    columns : sequence of str
        Result columns.

    Returns
    -------
    tuple
        The distinct rows of the restricted table and *point*.
    """
    kept = [name for name in frame.columns if name in point or name in columns]
    if len(kept) < len(frame.columns):
        frame = frame[kept].drop_duplicates()
    return frame, list(point)


def _row_pairs(small, large, radius, compare, with_distances=True):
    """
    Return the pairs of rows of two tables whose points are within *radius*.

    Parameters
    ----------
    small, large : tuple
        Each table and the names of its coordinate columns, *small* having
        fewer rows.
    radius : float
        Distance bound.
    compare : callable
        ``operator.lt`` or ``operator.le``.
    with_distances : bool, optional
        Whether the distances are returned, rather than None.

    Returns
    -------
    tuple
        The row in *small* of each pair, its row in *large* and the
        distance between their points, as numpy arrays.
    """
    points, groups = _distinct_points(*large)
    small_rows, large_points, distances = _pairs(
        _as_points(small[0][small[1]]), points, radius, compare,
        with_distances,
    )
    if groups is None:
        return small_rows, large_points, distances
    repeat, large_rows = _expand(large_points, groups)
    small_rows = small_rows[repeat]
    if distances is not None:
        distances = distances[repeat]
    return small_rows, large_rows, distances


def _distinct_points(frame, point):
    """
    Return the distinct points of a table and the rows of each one.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows of the table.
    point : list of str
        Names of the coordinate columns.

    Returns
    -------
    tuple
        The points, one per row, and the order of the rows sorted by point
        with the start and the number of rows of each point in it, or None
        when each row is a point.
    """
    coordinates = _as_points(frame[point])
    if len(point) == len(frame.columns):
        return coordinates, None
    point_of_row = frame.groupby(point, sort=False).ngroup().to_numpy()
    order = numpy.argsort(point_of_row, kind="stable")
    counts = numpy.bincount(point_of_row, minlength=1)
    starts = numpy.cumsum(counts) - counts
    return coordinates[order[starts]], (order, starts, counts)


def _expand(point_index, groups):
    """
    Return each element of *point_index* repeated for the rows of its point.

    Parameters
    ----------
    point_index : numpy.ndarray
        Point of each element.
    groups : tuple of numpy.ndarray
        Order of the rows sorted by point, and the start and the number of
        rows of each point in it.

    Returns
    -------
    tuple of numpy.ndarray
        The element and the row of each output.
    """
    order, starts, counts = groups
    sizes = counts[point_index]
    total = int(sizes.sum())
    index = _index_type(max(total, len(order)))
    repeat = numpy.repeat(numpy.arange(len(point_index), dtype=index), sizes)
    # The rows of each point follow from its start in *order*.
    rows = numpy.arange(total, dtype=index)
    rows += numpy.repeat((starts[point_index] - numpy.cumsum(sizes) + sizes)
                         .astype(index), sizes)
    return repeat, order.astype(index, copy=False)[rows]


def _index_type(size):
    """Return the smallest integer type numbering *size* elements."""
    return numpy.int32 if size < 2 ** 31 else numpy.int64


def _distinct_pairs(small, large, indices, columns, distance):
    """
    Return pairs of rows giving each distinct result row once.

    Parameters
    ----------
    small, large : tuple
        Each table and the names of its coordinate columns.
    indices : tuple of numpy.ndarray
        The rows of the pairs in *small*, in *large* and their distances.
    columns : sequence of str
        Result columns.
    distance : str or None
        Name of the distance column.

    Returns
    -------
    tuple of numpy.ndarray
        The rows of the distinct pairs and their distances.
    """
    (small_codes, small_rows, n_small), (large_codes, large_rows, n_large) = (
        _value_codes(table[0], columns) for table in (small, large)
    )
    if distance in columns or n_small * n_large >= MAX_KEY:
        rows = _join_columns(small, large, indices, columns, distance)
        keep = numpy.flatnonzero(~rows.duplicated().to_numpy())
        return tuple(index[keep] for index in indices)
    keys = _take(large_codes, indices[1]).astype(numpy.int64)
    keys *= n_small
    keys += _take(small_codes, indices[0])
    keys.sort()
    first = numpy.ones(len(keys), dtype=bool)
    numpy.not_equal(keys[1:], keys[:-1], out=first[1:])
    large_code, small_code = numpy.divmod(keys[first], n_small)
    return _take(small_rows, small_code), _take(large_rows, large_code), None


def _value_codes(frame, columns):
    """
    Return codes of the distinct values of the result columns of a table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows of the table.
    columns : sequence of str
        Result columns.

    Returns
    -------
    tuple
        The code of each row, a row of each code and the number of codes;
        the codes and the rows are None when the codes are the rows.
    """
    selected = [name for name in frame.columns if name in columns]
    if len(selected) == len(frame.columns):
        return None, None, len(frame)
    if not selected:
        zeros = numpy.zeros(len(frame), dtype=int)
        return zeros, zeros[:1], 1
    codes = frame.groupby(selected, sort=False).ngroup().to_numpy()
    rows = numpy.unique(codes, return_index=True)[1]
    return codes, rows, len(rows)


def _take(values, index):
    """
    Return the *values* at *index*, *index* itself if *values* is None.

    Parameters
    ----------
    values : numpy.ndarray or None
        Values, None standing for the identity.
    index : numpy.ndarray
        Positions.

    Returns
    -------
    numpy.ndarray
        The values at the positions.
    """
    return index if values is None else values[index]


def _join_columns(small, large, indices, columns, distance):
    """
    Return the result columns of pairs of rows.

    Parameters
    ----------
    small, large : tuple
        Each table and the names of its coordinate columns.
    indices : tuple of numpy.ndarray
        The rows of the pairs in *small*, in *large* and their distances.
    columns : sequence of str
        Result columns.
    distance : str or None
        Name of the distance column.

    Returns
    -------
    pandas.DataFrame
        The result columns of the pairs.
    """
    values = {distance: indices[2]}
    for (frame, _), rows in zip((small, large), indices):
        values.update({
            name: frame[name].to_numpy()[rows]
            for name in frame.columns if name in columns
        })
    # The columns are new arrays: the frame holds them without a copy.
    return pandas.DataFrame(
        {name: values[name] for name in columns}, copy=False
    )


def _as_points(points):
    """
    Return *points* as a 2-D numpy array with a numeric type if possible.

    Parameters
    ----------
    points : array-like
        Point coordinates, one point per row.

    Returns
    -------
    numpy.ndarray
        The points, one per row.
    """
    points = numpy.asarray(points)
    if points.dtype == object and points.size > 0:
        points = numpy.array(points.tolist())
    if points.ndim != 2:
        points = points.reshape(len(points), -1)
    return points


def _pairs(small, large, radius, compare, with_distances=True):
    """
    Return the pairs of points within *radius*, by the fastest means.

    Parameters
    ----------
    small, large : numpy.ndarray
        Point sets, *small* being the smaller one; only *small* may
        repeat points.
    radius : float
        Distance bound.
    compare : callable
        ``operator.lt`` or ``operator.le``, comparing distances with
        *radius*.
    with_distances : bool, optional
        Whether the distances are returned, rather than None.

    Returns
    -------
    tuple of numpy.ndarray
        Indices of the points of *small*, of *large*, pairwise, and their
        distances.
    """
    if len(small) == 0 or len(large) == 0 or not radius >= 0:
        empty = numpy.empty(0, dtype=int)
        return empty, empty, numpy.empty(0) if with_distances else None
    pairs = None
    stencil = _integer_stencil(small, large, radius, compare)
    if stencil is not None:
        pairs = _stencil_pairs(small, large, *stencil, with_distances)
    if pairs is None:
        pairs = _grid_pairs(small, large, radius, compare, with_distances)
    if pairs is None:
        pairs = _kdtree_pairs(small, large, radius, compare, with_distances)
    return pairs


def _integer_stencil(small, large, radius, compare):
    """
    Return the integer offsets within *radius*, when worth using them.

    Parameters
    ----------
    small, large : numpy.ndarray
        Point sets, *small* being the smaller one; only *small* may
        repeat points.
    radius : float
        Distance bound.
    compare : callable
        ``operator.lt`` or ``operator.le``.

    Returns
    -------
    tuple of numpy.ndarray or None
        The offsets, one per row, and their lengths, or None when the
        points are not integers or a k-d tree is expected to be cheaper.
    """
    integers = all(
        numpy.issubdtype(points.dtype, numpy.integer)
        for points in (small, large)
    )
    if not integers or math.isinf(radius):
        return None
    reach = int(math.floor(radius))
    n_dims = small.shape[1]
    if (2 * reach + 1) ** n_dims > STENCIL_MAX_CELLS:
        return None
    axis = numpy.arange(-reach, reach + 1)
    cube = numpy.stack(
        numpy.meshgrid(*([axis] * n_dims), indexing="ij"), axis=-1
    ).reshape(-1, n_dims)
    lengths = euclidean_distance(*cube.T, *numpy.zeros_like(cube).T)
    keep = compare(lengths, radius)
    work = len(small) * int(keep.sum())
    cells = _key_space(large, reach)[2]
    dense = cells <= STENCIL_GRID_CELLS_PER_POINT * len(large) and (
        _uses_grid(cells, len(large))
    )
    if not dense and work > max(
        STENCIL_MIN_WORK, STENCIL_WORK_PER_POINT * len(large)
    ):
        return None
    return cube[keep], lengths[keep]


def _stencil_pairs(small, large, offsets, lengths, with_distances=True):
    """
    Return the points of *large* at the *offsets* of points of *small*.

    Points are encoded as integer keys over the bounding box of *large*,
    padded so that the points of *small* that can have a neighbour stay
    within it once shifted by any offset.

    Parameters
    ----------
    small, large : numpy.ndarray
        Integer point sets; only *small* may repeat points.
    offsets : numpy.ndarray
        Integer offsets, one per row.
    lengths : numpy.ndarray
        Lengths of the offsets.
    with_distances : bool, optional
        Whether the distances are returned, rather than None.

    Returns
    -------
    tuple of numpy.ndarray or None
        Indices of the points of *small*, of *large*, pairwise, and their
        distances, or None when the coordinates of *large* cannot be
        encoded as keys.
    """
    reach = int(numpy.abs(offsets).max(initial=0))
    low, high, size = _key_space(large, reach)
    if size >= MAX_KEY:
        return None
    extent = high - low + 1
    strides = numpy.append(numpy.cumprod(extent[:0:-1])[::-1], 1)
    lookup = _index_lookup((large - low) @ strides, size)
    near = numpy.flatnonzero(
        ((small >= low + reach) & (small <= high - reach)).all(axis=1)
    ).astype(_index_type(len(small)))
    keys = (small[near] - low) @ strides
    found = [
        _found(lookup(keys + shift), near) for shift in offsets @ strides
    ]
    counts = [len(small_found) for small_found, _ in found]
    small_index, large_index = _concatenate(found)
    distances = numpy.repeat(lengths, counts) if with_distances else None
    return small_index, large_index, distances


def _grid_step(*point_sets):
    """
    Return the step of the grid the points of float sets lie on, if any.

    Parameters
    ----------
    *point_sets : numpy.ndarray
        Point sets.

    Returns
    -------
    float or None
        The coarsest of :data:`GRID_STEPS` of which all the coordinates are
        multiples, up to rounding errors, or None.
    """
    for step in GRID_STEPS:
        for points in point_sets:
            steps = points / step
            largest = numpy.abs(steps).max(initial=0)
            error = numpy.abs(steps - numpy.rint(steps)).max(initial=0)
            if largest > GRID_MAX_COORDINATE or error > 1e-9 * max(1, largest):
                break
        else:
            return step
    return None


def _grid_pairs(small, large, radius, compare, with_distances):
    """
    Return the pairs of points of float grids within *radius*, as integers.

    The points are numbered by their position in the grid, and their
    neighbours looked up with a stencil of integer offsets. The pairs whose
    offset is too close to *radius* to be told from the rounding errors of
    the floats are checked on the floats.

    Parameters
    ----------
    small, large : numpy.ndarray
        Point sets, *small* being the smaller one; only *small* may
        repeat points.
    radius : float
        Distance bound.
    compare : callable
        ``operator.lt`` or ``operator.le``.
    with_distances : bool
        Whether the distances are returned, rather than None.

    Returns
    -------
    tuple of numpy.ndarray or None
        Indices of the points of *small*, of *large*, pairwise, and their
        distances, or None when the points are not on a grid, or a stencil
        is not worth using.
    """
    if all(
        numpy.issubdtype(points.dtype, numpy.integer)
        for points in (small, large)
    ):
        return None
    step = _grid_step(small, large)
    if step is None or math.isinf(radius):
        return None
    cells = [numpy.rint(points / step).astype(numpy.int64)
             for points in (small, large)]
    in_steps = radius / step
    stencil = _integer_stencil(
        *cells, in_steps + GRID_TOLERANCE, numpy.less_equal
    )
    if stencil is None:
        return None
    offsets, lengths = stencil
    sure = lengths < in_steps - GRID_TOLERANCE
    inner = _stencil_pairs(*cells, offsets[sure], lengths[sure], False)
    edge = _stencil_pairs(*cells, offsets[~sure], lengths[~sure], False)
    if inner is None or edge is None:
        return None
    exact = _exact_distances(small, large, edge[0], edge[1])
    keep = compare(exact, radius)
    chunks = [inner[:2], (edge[0][keep], edge[1][keep])]
    small_index, large_index = _concatenate(chunks)
    distances = None
    if with_distances:
        distances = _exact_distances(small, large, small_index, large_index)
    return small_index, large_index, distances


def _exact_distances(small, large, small_index, large_index, block=1 << 20):
    """
    Return the distances of pairs of points, by blocks of pairs.

    Parameters
    ----------
    small, large : numpy.ndarray
        Point sets.
    small_index, large_index : numpy.ndarray
        Indices of the points of each pair.
    block : int, optional
        Number of pairs whose coordinates are held at once.

    Returns
    -------
    numpy.ndarray
        The distance of each pair.
    """
    distances = numpy.empty(len(small_index))
    for start in range(0, len(small_index), block):
        rows = slice(start, start + block)
        distances[rows] = euclidean_distance(
            *small[small_index[rows]].T, *large[large_index[rows]].T
        )
    return distances


def _key_space(points, reach):
    """
    Return the box encoding the stencil neighbours of the points as keys.

    Parameters
    ----------
    points : numpy.ndarray
        Integer points.
    reach : int
        Largest coordinate of the stencil offsets, the box being padded
        by twice this on each side.

    Returns
    -------
    tuple
        The lowest and highest corners of the box and its number of cells.
    """
    low = points.min(axis=0) - 2 * reach
    high = points.max(axis=0) + 2 * reach
    return low, high, math.prod(int(e) for e in high - low + 1)


def _uses_grid(size, n_keys):
    """
    Return whether keys of a key space of *size* are looked up in a grid.

    Parameters
    ----------
    size : int
        Size of the key space.
    n_keys : int
        Number of keys.

    Returns
    -------
    bool
        Whether the keys are looked up in an occupancy grid rather than
        searched in sorted order.
    """
    return size <= max(GRID_MIN_CELLS, GRID_CELLS_PER_POINT * n_keys)


def _concatenate(chunks):
    """
    Return the arrays of the chunks of pairs, each chunk released once copied.

    Concatenating the chunks while they are all alive would hold the pairs
    twice; the list *chunks* is emptied instead, as its chunks are copied
    into the result.

    Parameters
    ----------
    chunks : list of tuple of numpy.ndarray
        Chunks, each the indices of the points of two sets, pairwise,
        optionally followed by the distances; the chunks are all alike.

    Returns
    -------
    tuple of numpy.ndarray
        The arrays of the chunks, in the order of the chunks.
    """
    if not chunks:
        return numpy.empty(0, dtype=int), numpy.empty(0, dtype=int)
    total = sum(len(chunk[0]) for chunk in chunks)
    result = tuple(
        numpy.empty(total, dtype=array.dtype) for array in chunks[0]
    )
    end = total
    while chunks:
        chunk = chunks.pop()
        for out, array in zip(result, chunk):
            out[end - len(array):end] = array
        end -= len(chunk[0])
    return result


def _found(index, near):
    """
    Return the points of a stencil look-up that found a neighbour.

    Parameters
    ----------
    index : numpy.ndarray
        Position of the neighbour of each looked-up point, -1 if missing.
    near : numpy.ndarray
        Index of each looked-up point.

    Returns
    -------
    tuple of numpy.ndarray
        The index of the points with a neighbour and the position of their
        neighbour.
    """
    hit = index >= 0
    return near[hit], index[hit]


def _index_lookup(keys, size):
    """
    Return a vectorised lookup of the position of integer keys.

    Parameters
    ----------
    keys : numpy.ndarray
        Distinct integer keys, all lower than *size*.
    size : int
        Size of the key space.

    Returns
    -------
    callable
        Function mapping an array of keys to their positions in *keys*,
        -1 for those missing.
    """
    index_type = _index_type(len(keys))
    if _uses_grid(size, len(keys)):
        grid = numpy.full(size, -1, dtype=index_type)
        grid[keys] = numpy.arange(len(keys), dtype=index_type)
        return grid.__getitem__
    order = numpy.argsort(keys)
    sorted_keys = keys[order]

    def lookup(query):
        position = numpy.searchsorted(sorted_keys, query)
        position = numpy.minimum(position, len(sorted_keys) - 1)
        return numpy.where(sorted_keys[position] == query, order[position], -1)

    return lookup


def _kdtree_pairs(small, large, radius, compare, with_distances=True):
    """
    Return the pairs of points within *radius* using k-d trees.

    The trees are searched with a slightly larger radius, so that their
    rounding errors lose no pair, and the exact bound applied afterwards.
    The points of *small* are searched in chunks expected to return
    :data:`KDTREE_CHUNK_PAIRS` pairs, so that the trees' pairs, which are
    more than those within *radius* and several arrays each, are never all
    alive at once.

    Parameters
    ----------
    small, large : numpy.ndarray
        Point sets, *small* being the smaller one; only *small* may
        repeat points.
    radius : float
        Distance bound.
    compare : callable
        ``operator.lt`` or ``operator.le``.
    with_distances : bool, optional
        Whether the distances are returned, rather than None.

    Returns
    -------
    tuple of numpy.ndarray
        Indices of the points of *small*, of *large*, pairwise, and their
        distances.
    """
    search_radius = radius * (1 + 1e-9) + 1e-9
    low = small.min(axis=0) - search_radius
    high = small.max(axis=0) + search_radius
    near = numpy.flatnonzero(((large >= low) & (large <= high)).all(axis=1))
    index_type = _index_type(max(len(small), len(large)))
    chunks = []
    if len(near) > 0:
        tree = scipy.spatial.cKDTree(large[near])
        size = _kdtree_chunk_size(tree, small, search_radius)

        def search(start):
            points = small[start:start + size]
            pairs = scipy.spatial.cKDTree(points).sparse_distance_matrix(
                tree, search_radius, output_type="ndarray",
            )
            small_index, large_index = pairs["i"], near[pairs["j"]]
            distances = numpy.asarray(euclidean_distance(
                *points[small_index].T, *large[large_index].T
            ), dtype=float)
            keep = compare(distances, radius)
            found = (
                (small_index[keep] + start).astype(index_type),
                large_index[keep].astype(index_type),
            )
            return found + (distances[keep],) * with_distances

        chunks = [search(start) for start in range(0, len(small), size)]
    if not chunks:
        empty = numpy.empty(0, dtype=index_type)
        chunks = [(empty, empty) + (numpy.empty(0),) * with_distances]
    arrays = _concatenate(chunks)
    return arrays if with_distances else arrays + (None,)


def _kdtree_chunk_size(tree, small, search_radius):
    """
    Return the number of points of *small* whose search returns few pairs.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        Tree of the points of the larger set.
    small : numpy.ndarray
        Point set searched.
    search_radius : float
        Radius of the searches.

    Returns
    -------
    int
        The number of points, at least one, estimated from the neighbours
        of a sample of the points.
    """
    sample = small[:: max(1, len(small) // KDTREE_SAMPLE)]
    neighbours = tree.query_ball_point(
        sample, search_radius, return_length=True
    ).mean()
    return max(1, int(KDTREE_CHUNK_PAIRS / max(neighbours, 1)))
