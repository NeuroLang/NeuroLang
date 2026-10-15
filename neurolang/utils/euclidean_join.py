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
Otherwise a k-d tree is built on the smaller set and on the points of the
larger set that lie within the radius of its bounding box.
"""

import logging
import math
import numbers
import operator

import numpy
import pandas
import scipy.spatial

from .various import log_performance

LOG = logging.getLogger(__name__)

#: Maximum number of cells of the cube enclosing the offset stencil.
STENCIL_MAX_CELLS = 1_000_000
#: Number of stencil look-ups always allowed before using a k-d tree.
STENCIL_MIN_WORK = 20_000_000
#: Number of stencil look-ups allowed per point of the larger set.
STENCIL_WORK_PER_POINT = 32
#: Number of point and offset combinations processed at once.
STENCIL_CHUNK = 1 << 22
#: Occupancy grids are used up to this many cells per point...
GRID_CELLS_PER_POINT = 4
#: ... or this many cells; sorted keys are searched otherwise.
GRID_MIN_CELLS = 1 << 24
#: Coordinate keys must be smaller than this to fit an int64.
MAX_KEY = 1 << 62


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
        indices = _row_pairs(small, large, radius, compare)
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


def _row_pairs(small, large, radius, compare):
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

    Returns
    -------
    tuple of numpy.ndarray
        The row in *small* of each pair, its row in *large* and the
        distance between their points.
    """
    points, groups = _distinct_points(*large)
    small_rows, large_points, distances = _pairs(
        _as_points(small[0][small[1]]), points, radius, compare
    )
    if groups is None:
        return small_rows, large_points, distances
    repeat, large_rows = _expand(large_points, groups)
    return small_rows[repeat], large_rows, distances[repeat]


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
    repeat = numpy.repeat(numpy.arange(len(point_index)), sizes)
    position = numpy.arange(len(repeat)) - numpy.repeat(
        numpy.cumsum(sizes) - sizes, sizes
    )
    return repeat, order[numpy.repeat(starts[point_index], sizes) + position]


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
    keys = numpy.sort(
        _take(large_codes, indices[1]) * n_small
        + _take(small_codes, indices[0])
    )
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
    return pandas.DataFrame({name: values[name] for name in columns})


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


def _pairs(small, large, radius, compare):
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

    Returns
    -------
    tuple of numpy.ndarray
        Indices of the points of *small*, of *large*, pairwise, and their
        distances.
    """
    if len(small) == 0 or len(large) == 0 or not radius >= 0:
        empty = numpy.empty(0, dtype=int)
        return empty, empty, numpy.empty(0)
    pairs = None
    stencil = _integer_stencil(small, large, radius, compare)
    if stencil is not None:
        pairs = _stencil_pairs(small, large, *stencil)
    if pairs is None:
        pairs = _kdtree_pairs(small, large, radius, compare)
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
    if work > max(STENCIL_MIN_WORK, STENCIL_WORK_PER_POINT * len(large)):
        return None
    return cube[keep], lengths[keep]


def _stencil_pairs(small, large, offsets, lengths):
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

    Returns
    -------
    tuple of numpy.ndarray or None
        Indices of the points of *small*, of *large*, pairwise, and their
        distances, or None when the coordinates of *large* cannot be
        encoded as keys.
    """
    reach = int(numpy.abs(offsets).max(initial=0))
    low = large.min(axis=0) - 2 * reach
    high = large.max(axis=0) + 2 * reach
    extent = high - low + 1
    size = math.prod(int(e) for e in extent)
    if size >= MAX_KEY:
        return None
    strides = numpy.append(numpy.cumprod(extent[:0:-1])[::-1], 1)
    lookup = _index_lookup((large - low) @ strides, size)
    near = numpy.flatnonzero(
        ((small >= low + reach) & (small <= high - reach)).all(axis=1)
    )
    keys = (small[near] - low) @ strides
    shifts = offsets @ strides
    block = max(1, STENCIL_CHUNK // max(len(near), 1))
    found = [(numpy.empty(0, dtype=int),) * 3]
    for start in range(0, len(offsets), block):
        index = lookup(keys[:, None] + shifts[None, start:start + block])
        point, offset = numpy.nonzero(index >= 0)
        found.append((near[point], index[point, offset], offset + start))
    small_index, large_index, offset_index = (
        numpy.concatenate(arrays) for arrays in zip(*found)
    )
    return small_index, large_index, lengths[offset_index]


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
    index_type = numpy.int32 if len(keys) < 2 ** 31 else numpy.int64
    if size <= max(GRID_MIN_CELLS, GRID_CELLS_PER_POINT * len(keys)):
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


def _kdtree_pairs(small, large, radius, compare):
    """
    Return the pairs of points within *radius* using k-d trees.

    The trees are searched with a slightly larger radius, so that their
    rounding errors lose no pair, and the exact bound applied afterwards.

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
    tuple of numpy.ndarray
        Indices of the points of *small*, of *large*, pairwise, and their
        distances.
    """
    search_radius = radius * (1 + 1e-9) + 1e-9
    low = small.min(axis=0) - search_radius
    high = small.max(axis=0) + search_radius
    near = numpy.flatnonzero(((large >= low) & (large <= high)).all(axis=1))
    pairs = numpy.empty(0, dtype=[("i", int), ("j", int)])
    if len(near) > 0:
        pairs = scipy.spatial.cKDTree(small).sparse_distance_matrix(
            scipy.spatial.cKDTree(large[near]), search_radius,
            output_type="ndarray",
        )
    small_index, large_index = pairs["i"], near[pairs["j"]]
    distances = numpy.asarray(
        euclidean_distance(*small[small_index].T, *large[large_index].T),
        dtype=float,
    )
    keep = compare(distances, radius)
    return small_index[keep], large_index[keep], distances[keep]
