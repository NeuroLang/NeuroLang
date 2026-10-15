"""
Pairs of points within a Euclidean distance, computed with numpy.

This is the kernel of the Euclidean spatial-bound syntactic sugar (see
:mod:`neurolang.frontend.datalog.sugar.spatial`). Given two point sets
and a radius it returns every pair of points, one from each set, whose
distance is below (or, non-strictly, at most) the radius. The distances
are those :func:`euclidean_distance` computes, so that the pairs are
exactly the ones the ``euclidean`` built-in would select.

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
        Point sets, *small* being the smaller one.
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
        Point sets, *small* being the smaller one.
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
        Integer point sets.
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
        Point sets, *small* being the smaller one.
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
