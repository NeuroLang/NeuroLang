from functools import lru_cache
from itertools import product

import numpy as np
from scipy.linalg import kron
from ncls import FNCLS

from .interval_algebra import (v_before, v_during, v_equals, v_finishes,
                               v_meets, v_overlaps, v_starts)
from .regions import ExplicitVBR, ImplicitVBR, Region

__all__ = ['cardinal_relation']

directions_dim_space = {
    'L': [0],
    'R': [0],
    'P': [1],
    'A': [1],
    'O': [0, 1, 2],
    'I': [2],
    'S': [2],
    'F': [3]
}

matrix_positions_per_directions = {
    'L': [0],
    'O': [1],
    'R': [2],
    'P': [0],
    'A': [2],
    'I': [0],
    'S': [2],
    'F': [2]
}

inverse_directions = {
    'R': 'L',
    'A': 'P',
    'S': 'I',
    'O': 'O',
    'L': 'R',
    'P': 'A',
    'I': 'S'
}


relations = [
    v_before, v_overlaps, v_during, v_meets, v_starts, v_finishes, v_equals
]


def fast_overlaps(region, other_region):
    """
    A region overlaps with another region if at least one of its voxels
    overlaps with a voxel from the other region. They do not overlap if no
    voxels overlap.
    """
    if len(region.intervals[0]) < len(other_region.intervals[0]):
        r0, r1 = region, other_region
    else:
        r0, r1 = other_region, region
    # 1. loop on each voxel of the smallest region
    for row in zip(*r0.intervals):
        # 2. for each voxel, compare its x, y, z intervals with
        # the intervals of all voxels in the other region.
        mask = None
        for i, c in enumerate(row):
            if mask is None:
                mask = r1.intervals[i].overlaps(c)
            else:
                mask &= r1.intervals[i].overlaps(c)
            if not any(mask):
                break
        overlaps = any(mask)
        if overlaps:
            break
    return overlaps


def vectorized_overlaps(region, other_region):
    # Create the cross product of x, y, z intervals from region and other_region
    prod = region.intervals_df.merge(
        other_region.intervals_df, how="cross", suffixes=("_left", "_right")
    )
    # Find elements for which x, y, and z intervals from left and right df overlap.
    # this is equivalent to using pd.arrays.IntervalArray.overlaps method :
    # https://github.com/pandas-dev/pandas/blob/5a404d5b70c6c611b204ba27b1d5c96a3b58f956/pandas/core/arrays/interval.py#L1275
    overlaps = (
        (prod.x_right.array.left <= prod.x_left.array.right)
        & (prod.x_left.array.left <= prod.x_right.array.right)
        & (prod.y_right.array.left <= prod.y_left.array.right)
        & (prod.y_left.array.left <= prod.y_right.array.right)
        & (prod.z_right.array.left <= prod.z_left.array.right)
        & (prod.z_left.array.left <= prod.z_right.array.right)
    )
    return overlaps.any()


def ncls_overlaps(region, other_region):
    """
    Use NCLS for querying overlaps between regions
    """
    if region.voxels_xyz.shape[0] < other_region.voxels_xyz.shape[0]:
        r0, r1 = region, other_region
    else:
        r0, r1 = other_region, region

    intersects = None
    for i in range(len(r0.ncls)):
        ncls = r0.ncls[i]
        starts = r1.voxels_xyz[:, i].astype(np.double) + 10000
        ends = starts + r1.affine[i, i]
        ids = np.arange(r1.voxels_xyz.shape[0])
        l_idx, r_ids = ncls.all_overlaps_both(starts, ends, ids)
        if intersects is None:
            intersects = set(zip(l_idx, r_ids))
        else:
            intersects &= set(zip(l_idx, r_ids))
        if len(intersects) == 0:
            break
    return len(intersects) > 0


def ncls_overlaps_3(region, other_region):
    """
    Same as ncls_overlaps but creates the NCLS tree in the loop.
    """
    if region.voxels_xyz.shape[0] > other_region.voxels_xyz.shape[0]:
        r0, r1 = region, other_region
    else:
        r0, r1 = other_region, region

    eps = 0.00001
    intersects = None
    r_ids = None
    for i in range(r0.voxels_xyz.shape[1]):
        r_starts = r0.voxels_xyz[:, i].astype(np.double) + 10000
        r_ends = r_starts + r0.affine[i, i]
        if r_ids is None:
            r_ids = np.arange(r0.voxels_xyz.shape[0])
        else:
            r_ids = np.array(list(set(r_ids))).astype(int)
            r_starts = r_starts[r_ids]
            r_ends = r_ends[r_ids]
        ncls = FNCLS(r_starts - eps, r_ends + eps, r_ids)
        l_starts = r1.voxels_xyz[:, i].astype(np.double) + 10000
        l_ends = l_starts + r1.affine[i, i]
        l_ids = np.arange(r1.voxels_xyz.shape[0])
        l_idx, r_ids = ncls.all_overlaps_both(l_starts, l_ends, l_ids)
        if intersects is None:
            intersects = set(zip(l_idx, r_ids))
        else:
            intersects &= set(zip(l_idx, r_ids))
        if len(intersects) == 0:
            break
    return len(intersects) > 0


def pygeos_overlaps(region, other_region):
    intersects = None
    for i in range(len(region.trees)):
        tree = region.trees[i]
        query_lines = other_region.lines[i]
        l_idx, r_ids = tree.query_bulk(query_lines)
        if intersects is None:
            intersects = set(zip(l_idx, r_ids))
        else:
            intersects &= set(zip(l_idx, r_ids))
        if len(intersects) == 0:
            break
    return len(intersects) > 0


def cardinal_relation_fast(
    region,
    reference_region,
    directions,
    refine_overlapping=False,
    stop_at=None,
):
    if region is reference_region:
        return False

    if type(region) is Region and type(reference_region) is Region:
        mat = direction_matrix(
            region.bounding_box,
            reference_region.bounding_box,
        )
        return is_in_direction(mat, directions)

    region, reference_region = cardinal_relation_prepare_regions(
        region, reference_region
    )

    if region == reference_region:
        return directions == "O"

    # 1. first check the bounding boxes
    mat = direction_matrix(region.bounding_box, reference_region.bounding_box)
    # 2. if they don't overlap, easy, check if the boxes are in the right direction
    if not is_in_direction(mat, "O") or not refine_overlapping:
        return is_in_direction(mat, directions)

    # 3. if they overlap, we have to check if they actually do overlap
    actual_overlap = vectorized_overlaps(region, reference_region)
    if directions == "O":
        return actual_overlap

    return is_in_direction(mat, directions)


def cardinal_relation(
    region,
    reference_region,
    directions,
    refine_overlapping=False,
    stop_at=None
):
    if region is reference_region:
        return False

    if type(region) is Region and type(reference_region) is Region:
        mat = direction_matrix(region.bounding_box,
                               reference_region.bounding_box)
        return is_in_direction(mat, directions)

    region, reference_region = cardinal_relation_prepare_regions(
        region, reference_region
    )

    if region == reference_region:
        result = directions == 'O'
    else:
        mat = direction_matrix(region.bounding_box,
                               reference_region.bounding_box)
        if not (refine_overlapping and 'O' in directions) and is_in_direction(
            mat, directions
        ):
            result = True
        else:
            overlap = is_in_direction(mat, 'O')
            if (
                    overlap and
                    refine_overlapping and
                    isinstance(region, ExplicitVBR)
            ):
                mat = overlap_resolution(
                        region, reference_region, directions, stop_at
                    )
            result = is_in_direction(mat, directions)

    return result


def cardinal_relation_prepare_regions(region, reference_region):
    if isinstance(region, ImplicitVBR):
        if isinstance(reference_region, ImplicitVBR):
            raise NotImplementedError(
                f'Comparison between two implicit regions '
                f'can\'t be performed: {region}, {reference_region}'
            )
        region = region.to_explicit_vbr(
            reference_region.affine, reference_region.image_dim
        )
    if isinstance(reference_region, ImplicitVBR):
        reference_region = reference_region.to_explicit_vbr(
            region.affine, region.image_dim
        )

    return region, reference_region


def overlap_resolution(
    region, reference_region, directions=None, stop_at=None
):
    if stop_at == 0:
        raise ValueError("stop_at must be larger than 0")

    region_stack = [
        (region.aabb_tree.root, reference_region.aabb_tree.root, 0)
    ]

    directions = directions.replace('O', '')
    if len(directions) == 0:
        directions = None

    total_mat = direction_matrix(
        region_stack[0][0].box, region_stack[0][1].box
    ) * 0
    overlap_indices = is_in_direction_indices(total_mat.ndim, 'O')
    while region_stack:
        region, reference_region, level = region_stack.pop()

        mat = direction_matrix(region.box, reference_region.box)
        total_mat += mat

        if (
            is_in_direction(mat, 'O') and
            (stop_at is None or level < stop_at - 1)
        ):
            region_nl = region.children
            ref_region_nl = reference_region.children
            if len(region_nl) == 0 and len(ref_region_nl) > 0:
                region_nl = (region,)
            elif len(region_nl) > 0 and len(ref_region_nl) == 0:
                ref_region_nl = (reference_region,)

            stack_update = [
                (r, ref, level + 1)
                for r, ref in product(region_nl, ref_region_nl)
            ]

            if len(stack_update) > 0:
                total_mat[overlap_indices] = 0
                region_stack += stack_update
            elif directions is None:
                break

        elif directions is not None and is_in_direction(mat, directions):
            break

    return total_mat.clip(0, 1)


@lru_cache(maxsize=128)
def is_in_direction_indices(n, direction):
    indices = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            indices[n - 1 - dim] = matrix_positions_per_directions[i]
    return np.ix_(*indices)


def is_in_direction(matrix, direction):
    indices = is_in_direction_indices(matrix.ndim, direction)
    return np.any(matrix[indices] == 1)


def relation_vectors(intervals, other_region_intervals):
    obtained_vectors = np.empty(
        (min(len(intervals), len(other_region_intervals)), 3)
    )
    ov = 0
    for interval, other_region_interval in zip(
        intervals, other_region_intervals
    ):
        for f in relations:
            vector = f(interval, other_region_interval)
            if vector is not None:
                obtained_vectors[ov] = vector
                ov += 1
                break
    return obtained_vectors[:ov]


def direction_matrix(region_bb, another_region_bb):
    rp_vector = relation_vectors(region_bb.limits, another_region_bb.limits)
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(rp_vector)):
        tensor = kron(
            rp_vector[i].reshape((3, ) + (1, ) * i),
            tensor
        ).squeeze()
    return tensor.clip(0, 1)
