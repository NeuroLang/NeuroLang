from functools import lru_cache
from itertools import product
from typing import Collection

import numpy as np
import pandas as pd
from scipy.linalg import kron

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


def cardinal_relation(region, reference_region, directions, refine_overlapping=False,
    stop_at=None):
    if region is reference_region:
        return False

    if type(region) is Region and type(reference_region) is Region:
        mat = direction_matrix_int(region.bounding_box.intervals,
                               reference_region.bounding_box.intervals)
        return is_in_direction(mat, directions)

    region, reference_region = cardinal_relation_prepare_regions(
        region, reference_region
    )

    if region == reference_region:
        return directions == 'O'
    
    # 1. first check the bounding boxes
    mat = direction_matrix_int(region.bounding_box.intervals,
                            reference_region.bounding_box.intervals)
    # 2. if they don't overlap, easy, check if the boxes are in the right direction
    if not is_in_direction(mat, 'O') or not refine_overlapping:
        return is_in_direction(mat, directions)
    
    # 3. if they overlap, we check the product of all voxels until we find a pair
    # that is in the right direction
    indices = is_in_direction_indices(mat.ndim, directions)
    for voxel, other_voxel in product(region.intervals, reference_region.intervals):
        mat = direction_matrix_int(voxel, other_voxel)
        if np.any(mat[indices] == 1):
            return True
    
    return False


def cardinal_relation_2(
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


def relation_vectors_intervals(intervals: Collection[pd.Interval], other_intervals: Collection[pd.Interval]):
    obtained_vectors = np.empty(
        (min(len(intervals), len(other_intervals)), 3)
    )
    ov = 0
    for interval, other_interval in zip(
        intervals, other_intervals
    ):
        if not interval.overlaps(other_interval):
            if interval < other_interval:
                obtained_vectors[ov] = (1, 0, 0)
            else:
                obtained_vectors[ov] = (0, 0, 1)
        else:
            left = interval.left not in other_interval and (interval.left != other_interval.left or other_interval.closed_left)
            right = interval.right not in other_interval and (interval.right != other_interval.right or other_interval.closed_right)
            res = (1 if left else 0, 1, 1 if right else 0)
            obtained_vectors[ov] = res
        ov += 1
    return obtained_vectors[:ov]


def relation_vectors(intervals, other_region_intervals):
    """
    intervals = [[0., 7.],
                 [0., 7.],
                 [0., 2.]]
    = interval_x, interval_y, interval_z
    
    renvoie 
                [[0, 1, 0],
                 [1, 0, 1],
                 [0, 0, 1]]
    """
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


def direction_matrix_int(intervals: Collection[pd.Interval], other_intervals: Collection[pd.Interval]):
    rp_vector = relation_vectors_intervals(intervals, other_intervals)
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(rp_vector)):
        tensor = kron(
            rp_vector[i].reshape((3, ) + (1, ) * i),
            tensor
        ).squeeze()
    return tensor.clip(0, 1)


def direction_matrix(region_bb, another_region_bb):
    rp_vector = relation_vectors(region_bb.limits, another_region_bb.limits)
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(rp_vector)):
        tensor = kron(
            rp_vector[i].reshape((3, ) + (1, ) * i),
            tensor
        ).squeeze()
    return tensor.clip(0, 1)
