from functools import lru_cache

import numpy as np
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
        mat = direction_matrix([region.bounding_box],
                               [reference_region.bounding_box])
        return is_in_direction(mat, directions)

    region, reference_region = cardinal_relation_prepare_regions(
        region, reference_region
    )

    if region == reference_region:
        result = False
    else:
        mat = direction_matrix([region.bounding_box],
                               [reference_region.bounding_box])
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

    if not np.array_equiv(region.affine, reference_region.affine):
        region = ExplicitVBR(
            region.to_ijk(reference_region.affine),
            affine_matrix=reference_region.affine,
            image_dim=reference_region.image_dim
        )

    return region, reference_region


def overlap_resolution(
    region, reference_region, directions=None, stop_at=None
):
    if stop_at == 0:
        raise ValueError("stop_at must be larger than 0")

    current_region_level = [region.aabb_tree.root]
    current_reference_region_level = [reference_region.aabb_tree.root]
    level = 0

    overlap = True

    max_depth_reached_reg = False
    max_depth_reached_ref = False

    directions = directions.replace('O', '')
    if len(directions) == 0:
        directions = None

    while (((stop_at is None) or (level < stop_at)) and
           (not (max_depth_reached_reg and max_depth_reached_ref)) and
           overlap):
        if not max_depth_reached_reg:
            current_region_next_level = tree_next_level(current_region_level)
            if current_region_next_level:
                current_region_level = current_region_next_level
            else:
                max_depth_reached_reg = True

        if not max_depth_reached_ref:
            current_ref_region_next_level = tree_next_level(
                current_reference_region_level
            )
            if current_ref_region_next_level:
                current_reference_region_level = current_ref_region_next_level
            else:
                max_depth_reached_ref = True
        mat = direction_matrix(
            [reg.box for reg in current_region_level],
            [reg.box for reg in current_reference_region_level]
        )

        if directions is not None and is_in_direction(mat, directions):
            break

        overlap = is_in_direction(mat, 'O')
        level += 1

    return mat


def tree_next_level(nodes):
    result = []
    for node in nodes:
        if node.left is not None:
            result.append(node.left)
        if node.right is not None:
            result.append(node.right)

    return result


@lru_cache(maxsize=128)
def is_in_direction_indices(n, direction):
    indices = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            indices[n - 1 - dim] = matrix_positions_per_directions[i]
    return indices


def is_in_direction(matrix, direction):
    indices = is_in_direction_indices(matrix.ndim, direction)
    return np.any(matrix[np.ix_(*indices)] == 1)


def relation_vectors(intervals, other_region_intervals):
    obtained_vectors = []
    relations = [
        v_before, v_overlaps, v_during, v_meets, v_starts, v_finishes, v_equals
    ]
    for interval, other_region_interval in zip(
        intervals, other_region_intervals
    ):
        for f in relations:
            vector = f(interval, other_region_interval)
            if vector is not None:
                obtained_vectors.append(vector)
                break
    return np.array(obtained_vectors)


def direction_matrix(region_bbs, another_region_bbs):
    res = np.zeros((3, ) * region_bbs[0].dim, dtype=bool)
    for bb in region_bbs:
        for another_region_bb in another_region_bbs:
            rp_vector = relation_vectors(bb.limits, another_region_bb.limits)
            tensor = rp_vector[0].reshape(1, 3)
            for i in range(1, len(rp_vector)):
                tensor = kron(rp_vector[i].reshape((3, ) + (1, ) * i),
                              tensor).squeeze()
            res += tensor.astype(
                bool, copy=False
            )
    return res.astype(int, copy=False)
