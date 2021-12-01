from functools import lru_cache
from itertools import product

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

anatomical_restric_axes = directions_dim_space["A"] + directions_dim_space["S"]
anatomical_restrict_dirs = ["A", "P", "I", "S"]

relations = [
    v_before, v_overlaps, v_during, v_meets, v_starts, v_finishes, v_equals
]


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
        mat = direction_matrix(
            region.bounding_box, reference_region.bounding_box
        )
        dir_counts = None
        if not (refine_overlapping and "O" in directions) and is_in_direction(
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
                mat, dir_counts = overlap_resolution(
                    region, reference_region, directions, stop_at
                )
            result = is_in_direction(mat, directions, dir_counts)

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
    dir_counts = np.zeros(len(anatomical_restrict_dirs), dtype=int)
    while region_stack:
        region, reference_region, level = region_stack.pop()

        mat = direction_matrix(region.box, reference_region.box, True)
        total_mat += mat
        dir_counts += np.array(
            [is_in_direction(mat, d) for d in anatomical_restrict_dirs], dtype=int
        )

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
                total_mat -= mat
                region_stack += stack_update
            elif directions is None:
                break

    return total_mat.clip(0, 1), dir_counts


@lru_cache(maxsize=128)
def is_in_direction_indices(n, direction):
    indices = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            indices[n - 1 - dim] = matrix_positions_per_directions[i]
    return np.ix_(*indices)


def is_in_direction(matrix, direction, dir_counts=None):
    if dir_counts is not None:
        for dir in direction:
            if dir in anatomical_restrict_dirs:
                c0, c1 = (
                    dir_counts[anatomical_restrict_dirs.index(dir)],
                    dir_counts[
                        anatomical_restrict_dirs.index(inverse_directions[dir])
                    ],
                )
                ratio = c0 / (c0 + c1)
                if ratio > 0.8:
                    return True
                else:
                    direction = direction.replace(dir, "")

        if len(direction) == 0:
            return False

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


def direction_matrix(
    region_bb, another_region_bb, restrict_axes_directions=False
):
    rp_vector = relation_vectors(region_bb.limits, another_region_bb.limits)
    if restrict_axes_directions:
        restrict_axis_relation_to_overlaping_regions(rp_vector)
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(rp_vector)):
        tensor = kron(
            rp_vector[i].reshape((3, ) + (1, ) * i),
            tensor
        ).squeeze()
    return tensor.clip(0, 1)


def restrict_axis_relation_to_overlaping_regions(rp_vector):
    """
    When checking for directions on the anterior/posterior or on the
    inferior/posterior axis, we only consider intervals which overlap on the
    other axis.
    """
    # for each axis
    for dir, orth_dir in zip(
        anatomical_restric_axes, anatomical_restric_axes[::-1]
    ):
        # check that the other axis is overlaping
        if not rp_vector[orth_dir, 1]:
            # otherwise remove non-overlaping relations from current axis
            rp_vector[dir] = (0, 1, 0)
