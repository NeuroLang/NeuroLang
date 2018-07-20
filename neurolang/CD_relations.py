from .interval_algebra import v_before, v_overlaps, v_during, v_meets, v_starts, v_finishes, v_equals
from .regions import ExplicitVBR
import numpy as np

__all__ = ['cardinal_relation']

directions_dim_space = {'L': [0], 'R': [0], 'P': [1], 'A': [1], 'O': [0, 1, 2],
                        'I': [2], 'S': [2], 'F': [3]}

matrix_positions_per_directions = {'L': [0], 'O': [1], 'R': [2], 'P': [0],
                                   'A': [2], 'I': [0], 'S': [2], 'F': [2]}

inverse_directions = {'R': 'L', 'A': 'P', 'S': 'I',
                      'O': 'O', 'L': 'R', 'P': 'A', 'I': 'S'}

direction_from_relation = {'I': -1, 'S': 1,
                           'P': -1, 'A': 1,
                           'L': -1, 'R': 1}


def cardinal_relation(region, reference_region, directions, refine_overlapping=False, stop_at=None):

    mat = direction_matrix([region.bounding_box], [reference_region.bounding_box])
    overlap = is_in_direction(mat, 'O')
    if overlap and refine_overlapping and isinstance(region, ExplicitVBR):
        mat = overlap_resolution(region, reference_region, stop_at)
    return is_in_direction(mat, directions)


def overlap_resolution(region, reference_region, stop_at=None):
    if stop_at == 0:
        raise ValueError("stop_at must be larger than 0")

    current_region_level = [region.aabb_tree.root]
    current_reference_region_level = [reference_region.aabb_tree.root]
    level = 0

    overlap = True

    max_depth_reached_reg = False
    max_depth_reached_ref = False

    while (
        ((stop_at is None) or (level < stop_at)) and
        (not (max_depth_reached_reg and max_depth_reached_ref)) and
        overlap
    ):
        if not max_depth_reached_reg:
            current_region_next_level = tree_next_level(current_region_level)
            if current_region_next_level:
                current_region_level = current_region_next_level
            else:
                max_depth_reached_reg = True

        if not max_depth_reached_ref:
            current_reference_region_next_level = tree_next_level(current_reference_region_level)
            if current_reference_region_next_level:
                current_reference_region_level = current_reference_region_next_level
            else:
                max_depth_reached_ref = True

        mat = direction_matrix(
            [reg.box for reg in current_region_level],
            [reg.box for reg in current_reference_region_level]
        )
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


def is_in_direction(matrix, direction):
    n = matrix.ndim
    indices = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            indices[n - 1 - dim] = matrix_positions_per_directions[i]
    return np.any(matrix[np.ix_(*indices)] == 1)


def relation_vectors(intervals, other_region_intervals):
    obtained_vectors = []
    relations = [v_before, v_overlaps, v_during, v_meets, v_starts, v_finishes, v_equals]
    for i in range(len(intervals)):
        for f in relations:
            vector = f(intervals[i], other_region_intervals[i])
            if vector is not None:
                obtained_vectors.append(vector)
                break
    return obtained_vectors


def direction_matrix(region_bbs, another_region_bbs):
    res = np.zeros((3,) * region_bbs[0].dim)
    for bb in region_bbs:
        for another_region_bb in another_region_bbs:
            rp_vector = relation_vectors(bb.limits, another_region_bb.limits)
            tensor = rp_vector[0].reshape(1, 3)
            for i in range(1, len(rp_vector)):
                tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
            res = np.logical_or(res, tensor).astype(int)
    return res
