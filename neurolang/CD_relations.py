from .interval_algebra import *
from .regions import *
from functools import singledispatch
import numpy as np

directions_dim_space = {'L': [0], 'R': [0], 'P': [1], 'A': [1], 'O': [0, 1, 2],
                          'I': [2], 'S': [2], 'F': [3]}

matrix_positions_per_directions = {'L': [0], 'O': [1], 'R': [2], 'P': [0],
                                   'A': [2], 'I': [0], 'S': [2], 'F': [2]}

inverse_directions = {'R': 'L', 'A': 'P', 'S': 'I',
                    'O': 'O', 'L': 'R', 'P': 'A', 'I': 'S'}


def cardinal_relation(region, reference_region, directions, refine_overlapping=False, stop_at=10000000):

    mat = direction_matrix([region.bounding_box], [reference_region.bounding_box])
    obtained = is_in_direction(mat, directions)

    if 'O' in directions and obtained and refine_overlapping and isinstance(region, ExplicitVBR):

        current_region = [region.aabb_tree.root]
        current_reference_region = [reference_region.aabb_tree.root]
        continue_refinement = True
        max_depth_reached_ref = max_depth_reached_reg = False
        level = 0
        while obtained and continue_refinement and (level < stop_at):
            if not max_depth_reached_reg:
                current_region, max_depth_reached_reg = children_of_tree_node(current_region)
            if not max_depth_reached_ref:
                current_reference_region, max_depth_reached_ref = children_of_tree_node(current_reference_region)
            continue_refinement = not (max_depth_reached_reg and max_depth_reached_ref)

            mat = direction_matrix([reg.box for reg in current_region], [reg.box for reg in current_reference_region])
            obtained = is_in_direction(mat, directions)
            level += 1
    return obtained


def children_of_tree_node(nodes):
    result = []
    for node in nodes:
        result += [node for node in [node.left, node.right] if node is not None]
        if len(result) == 0:
            return nodes, True
    return result, False


def is_in_direction(matrix, direction):
    n = matrix.ndim
    idxs = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            idxs[n-1-dim] = matrix_positions_per_directions[i]
    return np.any(matrix[idxs] == 1)


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


@singledispatch
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


@direction_matrix.register(Region)
def _(region, other_region):
    rp_vector = relation_vectors(region.bounding_box.limits, other_region.bounding_box.limits)
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(rp_vector)):
        tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
    return tensor
