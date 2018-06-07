from .interval_algebra import *
from .regions import *
import numpy as np


directions_dim_space = {'L': [0], 'R': [0], 'P': [1], 'A': [1], 'O': [0, 1],
                          'I': [2], 'S': [2], 'F': [3]}

matrix_positions_per_directions = {'L': [0], 'O': [1], 'R': [2], 'P': [0],
                                   'A': [2], 'I': [0], 'S': [2], 'F': [2]}

inverse_directions = {'R': 'L', 'A': 'P', 'S': 'I',
                    'O': 'O', 'L': 'R', 'P': 'A', 'I': 'S'}

#
# def direction_matrix(region, another_region):
#
#     relations = intervals_relations_from_regions(region, another_region)
#     rp_vector = [relative_position_vector(r) for r in relations]
#     result = rp_vector[0].reshape(1, 3)
#
#     for i in range(1, len(relations)):
#         result = np.kron(rp_vector[i].reshape((3,) + (1,) * i), result)
#     return result


def direction_matrix(region, another_region):

    res = np.zeros((3,) * region.dim)
    for bb in region.bounding_box:
        for another_region_bb in another_region.bounding_box:
            relations = get_intervals_relations(bb, another_region_bb)
            rp_vector = [relative_position_vector(r) for r in relations]
            tensor = rp_vector[0].reshape(1, 3)
            for i in range(1, len(relations)):
                tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
            res = np.logical_or(res, tensor).astype(int)
    return res

#
# def intervals_relations_from_bounding_box(bb, another_bb):
#     intervals = bb.bounding_box
#     other_region_intervals = another_bb.bounding_box
#     return get_intervals_relations(intervals, other_region_intervals)
#
#

def is_in_direction(matrix, direction):

    n = matrix.ndim
    idxs = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            idxs[n-1-dim] = matrix_positions_per_directions[i]
    return np.any(matrix[idxs] == 1)


# def intervals_relations_from_regions(region, another_region):
#     intervals = region.bounding_box
#     other_region_intervals = another_region.bounding_box
#     return get_intervals_relations(intervals, other_region_intervals)


def get_intervals_relations(intervals, other_region_intervals):
    obtained_relation_per_axis = [''] * len(intervals)
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    for f in relations:
        for i in range(len(obtained_relation_per_axis)):
            if f(intervals[i], other_region_intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0])
            elif f(other_region_intervals[i], intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0] + 'i')
        if np.all(np.array(obtained_relation_per_axis) != ''):
            break
    return tuple(obtained_relation_per_axis)


def relative_position_vector(relation):
    if relation in ['d', 's', 'f', 'e']:
        return np.array((0, 1, 0))
    elif relation in ['m', 'b']:
        return np.array((1, 0, 0))
    elif relation in ['mi', 'bi']:
        return np.array((0, 0, 1))
    elif relation in ['o', 'fi']:
        return np.array((1, 1, 0))
    elif relation in ['oi', 'si']:
        return np.array((0, 1, 1))
    return np.array((1, 1, 1))
