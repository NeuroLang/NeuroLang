from .interval_algebra import *
from .regions import *
import numpy as np

directions = {'B': [[1], [1]], 'L': [[0, 1, 2], [0]],
              'R': [[0, 1, 2], [2]],
              'P': [[0], [0, 1, 2]],
              'A': [[2], [0, 1, 2]],
              'O': [[1], [1]]}

inverse_directions = {'R': 'L', 'A': 'P', 'S': 'I',
              'O': 'O', 'L': 'R', 'P': 'A', 'I': 'S'}


def intervals_relations_from_boxes(bounding_box, another_bounding_box):
    ''' retrieve interval relations of the axes between two objects '''
    intervals = bounding_box.axis_intervals()
    other_box_intervals = another_bounding_box.axis_intervals()
    return get_intervals_relations(intervals, other_box_intervals)


def get_intervals_relations(intervals, other_box_intervals):
    obtained_relation_per_axis = ['' for _ in range(len(intervals))]
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    for f in relations:
        for i in range(len(obtained_relation_per_axis)):
            if f(intervals[i], other_box_intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0])
            elif f(other_box_intervals[i], intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0] + 'i')
        if np.all(np.array(obtained_relation_per_axis) != ''):
            break
    return tuple(obtained_relation_per_axis)


def is_in_direction(matrix, direction):
    if direction in ['I', 'C', 'S']:
        dir_index = int(np.where(np.array(['I', 'C', 'S']) == direction)[0])
        return np.any(matrix[dir_index] == 1)

    for m in matrix:
        if np.any(m[directions[direction]] == 1):
            return True
    return False


def direction_matrix(bounding_box, another_bounding_box):

    relations = intervals_relations_from_boxes(bounding_box, another_bounding_box)
    rp_vector = [relative_position_vector(r) for r in relations]
    result = rp_vector[0].reshape(1, 3)

    for i in range(1, len(relations)):
        result = np.kron(rp_vector[i].reshape((3,) + (1,) * i), result)
    return result


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
