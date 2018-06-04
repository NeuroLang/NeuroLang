from .interval_algebra import *
from .regions import *
import numpy as np


directions = {'SP': (0, 0), 'S': (0, 1), 'SA': (0, 2),
          'P': (1, 0), 'O': (1, 1), 'A': (1, 2),
          'IP': (2, 0), 'I': (2, 1), 'IA': (2, 2)}

inverse_directions = {'SP': 'IA', 'S': 'I', 'SA': 'IP',
            'P': 'A', 'O': 'O', 'A': 'P',
            'IP': 'SA', 'I': 'S', 'IA': 'SP'}


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
    if direction in ['L', 'C', 'R']:
        index = int(np.where(np.array(['L', 'C', 'R']) == direction)[0])
        return np.any(matrix[index] == 1)

    (a, s) = directions[direction]
    #todo remove if
    if matrix.shape == (3, 3, 3):
        return matrix[1][a, s] == 1
    return matrix[a, s] == 1


def direction_matrix(bounding_box, another_bounding_box):
    relations = intervals_relations_from_boxes(bounding_box, another_bounding_box)
    n = len(relations)

    #patch kronecker order such that: res = kron(r,kron(s,a))
    if n == 3:
        relations = [relations[1], relations[2], relations[0]]
    rp_vector = [relative_position_vector(r) for r in relations]
    result = rp_vector[0]
    for i in range(1, n):
        result = np.kron(rp_vector[i], result)
    result = result.reshape((3,) * n)

    #patch to return matrices with the original standard: superior in upper rows, inferior in lower ones.
    if n == 3:
        for k in range(3):
            result[:, [0, 2], :] = result[:, [2, 0], :]
    if n == 2:
        result[[0, 2]] = result[[2, 0]]
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

