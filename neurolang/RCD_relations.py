from .interval_algebra import *
from .regions import *
import numpy as np


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

    (a, s) = directions_map(direction)
    #todo remove if
    if matrix.shape == (3, 3, 3):
        return matrix[1][a, s] == 1
    return matrix[a, s] == 1

def directions_map(d):
    return {'SP': (0, 0), 'S': (0, 1), 'SA': (0, 2),
            'P': (1, 0), 'O': (1, 1), 'A': (1, 2),
            'IP': (2, 0), 'I': (2, 1), 'IA': (2, 2)}[d]


def inverse_direction(d):
    return {'SP': 'IA', 'S': 'I', 'SA': 'IP',
            'P': 'A', 'O': 'O', 'A': 'P',
            'IP': 'SA', 'I': 'S', 'IA': 'SP'}[d]

def direction_matrix(bounding_box, another_bounding_box):
    relations = intervals_relations_from_boxes(bounding_box, another_bounding_box)
    return translate_ia_relations(*relations[::-1])     # todo: relations reordering can be avoided

def translate_ia_relations(*args):
    n = len(args)
    result = np.zeros(shape=(3,) * n)

    # todo remove if
    if n == 3:
        args = [args[2], args[0], args[1]]

    ixs_per_ax = np.empty((n, n), dtype=list)
    ixs_prod_match = np.empty(n, dtype=list)
    for i in range(n):
        ixs_per_ax[i, :] = relations_to_matrix_idxs(n, i, args[i])

    for i in range(n):
        ixs_per_coordinate = [set(elem) for elem in ixs_per_ax[:, i]]
        ixs_prod_match[i] = list(set.intersection(*ixs_per_coordinate))

    result[np.ix_(*ixs_prod_match)] = 1

    #todo remove ifs
    if n == 3:
        for k in range(3):
            result[:, [0, 2], :] = result[:, [2, 0], :]
    if n == 2:
        result[[0, 2]] = result[[2, 0]]
    return result


def relations_to_matrix_idxs(n, dim, relation):

    res = [[0, 1, 2]] * n
    if relation in ['d', 's', 'f', 'e']:
        res[dim] = [1]
    elif relation in ['m', 'b']:
        res[dim] = [0]
    elif relation in ['mi', 'bi']:
        res[dim] = [2]
    elif relation in ['o', 'fi']:
        res[dim] = [0, 1]
    elif relation in ['oi', 'si']:
        res[dim] = [1, 2]
    return res
