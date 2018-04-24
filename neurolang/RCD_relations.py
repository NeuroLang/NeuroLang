from .interval_algebra import *
from .regions import *
import numpy as np


def get_interval_relation_to(bounding_box, another_bounding_box):
    ''' retrieve interval relations of the axes between two objects '''
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    intervals = bounding_box.axis_intervals()
    other_box_intervals = another_bounding_box.axis_intervals()

    obtained_relation_per_axis = ['' for _ in range(len(intervals))]

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
    return np.any(matrix[:, a, s] == 1)


def directions_map(d):
    return {'SP': (0, 0), 'S': (0, 1), 'SA': (0, 2),
            'P': (1, 0), 'O': (1, 1), 'A': (1, 2),
            'IP': (2, 0), 'I': (2, 1), 'IA': (2, 2)}[d]


def inverse_direction(d):
    return {'SP': 'IA', 'S': 'I', 'SA': 'IP',
            'P': 'A', 'O': 'O', 'A': 'P',
            'IP': 'SA', 'I': 'S', 'IA': 'SP'}[d]


def direction_matrix(bounding_box, another_bounding_box):
    ''' direction matrix of two bounding boxes '''
    intervals_relations = get_interval_relation_to(bounding_box, another_bounding_box)
    res = tensor_direction_matrix_wrapper(intervals_relations)
    return res


def tensor_direction_matrix_wrapper(ia_relations):
    res = np.zeros(shape=(3, 3, 3))
    if len(ia_relations) != 3:
        a = ia_relations[0]
        s = ia_relations[1]
        as_matrix = translate_ia_relation(s, a)
        res[1] = as_matrix
    else:
        a = ia_relations[1]
        s = ia_relations[2]
        as_matrix = translate_ia_relation(s, a)
        r = ia_relations[0]
        if r in ['d', 's', 'f', 'e']:
            res[1] = as_matrix
        elif r in ['m', 'b']:
            res[0] = as_matrix
        elif r in ['mi', 'bi']:
            res[2] = as_matrix
        elif r in ['o', 'fi']:
            res[0] = res[1] = as_matrix
        elif r in ['oi', 'si']:
            res[0] = res[2] = as_matrix
        elif r in ['di']:
            res[0] = res[1] = res[2] = as_matrix
    return res


def translate_ia_relation(x, y):
    '''' IA to RCD mapping '''
    if x in ['d', 's', 'f', 'e'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['mi', 'bi']:
        return np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['mi', 'bi']:
        return np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['mi', 'bi']:
        return np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    elif x in ['m', 'b'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

    elif x in ['fi', 'o'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    elif x in ['si', 'oi'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
    elif x in ['fi', 'o'] and y in ['mi', 'bi']:
        return np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['si', 'oi'] and y in ['mi', 'bi']:
        return np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    elif x in ['fi', 'o'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    elif x in ['si', 'oi'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['si', 'oi']:
        return np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])

    elif x in ['m', 'b'] and y in ['si', 'oi']:
        return np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]])
    elif x in ['mi', 'bi'] and y in ['si', 'oi']:
        return np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    elif x in ['di'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    elif x in ['di'] and y in ['mi', 'bi']:
        return np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    elif x in ['di'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['di']:
        return np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    elif x in ['m', 'b'] and y in ['di']:
        return np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['di']:
        return np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    elif x in ['o', 'fi'] and y in ['o', 'fi']:
        return np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
    elif x in ['o', 'fi'] and y in ['si', 'oi']:
        return np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    elif x in ['si', 'oi'] and y in ['o', 'fi']:
        return np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])
    elif x in ['si', 'oi'] and y in ['si', 'oi']:
        return np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    elif x in ['o', 'fi'] and y in ['di']:
        return np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    elif x in ['si', 'oi'] and y in ['di']:
        return np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
    elif x in ['di'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    elif x in ['di'] and y in ['si', 'oi']:
        return np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    elif x in ['di'] and y in ['di']:
        return np.array(np.ones(shape=(3, 3)))
