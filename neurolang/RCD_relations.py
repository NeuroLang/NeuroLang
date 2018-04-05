from .interval_algebra import *
from .regions import *
import numpy as np


def get_interval_relation_to(bounding_box, another_bounding_box):
    ''' retrieve interval relations of the axes between two objects '''
    [x_rel, y_rel] = ['', '']
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    [x, y] = bounding_box.axis_intervals()
    [other_x, other_y] = another_bounding_box.axis_intervals()
    for f in relations:
        if f(x, other_x):
            x_rel = str(f.__name__[0])
        elif f(other_x, x):
            x_rel = str(f.__name__[0] + 'i')
        if f(y, other_y):
            y_rel = str(f.__name__[0])
        elif f(other_y, y):
            y_rel = str(f.__name__[0] + 'i')
        if x_rel != '' and y_rel != '':
            break
    return tuple([x_rel, y_rel])


def direction_matrix(bounding_box, another_bounding_box):
    ''' direction matrix of two bounding boxes '''
    intervals_relations = get_interval_relation_to(bounding_box, another_bounding_box)
    return translate_ia_relation(intervals_relations[0], intervals_relations[1])


def translate_ia_relation(x, y):
    '''' IA to RCD mapping '''
    if x in ['d', 's', 'f', 'e'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['mi', 'bi']:
        return np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
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
        return np.array([[0, 1, 1], [0, 0, 0], [0, 1, 1]])
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
