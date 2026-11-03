import numpy as np


def before(i1, i2) -> bool:
    return i1[0] < i1[1] < i2[0] < i2[1]


def overlaps(i1, i2) -> bool:
    return i1[0] < i2[0] < i1[1] < i2[1]


def during(i1, i2) -> bool:
    return i2[0] < i1[0] < i1[1] < i2[1]


def meets(i1, i2) -> bool:
    return i1[0] < i1[1] == i2[0] < i2[1]


def starts(i1, i2) -> bool:
    return i1[0] == i2[0] < i1[1] < i2[1]


def finishes(i1, i2) -> bool:
    return i2[0] < i1[0] < i1[1] == i2[1]


def equals(i1, i2) -> bool:
    return i1[0] == i2[0] < i1[1] == i2[1]


def converse(operation):
    return lambda x, y: operation(y, x)


def negate(operation):
    return lambda x, y: not operation(x, y)


def get_intervals_relations(intervals, other_region_intervals):
    obtained_relation_per_axis = [''] * len(intervals)
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    for f in relations:
        for i, _ in enumerate(obtained_relation_per_axis):
            if f(intervals[i], other_region_intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0])
            elif f(other_region_intervals[i], intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0] + 'i')
        if np.all(np.array(obtained_relation_per_axis) != ''):
            break
    return tuple(obtained_relation_per_axis)


def v_before(i1, i2):
    if i1[0] < i1[1] < i2[0] < i2[1]:
        return (1, 0, 0)
    elif i2[0] < i2[1] < i1[0] < i1[1]:
        return (0, 0, 1)
    return None


def v_overlaps(i1, i2):
    if i1[0] < i2[0] < i1[1] < i2[1]:
        return (1, 1, 0)
    elif i2[0] < i1[0] < i2[1] < i1[1]:
        return (0, 1, 1)
    return None


def v_during(i1, i2):
    if i2[0] < i1[0] < i1[1] < i2[1]:
        return (0, 1, 0)
    elif i1[0] < i2[0] < i2[1] < i1[1]:
        return (1, 1, 1)
    return None


def v_meets(i1, i2):
    if i1[0] < i1[1] == i2[0] < i2[1]:
        return (1, 0, 0)
    elif i2[0] < i2[1] == i1[0] < i1[1]:
        return (0, 0, 1)
    return None


def v_starts(i1, i2):
    if i1[0] == i2[0] < i1[1] < i2[1]:
        return (0, 1, 0)
    elif i2[0] == i1[0] < i2[1] < i1[1]:
        return (0, 1, 1)
    return None


def v_finishes(i1, i2):
    if i2[0] < i1[0] < i1[1] == i2[1]:
        return (0, 1, 0)
    elif i1[0] < i2[0] < i2[1] == i1[1]:
        return (1, 1, 0)


def v_equals(i1, i2):
    if i1[0] == i2[0] < i1[1] == i2[1]:
        return (0, 1, 0)
    return None
