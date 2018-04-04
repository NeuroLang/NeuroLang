from typing import Tuple
import numpy as np

class Region:
    def __init__(self, lb, ub) -> None:
        self._lb = np.asanyarray(lb)
        self._ub = np.asanyarray(ub)

    def generate_random_box(x_bounds, y_bounds, size_bounds):
        lower_bound = np.asanyarray([np.random.uniform(*b) for b in (x_bounds, y_bounds)])
        upper_bound = lower_bound + np.random.uniform(*size_bounds, size=2)
        return Region(lower_bound, upper_bound)

    @property
    def center(self) -> np.array:
        return 0.5 * (self._lb + self._ub)

    @property
    def width(self) -> Tuple:
        return tuple(self._ub - self._lb)

    def get_interval_relation_to(self, other: 'Region') -> tuple:
        [x_rel, y_rel] = ['', '']
        relations = [before, overlaps, during, meets, starts, finishes, equals]
        [x, y] = self.axis_intervals()
        [other_x, other_y] = other.axis_intervals()
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

    def axis_intervals(self) -> np.array:
        return np.array([tuple([self._lb[0], self._ub[0]]), tuple([self._lb[1], self._ub[1]])])


    def __eq__(self, other) -> bool:
        return np.all(self._lb == other._lb) and np.all(self._ub == other._ub)

    def __repr__(self):
        return 'Region(lb={}, up={})'.format(tuple(self._lb), tuple(self._ub))

#TODO refa into class
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