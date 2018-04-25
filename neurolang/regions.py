from .RCD_relations import *
import numpy as np


class Region:

    def __init__(self, lb, ub) -> None:
        if not np.all([lb[i] < ub[i] for i in range(len(lb))]):
            raise Exception('Lower bounds must lower (and not equal) than upper bounds when creating rectangular regions')
        self._bounding_box = np.c_[lb, ub]
        self._bounding_box.setflags(write=False)

    def __hash__(self):
        return hash(self._bounding_box.tobytes())

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def _lb(self):
        return self._bounding_box[:, 0]

    @property
    def _ub(self):
        return self._bounding_box[:, 1]

    @property
    def center(self) -> np.array:
        return 0.5 * (self._lb + self._ub)

    @property
    def width(self) -> np.array:
        return self._ub - self._lb

    def axis_intervals(self) -> np.array:
        return np.array([tuple([self._lb[i], self._ub[i]]) for i in range(len(self._lb))])

    def __eq__(self, other) -> bool:
        return np.all(self._lb == other._lb) and np.all(self._ub == other._ub)

    def __repr__(self):
        return 'Region(lb={}, up={})'.format(tuple(self._lb), tuple(self._ub))
