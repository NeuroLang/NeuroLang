from .RCD_relations import *


class Region:

    def __init__(self, lb, ub) -> None:
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

    def direction_matrix(self, other_region) -> np.array:
        return direction_matrix(self._bounding_box, other_region.bounding_box)

    def axis_intervals(self) -> np.array:
        return np.array([tuple([self._lb[0], self._ub[0]]), tuple([self._lb[1], self._ub[1]])])

    def __eq__(self, other) -> bool:
        return np.all(self._lb == other._lb) and np.all(self._ub == other._ub)

    def __repr__(self):
        return 'Region(lb={}, up={})'.format(tuple(self._lb), tuple(self._ub))
