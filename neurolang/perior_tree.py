from typing import Union, Tuple, List, Set, Optional
from .brain_tree import AABB, Tree
import numpy as np
from multimethod import multimethod
from functools import singledispatch, update_wrapper


def methdispatch(func):
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


class BoundedAABB(AABB):

    def __init__(self, lb: tuple, ub: tuple, bounded_area: 'Boundary') -> None:
        super().__init__(lb, ub)
        self._bound_area = bounded_area

    def __getitem__(self, i):
            return self._lb if i == 0 else self._ub

    def __repr__(self):
        return 'BoundedAABB(lb={}, up={})'.format(tuple(self._lb), tuple(self._ub))


    # def restrict_position(self, bounded_area : 'Boundary):
    #     rng = self.range()
    #     center = tuple([self._lb[i] + rng[i]/2 for i in range(len(self._lb))])
    #     center = bounded_area.adjust_position(center)
    #     self._lb = np.asanyarray([center[i] - rng[i]/2 for i in range(len(self._lb))])
    #     self._ub = np.asanyarray([center[i] + rng[i]/2 for i in range(len(self._lb))])

    # def methdispatch(func):
    #     dispatcher = singledispatch(func)
    #
    #     def wrapper(*args, **kw):
    #         return dispatcher.dispatch(args[2].__class__)(*args, **kw)
    #
    #     wrapper.register = dispatcher.register
    #     update_wrapper(wrapper, func)
    #     return wrapper

    @methdispatch
    def expand(self, arg):
        print(arg)

    @expand.register(tuple)
    def _(self, point: tuple):
        center = np.array([self._lb[i] + self.range()[i] / 2 for i in range(len(self._lb))])
        dc = np.array(self._bound_area.adjust_direction(tuple(np.array(point) - center)))
        p = center + dc
        l = self._bound_area.adjust_position(np.array(min(tuple(self._lb), tuple(p))))
        u = self._bound_area.adjust_position(np.array(max(tuple(self._ub), tuple(p))))

        return BoundedAABB(l, u, self._bound_area)

    def adjust_to_bound(self, bound_area: 'Boundary') -> None:
        self._lb, self._ub = bound_area.adjust_position(np.array(self._lb)), bound_area.adjust_position(np.array(self._ub))

    @expand.register(object)
    def _(self, another_box: 'BoundedAABB'):
        center = np.array([self._lb[i] + self.range()[i] / 2 for i in range(len(self._lb))])
        box2_center = np.array([another_box[0][i] + another_box.range()[i] / 2 for i in range(len(another_box[0]))])

        dc = np.array(self._bound_area.adjust_direction(tuple(box2_center - center)))
        l1, u1 = tuple(self._lb), tuple(self._ub)
        l2, u2 = tuple((center + dc) - another_box.range()/2), tuple((center + dc) + another_box.range()/2)

        l, u = self._bound_area.adjust_position(np.array(min(l1, l2))), self._bound_area.adjust_position(np.array(max(u1, u2)))
        return BoundedAABB(l, u, self._bound_area)

    @multimethod(object, object)
    def intersects(self, other: 'BoundedAABB', bound_area: 'Boundary') -> bool:
        center = np.array([self._lb[i] + self.range()[i] / 2 for i in range(len(self._lb))])
        box2_center = np.array([other[0][i] + other.range()[i] / 2 for i in range(len(other[0]))])
        dc = np.array(bound_area.adjust_direction(tuple(center - box2_center)))
        radius = self.width()/2
        other_radius = other.width()/2
        for i in range(len(radius)):
            if abs(dc[i]) <= radius[i] + other_radius[i]:
                return True
        return False

    @multimethod(object, object)
    def contains(self, other: 'BoundedAABB', bound_area: 'Boundary') -> bool:
        center = np.array([self._lb[i] + self.range()[i] / 2 for i in range(len(self._lb))])
        box2_center = np.array([other[0][i] + other.range()[i] / 2 for i in range(len(other[0]))])
        dc = np.array(bound_area.adjust_direction(tuple(center - box2_center)))
        radius = self.width()/2
        other_radius = other.width()/2
        for i in range(len(radius)):
            if abs(dc[i]) <= radius[i] + other_radius[i]:
                return True
        return False

    @multimethod(object, tuple)
    def contains(self, point: tuple, bound_area: 'Boundary') -> bool:
        center = np.array([self._lb[i] + self.range()[i] / 2 for i in range(len(self._lb))])

        dc = np.array(bound_area.adjust_direction(tuple(point - center)))
        radius = self.width() / 2
        for i in range(len(radius)):
            if abs(dc[i]) > radius[i]:
                return False
        return True

    def range(self) -> np.array:
        return self._ub - self._lb

    def width(self) -> Tuple:
        return self._ub - self._lb


    def height(self) -> Tuple:
        return self._ub - self._lb

class Boundary(BoundedAABB):

    def __init__(self, lb, ub) -> None:
        self._lb = np.asanyarray(lb)
        self._ub = np.asanyarray(ub)

    def adjust_position(self, point) -> Tuple:
        res = np.copy(point)
        for i in range(len(point)):
            if point[i] < self._lb[i]:
                res[i] = point[i] + self.width()[i]
            elif point[i] > self._ub[i]:
                res[i] = point[i] - self.width()[i]
        return tuple(res)

    def adjust_direction(self, point):
        res = np.copy(point)
        for i in range(len(point)):
            if point[i] < -(self.width()[i] / 2):
                res[i] = point[i] + self.width()[i]
            elif point[i] >= (self.width()[i] / 2):
                res[i] = point[i] - self.width()[i]
        return tuple(res)