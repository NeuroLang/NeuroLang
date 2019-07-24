from typing import Union, Tuple, Set
from .aabb_tree import AABB
import numpy as np
from copy import deepcopy
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
        return 'BoundedAABB(lb={}, up={}, boundedTo={})'.format(
            tuple(self._lb), tuple(self._ub), self._bound_area
        )

    def __eq__(self, other: 'BoundedAABB') -> bool:
        return np.all(self._lb == other._lb) and np.all(
            self._ub == other._ub
        ) and self._bound_area == other._bound_area

    @property
    def center(self) -> np.array:
        return 0.5 * (self._lb + self._ub)

    @property
    def volume(self) -> float:
        return (self._ub - self._lb).prod()

    @property
    def width(self) -> Tuple:
        return self._ub - self._lb

    def adjust_to_bound(self) -> None:
        self._lb, self._ub = self._bound_area.adjust_position(
            np.array(self._lb)
        ), self._bound_area.adjust_position(np.array(self._ub))

    @methdispatch
    def expand(self, arg):
        print(arg)

    @expand.register(tuple)
    def expand_tuple(self, point: tuple):
        center = np.array([
            self._lb[i] + self.width[i] / 2 for i in range(len(self._lb))
        ])
        dc = np.array(
            self._bound_area.adjust_direction(tuple(np.array(point) - center))
        )
        p = center + dc
        lb = self._bound_area.adjust_position(
            np.array(min(tuple(self._lb), tuple(p)))
        )
        ub = self._bound_area.adjust_position(
            np.array(max(tuple(self._ub), tuple(p)))
        )

        return BoundedAABB(lb, ub, self._bound_area)

    @expand.register(object)
    def expand_object(self, another_box: 'BoundedAABB'):
        center = np.array([
            self._lb[i] + self.width[i] / 2 for i in range(len(self._lb))
        ])
        box2_center = np.array([
            another_box[0][i] + another_box.width[i] / 2
            for i in range(len(another_box[0]))
        ])

        dc = np.array(
            self._bound_area.adjust_direction(tuple(box2_center - center))
        )
        l1, u1 = tuple(self._lb), tuple(self._ub)
        l2, u2 = tuple((center + dc) -
                       another_box.width / 2), tuple((center + dc) +
                                                     another_box.width / 2)

        l, u = self._bound_area.adjust_position(
            np.array(min(l1, l2))
        ), self._bound_area.adjust_position(np.array(max(u1, u2)))
        return BoundedAABB(l, u, self._bound_area)

    def intersects(self, other: 'BoundedAABB') -> bool:
        center = np.array([
            self._lb[i] + self.width[i] / 2 for i in range(len(self._lb))
        ])
        box2_center = np.array([
            other[0][i] + other.width[i] / 2 for i in range(len(other[0]))
        ])
        dc = np.array(
            self._bound_area.adjust_direction(tuple(center - box2_center))
        )
        radius = self.width / 2
        other_radius = other.width / 2
        return np.all(abs(dc) <= other_radius + radius)

    @methdispatch
    def contains(self, arg):
        print(arg)

    @contains.register(object)
    def _(self, other: 'BoundedAABB') -> bool:
        center = np.array([
            self._lb[i] + self.width[i] / 2 for i in range(len(self._lb))
        ])
        box2_center = np.array([
            other._lb[i] + other.width[i] / 2 for i in range(len(other._lb))
        ])
        dc = np.array(
            self._bound_area.adjust_direction(tuple(center - box2_center))
        )
        radius = self.width / 2
        other_radius = other.width / 2
        return np.all(abs(dc) <= radius - other_radius)

    @contains.register(tuple)
    def contains_tuple(self, point: tuple) -> bool:
        center = np.array([
            self._lb[i] + self.width[i] / 2 for i in range(len(self._lb))
        ])

        dc = np.array(self._bound_area.adjust_direction(tuple(point - center)))
        radius = self.width / 2
        for i, r in enumerate(radius):
            if abs(dc[i]) > r:
                return False
        return True

    def direction_matrix(self, other: 'BoundedAABB') -> np.matrix:
        res = np.zeros(shape=(1, 9))
        tiles = self.cardinal_tiles()
        for i in range(0, 9):
            if other.intersects(tiles[0, i]):
                res[0, i] = 1
        return np.asmatrix(np.reshape(res, (3, 3)))

    def cardinal_tiles(self) -> np.array:
        res = np.empty((1, 9), dtype=object)
        range_ = abs(self.width)

        index = 0
        for j in [1, 0, -1]:
            lb = self._lb + np.asanyarray((-range_[0], j * range_[1]))
            ub = self._ub + np.asanyarray((-range_[0], j * range_[1]))
            for i in [0, 1, 2]:
                increase = np.asanyarray((range_[0], 0)) * i
                res[0, index] = BoundedAABB(
                    tuple(lb + increase), tuple(ub + increase),
                    self._bound_area
                )
                index += 1

        return res


class Boundary(BoundedAABB):
    def __init__(self, lb, ub) -> None:
        self._lb = np.asanyarray(lb)
        self._ub = np.asanyarray(ub)

    def adjust_position(self, point) -> Tuple:
        res = np.copy(point)
        for i, p in enumerate(point):
            if p < self._lb[i]:
                res[i] = p + self.width[i]
            elif p > self._ub[i]:
                res[i] = p - self.width[i]
        return tuple(res)

    def adjust_direction(self, point):
        res = np.copy(point)
        for i, p in enumerate(point):
            if p < -(self.width[i] / 2):
                res[i] = p + self.width[i]
            elif p >= (self.width[i] / 2):
                res[i] = p - self.width[i]
        return tuple(res)

    def __repr__(self):
        return 'Boundry(lowerBound={}, upperBound={})'.format(
            tuple(self._lb), tuple(self._ub)
        )

    def __eq__(self, other) -> bool:
        return np.all(self._lb == other._lb) and np.all(self._ub == other._ub)


class Node(object):

    def __init__(self,
                 box: BoundedAABB,
                 parent: Union[None, 'Node'] = None,
                 left: Union[None, 'Node'] = None,
                 right: Union[None, 'Node'] = None,
                 height: int = 0,
                 region_ids: Set[int] = None) -> None:

        self.box = box
        self.parent = parent
        self.left = left
        self.right = right
        self.height = height
        if region_ids is None:
            region_ids = set()
        self.region_ids = region_ids

    @property
    def is_leaf(self):
        return self.left is None


class Tree(object):

    def __init__(self) -> None:
        self.root = None
        self.region_boxes = dict()
        self.height = 0

    def expand_region_box(
        self, region_id: int, added_box: BoundedAABB
    ) -> None:
        if region_id not in self.region_boxes:
            self.region_boxes[region_id] = added_box
        else:
            self.region_boxes[region_id] = \
                self.region_boxes[region_id].expand(added_box)

    def add(self, box: BoundedAABB, region_ids: Set[int] = None) -> None:
        if region_ids is None:
            region_ids = set()

        for region_id in region_ids:
            self.expand_region_box(region_id, box)

        # if the tree is empty, just set root to the given node
        if self.root is None:
            self.root = Node(box=box, region_ids=region_ids)
            return

        n = self.root  # type: Node
        # go down the tree until we reach a leaf
        while not n.is_leaf:
            # to decide whether to go to the left or right branch
            # we use a heuristic that takes into account the volume increase
            # of the left and right boxes after adding the new box
            combined_volume = box.expand(n.box).volume
            cost = 2 * combined_volume
            inherit_cost = 2 * (combined_volume - n.box.volume)
            if n.left.is_leaf:
                cost_left = box.expand(n.left.box).volume + inherit_cost
            else:
                cost_left = (
                    box.expand(n.left.box).volume - n.left.box.volume +
                    inherit_cost
                )
            if n.right.is_leaf:
                cost_right = box.expand(n.right.box).volume + inherit_cost
            else:
                cost_right = (
                    box.expand(n.right.box).volume - n.right.box.volume +
                    inherit_cost
                )
            if (cost < cost_left) and (cost < cost_right):
                break
            n = n.left if cost_left < cost_right else n.right

        old_parent = n.parent
        new_parent = Node(
            box=box.expand(n.box),
            left=n,
            parent=old_parent,
            height=n.height + 1,
            region_ids=region_ids
        )
        n.parent = new_parent
        new_node = Node(box=box, parent=new_parent, region_ids=region_ids)
        new_parent.right = new_node
        if old_parent is None:
            self.root = new_parent
        else:
            if n is old_parent.left:
                old_parent.left = new_parent
            else:
                old_parent.right = new_parent

        # recalculate heights and aabbs to take into account new node
        n = new_node.parent
        while n is not None:
            # update sets of regions partially contained by parent nodes
            n.region_ids = n.region_ids.union(region_ids)
            n.height = 1 + max(n.left.height, n.right.height)
            n.box = n.left.box.expand(n.right.box)
            n = n.parent

    def query_regions_contained_in_box(self, box: BoundedAABB) -> Set[int]:
        if self.root is None:
            return set()
        if self.root.is_leaf:
            if box.contains(self.root.box):
                return self.root.region_ids
            else:
                return set()
        matching_regions = set()  # type: Set[int]
        for n in [self.root.left, self.root.right]:
            if n.box.overlaps(box):
                for region_id in n.region_ids:
                    if box.contains(self.region_boxes[region_id]):
                        matching_regions.add(region_id)
        return matching_regions

    def query_regions_axdir(self, region_id: int, axis: int,
                            direction: int) -> Set[int]:
        if direction not in {-1, 1}:
            raise Exception(
                'bad direction value: {}, expected to be in {}'.format(
                    direction, {-1, 1}
                )
            )
        if axis not in {0, 1, 2}:
            raise Exception(
                'bad axis value: {}, expected to be in {}'.format(
                    axis, {0, 1, 2}
                )
            )
        if region_id not in self.region_boxes or self.root is None:
            return set()
        region_box = self.region_boxes[region_id]
        box = deepcopy(self.root.box)
        if direction == 1:
            box._lb[axis] = region_box._ub[axis]
        else:
            box._ub[axis] = region_box._lb[axis]
        return self.query_regions_contained_in_box(box)
