from typing import Union, Tuple, List, Set
from copy import deepcopy

import numpy as np


class AABB:

    def __init__(self, lb, ub) -> None:
        _lb = np.asanyarray(lb, dtype=float)
        _ub = np.asanyarray(ub, dtype=float)
        _lb.setflags(write=False)
        _ub.setflags(write=False)
        self._limits = np.c_[_lb, _ub]
        self._limits.setflags(write=False)
        self._lb = self._limits[:, 0]
        self._ub = self._limits[:, 1]

    @property
    def lb(self) -> np.array:
        return self._lb

    @property
    def ub(self) -> np.array:
        return self._ub

    @property
    def center(self) -> np.array:
        return 0.5 * (self.lb + self.ub)

    @property
    def volume(self) -> float:
        return (self.ub - self.lb).prod()

    @property
    def width(self) -> np.array:
        return self.ub - self.lb

    @property
    def limits(self):
        return self._limits

    @property
    def dim(self):
        return len(self.lb)

    def union(self, other: 'AABB') -> 'AABB':
        return AABB(np.minimum(self.lb, other.lb),
                    np.maximum(self.ub, other.ub))

    def contains(self, other: 'AABB') -> bool:
        return ((other.lb >= self.lb).sum() + (other.ub <= self.ub).sum()) == 6

    def overlaps(self, other: 'AABB') -> bool:
        return ((other.ub > self.lb).sum() +
                (other.lb < self.ub).sum()) == 6

    def __eq__(self, other) -> bool:
        return np.all(self.lb == other.lb) and np.all(self.ub == other.ub)

    def __repr__(self):
        return 'AABB(lb={}, up={})'.format(tuple(self.lb), tuple(self.ub))

    def __hash__(self):
        return hash(self.limits.tobytes())

def _aabb_from_vertices(vertices) -> AABB:
    stacked = np.vstack(vertices)
    # take min and max in each dimension to get the triangle's bounding box
    return AABB(np.min(stacked, axis=0), np.max(stacked, axis=0))


class Node:

    def __init__(self,
                 box: AABB,
                 parent: Union[None, 'Node'] = None,
                 left: Union[None, 'Node'] = None,
                 right: Union[None, 'Node'] = None,
                 height: int = 0,
                 region_ids: Set[int] = set()) -> None:

        self.box = box
        self.parent = parent
        self.left = left
        self.right = right
        self.height = height
        self.region_ids = region_ids

    @property
    def is_leaf(self):
        return self.left is None


class Tree:

    def __init__(self) -> None:
        self.root = None  # type: Union[Node, None]
        self.region_boxes = dict()  # type: Dict[int, AABB]
        self.height = 0

    def expand_region_box(self, region_id: int, added_box: AABB) -> None:
        if region_id not in self.region_boxes:
            self.region_boxes[region_id] = added_box
        else:
            self.region_boxes[region_id] = \
                self.region_boxes[region_id].union(added_box)

    def add(self, box: AABB, region_ids: Set[int] = set()) -> None:

        for region_id in region_ids:
            self.expand_region_box(region_id, box)

        # if the tree is empty, just set root to the given node
        if self.root is None:
            self.root = Node(box=box, region_ids=region_ids)
            return

        n = self.root  # type: Node
        # go down until the tree until we reach a leaf
        while not n.is_leaf:
            if n.left.box.contains(box):
                n = n.left
                continue
            elif n.right is not None and n.right.box.contains(box):
                n = n.right
                continue
            elif n.box.contains(box):
                break
            # to decide whether to go to the left or right branch
            # we use a heuristic that takes into account the volume increase
            # of the left and right boxes after adding the new box
            combined_volume = box.union(n.box).volume
            cost = 2 * combined_volume
            inherit_cost = 2 * (combined_volume - n.box.volume)
            if n.left.is_leaf:
                cost_left = box.union(n.left.box).volume + inherit_cost
            else:
                cost_left = (box.union(n.left.box).volume -
                             n.left.box.volume + inherit_cost)
            if n.right.is_leaf:
                cost_right = box.union(n.right.box).volume + inherit_cost
            else:
                cost_right = (box.union(n.right.box).volume -
                              n.right.box.volume + inherit_cost)
            if (cost < cost_left) and (cost < cost_right):
                break
            n = n.left if cost_left < cost_right else n.right

        if n.box.contains(box):
            new_node = Node(box=box, parent=n, region_ids=region_ids)
            if n.left is None:
                n.left = new_node
            else:
                n.right = new_node
            n.region_ids = n.region_ids.union(region_ids)
            while n is not None:
                # update sets of regions partially contained by parent nodes
                n.region_ids = n.region_ids.union(region_ids)
                hrec = [n.left, n.right]
                n.height = 1 + max(h.height for h in hrec if h is not None) if hrec != [None, None] else 0
                n = n.parent
        else:
            old_parent = n.parent
            new_parent = Node(box=box.union(n.box),
                              left=n,
                              parent=old_parent,
                              height=n.height + 1,
                              region_ids=region_ids)
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
                n.box = n.left.box.union(n.right.box)
                n = n.parent

    def query_regions_contained_in_box(self, box: AABB) -> Set[int]:
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
            raise Exception('bad direction value: {}, expected to be in {}'
                            .format(direction, {-1, 1}))
        if axis not in {0, 1, 2}:
            raise Exception('bad axis value: {}, expected to be in {}'
                            .format(axis, {0, 1, 2}))
        if region_id not in self.region_boxes or self.root is None:
            return set()
        region_box = self.region_boxes[region_id]
        box = deepcopy(self.root.box)
        if direction == 1:
            box._lb[axis] = region_box._ub[axis]
        else:
            box._ub[axis] = region_box._lb[axis]
        return self.query_regions_contained_in_box(box)
