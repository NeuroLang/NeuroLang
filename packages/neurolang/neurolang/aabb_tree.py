from copy import deepcopy

import numpy as np


class AABB(object):

    def __init__(self, lb, ub):
        self._limits = np.asarray((lb, ub), dtype=float).T
        self._limits.setflags(write=False)
        self._lb = self._limits[:, 0]
        self._ub = self._limits[:, 1]

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def center(self):
        return 0.5 * (self.lb + self.ub)

    @property
    def volume(self):
        return (self.ub - self.lb).prod()

    @property
    def width(self):
        return self.ub - self.lb

    @property
    def limits(self):
        return self._limits

    @property
    def dim(self):
        return len(self.lb)

    def union(self, other):
        return AABB(np.minimum(self.lb, other.lb),
                    np.maximum(self.ub, other.ub))

    def contains(self, other):
        return all(other.lb >= self.lb) and all(other.ub <= self.ub)

    def overlaps(self, other):
        return ((other.ub > self.lb).sum() +
                (other.lb < self.ub).sum()) == 6

    def __eq__(self, other):
        return np.all(self.lb == other.lb) and np.all(self.ub == other.ub)

    def __repr__(self):
        return 'AABB(lb={}, up={})'.format(tuple(self.lb), tuple(self.ub))

    def __hash__(self):
        return hash(self.limits.tobytes())


def aabb_from_vertices(vertices):
    vertices = np.atleast_2d(vertices)
    lb, ub = np.min(vertices, axis=0), np.max(vertices, axis=0)
    return AABB(lb, ub)


class Node(object):

    def __init__(self, box, parent=None, left=None, right=None, height=0,
                 regions=None):
        self.box = box
        self.parent = parent
        self._left = left
        self._right = right
        if regions is None:
            regions = set()
        self.regions = regions
        self.height = height
        self._needs_update = True

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @left.setter
    def left(self, value):
        self._left = value
        self._needs_update = True

    @right.setter
    def right(self, value):
        self._right = value
        self._needs_update = True

    @property
    def is_leaf(self):
        self._update_state()
        return self._is_leaf

    @property
    def children(self):
        self._update_state()
        return self._children

    def __repr__(self):
        self._update_state()
        return ('Node(regions={}, box={}, is_leaf={})'
                .format(self.regions, self.box, self.is_leaf))

    def _update_state(self):
        if not self._needs_update:
            return

        self._is_leaf = self.left is None and self.right is None
        self._children = []
        if self.left is not None:
            self._children.append(self.left)
        if self.right is not None:
            self._children.append(self.right)
        self._needs_update = False


class Tree(object):

    def __init__(self):
        self.root = None
        self.region_boxes = dict()
        self.height = 0

    def expand_region_box(self, region_id, added_box):
        if region_id not in self.region_boxes:
            self.region_boxes[region_id] = added_box
        else:
            self.region_boxes[region_id] = \
                self.region_boxes[region_id].union(added_box)

    def add_left(self, box, regions=None):
        return self.add_in_direction('left', box, regions)

    def add_right(self, box, regions=None):
        return self.add_in_direction('right', box, regions)

    def add_in_direction(self, direction, box, regions=None):
        if regions is None:
            regions = set()
        for region_id in regions:
            self.expand_region_box(region_id, box)
        n = self.root
        while not n.is_leaf:
            if n.left is not None and n.left.box.contains(box):
                n = n.left
                continue
            elif n.right is not None and n.right.box.contains(box):
                n = n.right
                continue
            elif n.box.contains(box):
                break
        new_node = Node(box=box, parent=n, regions=regions)
        if direction == 'left':
            n.left = new_node
        elif direction == 'right':
            n.right = new_node
        while n is not None:
            n.regions = n.regions.union(regions)
            hrec = (n.left, n.right)
            n.height = 1 + max(h.height for h in hrec if h is not None)
            self.height = max(self.height, n.height)
            n = n.parent

    def add(self, box, regions=None):
        if regions is None:
            regions = set()
        for region_id in regions:
            self.expand_region_box(region_id, box)

        # if the tree is empty, just set root to the given node
        if self.root is None:
            self.root = Node(box=box, regions=regions)
            return

        n = self.root
        # go down until the tree until we reach a leaf
        while not n.is_leaf:
            # if we stumble upon the same box
            if n.box == box:
                break
            # if left box contains the new box, go there
            if n.left is not None and n.left.box.contains(box):
                n = n.left
                continue
            # if right box contains the new box, go there
            elif n.right is not None and n.right.box.contains(box):
                n = n.right
                continue
            # to decide whether to go to the left or right branch
            # we use a heuristic that takes into account the volume increase
            # of the left and right boxes after adding the new box
            combined_volume = box.union(n.box).volume
            # cost of merging new box with current box
            cost = 2 * combined_volume
            inherit_cost = 2 * (combined_volume - n.box.volume)
            cost_left = box.union(n.left.box).volume + inherit_cost
            if n.left is None or not n.left.is_leaf:
                cost_left -= n.left.box.volume

            cost_right = box.union(n.right.box).volume + inherit_cost
            if n.right is None or not n.right.is_leaf:
                cost_right -= n.right.box.volume

            if (cost < cost_left) and (cost < cost_right):
                break
            # if it's cheaper to go left, we go left
            if cost_left < cost_right:
                n = n.left
            # otherwise, we go right
            else:
                n = n.right

        # if we ended up on the same box
        if n.box == box:
            # update regions to include the newly added regions
            n.regions = n.regions.union(regions)
            return

        new_node = Node(box=box, regions=regions)
        new_parent = Node(box=n.box.union(box), parent=n.parent,
                          left=n, right=new_node)
        new_node.parent = new_parent
        # if node was the root of the tree, update the root
        if n.parent is None:
            self.root = new_parent
        # else, update the corresponding child (left or right) of the
        # node's parent to be the new parent
        else:
            if n.parent.left is n:
                n.parent.left = new_parent
            elif n.parent.right is n:
                n.parent.right = new_parent
            else:
                raise Exception('Something went wrong')
        # update the node's parent to the newly created parent
        n.parent = new_parent
        # recalculate heights and aabbs to take into account new node
        self._update_parents(new_node)

    @staticmethod
    def _update_parents(starting_node):
        n = starting_node.parent
        while n is not None:
            if n.left is not None:
                n.regions = n.left.regions.union(
                    n.right.regions if n.right is not None else set()
                )
            elif n.right is not None:
                n.regions = n.right.regions
            hrec = [n.left, n.right]
            n.height = 1 + max(h.height for h in hrec if h is not None)
            if n.right is not None:
                n.box = n.left.box.union(n.right.box)
            n = n.parent

    def query_regions_contained_in_box(self, box):
        if self.root is None:
            return set()
        if self.root.is_leaf:
            if box.contains(self.root.box):
                return self.root.regions
            else:
                return set()
        matching_regions = set()
        for n in (self.root.left, self.root.right):
            if n.box.overlaps(box):
                for region_id in n.regions:
                    if box.contains(self.region_boxes[region_id]):
                        matching_regions.add(region_id)
        return matching_regions

    def query_regions_axdir(self, region_id, axis, direction):
        if direction not in (-1, 1):
            raise Exception('bad direction value: {}, expected to be in {}'
                            .format(direction, (-1, 1)))
        if axis not in (0, 1, 2):
            raise Exception('bad axis value: {}, expected to be in {}'
                            .format(axis, (0, 1, 2)))
        if region_id not in self.region_boxes or self.root is None:
            return set()
        region_box = self.region_boxes[region_id]
        box = deepcopy(self.root.box)
        if direction == 1:
            box._lb[axis] = region_box._ub[axis]
        else:
            box._ub[axis] = region_box._lb[axis]
        return self.query_regions_contained_in_box(box)

    def _query_overlapping_regions_rec(self, node, region_box):
        if node is None or not node.box.overlaps(region_box):
            return set()
        elif node.is_leaf:
            return set(
                region for region in node.regions
                if self.region_boxes[region].overlaps(region_box)
            )
        else:
            left_matches = self._query_overlapping_regions_rec(node.left,
                                                               region_box)
            right_matches = self._query_overlapping_regions_rec(node.right,
                                                                region_box)
            return left_matches.union(right_matches)

    def query_overlapping_regions(self, region):
        if region not in self.region_boxes or self.root is None:
            return set()
        region_box = self.region_boxes[region]
        # normally, this should never happen
        if not self.root.box.contains(region_box):
            return set()
        matching = self._query_overlapping_regions_rec(self.root, region_box)
        matching = matching.difference({region})
        return matching
