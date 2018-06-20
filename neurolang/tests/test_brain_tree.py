import numpy as np

from ..brain_tree import AABB, Tree


def _generate_random_box(x_bounds, y_bounds, z_bounds, size_bounds):
    lower_bound = np.array([np.random.uniform(*b) for b in (x_bounds, y_bounds, z_bounds)])
    upper_bound = lower_bound + np.random.uniform(*size_bounds, size=3)
    return AABB(lower_bound, upper_bound)


def test_aabb_contains():
    box1 = AABB((0, 0, 0), (1, 1, 1))
    box2 = AABB((0.3, 0.3, 0.3), (0.7, 0.7, 0.7))
    assert box1.contains(box2)
    assert not box2.contains(box1)
    assert box1.contains(box1)
    assert box2.contains(box2)

    box1 = AABB((-1, -1, -1), (1, 1, 1))
    box2 = AABB((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    assert box1.contains(box2)
    assert not box2.contains(box1)
    assert box1.contains(box1)
    assert box2.contains(box2)


def test_aabb_overlaps():
    box1 = AABB((0, 0, 0), (1, 1, 1))
    box2 = AABB((2, 2, 2), (3, 3, 3))
    assert not box1.overlaps(box2)
    box3 = AABB((0.9, 0.9, 0.9), (3, 3, 3))
    assert box1.overlaps(box3)
    assert box3.overlaps(box1)
    box3 = AABB((0.9, 0.9, 1.1), (3, 3, 3))
    assert not box1.overlaps(box3)
    assert not box3.overlaps(box1)


def test_aabb_union():
    box1 = AABB((0, 0, 0), (1, 1, 1))
    box2 = AABB((2, 2, 2), (3, 3, 3))
    assert box1.union(box2) == AABB((0, 0, 0), (3, 3, 3))

def test_tree_construction():
    tree = Tree()
    assert tree.root is None


def test_tree_add_bt():
    tree = Tree()
    box1 = AABB((0, 0, 0), (1, 1, 1))
    box2 = AABB((0.5, 0.5, 0.5), (2, 2, 2))
    tree.add(box1)
    assert tree.root is not None
    tree.add(box2)
    assert tree.root.box == box1.union(box2)
    assert tree.root.left.box == box1
    assert tree.root.right.box == box2

    tree = Tree()
    for _ in range(100):
        tree.add(_generate_random_box((-2, -1), (0, 1), (0, 1), (0.2, 0.7)))
        tree.add(_generate_random_box((1, 2), (0, 1), (0, 1), (0.2, 0.7)))


def test_tree_add_with_region_id():
    tree = Tree()
    box1 = AABB((1000, 1000, 1000), (2000, 500, 2000))
    tree.add(box1, region_ids={4})
    assert tree.root is not None
    assert tree.root.box == box1
    assert 4 in tree.root.region_ids


def test_tree_query_regions_contained_in_box():
    tree = Tree()
    tree.add(AABB((0, 0, 0), (1, 1, 1)), region_ids={1})
    box = AABB((-1, -1, -1), (2, 2, 2))
    assert tree.query_regions_contained_in_box(box) == {1}
    box = AABB((0.5, 0.5, 0.5), (2, 2, 2))
    assert tree.query_regions_contained_in_box(box) == set()


def test_tree_query_regions_axdir():
    tree = Tree()
    tree.add(AABB((0, 0, 0), (1, 1, 1)), region_ids={1})
    tree.add(AABB((2, 0, 0), (3, 1, 1)), region_ids={2})
    assert tree.query_regions_axdir(region_id=1, axis=0, direction=1) == {2}
    assert tree.query_regions_axdir(region_id=1, axis=1, direction=1) == set()
    assert tree.query_regions_axdir(region_id=2, axis=0, direction=-1) == {1}

    tree = Tree()
    tree.add(AABB((-0.5, -0.5, -20.5), (0.6, 0.6, -20)), region_ids={1})
    tree.add(AABB((0.5, 0.5, 20), (1.5, 1.5, 20.5)), region_ids={2})
    assert tree.query_regions_axdir(region_id=1, axis=2, direction=1) == {2}
    assert tree.query_regions_axdir(region_id=1, axis=0, direction=1) == set()
    assert tree.query_regions_axdir(region_id=1, axis=1, direction=1) == set()
    assert tree.query_regions_axdir(region_id=2, axis=2, direction=-1) == {1}
