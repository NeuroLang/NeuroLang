import numpy as np

from ..perior_tree import BoundedAABB, Boundary, Tree

def _generate_random_box(x_bounds, y_bounds, boundry, size_bounds):
    lower_bound = np.array([np.random.uniform(*b) for b in (x_bounds, y_bounds)])
    upper_bound = lower_bound + np.random.uniform(*size_bounds, size=2)
    return BoundedAABB(lower_bound, upper_bound, boundry)

def test_point_adjust_position():
    period_bound = Boundary((0, 0), (10, 10))
    point = np.asanyarray((-3, -7))
    inbound_point = period_bound.adjust_position(point)
    assert inbound_point == (7, 3)

def test_vector_adjust_direction():
    period_bound = Boundary((0, 0), (10, 10))
    point = np.asanyarray((9.5, 0.5))
    dir_vec = period_bound.adjust_direction(point)
    assert dir_vec == (-0.5, 0.5)


    # #TODO: refa ploting
    # rect = plt.Rectangle(period_bound[0], period_bound.width(), period_bound.height(), facecolor="#aaaaaa")
    # fig, ax = plt.subplots()
    # ax.add_patch(rect)
    #
    # plt.plot(point[0], point[1], 'ro')
    # plt.show()


def test_adjust_aabb_out_of_bound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((-3, -3), (-1,-1), period_bound)
    box1.adjust_to_bound()
    assert box1 == BoundedAABB((7, 7), (9, 9), period_bound)

    # fig, ax = plt.subplots()
    # ax.add_patch(patches.Rectangle(box1[0], box1.width(), box1.height(), hatch='+', fill=False))
    # ax.axis([period_bound[0][0], period_bound[1][0], period_bound[0][1], period_bound[1][1]])
    # plt.show()

def test_aabbs_union_in_bound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((5, 6), (6, 8), period_bound)
    assert box1.expand(box2) == BoundedAABB((1, 1), (6, 8), period_bound)


    # #Todo: refa ploting
    # fig, ax = plt.subplots()
    # ax.add_patch(patches.Rectangle(box1[0], box1.width(), box1.height(), hatch='+', fill=False))
    # ax.add_patch(patches.Rectangle(box2[0], box2.width(), box2.height(), hatch='*', fill=False))
    # ax.axis([0, 10, 0, 10])
    # plt.show()

def test_contains_aabbs():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((0, 0), (2,2), period_bound)
    assert not box1.contains(box2)

    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((1.5, 1.5), (2,2), period_bound)
    assert box1.contains(box2)

    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((1, 1), (3, 3), period_bound)
    assert box1.contains(box2)

    box1 = BoundedAABB((-10, -10), (0, 0), period_bound)
    box2 = BoundedAABB((1, 1), (3, 3), period_bound)
    assert box1.contains(box2)


def test_aabbs_intersect():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((2, 2), (4, 4), period_bound)
    assert box1.intersects(box1)
    assert box1.intersects(box2)
    assert box2.intersects(box1)

    box1 = BoundedAABB((1, 1), (2, 2), period_bound)
    box2 = BoundedAABB((3, 3), (4, 4), period_bound)
    assert not box1.intersects(box2)

    box1 = BoundedAABB((-2, -2), (0, 0), period_bound)
    box2 = BoundedAABB((9, 9), (10, 10), period_bound)
    assert box1.intersects(box2)


def test_expand_aabb_point():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((0, 0), (5, 5), period_bound)
    assert box1.expand((7, 7)) == BoundedAABB((0, 0), (7, 7), period_bound)

def test_expand_aabb_point_outbound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((9, 9), (10, 10), period_bound)
    assert box1.expand((14, 9)) == BoundedAABB((9, 9), (4, 9), period_bound)

def test_expand_aabbs_inbound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((0, 0), (5, 5), period_bound)
    box2 = BoundedAABB((2, 2), (7, 7), period_bound)
    assert box1.expand(box2) == BoundedAABB((0, 0), (7, 7), period_bound)


def test_expand_aabbs_outside_bound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((0, 0), (1, 1), period_bound)
    box2 = BoundedAABB((9, 0), (10, 1), period_bound)
    assert box1.expand(box2) == BoundedAABB((9, 0), (1, 1), period_bound)


def test_boundry_eq():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((0, 0), (5, 5), period_bound)
    box2 = BoundedAABB((0, 0), (5, 5), period_bound)
    assert box1 == box2


def test_tree_construction():
    tree = Tree()
    assert tree.root is None


def test_tree_add():
    period_bound = Boundary((0, 0), (10, 10))
    tree = Tree()
    box1 = BoundedAABB((0, 0), (1, 1), period_bound)
    box2 = BoundedAABB((0.5, 0.5), (2, 2), period_bound)
    tree.add(box1)
    assert tree.root is not None
    tree.add(box2)
    assert tree.root.box == box1.expand(box2)
    assert tree.root.left.box == box1
    assert tree.root.right.box == box2

    for _ in range(100):
        tree.add(_generate_random_box((-2, -1), (0, 1), period_bound, (0.2, 0.7)))
        tree.add(_generate_random_box((1, 2), (0, 1), period_bound, (0.2, 0.7)))

def test_tree_query_regions_contained_in_box():
    period_bound = Boundary((0, 0), (10, 10))
    tree = Tree()
    tree.add(BoundedAABB((2, 2), (3, 3), period_bound), region_ids={1})
    box = BoundedAABB((1, 1), (4, 4), period_bound)
    assert tree.query_regions_contained_in_box(box) == {1}
    box = BoundedAABB((2.5, 2.5), (4, 4), period_bound)
    assert tree.query_regions_contained_in_box(box) == set()
