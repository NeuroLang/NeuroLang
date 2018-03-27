import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..perior_tree import BoundedAABB, Boundary

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


def test_aabb_out_of_bound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((-3, -3), (-1,-1), period_bound)
    box1.adjust_to_bound(period_bound)
    assert box1 == BoundedAABB((7, 7), (9, 9), period_bound)

    # fig, ax = plt.subplots()
    # ax.add_patch(patches.Rectangle(box1[0], box1.width(), box1.height(), hatch='+', fill=False))
    # ax.axis([period_bound[0][0], period_bound[1][0], period_bound[0][1], period_bound[1][1]])
    # plt.show()

def test_aabbs_union_in_bound():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((5, 6), (6, 8), period_bound)
    assert box1.union(box2) == BoundedAABB((1, 1), (6, 8), period_bound)


    # #Todo: refa ploting
    # fig, ax = plt.subplots()
    # ax.add_patch(patches.Rectangle(box1[0], box1.width(), box1.height(), hatch='+', fill=False))
    # ax.add_patch(patches.Rectangle(box2[0], box2.width(), box2.height(), hatch='*', fill=False))
    # ax.axis([0, 10, 0, 10])
    # plt.show()

def test_aabbs_intersect():
    period_bound = Boundary((0, 0), (10, 10))
    box1 = BoundedAABB((1, 1), (3, 3), period_bound)
    box2 = BoundedAABB((2, 2), (4, 4), period_bound)
    assert box1.intersects(box1, period_bound)
    assert box1.intersects(box2, period_bound)
    assert box2.intersects(box1, period_bound)

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