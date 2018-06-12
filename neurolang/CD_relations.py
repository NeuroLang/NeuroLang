from .interval_algebra import *
from .regions import *
from functools import singledispatch
import numpy as np
import itertools

directions_dim_space = {'L': [0], 'R': [0], 'P': [1], 'A': [1], 'O': [0, 1],
                          'I': [2], 'S': [2], 'F': [3]}

matrix_positions_per_directions = {'L': [0], 'O': [1], 'R': [2], 'P': [0],
                                   'A': [2], 'I': [0], 'S': [2], 'F': [2]}

inverse_directions = {'R': 'L', 'A': 'P', 'S': 'I',
                    'O': 'O', 'L': 'R', 'P': 'A', 'I': 'S'}


def cardinal_relation(region, reference_region, directions, refine_overlapping=False, max_granularity=5):

    region_tree = region.aabb_tree
    reference_tree = reference_region.aabb_tree
    region_tree_nodes, reference_tree_nodes = [region_tree.root], [reference_tree.root]
    mat = direction_matrix([region_tree.root.box], [reference_tree.root.box])
    obtained = is_in_direction(mat, directions)

    if 'O' in directions and refine_overlapping and isinstance(region, ExplicitVBR):
        region_elements, reference_elements = region.to_xyz(), reference_region.to_xyz()
        granularity_level = 0
        while obtained and granularity_level < max_granularity:

            region_bbs = get_next_level_of_granularity_bbs(region_tree, region_tree_nodes, region_elements)
            mat = direction_matrix(region_bbs, [node.box for node in reference_tree_nodes])
            obtained = is_in_direction(mat, directions)
            if not obtained:
                return obtained

            reference_bbs = get_next_level_of_granularity_bbs(reference_tree, reference_tree_nodes, reference_elements)
            mat = direction_matrix(region_bbs, reference_bbs)
            obtained = is_in_direction(mat, directions)
            if not obtained:
                return obtained

            region_tree_nodes = children_of_tree_node(region_tree_nodes)
            reference_tree_nodes = children_of_tree_node(reference_tree_nodes)
            granularity_level += 1
    return obtained


def children_of_tree_node(nodes):
    result = []
    for node in nodes:
        result += [node for node in [node.left, node.right] if node is not None]
    return result


def get_next_level_of_granularity_bbs(tree, nodes, elements):
    if np.all([node.is_leaf for node in nodes]):
        children_boxes = list(map(lambda node: data_manipulation.split_bounding_box(node.box.limits), nodes))
        bbs = np.array([v for split in children_boxes for v in [split[0], split[1]]])
        bbs = data_manipulation.add_non_empty_bb_to_tree(bbs, tree, elements)
    else:
        children = children_of_tree_node(nodes)
        bbs = [v.box for v in children]
    return bbs


@singledispatch
def direction_matrix(region_bbs, another_region_bbs):
    res = np.zeros((3,) * region_bbs[0].dim)
    for bb in region_bbs:
        for another_region_bb in another_region_bbs:
            relations = get_intervals_relations(bb.limits, another_region_bb.limits)
            rp_vector = [relative_position_vector(r) for r in relations]
            tensor = rp_vector[0].reshape(1, 3)
            for i in range(1, len(relations)):
                tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
            res = np.logical_or(res, tensor).astype(int)
    return res


@direction_matrix.register(Region)
def _(region, other_region):
    relations = get_intervals_relations(region.bounding_box.limits, other_region.bounding_box.limits)
    rp_vector = [relative_position_vector(r) for r in relations]
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(relations)):
        tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
    return tensor


def is_in_direction(matrix, direction):

    n = matrix.ndim
    idxs = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            idxs[n-1-dim] = matrix_positions_per_directions[i]
    return np.any(matrix[idxs] == 1)


def get_intervals_relations(intervals, other_region_intervals):
    obtained_relation_per_axis = [''] * len(intervals)
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    for f in relations:
        for i in range(len(obtained_relation_per_axis)):
            if f(intervals[i], other_region_intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0])
            elif f(other_region_intervals[i], intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0] + 'i')
        if np.all(np.array(obtained_relation_per_axis) != ''):
            break
    return tuple(obtained_relation_per_axis)


def relative_position_vector(relation):
    if relation in ['d', 's', 'f', 'e']:
        return np.array((0, 1, 0))
    elif relation in ['m', 'b']:
        return np.array((1, 0, 0))
    elif relation in ['mi', 'bi']:
        return np.array((0, 0, 1))
    elif relation in ['o', 'fi']:
        return np.array((1, 1, 0))
    elif relation in ['oi', 'si']:
        return np.array((0, 1, 1))
    return np.array((1, 1, 1))
