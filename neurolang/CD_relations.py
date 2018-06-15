from .interval_algebra import *
from .regions import *
from functools import singledispatch
import numpy as np
import itertools

directions_dim_space = {'L': [0], 'R': [0], 'P': [1], 'A': [1], 'O': [0, 1, 2],
                          'I': [2], 'S': [2], 'F': [3]}

matrix_positions_per_directions = {'L': [0], 'O': [1], 'R': [2], 'P': [0],
                                   'A': [2], 'I': [0], 'S': [2], 'F': [2]}

inverse_directions = {'R': 'L', 'A': 'P', 'S': 'I',
                    'O': 'O', 'L': 'R', 'P': 'A', 'I': 'S'}


def cardinal_relation(region, reference_region, directions, refine_overlapping=False, max_granularity=20):

    region_tree, region_box = region.aabb_tree, region.bounding_box
    reference_tree, reference_box = reference_region.aabb_tree, reference_region.bounding_box
    region_tree_nodes, reference_tree_nodes = [region_tree.root], [reference_tree.root]
    mat = direction_matrix([region_tree.root.box], [reference_tree.root.box])
    dim = region_box.dim
    obtained = is_in_direction(mat, directions)

    if 'O' in directions and obtained and refine_overlapping and isinstance(region, ExplicitVBR):
        max_resolution_region = abs(np.linalg.eigvals(region._affine_matrix)[0:dim]) / 2
        max_resolution_reference = abs(np.linalg.eigvals(reference_region._affine_matrix)[0:dim]) / 2
        region_width, reference_width = region_box.width, reference_box.width

        max_resolution_reached_region = max_resolution_reached_reference = False
        region_elements, reference_elements = region.to_xyz(), reference_region.to_xyz()
        granularity_level = 1

        region_bbs = [region_box]
        reference_bbs = [reference_box]
        while obtained and granularity_level < max_granularity:
            if not max_resolution_reached_region:
                region_bbs = get_next_level_of_granularity_bbs(region_tree, region_tree_nodes, region_elements)
                max_resolution_reached_region = (max_resolution_region > region_width / (2 * granularity_level)).sum() == 3
                mat = direction_matrix(region_bbs, reference_bbs)
                obtained = is_in_direction(mat, directions)
                if not obtained:
                    return obtained

            if not max_resolution_reached_reference:
                reference_bbs = get_next_level_of_granularity_bbs(reference_tree, reference_tree_nodes, reference_elements)
                max_resolution_reached_reference = (max_resolution_reference > reference_width / (2 * granularity_level)).sum() == 3
                mat = direction_matrix(region_bbs, reference_bbs)
                obtained = is_in_direction(mat, directions)
                if not obtained:
                    return obtained

            if max_resolution_reached_region and max_resolution_reached_reference:
                return obtained
            granularity_level += 1
    return obtained


def get_next_level_of_granularity_bbs(tree, nodes, elements):

    def nodes_children_boxes(nodes):
        result = []
        for node in nodes:
            children = [node.left, node.right]
            if None in children:
                return list(map(lambda node: data_manipulation.split_bounding_box(node.box.limits), nodes)), False
            result += children
        nodes[:] = result
        return [node.box for node in result], True

    children_nodes_boxes, children_def = nodes_children_boxes(nodes)
    if children_def:
        return children_nodes_boxes
    bbs = np.array(list(itertools.chain.from_iterable(children_nodes_boxes)))
    bbs = data_manipulation.add_non_empty_bbs_to_tree(nodes, bbs, tree, elements)
    return bbs


def is_in_direction(matrix, direction):
    n = matrix.ndim
    idxs = [[0, 1, 2]] * n
    for i in direction:
        for dim in directions_dim_space[i]:
            idxs[n-1-dim] = matrix_positions_per_directions[i]
    return np.any(matrix[idxs] == 1)


def relation_vectors(intervals, other_region_intervals):
    obtained_vectors = []
    relations = [v_before, v_overlaps, v_during, v_meets, v_starts, v_finishes, v_equals]
    for i in range(len(intervals)):
        for f in relations:
            vector = f(intervals[i], other_region_intervals[i])
            if vector is not None:
                obtained_vectors.append(vector)
                break
    return obtained_vectors


@singledispatch
def direction_matrix(region_bbs, another_region_bbs):
    res = np.zeros((3,) * region_bbs[0].dim)
    for bb in region_bbs:
        for another_region_bb in another_region_bbs:
            rp_vector = relation_vectors(bb.limits, another_region_bb.limits)
            tensor = rp_vector[0].reshape(1, 3)
            for i in range(1, len(rp_vector)):
                tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
            res = np.logical_or(res, tensor).astype(int)
    return res


@direction_matrix.register(Region)
def _(region, other_region):
    rp_vector = relation_vectors(region.bounding_box.limits, other_region.bounding_box.limits)
    tensor = rp_vector[0].reshape(1, 3)
    for i in range(1, len(rp_vector)):
        tensor = np.kron(rp_vector[i].reshape((3,) + (1,) * i), tensor)
    return tensor
