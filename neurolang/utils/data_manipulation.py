import nibabel as nib
import numpy as np
import copy
from xml.etree import ElementTree
from ..brain_tree import AABB


def parse_region_label_map(labeled_im):
    extension_header = ElementTree.fromstring(labeled_im.header.extensions[0].get_content())
    labeltable = {
        l.text: int(l.get('Key'))
        for l in extension_header.findall(".//Label")
        }
    del labeltable['???']
    return labeltable


def transform_to_ras_coordinate_system(labels_im, region_key):
    labels = labels_im.get_data()
    voxel_space_data = np.transpose((labels == region_key).nonzero())
    ras_space_data = nib.affines.apply_affine(
        labels_im.affine,
        voxel_space_data
    )
    return ras_space_data


def split_bounding_box(box_limits):
    bb1, bb2 = copy.copy(box_limits), copy.copy(box_limits)
    ax = np.argmax(box_limits[:, 1] - box_limits[:, 0])
    middle = (box_limits[ax, 0] + box_limits[ax, 1]) / 2
    bb1[ax] = [box_limits[ax, 0], middle]
    bb2[ax] = [middle, box_limits[ax, 1]]
    return AABB(tuple(bb1[:, 0]), tuple(bb1[:, 1])), AABB(tuple(bb2[:, 0]), tuple(bb2[:, 1]))


def add_non_empty_bbs_to_tree(nodes, boxes, tree, elements):
    nodes[:] = []
    bb_has_elements = np.zeros(len(boxes))
    dim = boxes[0].dim
    for value in elements:
        for i in range(len(boxes)):
            if bb_has_elements[i] == 0 and ((boxes[i].lb <= value).sum() + (value <= boxes[i].ub).sum()) == (2 * 3):
                bb_has_elements[i] = 1
                node = tree.add(boxes[i])
                nodes.append(node)
                break
        if sum(bb_has_elements) == len(boxes):
            break
    boxes = boxes[(bb_has_elements != 0)]
    return boxes
