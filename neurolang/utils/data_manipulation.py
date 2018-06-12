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


def region_data_limits(data):
    _, m = data.shape
    limits = np.array([(min(data[:, i]), max(data[:, i])) for i in range(m)])
    return tuple([tuple(limits[:, 0]), tuple(limits[:, 1])])


def split_bounding_box(box_limits):
    bb1, bb2 = copy.copy(box_limits), copy.copy(box_limits)
    ax = np.argmax([rng[1] - rng[0] for rng in box_limits])
    middle = (box_limits[ax, 0] + box_limits[ax, 1]) / 2
    bb1[ax] = [box_limits[ax, 0], middle]
    bb2[ax] = [middle, box_limits[ax, 1]]
    return AABB(tuple(bb1[:, 0]), tuple(bb1[:, 1])), AABB(tuple(bb2[:, 0]), tuple(bb2[:, 1]))


def add_non_empty_bb_to_tree(boxes, tree, elements):
    bb_elements = np.zeros(len(boxes))
    for i in range(len(boxes)):
        for value in elements:
            if element_in_bb_limits(value, boxes[i].limits):
                bb_elements[i] += 1
                break
    boxes = boxes[(bb_elements != 0)]
    for box in boxes:
        tree.add(box)
    return boxes


def element_in_bb_limits(coordinate, box_limits):
    for i in range(len(box_limits)):
        if not box_limits[i, 0] <= coordinate[i] <= box_limits[i, 1]:
            return False
    return True
