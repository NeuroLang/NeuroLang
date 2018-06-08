import nibabel as nib
import numpy as np
from xml.etree import ElementTree


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


def coordinate_in_limits(coordinate, region_limits):
    for i in range(len(region_limits)):
        if not region_limits[i, 0] <= coordinate[i] <= region_limits[i, 1]:
            return False
    return True
