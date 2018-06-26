from xml.etree import ElementTree
from scipy.ndimage import label, generate_binary_structure
from ..regions import ExplicitVBR
import neurosynth as ns
import random
import numpy as np
import os


def parse_region_label_map(labeled_im, first_n=None):
    extension_header = ElementTree.fromstring(labeled_im.header.extensions[0].get_content())
    aList = extension_header.findall(".//Label")
    random.shuffle(aList)
    labeltable = {
        l.text: int(l.get('Key'))
        for l in aList[:first_n]
        }
    if '???' in labeltable.keys():
        del labeltable['???']
    return labeltable


def generate_neurosynth_dataset(path):
    if not os.path.exists(path):
        os.makedirs(path)
    ns.dataset.download(path=path, unpack=True)
    dataset = ns.Dataset(path + '/database.txt')
    dataset.add_features(path + '/features.txt')
    dataset.save(path + '/dataset.pkl')
    return dataset


def generate_connected_regions_set(data, affine, k=None):
    s = generate_binary_structure(3, 2)
    labeled_array, num_features = label(data, structure=s)
    emotion_regions = []
    for i in range(1, num_features):
        region_voxels = list(zip(*np.where(labeled_array == i)))
        emotion_regions.append(ExplicitVBR(region_voxels, affine))

    emotion_regions = sorted(emotion_regions, key=lambda x: len(x._voxels), reverse=True)[:k]
    return set(emotion_regions)
