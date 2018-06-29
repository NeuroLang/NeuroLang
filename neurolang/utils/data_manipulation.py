from xml.etree import ElementTree
import neurosynth as ns
import os
import random


def parse_region_label_map(labeled_im, first_n=None):
    extension_header = ElementTree.fromstring(labeled_im.header.extensions[0].get_content())
    labels = extension_header.findall(".//Label")
    random.shuffle(labels)
    labeltable = {
        l.text: int(l.get('Key'))
        for l in labels[:first_n]
        }
    if '???' in labeltable.keys():
        del labeltable['???']
    return labeltable


def fetch_neurosynth_dataset(path):
    if not os.path.exists(path):
        os.makedirs(path)
    ns.dataset.download(path=path, unpack=True)
    dataset = ns.Dataset(os.path.join(path, 'database.txt'))
    dataset.add_features(os.path.join(path, 'features.txt'))
    dataset.save(os.path.join(path, 'dataset.pkl'))
    return dataset