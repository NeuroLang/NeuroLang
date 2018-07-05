from xml.etree import ElementTree
import neurosynth as ns
from collections import Iterable
from fnmatch import fnmatch
import os


def parse_region_label_map(labeled_im, selected_labels=None):
    extension_header = ElementTree.fromstring(labeled_im.header.extensions[0].get_content())
    labels = extension_header.findall(".//Label")

    if selected_labels is None:
        labeltable = {
            l.text: int(l.get('Key'))
            for l in labels
            }
    elif isinstance(selected_labels, str):
        labeltable = {
            l.text: int(l.get('Key'))
            for l in labels
            if fnmatch(l.text, selected_labels)
        }
    elif isinstance(selected_labels, Iterable):
        labeltable = {
            l.text: int(l.get('Key'))
            for l in labels
            if l.text in selected_labels
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