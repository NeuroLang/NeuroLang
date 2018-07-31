from xml.etree import ElementTree
from collections import Iterable
from fnmatch import fnmatch
import os
import logging

try:
    import neurosynth as ns
    __has_neurosynth__ = True
except ModuleNotFoundError:
    __has_neurosynth__ = False


def parse_region_label_map(
    labeled_im, selected_labels=None
):
    extension_header = ElementTree.fromstring(
        labeled_im.header.extensions[0].get_content()
    )
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


def fetch_neurosynth_data(terms, frequency_threshold=0.05, q=0.01, prior=0.5):
    if not __has_neurosynth__:
        raise NotImplemented("Neurosynth not installed")

    file_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(file_dir, '../utils/neurosynth')
    file = os.path.join(path, 'dataset.pkl')

    if not os.path.isfile(file):
        logging.info(
            f'Downloading neurosynth database and features in path: {path}'
        )
        dataset = download_ns_dataset(path)
    else:
        dataset = ns.Dataset.load(file)

    studies_ids = dataset.get_studies(
        features=terms, frequency_threshold=frequency_threshold
    )
    ma = ns.meta.MetaAnalysis(dataset, studies_ids, q=q, prior=prior)
    data = ma.images['pAgF_z_FDR_0.01']
    masked_data = dataset.masker.unmask(data)
    affine = dataset.masker.get_header().get_sform()
    dim = dataset.masker.dims
    return masked_data, affine, dim


def download_ns_dataset(path):
    if __has_neurosynth__:
        if not os.path.exists(path):
            os.makedirs(path)
        ns.dataset.download(path=path, unpack=True)
        dataset = ns.Dataset(os.path.join(path, 'database.txt'))
        dataset.add_features(os.path.join(path, 'features.txt'))
        dataset.save(os.path.join(path, 'dataset.pkl'))
        return dataset
    else:
        raise NotImplemented("Neurosynth not installed in the system")
