from collections.abc import Iterable
from fnmatch import fnmatch
from xml.etree import ElementTree


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
