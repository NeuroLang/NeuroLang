from xml.etree import ElementTree


def parse_region_label_map(labeled_im, first_n=None):
    extension_header = ElementTree.fromstring(labeled_im.header.extensions[0].get_content())
    labeltable = {
        l.text: int(l.get('Key'))
        for l in extension_header.findall(".//Label")[:first_n]
        }
    del labeltable['???']
    return labeltable
