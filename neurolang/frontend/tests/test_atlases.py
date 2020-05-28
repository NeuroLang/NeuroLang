import nibabel
import nilearn.datasets
import pytest

from ...expressions import Symbol
from .. import RegionFrontend


def test_add_atlas_set_destrieux():
    destrieux_dataset = nilearn.datasets.fetch_atlas_destrieux_2009()
    labels = destrieux_dataset["labels"]
    image = nibabel.load(destrieux_dataset["maps"])
    name = "destrieux"
    frontend = RegionFrontend()
    frontend.add_atlas_set(name, labels, image)
    assert Symbol(name) in frontend.symbol_table
    assert len(frontend.symbol_table[Symbol(name)].value) == 149
