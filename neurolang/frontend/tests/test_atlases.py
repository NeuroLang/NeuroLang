import nibabel
import nilearn.datasets
import pytest

from ...expressions import Symbol
from .. import (
    RegionFrontend,
    SymbolAlreadyExistsException,
    UnsupportedAtlasException,
)


def test_add_atlas_set_destrieux():
    destrieux_dataset = nilearn.datasets.fetch_atlas_destrieux_2009()
    labels = destrieux_dataset["labels"]
    image = nibabel.load(destrieux_dataset["maps"])
    name = "destrieux"
    frontend = RegionFrontend()
    frontend.add_atlas_set(name, labels, image)
    assert Symbol(name) in frontend.symbol_table
    assert len(frontend.symbol_table[Symbol(name)].value) == 149


def test_load_atlas_destrieux():
    frontend = RegionFrontend()
    frontend.load_atlas("destrieux")
    assert Symbol("destrieux") in frontend.symbol_table
    with pytest.raises(SymbolAlreadyExistsException):
        frontend.load_atlas("destrieux")
    frontend.load_atlas("destrieux", "other_name")
    assert Symbol("other_name") in frontend.symbol_table


def test_load_unsupported_atlas_exception():
    frontend = RegionFrontend()
    with pytest.raises(UnsupportedAtlasException):
        frontend.load_atlas("unknown_atlas")
