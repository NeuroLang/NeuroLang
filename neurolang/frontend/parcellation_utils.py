from ..regions import ExplicitVBR
from ..utils.data_manipulation import parse_region_label_map
from .. import neurolang as nl
import numpy as np
from typing import AbstractSet

__all__ = ['load_parcellation_regions_to_solver']

def load_parcellation_regions_to_solver(solver, parc_im, k=None):
    labels = parc_im.get_data()
    label_regions_map = parse_region_label_map(parc_im, k)
    for region_name, region_key in label_regions_map.items():
        voxel_coordinates = np.transpose((labels == region_key).nonzero())
        region = ExplicitVBR(voxel_coordinates, parc_im.affine, parc_im.shape)
        solver.symbol_table[nl.Symbol[solver.type](region_name)] = \
            nl.Constant[AbstractSet[solver.type]](frozenset([region]))
