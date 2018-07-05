from .query_resolution import *
from ..region_solver import RegionsSetSolver
from ..symbols_and_types import TypedSymbolTable
from ..regions import ExplicitVBR
from ..utils.data_manipulation import parse_region_label_map
import numpy as np

__all__ = ['RegionFrontend', 'QueryBuilder']


class RegionFrontend(QueryBuilder):

    def __init__(self):
        super().__init__(RegionsSetSolver(TypedSymbolTable()))

    def load_parcellation(self, parc_im, selected_labels=None):
        labels = parc_im.get_data()
        label_regions_map = parse_region_label_map(parc_im, selected_labels=selected_labels)
        res = []
        for region_name, region_key in label_regions_map.items():
            voxel_coordinates = np.transpose((labels == region_key).nonzero())
            region = ExplicitVBR(voxel_coordinates, parc_im.affine, parc_im.shape)
            res.append(self.add_region(region, result_symbol_name=region_name))

        return res
