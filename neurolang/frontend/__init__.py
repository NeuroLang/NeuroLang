from .query_resolution import QueryBuilder
from ..solver import DatalogSolver
from ..region_solver_ds import RegionSolver
from ..regions import ExplicitVBR
from ..utils.data_manipulation import parse_region_label_map
from .. import neurolang as nl
import numpy as np

__all__ = ['RegionFrontend', 'QueryBuilder']


class RegionFrontendSolver(
        RegionSolver,
        DatalogSolver
):
    pass


class RegionFrontend(QueryBuilder):

    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendSolver()
        super().__init__(solver)

    def load_parcellation(self, parc_im, selected_labels=None):
        labels = parc_im.get_data()
        label_regions_map = parse_region_label_map(
            parc_im, selected_labels=selected_labels
        )
        res = []
        for region_name, region_key in label_regions_map.items():
            voxel_coordinates = np.transpose((labels == region_key).nonzero())
            region = ExplicitVBR(
                voxel_coordinates, parc_im.affine, parc_im.shape
            )
            # res.append(
            #   self.add_region(region, result_symbol_name=region_name)
            # )
            c = nl.Constant[self.solver.type](region)
            s = nl.Symbol[self.solver.type](region_name)
            self.solver.symbol_table[s] = c
            res.append(s)

        return res
