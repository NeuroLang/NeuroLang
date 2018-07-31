from .query_resolution import QueryBuilder
from ..region_solver_ds import RegionSolver
from ..symbols_and_types import TypedSymbolTable
from ..regions import (
    ExplicitVBR, ImplicitVBR, SphericalVolume,
    region_set_from_masked_data, take_principal_regions
)
from ..utils.data_manipulation import (
    parse_region_label_map, fetch_neurosynth_data
)
from .. import neurolang as nl
import numpy as np

__all__ = ['RegionFrontend', 'QueryBuilder']


class RegionFrontend(QueryBuilder):
    def __init__(self, solver=None):
        if solver is None:
            solver = RegionSolver(TypedSymbolTable())
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
            c = nl.Constant[self.solver.type](region)
            s = nl.Symbol[self.solver.type](region_name)
            self.solver.symbol_table[s] = c
            res.append(s)

        return res

    def sphere(self, center, radius, result_symbol_name=None):

        sr = SphericalVolume(center, radius)
        symbol = self.add_region(sr, result_symbol_name)
        return symbol

    def make_implicit_regions_explicit(self, affine, dim):

        for region_symbol_name in self.region_names:
            region_symbol = self.get_symbol(region_symbol_name)
            region = region_symbol.value
            if isinstance(region, ImplicitVBR):
                self.add_region(
                    region.to_explicit_vbr(affine, dim), region_symbol_name
                )

    def load_neurosynth_term_region(
        self, term: str, components=None, result_symbol_name=None
    ):

        if not result_symbol_name:
            result_symbol_name = \
                term.replace(" ", "_") + '_region'  # this or uuid

        data, affine, dim = fetch_neurosynth_data(term)

        region_set = region_set_from_masked_data(data, affine, dim)
        if components:
            region_set = take_principal_regions(region_set, components)

        c = nl.Constant[self.solver.set_type](region_set)
        s = nl.Symbol[self.solver.set_type](result_symbol_name)
        self.solver.symbol_table[s] = c
        return s
