from .query_resolution import QueryBuilder
from ..region_solver import RegionsSetSolver
from ..symbols_and_types import TypedSymbolTable
from ..regions import ExplicitVBR, SphericalVolume, region_set_from_masked_data, region_union, take_principal_regions
from ..utils.data_manipulation import parse_region_label_map, fetch_neurosynth_dataset
from .. import neurolang as nl

import numpy as np
import os
from uuid import uuid1
try:
    from neurosynth import Dataset, meta
    __has_neurosynth__ = True
except ModuleNotFoundError:
    __has_neurosynth__ = False


__all__ = ['RegionFrontend', 'QueryBuilder']


class RegionFrontend(QueryBuilder):

    def __init__(self, solver=None):
        if solver is None:
            solver = RegionsSetSolver(TypedSymbolTable())
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
            #res.append(self.add_region(region, result_symbol_name=region_name))
            c = nl.Constant[self.solver.type](region)
            s = nl.Symbol[self.solver.type](region_name)
            self.solver.symbol_table[s] = c
            res.append(s)
        
        return res

    def load_neurosynth_term_regions(self, term: str, k=5, result_symbol_name=None):

        if not __has_neurosynth__:
            raise NotImplemented("Neurosynth not installed")

        if not result_symbol_name:
            result_symbol_name = term.replace(" ", "_") + '_region'
        file_dir = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(file_dir, '../utils/neurosynth')
        file = os.path.join(path, 'dataset.pkl')
        if not os.path.isfile(file):
            dataset = fetch_neurosynth_dataset(path)
        else:
            dataset = Dataset.load(file)

        studies_ids = dataset.get_studies(features=term, frequency_threshold=0.05)
        ma = meta.MetaAnalysis(dataset, studies_ids, q=0.01, prior=0.5)
        data = ma.images['pAgF_z_FDR_0.01']
        affine = dataset.masker.get_header().get_sform()
        dim = dataset.masker.dims
        masked_data = dataset.masker.unmask(data)
        components = take_principal_regions(region_set_from_masked_data(masked_data, affine, dim), k)
        c = nl.Constant[self.solver.type](region_union(components))
        s = nl.Symbol[self.solver.type](result_symbol_name)
        self.solver.symbol_table[s] = c

        # res = []
        # for i, region in enumerate(components):
        #     c = nl.Constant[self.solver.type](region)
        #     s = nl.Symbol[self.solver.type](term + f'_{i}')
        #     self.solver.symbol_table[s] = c
        #     res.append(s)
        return s

    def sphere(self, center, radius, affine, im_shape, result_symbol_name=None):

        if result_symbol_name is None:
            result_symbol_name = str(uuid1())

        sr = SphericalVolume(center, radius)
        explicit_sr = sr.to_explicit_vbr(affine, im_shape)
        c = nl.Constant[self.solver.type](explicit_sr)
        s = nl.Symbol[self.solver.type](result_symbol_name)
        self.solver.symbol_table[s] = c
        return s
