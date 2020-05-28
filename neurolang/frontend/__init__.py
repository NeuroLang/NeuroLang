"""
``neurolang.frontend``
=====================

This module provides a user-friendly interface with NeuroLang through languages
with sugar syntax which are translated to intermediate representations and
compiled by the NeuroLang backend.

The frontend provides functions and spatial operators for loading,
constructing, transforming and combining brain regions.

Regions can be loaded from atlases such as Destrieux et al. [1]_ or from
coordinate-based meta-analysis database such as Neurosynth [2]_.

References
----------

.. [1] Destrieux, Christophe, Bruce Fischl, Anders Dale, and Eric Halgren.
   “Automatic Parcellation of Human Cortical Gyri and Sulci Using Standard
   Anatomical Nomenclature.” NeuroImage 53, no. 1 (October 15, 2010): 1–15.
   https://doi.org/10.1016/j.neuroimage.2010.06.010.

.. [2] Yarkoni, Tal, Russell A Poldrack, Thomas E Nichols, David C Van Essen,
   and Tor D Wager. “Large-Scale Automated Synthesis of Human Functional
   Neuroimaging Data.” Nature Methods 8, no. 8 (August 2011): 665–70.
   https://doi.org/10.1038/nmeth.1635.

Examples
--------

Examples of how to use the front-end are provided in the ``examples/`` folder
distributed with the ``neurolang`` package.

"""
from typing import AbstractSet, Any, Callable

import numpy as np
import nilearn.datasets
import nibabel

from ..exceptions import NeuroLangException
from .. import neurolang as nl
from ..datalog import DatalogProgram
from ..datalog.aggregation import Chase, DatalogWithAggregationMixin
from ..expression_walker import (Constant, ExpressionBasicEvaluator,
                                 FunctionApplication, Symbol, add_match)
from ..region_solver import RegionSolver
from ..regions import ExplicitVBR
from ..solver import FirstOrderLogicSolver
from ..solver_datalog_extensional_db import ExtensionalDatabaseSolver
from ..utils.data_manipulation import parse_region_label_map
from .query_resolution import QueryBuilderFirstOrder
from .query_resolution_datalog import QueryBuilderDatalog

__all__ = [
    'NeurolangDL', 'RegionFrontend',
    'QueryBuilderDatalog', 'QueryBuilderFirstOrder'
]


ATLAS_LOADERS = {
    "destrieux": nilearn.datasets.fetch_atlas_destrieux_2009,
}


def function_isin(element: Any, set_: AbstractSet) -> bool:
    '''Function for checking that an element is in a set'''
    return element in set_


class RegionFrontendSolver(
        ExtensionalDatabaseSolver,
        RegionSolver,
        FirstOrderLogicSolver
):
    @add_match(
        FunctionApplication(
            Constant(function_isin),
            (Constant, Constant[AbstractSet])
        )
    )
    def rewrite_isin(self, expression):
        '''Rewrite `isin` in Datalog syntax'''
        return self.walk(expression.args[1](expression.args[0]))


class UnsupportedAtlas(NeuroLangException):
    pass


class SymbolAlreadyExists(NeuroLangException):
    pass


class RegionFrontend(QueryBuilderFirstOrder):

    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendSolver()
        super().__init__(solver)
        isin_symbol = Symbol[Callable[[Any, AbstractSet[Any]], bool]]('isin')
        self.solver.symbol_table[isin_symbol] = Constant(function_isin)

    def load_parcellation(self, parc_im, selected_labels=None):
        labels = np.asanyarray(parc_im.dataobj)
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

    def load_atlas(self, atlas_name, symbol_name=None):
        if symbol_name is None:
            symbol_name = atlas_name
        if Symbol(symbol_name) in self.symbol_table:
            raise SymbolAlreadyExists(symbol_name)
        if atlas_name not in ATLAS_LOADERS:
            raise UnsupportedAtlas(atlas_name)
        atlas = ATLAS_LOADERS[atlas_name]()
        labels = atlas["labels"]
        image = nibabel.load(atlas["maps"])
        return self.add_atlas_set(symbol_name, labels, image)


class NeurolangDL(QueryBuilderDatalog):

    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendDatalogSolver()
        super().__init__(solver, chase_class=Chase)


class RegionFrontendDatalogSolver(
        RegionSolver,
        DatalogWithAggregationMixin,
        DatalogProgram,
        ExpressionBasicEvaluator
):
    pass
