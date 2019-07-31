from .query_resolution import QueryBuilderFirstOrder, QueryBuilderDatalog
from ..expression_walker import (
    add_match, Symbol, FunctionApplication, Constant
)
from ..solver import FirstOrderLogicSolver
from ..solver_datalog_extensional_db import ExtensionalDatabaseSolver
from ..region_solver import RegionSolver
from ..regions import (
    ExplicitVBR, ImplicitVBR, SphericalVolume
)
from ..utils.data_manipulation import parse_region_label_map
from .. import neurolang as nl
from ..solver_datalog_naive import DatalogBasic
from ..expression_walker import ExpressionBasicEvaluator

from typing import Any, AbstractSet, Callable
import numpy as np


__all__ = ['RegionFrontend', 'QueryBuilderDatalog', 'QueryBuilderFirstOrder']


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


class RegionFrontend(QueryBuilderFirstOrder):

    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendSolver()
        super().__init__(solver)
        isin_symbol = Symbol[Callable[[Any, AbstractSet[Any]], bool]]('isin')
        self.solver.symbol_table[isin_symbol] = Constant(function_isin)

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


class NeurolangDL(QueryBuilderDatalog):

    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendDatalogSolver()
        super().__init__(solver)
        isin_symbol = Symbol[Callable[[Any, AbstractSet[Any]], bool]]('isin')
        self.solver.symbol_table[isin_symbol] = Constant(function_isin)

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


class RegionFrontendDatalogSolver(
        RegionSolver,
        DatalogBasic,
        ExpressionBasicEvaluator
):
    pass
