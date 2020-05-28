from typing import AbstractSet, Any, Callable

import numpy as np

from .. import neurolang as nl
from ..datalog import DatalogProgram
from ..datalog.aggregation import Chase, DatalogWithAggregationMixin
from ..expression_walker import (
    Constant,
    ExpressionBasicEvaluator,
    FunctionApplication,
    Symbol,
    add_match,
)
from ..region_solver import RegionSolver
from ..regions import ExplicitVBR
from ..solver import FirstOrderLogicSolver
from ..solver_datalog_extensional_db import ExtensionalDatabaseSolver
from ..utils.data_manipulation import parse_region_label_map
from .query_resolution import QueryBuilderFirstOrder
from .query_resolution_datalog import QueryBuilderDatalog
from .query_resolution_fol_through_datalog import (
    QueryBuilderFirstOrderThroughDatalog,
)

__all__ = [
    "NeurolangDL",
    "RegionFrontend",
    "QueryBuilderDatalog",
    "QueryBuilderFirstOrder",
]


def function_isin(element: Any, set_: AbstractSet) -> bool:
    """Function for checking that an element is in a set"""
    return element in set_


class RegionFrontendSolver(
    ExtensionalDatabaseSolver, RegionSolver, FirstOrderLogicSolver
):
    @add_match(
        FunctionApplication(
            Constant(function_isin), (Constant, Constant[AbstractSet])
        )
    )
    def rewrite_isin(self, expression):
        """Rewrite `isin` in Datalog syntax"""
        return self.walk(expression.args[1](expression.args[0]))


class LoadParcellationMixin:
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


class RegionFrontend(QueryBuilderFirstOrder, LoadParcellationMixin):
    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendSolver()
        super().__init__(solver)
        isin_symbol = Symbol[Callable[[Any, AbstractSet[Any]], bool]]("isin")
        self.solver.symbol_table[isin_symbol] = Constant(function_isin)


class RegionFrontendFolThroughDatalog(
    QueryBuilderFirstOrderThroughDatalog, LoadParcellationMixin
):
    def __init__(self, solver=None):
        super().__init__(solver)


class NeurolangDL(QueryBuilderDatalog):
    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendDatalogSolver()
        super().__init__(solver, chase_class=Chase)


class RegionFrontendDatalogSolver(
    RegionSolver,
    DatalogWithAggregationMixin,
    DatalogProgram,
    ExpressionBasicEvaluator,
):
    pass
