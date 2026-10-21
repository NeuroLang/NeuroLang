from typing import AbstractSet, Any

import numpy as np

from nl_expr import expressions as ir
from nl_core.datalog import DatalogProgram
from nl_core.datalog.chase import Chase
from nl_core.datalog.aggregation import (
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from nl_core.datalog.negation import DatalogProgramNegationMixin
from nl_expr.expression_walker import ExpressionBasicEvaluator
from nl_core.datalog.horn_clauses import Fol2DatalogMixin
from ..region_solver import RegionSolver
from ..regions import ExplicitVBR, ExplicitVBROverlay
from ..utils.data_manipulation import parse_region_label_map
from .datalog.squall import StripDimensionTypePredicatesMixin
from .datalog.sugar import (
    TranslateSSugarToSelectByColumn,
    TranslateSelectByFirstColumn,
    TranslateHeadConstantsToEqualities
)
from .datalog.sugar.spatial import (
    TranslateEuclideanDistanceBoundMatrixMixin,
    TranslateRegionDestroy,
)
from .type_resolution import TypeResolutionMixin
from ..commands import CommandsMixin
from .frontend_extensions import NumpyFunctionsMixin
from .query_resolution_datalog import QueryBuilderDatalog
from .query_resolution_expressions import Symbol

__all__ = ["NeurolangDL", "ExplicitVBR", "ExplicitVBROverlay", "Symbol"]


def function_isin(element: Any, set_: AbstractSet) -> bool:
    """Function for checking that an element is in a set"""
    return element in set_


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

            c = ir.Constant[self.program_ir.type](region)
            s = ir.Symbol[self.program_ir.type](region_name)
            self.program_ir.symbol_table[s] = c
            res.append(s)

        return res


class NeurolangDL(QueryBuilderDatalog):
    """
    Deterministic Datalog Frontend for Neurolang.
    QueryBuilderDatalog with a RegionFrontendDatalogSolver program
    intermediate representation
    """

    def __init__(self, program_ir=None, solver_class=None):
        if program_ir is None:
            if solver_class is not None:
                program_ir = solver_class()
            else:
                program_ir = RegionFrontendDatalogSolver()
        super().__init__(program_ir, chase_class=Chase)


class RegionFrontendDatalogSolver(
    TranslateToLogicWithAggregation,
    StripDimensionTypePredicatesMixin,
    TranslateSSugarToSelectByColumn,
    TranslateSelectByFirstColumn,
    TranslateHeadConstantsToEqualities,
    TranslateEuclideanDistanceBoundMatrixMixin,
    TypeResolutionMixin,
    TranslateRegionDestroy,
    Fol2DatalogMixin,
    RegionSolver,
    CommandsMixin,
    NumpyFunctionsMixin,
    DatalogWithAggregationMixin,
    BuiltinAggregationMixin,
    DatalogProgramNegationMixin,
    DatalogProgram,
    ExpressionBasicEvaluator,
):
    """
    DatalogProgram preloaded with Region symbols and builtins, and
    complemented with aggregation capabilities
    """

    pass
