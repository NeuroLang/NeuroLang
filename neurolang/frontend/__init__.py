from typing import AbstractSet, Any

import numpy as np

from .. import neurolang as nl
from ..datalog import DatalogProgram
from ..datalog.aggregation import (
    Chase,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..expression_walker import ExpressionBasicEvaluator
from ..region_solver import RegionSolver
from ..regions import ExplicitVBR, ExplicitVBROverlay
from ..utils.data_manipulation import parse_region_label_map
from .query_resolution_datalog import QueryBuilderDatalog

__all__ = [
    "NeurolangDL",
    "QueryBuilderDatalog",
    "ExplicitVBR",
    "ExplicitVBROverlay",
]


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

            c = nl.Constant[self.program_ir.type](region)
            s = nl.Symbol[self.program_ir.type](region_name)
            self.program_ir.symbol_table[s] = c
            res.append(s)

        return res


class NeurolangDL(QueryBuilderDatalog):
    def __init__(self, program_ir=None):
        if program_ir is None:
            program_ir = RegionFrontendDatalogSolver()
        super().__init__(program_ir, chase_class=Chase)


class RegionFrontendDatalogSolver(
    TranslateToLogicWithAggregation,
    RegionSolver,
    DatalogWithAggregationMixin,
    DatalogProgram,
    ExpressionBasicEvaluator,
):
    pass
