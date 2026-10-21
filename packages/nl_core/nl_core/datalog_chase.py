from warnings import warn

from .datalog.chase import (
    ChaseNode,
    ChaseNaive,
    Chase,
    ChaseMGUMixin,
    ChaseRelationalAlgebraPlusCeriMixin
)

warn("This module is going to be deprecated. Switch to datalog.chase")

__all__ = [
    "ChaseNode",
    "DatalogChaseGeneral", "DatalogChase",
    "DatalogChaseMGUMixin", "DatalogChaseRelationalAlgebraMixin"
]

DatalogChase = Chase
DatalogChaseMGUMixin = ChaseMGUMixin
DatalogChaseRelationalAlgebraMixin = ChaseRelationalAlgebraPlusCeriMixin


class DatalogChaseGeneral(Chase, ChaseNaive):
    pass
