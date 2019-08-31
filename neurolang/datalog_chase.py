from warnings import warn

from .datalog.chase import (
    ChaseNode,
    ChaseGeneral,
    Chase,
    ChaseMGUMixin,
    ChaseRelationalAlgebraMixin
)

warn("This module is going to be deprecated. Switch to datalog.chase")

__all__ = [
    "ChaseNode",
    "DatalogChaseGeneral", "DatalogChase",
    "DatalogChaseMGUMixin", "DatalogChaseRelationalAlgebraMixin"
]

DatalogChase = Chase
DatalogChaseGeneral = ChaseGeneral
DatalogChaseMGUMixin = ChaseMGUMixin
DatalogChaseRelationalAlgebraMixin = ChaseRelationalAlgebraMixin
