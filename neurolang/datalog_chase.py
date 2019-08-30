from warnings import warn

from .datalog.chase import (
    ChaseNode,
    DatalogChaseGeneral,
    DatalogChase,
    DatalogChaseMGUMixin,
    DatalogChaseRelationalAlgebraMixin
)

warn("This module is going to be deprecated. Switch to datalog.chase")

__all__ = [
    "ChaseNode",
    "DatalogChaseGeneral", "DatalogChase",
    "DatalogChaseMGUMixin", "DatalogChaseRelationalAlgebraMixin"
]
