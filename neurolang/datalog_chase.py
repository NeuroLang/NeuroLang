from warnings import warn

from .datalog.chase import (
    DatalogChase,
)

warn("This module is going to be deprecated. Switch to datalog.chase")

__all__ = ["DatalogChase"]
