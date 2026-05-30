from neurolang.config import config
from .abstract import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr

_backend = config["RAS"].get("backend") if "RAS" in config else None

if _backend == "dask":
    from .dask_sql import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
        RelationalAlgebraStringExpression,
    )
elif _backend == "polars":
    from .polars import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
        RelationalAlgebraStringExpression,
    )
else:
    from .pandas import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
        RelationalAlgebraStringExpression,
    )


__all__ = [
    "RelationalAlgebraColumnInt",
    "RelationalAlgebraColumnStr",
    "RelationalAlgebraStringExpression",
    "RelationalAlgebraFrozenSet",
    "RelationalAlgebraSet",
    "NamedRelationalAlgebraFrozenSet",
]
