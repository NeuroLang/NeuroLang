from neurolang.config import config
from .abstract import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr

if "RAS" in config:
    backend = config["RAS"].get("backend")
    if backend == "dask":
        from .dask_sql import (
            NamedRelationalAlgebraFrozenSet,
            RelationalAlgebraFrozenSet,
            RelationalAlgebraSet,
            RelationalAlgebraStringExpression,
        )
    elif backend == "polars":
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
