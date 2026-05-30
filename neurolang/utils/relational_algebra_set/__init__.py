from neurolang.config import config
from .abstract import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr

# Determine backend priority:
# 1. Explicit config.ini setting always wins
# 2. If polars is installed and no explicit backend set, default to polars
# 3. Otherwise fall back to pandas

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
    # No explicit backend set — auto-detect polars availability
    try:
        import polars as _pl
        from .polars import (
            NamedRelationalAlgebraFrozenSet,
            RelationalAlgebraFrozenSet,
            RelationalAlgebraSet,
            RelationalAlgebraStringExpression,
        )
    except ImportError:
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
