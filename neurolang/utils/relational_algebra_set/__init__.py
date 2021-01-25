import os
from .abstract import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr, RelationalAlgebraStringExpression

if os.getenv("NEURO_RAS_BACKEND", "pandas") == "sql":
    from .sql import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
    )
else:
    from .pandas import (
        NamedRelationalAlgebraFrozenSet,
        RelationalAlgebraFrozenSet,
        RelationalAlgebraSet,
    )


__all__ = [
    "RelationalAlgebraColumnInt",
    "RelationalAlgebraColumnStr",
    "RelationalAlgebraStringExpression",
    "RelationalAlgebraFrozenSet",
    "RelationalAlgebraSet",
    "NamedRelationalAlgebraFrozenSet",
]
