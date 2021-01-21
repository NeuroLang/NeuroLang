import os
from .pandas import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr

if os.getenv("NEURO_RAS_BACKEND", "pandas") == "sql":
    from .sql import (
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
