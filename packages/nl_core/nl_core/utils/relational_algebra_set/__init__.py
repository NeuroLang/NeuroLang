from nl_expr.config import check_query_backend, config
from .abstract import RelationalAlgebraColumnInt, RelationalAlgebraColumnStr

config.reload_on_backend_switch(__name__)

_backend = config["RAS"].get("backend") if "RAS" in config else None
if _backend is not None:
    try:
        check_query_backend(_backend)
    except ValueError as e:
        raise ValueError(
            f"Invalid [RAS] backend in the NeuroLang configuration: {e}"
        ) from None

if _backend == "polars":
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
