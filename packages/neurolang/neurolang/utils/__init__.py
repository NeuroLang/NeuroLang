from nl_expr.utils.orderedset import OrderedSet
from nl_expr.utils.various import log_performance, powerset

# The relational algebra sets are imported when first used, so that the
# expression engine, which uses ``OrderedSet``, needs neither the sets nor
# their pandas or polars backend.
_RELATIONAL_ALGEBRA_SETS = (
    "RelationalAlgebraSet",
    "RelationalAlgebraFrozenSet",
    "NamedRelationalAlgebraFrozenSet",
    "RelationalAlgebraStringExpression",
)

__all__ = [
    "OrderedSet",
    *_RELATIONAL_ALGEBRA_SETS,
    "log_performance",
    "powerset",
]


def __getattr__(name):
    if name in _RELATIONAL_ALGEBRA_SETS:
        from nl_core.utils import relational_algebra_set

        return getattr(relational_algebra_set, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
