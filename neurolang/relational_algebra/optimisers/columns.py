"""Column helpers shared by the relational algebra optimisers."""

from ..relational_algebra import (
    Projection,
    RelationalAlgebraOperation,
    _get_columns_for_RA_or_constant
)


def _ra_columns(expr):
    """Safely get output columns from any RA node or leaf Constant."""
    if isinstance(expr, RelationalAlgebraOperation):
        return expr.columns()
    try:
        return _get_columns_for_RA_or_constant(expr)
    except (NotImplementedError, AttributeError):
        return set()


def _project_onto_operand(operand, attributes):
    """
    Restrict a projection to the columns of one operand.

    Return ``Projection(operand, attributes ∩ operand.columns)`` when that
    drops at least one column, and ``operand`` itself otherwise.
    """
    operand_cols = _ra_columns(operand)
    pushed_attrs = tuple(a for a in attributes if a in operand_cols)
    if pushed_attrs and pushed_attrs != tuple(operand_cols):
        return Projection(operand, pushed_attrs)
    return operand
