from typing import AbstractSet

from ...expressions import Constant, Symbol
from ...relational_algebra import (
    ColumnInt,
    NamedRelationalAlgebraFrozenSet,
    NameColumns,
    NaturalJoin,
    Projection,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from ...utils.relational_algebra_set import RelationalAlgebraFrozenSet
from ..probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
)
from ..query_resolution import RAQueryGOOOptimiser


def _set_symbol(name, n_rows, arity, symbol_table):
    """Store an unnamed relation of ``n_rows`` rows under ``name``."""
    value_symbol = Symbol[AbstractSet](name + "_values")
    symbol_table[value_symbol] = Constant[AbstractSet](
        RelationalAlgebraFrozenSet([(i,) * arity for i in range(n_rows)])
    )
    return value_symbol


def _named_operand(symbol, columns):
    return NameColumns(
        Projection(
            symbol,
            tuple(
                Constant[ColumnInt](ColumnInt(i)) for i in range(len(columns))
            ),
        ),
        tuple(str2columnstr_constant(c) for c in columns),
    )


def _left_deep_operands(expression):
    operands = []
    while isinstance(expression, NaturalJoin):
        operands.append(expression.relation_right)
        expression = expression.relation_left
    return [expression] + operands[::-1]


def test_goo_optimiser_orders_joins_by_symbol_table_sizes():
    """
    Check that RAQueryGOOOptimiser orders joins by the size of the sets.

    Symbols are resolved through the symbol table, and deterministic or
    probabilistic sets through their relation, so the smallest relation
    ``labels`` is joined before the large ``voxel``.
    """
    symbol_table = {}
    reports, voxel, labels, choice = (
        Symbol[AbstractSet](n) for n in ("reports", "voxel", "labels", "c")
    )
    symbol_table[reports] = DeterministicFactSet(
        _set_symbol("reports", 50, 2, symbol_table)
    )
    symbol_table[voxel] = DeterministicFactSet(
        _set_symbol("voxel", 100, 1, symbol_table)
    )
    symbol_table[labels] = DeterministicFactSet(
        _set_symbol("labels", 5, 2, symbol_table)
    )
    symbol_table[choice] = ProbabilisticChoiceSet(
        _set_symbol("choice", 20, 2, symbol_table),
        Constant[ColumnInt](ColumnInt(0)),
    )
    r = _named_operand(reports, "sx")
    v = _named_operand(voxel, "x")
    l_ = _named_operand(labels, "lx")
    s = _named_operand(choice, "ps")
    query = NaturalJoin(NaturalJoin(NaturalJoin(r, v), l_), s)

    result = RAQueryGOOOptimiser(symbol_table=symbol_table).walk(query)
    assert _left_deep_operands(result) == [l_, r, s, v]

    assert RAQueryGOOOptimiser().walk(query) is query


def test_goo_optimiser_resolves_provenance_sets():
    """Check that provenance sets are resolved through their relation."""
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("p", "a"), [(0.5, 1), (0.5, 2)])
    )
    prov_set = ProvenanceAlgebraSet(relation, str2columnstr_constant("p"))
    assert RAQueryGOOOptimiser().operand_size(prov_set) == 2
