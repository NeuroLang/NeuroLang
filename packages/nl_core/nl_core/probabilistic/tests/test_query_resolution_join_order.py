from typing import AbstractSet

from nl_expr.expressions import Constant, Symbol
from ...relational_algebra import (
    ColumnInt,
    NamedRelationalAlgebraFrozenSet,
    NameColumns,
    NaturalJoin,
    Projection,
    str2columnstr_constant,
)
from ...relational_algebra.optimisers import _estimated_size
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
    ``labels`` starts, then ``voxel``, all of whose columns it has, before
    ``reports`` and the choice.
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
    assert _left_deep_operands(result) == [l_, v, r, s]

    assert RAQueryGOOOptimiser().walk(query) is query


def test_goo_optimiser_joins_an_image_before_the_studies():
    """
    Check that the voxels of an image are joined with the activations
    before the studies and their terms, the image having more voxels than
    there are studies.

    In the decoding of an image, ``voxel(v) ⋈ active(v) ⋈ activated(v, s)
    ⋈ study(s) ⋈ term(s, t)``, the choice of a study, of 20 rows, is
    smaller than the image, of 50; started from it, the order would join
    the terms of the studies, then all the activations of each of them.
    The join of ``voxel`` and ``active`` keeps at most 50 rows, the least.
    """
    symbol_table = {}
    names = ("voxel", "active", "activated", "study", "term")
    sizes = (50, 50, 1000, 20, 200)
    arities = (1, 1, 2, 1, 2)
    columns = ("v", "v", "vs", "s", "st")
    symbols = [Symbol[AbstractSet](n) for n in names]
    for symbol, name, size, arity in zip(symbols, names, sizes, arities):
        symbol_table[symbol] = DeterministicFactSet(
            _set_symbol(name, size, arity, symbol_table)
        )
    voxel, active, activated, study, term = (
        _named_operand(symbol, c) for symbol, c in zip(symbols, columns)
    )
    query = NaturalJoin(
        NaturalJoin(
            NaturalJoin(NaturalJoin(term, study), activated), active
        ),
        voxel,
    )

    result = RAQueryGOOOptimiser(symbol_table=symbol_table).walk(query)

    ordered = _left_deep_operands(result)
    assert set(ordered[:2]) == {voxel, active}
    assert ordered[2:] == [activated, study, term]


def test_goo_optimiser_resolves_provenance_sets():
    """Check that provenance sets are resolved through their relation."""
    relation = Constant[AbstractSet](
        NamedRelationalAlgebraFrozenSet(("p", "a"), [(0.5, 1), (0.5, 2)])
    )
    prov_set = ProvenanceAlgebraSet(relation, str2columnstr_constant("p"))
    resolve = RAQueryGOOOptimiser().resolve_relation
    assert _estimated_size(prov_set, resolve) == 2
