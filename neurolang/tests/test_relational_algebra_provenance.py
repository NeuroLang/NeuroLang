import pytest

from ..expressions import (Constant, Symbol)
from ..relational_algebra import (
    ColumnStr, EquiJoin, NaturalJoin, Product, Selection, eq_, RenameColumn
)
from ..relational_algebra_provenance import (
    RelationalAlgebraProvenanceCountingSolver, ProvenanceAlgebraSet, Union,
    ProjectionNonProvenance, Projection
)
from ..utils import NamedRelationalAlgebraFrozenSet

import numpy as np

C_ = Constant
S_ = Symbol

R1 = NamedRelationalAlgebraFrozenSet(
    columns=('col1', 'col2', '__provenance__'),
    iterable=[(i, i * 2, i) for i in range(10)]
)
provenance_set_r1 = ProvenanceAlgebraSet(R1, '__provenance__')


def test_selection():
    s = Selection(provenance_set_r1, eq_(C_(ColumnStr('col1')), C_(4)))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == R1.selection({'col1': 4})
    assert '__provenance__' in sol.columns
    assert sol._container['__provenance__'
                          ].values in R1._container['__provenance__'].values


def test_selection_columns():
    s = Selection(
        provenance_set_r1, eq_(C_(ColumnStr('col1')), C_(ColumnStr('col2')))
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == R1.selection_columns({'col1': 'col2'})
    assert sol == R1.selection({'col1': 0})
    assert '__provenance__' in sol.columns


def test_valid_rename():
    s = RenameColumn(
        provenance_set_r1, C_(ColumnStr('col1')), C_(ColumnStr('renamed'))
    )
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == R1.rename_column('col1', 'renamed')
    assert 'renamed' in sol.columns
    assert 'col1' not in sol.columns
    assert '__provenance__' in sol.columns
    assert np.array_equal(
        R1._container['__provenance__'].values,
        sol._container['__provenance__'].values
    )


def test_provenance_rename():
    s = RenameColumn(
        provenance_set_r1, C_(ColumnStr('__provenance__')),
        C_(ColumnStr('renamed'))
    )

    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    assert sol.provenance_column == 'renamed'
    sol = sol.value
    assert sol == R1.rename_column('__provenance__', 'renamed')
    assert 'renamed' in sol.columns
    assert '__provenance__' not in sol.columns
    assert np.array_equal(
        R1._container['__provenance__'].values,
        sol._container['renamed'].values
    )


def test_projections_non_provenance():
    s = ProjectionNonProvenance(provenance_set_r1, (C_(ColumnStr('col1')), ))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value
    R1proj = R1.projection('col1')

    assert sol == R1proj
    assert '__provenance__' not in sol.columns


def test_naturaljoin():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', '__provenance__'),
        iterable=[(i * 2, i) for i in range(10)]
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, '__provenance__')

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', 'colA', '__provenance__'),
        iterable=[(i % 5, i * 3, i) for i in range(20)]
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, '__provenance__')

    s = NaturalJoin(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s)

    RA1_np = NamedRelationalAlgebraFrozenSet(
        columns=('col1', ), iterable=[(i * 2) for i in range(10)]
    )

    RA2_np = NamedRelationalAlgebraFrozenSet(
        columns=('col1', 'colA'), iterable=[(i % 5, i * 3) for i in range(20)]
    )

    R1njR2 = RA1_np.naturaljoin(RA2_np)
    sol_np = sol.value.projection(*['col1', 'colA'])
    assert sol_np == R1njR2

    R1 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', '__provenance__1'),
        iterable=[(i * 2, i) for i in range(10)]
    )

    R2 = NamedRelationalAlgebraFrozenSet(
        columns=('colA', '__provenance__2'),
        iterable=[(i * 3, i) for i in range(20)]
    )

    R1cpR2 = R1.cross_product(R2)
    RnjR = R1cpR2.naturaljoin(R1njR2)
    res = RnjR._container.apply(
        lambda x: x['__provenance__1'] * x['__provenance__2'], axis=1
    )
    prov_sol = sol.value._container['__provenance__'].values
    prov_res = res.values
    assert np.all(prov_sol == prov_res)


def test_product():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', '__provenance__'),
        iterable=[(i * 2, i) for i in range(10)]
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, '__provenance__')

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=('colA', '__provenance__'),
        iterable=[(i * 3, i) for i in range(20)]
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, '__provenance__')

    s = Product((pset_r1, pset_r2))
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    RA1_np = NamedRelationalAlgebraFrozenSet(
        columns=('col1', ), iterable=[(i * 2) for i in range(10)]
    )

    RA2_np = NamedRelationalAlgebraFrozenSet(
        columns=('colA', ), iterable=[(i * 3) for i in range(20)]
    )

    R1njR2 = RA1_np.cross_product(RA2_np)
    sol_np = sol.projection(*['col1', 'colA'])
    assert sol_np == R1njR2

    R1 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', '__provenance__1'),
        iterable=[(i * 2, i) for i in range(10)]
    )

    R2 = NamedRelationalAlgebraFrozenSet(
        columns=('colA', '__provenance__2'),
        iterable=[(i * 3, i) for i in range(20)]
    )

    R1cpR2 = R1.cross_product(R2)
    RnjR = R1cpR2.naturaljoin(R1njR2)
    res = RnjR._container.apply(
        lambda x: x['__provenance__1'] * x['__provenance__2'], axis=1
    )
    prov_sol = sol._container['__provenance__'].values
    prov_res = res.values
    assert np.all(prov_sol == prov_res)


def test_union():
    RA1 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', '__provenance__'),
        iterable=[(i * 2, i) for i in range(10)]
    )
    pset_r1 = ProvenanceAlgebraSet(RA1, '__provenance__')

    RA2 = NamedRelationalAlgebraFrozenSet(
        columns=('col1', '__provenance__'),
        iterable=[(i % 5, i) for i in range(20)]
    )
    pset_r2 = ProvenanceAlgebraSet(RA2, '__provenance__')

    s = Union(pset_r1, pset_r2)
    sol = RelationalAlgebraProvenanceCountingSolver().walk(s).value

    assert sol == (RA1 | RA2)


def test_projection():
    relation = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[
                ("a", "b", 1),
                ("b", "a", 2),
                ("c", "a", 2),
                ("c", "a", 1),
                ("b", "a", 1),
            ],
            columns=["x", "y", '__provenance__'],
        ), '__provenance__'
    )
    expected = ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            iterable=[("a", "b", 1), ("b", "a", 3), ("c", "a", 3)],
            columns=["x", "y", '__provenance__'],
        ), '__provenance__'
    )
    sum_agg_op = Projection(
        relation,
        tuple([Constant(ColumnStr("x")),
               Constant(ColumnStr("y"))]),
    )
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(sum_agg_op)

    assert result == expected