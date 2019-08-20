from pytest import raises

from ..relational_algebra import (
    C_, Column, eq_,
    Selection, Projection, EquiJoin, Product,
    RelationalAlgebraSolver,
    RelationalAlgebraRewriteSelections,
    RelationalAlgebraOptimiser
)
from ..utils import RelationalAlgebraSet
from ..solver_datalog_naive import WrappedRelationalAlgebraSet


R1 = WrappedRelationalAlgebraSet([
    (i, i * 2)
    for i in range(10)
])

R2 = WrappedRelationalAlgebraSet([
    (i * 2, i * 3)
    for i in range(10)
])


def test_selection():
    s = Selection(C_(R1), eq_(C_(Column(0)), C_(0)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection({0: 0})


def test_selection_columns():
    s = Selection(C_(R1), eq_(C_(Column(0)), C_(Column(1))))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection_columns({0: 1})


def test_projections():
    s = Projection(C_(R1), (C_(Column(0)),))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.projection(0)


def test_equijoin():
    s = EquiJoin(
        C_(R1), (C_(Column(0)),),
        C_(R2), (C_(Column(0)),)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.equijoin(R2, [(0, 0)])


def test_product():
    s = Product((C_(R1), C_(R2)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2)

    s = Product((C_(R1), C_(R2), C_(R1)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2).cross_product(R1)
