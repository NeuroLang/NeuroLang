from typing import AbstractSet, Tuple

from ..expressions import Constant
from ..datalog.basic_representation import WrappedRelationalAlgebraSet
from ..relational_algebra import (ColumnInt, Difference, EquiJoin,
                                  NaturalJoin, Product, Projection,
                                  RelationalAlgebraOptimiser,
                                  RelationalAlgebraSolver, Selection, eq_)
from ..utils import NamedRelationalAlgebraFrozenSet

R1 = WrappedRelationalAlgebraSet([
    (i, i * 2)
    for i in range(10)
])

R2 = WrappedRelationalAlgebraSet([
    (i * 2, i * 3)
    for i in range(10)
])


C_ = Constant


def test_selection():
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(0)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection({0: 0})


def test_selection_columns():
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.selection_columns({0: 1})


def test_projections():
    s = Projection(C_(R1), (C_(ColumnInt(0)),))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.projection(0)


def test_equijoin():
    s = EquiJoin(
        C_(R1), (C_(ColumnInt(0)),),
        C_(R2), (C_(ColumnInt(0)),)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.equijoin(R2, [(0, 0)])


def test_naturaljoin():
    r1_named = NamedRelationalAlgebraFrozenSet(('x', 'y'), R1)
    r2_named = NamedRelationalAlgebraFrozenSet(('x', 'z'), R2)
    s = NaturalJoin(
        C_[AbstractSet[Tuple[int, int]]](r1_named),
        C_[AbstractSet[Tuple[int, int]]](r2_named)
    )
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == r1_named.naturaljoin(r2_named)


def test_product():
    s = Product((C_(R1), C_(R2)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2)

    s = Product((C_(R1), C_(R2), C_(R1)))
    sol = RelationalAlgebraSolver().walk(s).value

    assert sol == R1.cross_product(R2).cross_product(R1)

    s = Product(tuple())
    sol = RelationalAlgebraSolver().walk(s).value
    assert len(sol) == 0


def test_selection_reorder():
    raop = RelationalAlgebraOptimiser()
    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(1)))
    assert raop.walk(s) is s

    s1 = Selection(C_(R1), eq_(C_(1), C_(ColumnInt(0))))
    assert raop.walk(s1) == s

    s = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    assert raop.walk(s) is s

    s1 = Selection(C_(R1), eq_(C_(ColumnInt(1)), C_(ColumnInt(0))))
    assert raop.walk(s1) == s

    s_in = Selection(C_(R1), eq_(C_(ColumnInt(1)), C_(ColumnInt(1))))
    s_out = Selection(s_in, eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    assert raop.walk(s_out) is s_out

    s_in1 = Selection(C_(R1), eq_(C_(ColumnInt(0)), C_(ColumnInt(1))))
    s_out1 = Selection(s_in1, eq_(C_(ColumnInt(1)), C_(ColumnInt(1))))
    assert raop.walk(s_out1) == s_out


def test_push_selection_equijoins():
    raop = RelationalAlgebraOptimiser()
    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(0)), C_(1))
    )
    s2_res = EquiJoin(
        Selection(
            C_(R1),
            eq_(C_(ColumnInt(0)), C_(1))
        ),
        (C_(ColumnInt(0)),),
        C_(R2), (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(2)), C_(1))
    )
    s2_res = EquiJoin(
        C_(R1),
        (C_(ColumnInt(0)),),
        Selection(
            C_(R2),
            eq_(C_(ColumnInt(0)), C_(1))
        ),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    )
    s2_res = EquiJoin(
        Selection(
            C_(R1),
            eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
        ),
        (C_(ColumnInt(0)),),
        C_(R2),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(2)), C_(ColumnInt(3)))
    )
    s2_res = EquiJoin(
        C_(R1),
        (C_(ColumnInt(0)),),
        Selection(
            C_(R2),
            eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
        ),
        (C_(ColumnInt(0)),)
    )

    assert raop.walk(s2) == s2_res

    s2 = Selection(
        EquiJoin(
            C_(R1), (C_(ColumnInt(0)),),
            C_(R2), (C_(ColumnInt(0)),)
        ),
        eq_(C_(ColumnInt(1)), C_(ColumnInt(2)))
    )
    assert raop.walk(s2) == s2


def test_push_and_infer_equijoins():
    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(R1), C_(R2)))
    formula1 = eq_(C_(ColumnInt(0)), C_(ColumnInt(1)))
    s = Selection(inner, formula1)
    assert raop.walk(s) == Product((Selection(C_(R1), formula1), C_(R2)))

    inner = Product((C_(R1), C_(R2)))
    formula2 = eq_(C_(ColumnInt(2)), C_(ColumnInt(3)))
    s = Selection(inner, formula2)
    res = raop.walk(s)
    expected_res = Product((C_(R1), Selection(C_(R2), formula1)))
    assert res == expected_res

    inner = Product((C_(R1), C_(R2)))
    formula3 = eq_(C_(ColumnInt(0)), C_(ColumnInt(3)))
    s = Selection(inner, formula3)
    assert raop.walk(s) == EquiJoin(
        C_(R1),
        (C_(ColumnInt(0)),),
        C_(R2),
        (C_(ColumnInt(1)),),
    )

    inner = Product((C_(R1), C_(R2), C_(R1)))
    formula3 = eq_(C_(ColumnInt(0)), C_(ColumnInt(3)))
    s = Selection(inner, formula3)
    assert raop.walk(s) == Product((
        EquiJoin(
            C_(R1),
            (C_(ColumnInt(0)),),
            C_(R2),
            (C_(ColumnInt(1)),),
        ),
        C_(R1)
    ))

    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(R1), C_(R2)))
    formula4 = eq_(C_(ColumnInt(0)), C_(1))
    s = Selection(inner, formula4)
    assert raop.walk(s) == Product(
        (Selection(C_(R1), formula4), C_(R2))
    )

    raop = RelationalAlgebraOptimiser()
    inner = Product((C_(R1), C_(R2)))
    formula5 = eq_(C_(ColumnInt(2)), C_(1))
    s = Selection(inner, formula5)
    res = raop.walk(s)
    theoretical_res = Product(
        (C_(R1), Selection(C_(R2), formula4))
    )
    assert res == theoretical_res
