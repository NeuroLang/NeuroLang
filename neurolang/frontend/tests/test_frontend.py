from collections import namedtuple
from operator import contains
from typing import AbstractSet, Callable, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from neurolang import frontend
from neurolang.frontend import query_resolution

from ... import expressions as exp
from ...datalog import DatalogProgram, Fact, Implication
from ...expression_walker import ExpressionBasicEvaluator
from ...regions import ExplicitVBR, SphericalVolume
from ...type_system import Unknown
from .. import query_resolution_expressions as qre


def test_symbol_management():
    class Solver(DatalogProgram, ExpressionBasicEvaluator):
        pass

    neurolang = query_resolution.QueryBuilderBase(Solver())

    sym = neurolang.new_symbol(int)
    assert sym.expression.type is int

    sym_ = neurolang.new_symbol(type_=(float, int))
    assert sym_.expression.type is Tuple[float, int]
    assert sym.expression.name != sym_.expression.name

    a = neurolang.new_symbol(int, name="a")
    assert a.expression.name == "a"

    b = neurolang.add_symbol(1, name="b")
    assert "b" in neurolang.symbols
    assert b.value == 1
    assert b.type is int
    assert neurolang.symbols.b == b

    with pytest.raises(AttributeError):
        assert neurolang.symbols.c

    @neurolang.add_symbol
    def id(x: int) -> int:
        return x

    assert "id" in neurolang.symbols
    assert id == neurolang.symbols.id
    assert id == neurolang.symbols["id"]
    assert id.type == Callable[[int], int]

    f = neurolang.new_symbol()
    new_expression = f(..., 1)
    assert isinstance(new_expression.expression, exp.FunctionApplication)
    assert new_expression.expression.functor == f.expression
    assert isinstance(new_expression.expression.args[0], exp.Symbol)
    assert new_expression.expression.args[0].is_fresh
    assert isinstance(new_expression.expression.args[1], exp.Constant)
    assert new_expression.expression.args[1].value == 1


def test_symbol_environment():
    class Solver(DatalogProgram, ExpressionBasicEvaluator):
        pass

    neurolang = query_resolution.QueryBuilderBase(Solver())

    b = neurolang.add_symbol(1, name="b")
    neurolang.symbols._dynamic_mode = True
    assert "c" not in neurolang.symbols
    c = neurolang.symbols.c
    assert c.type is Unknown
    assert c.expression.name == "c"
    del neurolang.symbols.b
    assert b not in neurolang.symbols
    neurolang.symbols._dynamic_mode = False

    with neurolang.environment as e:
        assert "c" not in e
        c = e.c
        assert c.type is Unknown
        assert c.expression.name == "c"

        e.d = 5
        assert e.d.value == 5
        assert e.d.type is int

    assert neurolang.symbols.d.value == 5
    assert neurolang.symbols.d.type is int

    with neurolang.scope as e:
        assert "f" not in e
        f = e.f
        assert f.type is Unknown
        assert f.expression.name == "f"

        e.g = 5
        assert e.g.value == 5
        assert e.g.type is int

    assert "f" not in neurolang.symbols
    assert "g" not in neurolang.symbols


def test_add_set():
    neurolang = frontend.NeurolangDL()

    s = neurolang.add_tuple_set(range(10), int)
    res = neurolang[s]

    assert s.type is AbstractSet[int]
    assert res.type is AbstractSet[int]
    assert res.value == frozenset((i,) for i in range(10))
    assert isinstance(repr(res), str)

    v = frozenset(zip(("a", "b", "c"), range(3)))
    s = neurolang.add_tuple_set(v, (str, int))
    res = neurolang[s]

    assert s.type is AbstractSet[Tuple[str, int]]
    assert res.type is AbstractSet[Tuple[str, int]]
    assert res.value == v
    assert isinstance(repr(res), str)


def test_add_set_neurolangdl():
    neurolang = frontend.NeurolangDL()

    s = neurolang.add_tuple_set(range(10), int)
    res = neurolang[s]

    assert s.type is AbstractSet[int]
    assert res.type is AbstractSet[int]
    assert res.value == frozenset((i,) for i in range(10))

    v = frozenset(zip(("a", "b", "c"), range(3)))
    s = neurolang.add_tuple_set(v, (str, int))
    res = neurolang[s]

    assert s.type is AbstractSet[Tuple[str, int]]
    assert res.type is AbstractSet[Tuple[str, int]]
    assert res.value == v


def test_query_regions_from_region_set():
    neurolang = frontend.NeurolangDL()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))

    i1 = ExplicitVBR(np.array([[0, 0, 2], [1, 1, 3]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    regions_ = {(i1,), (i2,), (i3,)}
    regions = neurolang.add_tuple_set(regions_)

    x = neurolang.new_region_symbol(name="x")
    query_result = neurolang.query(
        (x,), regions(x) & neurolang.symbols.inferior_of(x, central)
    )

    assert len(query_result) == len(regions)
    assert query_result == {(i1,), (i2,), (i3,)}


def test_query_new_predicate():
    neurolang = frontend.NeurolangDL()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    inferior_posterior = ExplicitVBR(
        np.array([[0, -10, -10], [1, -5, -5]]), np.eye(4)
    )

    inferior_central = ExplicitVBR(
        np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4)
    )
    inferior_anterior = ExplicitVBR(
        np.array([[0, 2, 2], [1, 5, 3]]), np.eye(4)
    )

    regions_ = {
        (inferior_posterior,),
        (inferior_central,),
        (inferior_anterior,),
    }
    regions = neurolang.add_tuple_set(regions_)

    def posterior_and_inferior(y, z):
        return neurolang.symbols.anatomical_posterior_of(
            y, z
        ) & neurolang.symbols.anatomical_inferior_of(y, z)

    x = neurolang.new_region_symbol(name="x")
    query_result = neurolang.query(
        (x,), regions(x) & posterior_and_inferior(x, central)
    )
    assert len(query_result) == 1
    assert next(iter(query_result)) == (inferior_posterior,)


@pytest.mark.skip()
def test_load_spherical_volume_first_order():
    neurolang = frontend.RegionFrontend()

    inferior = ExplicitVBR(np.array([[0, 0, 0], [1, 1, 1]]), np.eye(4))

    neurolang.add_region(inferior, name="inferior_region")
    neurolang.sphere((0, 0, 0), 0.5, name="unit_sphere")
    assert neurolang.symbols["unit_sphere"].value == SphericalVolume(
        (0, 0, 0), 0.5
    )

    x = neurolang.new_region_symbol(name="x")
    query = neurolang.query(
        x, neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
    )
    query_result = query.do()
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior

    neurolang.make_implicit_regions_explicit(np.eye(4), (500, 500, 500))
    query = neurolang.query(
        x, neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
    )
    query_result = query.do()
    sphere_constant = neurolang.symbols["unit_sphere"].value
    assert (
        isinstance(sphere_constant, ExplicitVBR)
        and np.array_equal(sphere_constant.affine, np.eye(4))
        and sphere_constant.image_dim == (500, 500, 500)
        and np.array_equal(sphere_constant.voxels, [[0, 0, 0]])
    )
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior


def test_load_spherical_volume_datalog():
    neurolang = frontend.NeurolangDL()

    inferior = ExplicitVBR(np.array([[0, 0, 0], [1, 1, 1]]), np.eye(4))

    regions = neurolang.add_tuple_set(
        {(inferior, "inferior_region")}, name="regions"
    )
    neurolang.sphere((0, 0, 0), 0.5, name="unit_sphere")
    assert neurolang.symbols["unit_sphere"].value == SphericalVolume(
        (0, 0, 0), 0.5
    )

    q = neurolang.new_symbol()
    x = neurolang.new_region_symbol(name="x")
    n = neurolang.new_region_symbol(name="n")
    query = neurolang.query(
        q(x),
        neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
        & regions(x, n),
    )

    assert len(query.value) == 1
    assert next(iter(query.value))[0] == inferior

    neurolang.make_implicit_regions_explicit(np.eye(4), (500, 500, 500))
    query = neurolang.query(
        q(x),
        neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
        & regions(x, n),
    )

    assert len(query.value) == 1
    assert next(iter(query.value))[0] == inferior

    sphere_constant = neurolang.symbols["unit_sphere"].value
    assert (
        isinstance(sphere_constant, ExplicitVBR)
        and np.array_equal(sphere_constant.affine, np.eye(4))
        and sphere_constant.image_dim == (500, 500, 500)
        and np.array_equal(sphere_constant.voxels, [[0, 0, 0]])
    )


def test_neurolang_dl_query():
    neurolang = frontend.NeurolangDL()
    r = neurolang.new_symbol(name="r")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    z = neurolang.new_symbol(name="z")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    sol = neurolang.query((x, y), q(x, y))
    assert sol == dataset

    sol = neurolang.query(tuple(), q(x, x))
    assert sol
    assert neurolang.query(q(x, x))

    sol = neurolang.query(tuple(), q(100, x))
    assert not sol
    assert not neurolang.query(q(100, x))

    sol = neurolang.query((x,), q(x, y) & q(y, z))
    res = set((x,) for x in range(5))
    assert sol == res

    r[x, y] = q(x, y)
    r[x, z] = r[x, y] & q(y, z)
    sol = neurolang.query((y,), r(1, y))
    assert sol == set((x,) for x in (2, 4, 8, 16))


def test_neurolang_dl_solve_all():
    neurolang = frontend.NeurolangDL()
    r = neurolang.new_symbol(name="r")
    x = neurolang.new_symbol(name="x")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    r[x] = q(x, x)
    sol = neurolang.solve_all()
    assert sol["q"].to_unnamed() == dataset
    assert sol["r"].to_unnamed() == set((i,) for i, j in dataset if i == j)
    assert len(sol) == 2
    assert neurolang.predicate_parameter_names(r) == ("x",)


def test_neurolange_dl_get_param_names():
    neurolang = frontend.NeurolangDL()
    r = neurolang.new_symbol(name="r")
    x = neurolang.new_symbol(name="x")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    r[x] = q(x, x)

    @neurolang.add_symbol
    def test_fun(x: int) -> int:
        """
        HELP TEST
        """
        return 0

    assert neurolang.predicate_parameter_names("q") == ("0", "1")
    assert neurolang.predicate_parameter_names(q) == ("0", "1")
    assert neurolang.predicate_parameter_names(r) == ("x",)
    assert neurolang.symbols[r].predicate_parameter_names == ("x",)
    assert r[x].help() is not None
    assert neurolang.symbols["test_fun"].help().strip() == "HELP TEST"


def test_neurolange_dl_named_sets():
    neurolang = frontend.NeurolangDL()
    r = neurolang.new_symbol(name="r")
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    r[x] = q(x, x)
    s[x, y] = q(x, x) & (y == x)

    res = neurolang.solve_all()

    assert res["r"].columns == ("x",)
    assert res["r"].row_type == Tuple[int]
    assert res["r"].to_unnamed() == {(i,) for i, j in dataset if i == j}


def test_neurolange_dl_negation():
    neurolang = frontend.NeurolangDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    s[x, y] = ~q(x, x) & q(x, y)

    res = neurolang.solve_all()

    assert res["s"].to_unnamed() == {(i, j) for i, j in dataset if i != j}


def test_neurolang_dl_datalog_code_list_symbols():
    neurolang = frontend.NeurolangDL()
    original_symbols = set(neurolang.symbols)
    neurolang.execute_datalog_program(
        """
    A(4, 5)
    A(5, 6)
    A(6, 5)
    B(x,y) :- A(x, y)
    B(x,y) :- B(x, z),A(z, y)
    C(x) :- B(x, y), y == 5
    D("x")
    """
    )

    assert set(neurolang.symbols) == {"A", "B", "C", "D"} | original_symbols


def test_neurolang_dl_datalog_code():
    neurolang = frontend.NeurolangDL()
    neurolang.execute_datalog_program(
        """
    A(4, 5)
    A(5, 6)
    A(6, 5)
    B(x,y) :- A(x, y)
    B(x,y) :- B(x, z),A(z, y)
    C(x) :- B(x, y), y == 5
    D("x")
    """
    )

    res = neurolang.solve_all()

    assert res["A"].row_type == Tuple[int, int]
    assert res["A"].to_unnamed() == {(4, 5), (5, 6), (6, 5)}
    assert res["B"].to_unnamed() == {
        (4, 5),
        (5, 6),
        (6, 5),
        (4, 6),
        (5, 5),
        (6, 6),
    }
    assert res["C"].to_unnamed() == {(4,), (5,), (6,)}
    assert res["D"].to_unnamed() == {
        ("x",),
    }


def test_neurolang_dl_aggregation():
    neurolang = frontend.NeurolangDL()
    q = neurolang.new_symbol(name="q")
    p = neurolang.new_symbol(name="p")
    r = neurolang.new_symbol(name="r")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    for i in range(10):
        q[i % 2, i] = True

    p[x, sum_(y)] = q[x, y]

    sol = neurolang.query(r(x, y), p(x, y))

    res_q = {(0, 2 + 4 + 6 + 8), (1, 1 + 3 + 5 + 7 + 9)}

    assert len(sol) == 2
    assert sol[r] == res_q
    assert sol[p] == res_q


def test_neurolang_dl_aggregation_direct_query():
    neurolang = frontend.NeurolangDL()
    q = neurolang.new_symbol(name="q")
    p = neurolang.new_symbol(name="p")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    for i in range(10):
        q[i % 2, i] = True

    p[x, sum_(y)] = q[x, y]

    sol = neurolang.query((x, y), p(x, y))

    res_q = {(0, 2 + 4 + 6 + 8), (1, 1 + 3 + 5 + 7 + 9)}

    assert sol == res_q


def test_neurolang_dl_aggregation_environment():
    neurolang = frontend.NeurolangDL()

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    with neurolang.environment as e:
        for i in range(10):
            e.q[i % 2, i] = True

        e.p[e.x, sum_(e.y)] = e.q[e.x, e.y]
        sol = neurolang.query(e.r(e.x, e.y), e.p(e.x, e.y))

    res_q = {(0, 2 + 4 + 6 + 8), (1, 1 + 3 + 5 + 7 + 9)}

    assert len(sol) == 2
    assert sol["r"] == res_q
    assert sol["p"] == res_q


def test_neurolang_dl_aggregation_environment_direct_query():
    neurolang = frontend.NeurolangDL()

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    with neurolang.environment as e:
        for i in range(10):
            e.q[i % 2, i] = True

        e.p[e.x, sum_(e.y)] = e.q[e.x, e.y]
        sol = neurolang.query((e.x, e.y), e.p(e.x, e.y))

    res_q = {(0, 2 + 4 + 6 + 8), (1, 1 + 3 + 5 + 7 + 9)}

    assert sol == res_q


def test_aggregation_number_of_arrivals():
    neurolang = frontend.NeurolangDL()

    @neurolang.add_symbol
    def agg_count(x) -> int:
        return len(x)

    with neurolang.environment as e:
        e.A[0, 1] = True
        e.A[1, 2] = True
        e.A[2, 3] = True
        e.reachable[e.x, e.y] = e.A[e.x, e.y]
        e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.A[e.z, e.y]

        e.count_destinations[e.x, agg_count(e.y)] = e.reachable[e.x, e.y]

        res = neurolang.query((e.x, e.c), e.count_destinations(e.x, e.c))

    assert res == {(0, 3), (1, 2), (2, 1)}


def test_neurolang_dl_attribute_access():
    neurolang = frontend.NeurolangDL()
    one_element = namedtuple("t", ("x", "y"))(1, 2)

    a = neurolang.add_tuple_set([(one_element,)], name="a")
    with neurolang.scope as e:
        e.q[e.x] = a[e.x]
        e.r[e.y] = a[e.w] & (e.y == e.w.x)
        res = neurolang.solve_all()

    q = res["q"]
    r = res["r"]
    assert len(q) == 1
    el = next(q.to_unnamed().itervalues())[0]
    assert el == one_element
    assert r.to_unnamed() == {(one_element.x,)}


def test_neurolang_dl_set_destroy():
    neurolang = frontend.NeurolangDL()
    contains_ = neurolang.add_symbol(contains)

    a = neurolang.add_tuple_set([(frozenset((0, 1, 2)),)], name="a")
    with neurolang.scope as e:
        e.q[e.y] = a[e.x] & contains_(e.x, e.y)
        res = neurolang.solve_all()

    q = res["q"].to_unnamed()
    assert len(q) == 3
    assert set(q) == {(0,), (1,), (2,)}


@pytest.mark.skip
@patch(
    "neurolang.frontend.neurosynth_utils."
    "NeuroSynthHandler.ns_region_set_from_term"
)
def test_neurosynth_region(mock_ns_regions):
    mock_ns_regions.return_value = {
        ExplicitVBR(np.array([[1, 0, 0], [1, 1, 0]]), np.eye(4))
    }
    neurolang = frontend.NeurolangDL()
    s = neurolang.load_neurosynth_term_regions(
        "gambling", 10, "gambling_regions"
    )
    res = neurolang[s]
    mock_ns_regions.assert_called()

    assert res.type is AbstractSet[Tuple[ExplicitVBR]]
    assert res.value == frozenset((t,) for t in mock_ns_regions.return_value)


def test_translate_expression_to_fronted_expression():
    qr = frontend.NeurolangDL()
    tr = qre.TranslateExpressionToFrontEndExpression(qr)

    assert tr.walk(exp.Constant(1)) == 1

    symbol_exp = exp.Symbol("a")
    symbol_fe = tr.walk(symbol_exp)
    assert symbol_fe.expression == symbol_exp
    assert symbol_fe.query_builder == tr.query_builder

    fa_exp = symbol_exp(exp.Constant(1))
    fa_fe = symbol_fe(1)
    fa_fe_tr = tr.walk(fa_exp)
    assert fa_fe_tr.expression == fa_exp
    assert fa_fe_tr == fa_fe

    fact_exp = Fact(fa_exp)
    fact_fe = tr.walk(fact_exp)
    assert fact_fe.expression == fact_exp
    assert fact_fe.consequent == fa_fe

    imp_exp = Implication(
        symbol_exp(exp.Symbol("x")), exp.Symbol("b")(exp.Symbol("x"))
    )
    imp_fe = tr.walk(imp_exp)
    assert imp_fe.expression == imp_exp
    assert imp_fe.consequent == tr.walk(imp_exp.consequent)
    assert imp_fe.antecedent == tr.walk(imp_exp.antecedent)


def test_first_column_sugar_body_s():
    qr = frontend.NeurolangDL()
    qr.add_tuple_set({
        ('one', 1), ('two', 2)
    }, name='dd')

    with qr.scope as e:
        e.s[e.x] = (e.x == e.y) & e.dd('one', e.y)
        e.r[e.x] = (e.x == (e.dd.s['one']))
        res_all = qr.solve_all()

    assert res_all['r'] == res_all['s']


def test_first_column_sugar_head_s():
    qr = frontend.NeurolangDL()
    qr.add_tuple_set({
        (1, 'one'), (2, 'two')
    }, name='dd')

    with qr.scope as e:
        e.r.s['one'] = e.dd('one')
        res_all = qr.solve_all()

    assert set(res_all['r']) == {('one', 1)}


def test_head_constant():
    qr = frontend.NeurolangDL()
    qr.add_tuple_set({
        (1,)
    }, name='dd')

    with qr.scope as e:
        e.r['one', e.x] = e.dd(e.x)
        res_all = qr.solve_all()

    assert set(res_all['r']) == {('one', 1)}
