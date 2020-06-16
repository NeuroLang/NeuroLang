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
from ...regions import ExplicitVBR, Region, SphericalVolume
from ...type_system import Unknown
from .. import query_resolution_expressions as qre
from ..query_resolution_expressions import Symbol


def test_symbol_management():
    class Solver(
        DatalogProgram,
        ExpressionBasicEvaluator
    ):
        pass

    neurolang = query_resolution.QueryBuilderBase(Solver())

    sym = neurolang.new_symbol(int)
    assert sym.expression.type is int

    sym_ = neurolang.new_symbol(type_=(float, int))
    assert sym_.expression.type is Tuple[float, int]
    assert sym.expression.name != sym_.expression.name

    a = neurolang.new_symbol(int, name='a')
    assert a.expression.name == 'a'

    b = neurolang.add_symbol(1, name='b')
    assert 'b' in neurolang.symbols
    assert b.value == 1
    assert b.type is int
    assert neurolang.symbols.b == b

    with pytest.raises(AttributeError):
        assert neurolang.symbols.c

    @neurolang.add_symbol
    def id(x: int) -> int:
        return x

    assert 'id' in neurolang.symbols
    assert id == neurolang.symbols.id
    assert id == neurolang.symbols['id']
    assert id.type == Callable[[int], int]


def test_symbol_environment():
    class Solver(
        DatalogProgram,
        ExpressionBasicEvaluator
    ):
        pass

    neurolang = query_resolution.QueryBuilderBase(Solver())

    b = neurolang.add_symbol(1, name='b')
    neurolang.symbols._dynamic_mode = True
    assert 'c' not in neurolang.symbols
    c = neurolang.symbols.c
    assert c.type is Unknown
    assert c.expression.name == 'c'
    del neurolang.symbols.b
    assert b not in neurolang.symbols
    neurolang.symbols._dynamic_mode = False

    with neurolang.environment as e:
        assert 'c' not in e
        c = e.c
        assert c.type is Unknown
        assert c.expression.name == 'c'

        e.d = 5
        assert e.d.value == 5
        assert e.d.type is int

    assert neurolang.symbols.d.value == 5
    assert neurolang.symbols.d.type is int

    with neurolang.scope as e:
        assert 'f' not in e
        f = e.f
        assert f.type is Unknown
        assert f.expression.name == 'f'

        e.g = 5
        assert e.g.value == 5
        assert e.g.type is int

    assert 'f' not in neurolang.symbols
    assert 'g' not in neurolang.symbols


def test_add_set():
    neurolang = frontend.RegionFrontend()

    s = neurolang.add_tuple_set(range(10), int)
    res = neurolang[s]

    assert s.type is AbstractSet[int]
    assert res.type is AbstractSet[int]
    assert res.value == frozenset(range(10))

    v = frozenset(zip(('a', 'b', 'c'), range(3)))
    s = neurolang.add_tuple_set(v, (str, int))
    res = neurolang[s]

    assert s.type is AbstractSet[Tuple[str, int]]
    assert res.type is AbstractSet[Tuple[str, int]]
    assert res.value == v

    exp = neurolang.symbols.isin(next(iter(s)), s)
    assert exp.do().value is True


def test_add_set_neurolangdl():
    neurolang = frontend.NeurolangDL()

    s = neurolang.add_tuple_set(range(10), int)
    res = neurolang[s]

    assert s.type is AbstractSet[int]
    assert res.type is AbstractSet[int]
    assert res.value == frozenset((i,) for i in range(10))

    v = frozenset(zip(('a', 'b', 'c'), range(3)))
    s = neurolang.add_tuple_set(v, (str, int))
    res = neurolang[s]

    assert s.type is AbstractSet[Tuple[str, int]]
    assert res.type is AbstractSet[Tuple[str, int]]
    assert res.value == v


def test_add_regions_and_query():
    neurolang = frontend.RegionFrontend()

    inferior = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region(inferior, name='inferior_region')
    neurolang.add_region(superior, name='superior_region')
    assert neurolang.symbols.inferior_region.value == inferior
    assert neurolang.symbols.superior_region.value == superior

    result_symbol = neurolang.symbols.superior_of(
        superior, inferior
    ).do(name='is_superior_test')
    assert result_symbol.value
    assert neurolang.get_symbol('is_superior_test').value

    x = neurolang.new_region_symbol(name='x')
    query = neurolang.query(
        x, neurolang.symbols.superior_of(x, neurolang.symbols.inferior_region)
    )
    query_result = query.do(name='result_of_test_query')

    assert isinstance(query_result, Symbol)
    assert isinstance(query_result.value, frozenset)
    assert len(query_result.value) == 1
    assert superior == next(iter(query_result.value))


def test_query_regions_from_region_set():
    neurolang = frontend.RegionFrontend()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    neurolang.add_region(central, name='reference_region')

    i1 = ExplicitVBR(np.array([[0, 0, 2], [1, 1, 3]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    regions = {i1, i2, i3}
    neurolang.add_tuple_set(regions, ExplicitVBR)

    x = neurolang.new_region_symbol(name='x')
    query_result = neurolang.query(
        x,
        neurolang.symbols.inferior_of(x, neurolang.symbols.reference_region)
    ).do(name='result_of_test_query')

    assert len(query_result.value) == len(regions)
    assert query_result.value == {i1, i2, i3}


def test_query_new_predicate():
    neurolang = frontend.RegionFrontend()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    reference_symbol = neurolang.add_region(
        central, name='reference_region'
    )

    inferior_posterior = ExplicitVBR(
        np.array([[0, -10, -10], [1, -5, -5]]), np.eye(4)
    )

    inferior_central = ExplicitVBR(
        np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4)
    )
    inferior_anterior = ExplicitVBR(
        np.array([[0, 2, 2], [1, 5, 3]]), np.eye(4)
    )

    regions = {inferior_posterior, inferior_central, inferior_anterior}

    neurolang.add_tuple_set(regions, ExplicitVBR)

    def posterior_and_inferior(y, z):
        return (
            neurolang.symbols.anatomical_posterior_of(y, z) &
            neurolang.symbols.anatomical_inferior_of(y, z)
        )

    x = neurolang.new_region_symbol(name='x')
    query = neurolang.query(x, posterior_and_inferior(x, reference_symbol))
    query_result = query.do(name='result_of_test_query')
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior_posterior


@pytest.mark.skip()
def test_load_spherical_volume_first_order():
    neurolang = frontend.RegionFrontend()

    inferior = ExplicitVBR(np.array([[0, 0, 0], [1, 1, 1]]), np.eye(4))

    neurolang.add_region(inferior, name='inferior_region')
    neurolang.sphere((0, 0, 0), .5, name='unit_sphere')
    assert (
        neurolang.symbols['unit_sphere'].value ==
        SphericalVolume((0, 0, 0), .5)
    )

    x = neurolang.new_region_symbol(name='x')
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
    sphere_constant = neurolang.symbols['unit_sphere'].value
    assert (
        isinstance(sphere_constant, ExplicitVBR) and
        np.array_equal(sphere_constant.affine, np.eye(4)) and
        sphere_constant.image_dim == (500, 500, 500) and
        np.array_equal(sphere_constant.voxels, [[0, 0, 0]])
    )
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior


def test_load_spherical_volume_datalog():
    neurolang = frontend.NeurolangDL()

    inferior = ExplicitVBR(np.array([[0, 0, 0], [1, 1, 1]]), np.eye(4))

    regions = neurolang.add_tuple_set(
       {(inferior, 'inferior_region')}, name='regions'
    )
    neurolang.sphere((0, 0, 0), .5, name='unit_sphere')
    assert (
        neurolang.symbols['unit_sphere'].value ==
        SphericalVolume((0, 0, 0), .5)
    )

    q = neurolang.new_symbol()
    x = neurolang.new_region_symbol(name='x')
    n = neurolang.new_region_symbol(name='n')
    query = neurolang.query(
        q(x),
        neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere) &
        regions(x, n)
    )

    assert len(query.value) == 1
    assert next(iter(query.value))[0] == inferior

    neurolang.make_implicit_regions_explicit(np.eye(4), (500, 500, 500))
    query = neurolang.query(
        q(x),
        neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere) &
        regions(x, n)
    )

    assert len(query.value) == 1
    assert next(iter(query.value))[0] == inferior

    sphere_constant = neurolang.symbols['unit_sphere'].value
    assert (
        isinstance(sphere_constant, ExplicitVBR) and
        np.array_equal(sphere_constant.affine, np.eye(4)) and
        sphere_constant.image_dim == (500, 500, 500) and
        np.array_equal(sphere_constant.voxels, [[0, 0, 0]])
    )


def test_neurolang_dl_query():
    neurolang = frontend.NeurolangDL()
    r = neurolang.new_symbol(name='r')
    x = neurolang.new_symbol(name='x')
    y = neurolang.new_symbol(name='y')
    z = neurolang.new_symbol(name='z')

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name='q')
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
    r = neurolang.new_symbol(name='r')
    x = neurolang.new_symbol(name='x')

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name='q')
    r[x] = q(x, x)
    sol = neurolang.solve_all()
    assert sol['q'] == dataset
    assert sol['r'] == set((i,) for i, j in dataset if i == j)
    assert len(sol) == 2
    assert neurolang.predicate_parameter_names(r) == ('x',)


def test_neurolange_dl_get_param_names():
    neurolang = frontend.NeurolangDL()
    r = neurolang.new_symbol(name='r')
    x = neurolang.new_symbol(name='x')

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name='q')
    r[x] = q(x, x)

    assert neurolang.predicate_parameter_names('q') == ('0', '1')
    assert neurolang.predicate_parameter_names(q) == ('0', '1')
    assert neurolang.predicate_parameter_names(r) == ('x',)
    assert neurolang.symbols[r].predicate_parameter_names == ('x',)
    assert r[x].help() is not None


def test_neurolang_dl_datalog_code_list_symbols():
    neurolang = frontend.NeurolangDL()
    original_symbols = set(neurolang.symbols)
    neurolang.execute_datalog_program('''
    A(4, 5)
    A(5, 6)
    A(6, 5)
    B(x,y) :- A(x, y)
    B(x,y) :- B(x, z),A(z, y)
    C(x) :- B(x, y), y == 5
    D("x")
    ''')

    assert set(neurolang.symbols) == {'A', 'B', 'C', 'D'} | original_symbols


def test_neurolang_dl_datalog_code():
    neurolang = frontend.NeurolangDL()
    neurolang.execute_datalog_program('''
    A(4, 5)
    A(5, 6)
    A(6, 5)
    B(x,y) :- A(x, y)
    B(x,y) :- B(x, z),A(z, y)
    C(x) :- B(x, y), y == 5
    D("x")
    ''')

    res = neurolang.solve_all()

    assert res['A'] == {(4, 5), (5, 6), (6, 5)}
    assert res['B'] == {
        (4, 5), (5, 6), (6, 5), (4, 6), (5, 5), (6, 6)
    }
    assert res['C'] == {
        (4,), (5,), (6,)
    }
    assert res['D'] == {
        ('x',),
    }


def test_neurolang_dl_aggregation():
    neurolang = frontend.NeurolangDL()
    q = neurolang.new_symbol(name='q')
    p = neurolang.new_symbol(name='p')
    r = neurolang.new_symbol(name='r')
    x = neurolang.new_symbol(name='x')
    y = neurolang.new_symbol(name='y')

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    for i in range(10):
        q[i % 2, i] = True

    p[x, sum_(y)] = q[x, y]

    sol = neurolang.query(r(x, y), p(x, y))

    res_q = {
        (0, 2 + 4 + 8),
        (1, 1 + 5 + 9)
    }

    assert len(sol) == 2
    assert sol[r] == res_q
    assert sol[p] == res_q


def test_neurolang_dl_attribute_access():
    neurolang = frontend.NeurolangDL()
    one_element = namedtuple('t', ('x', 'y'))(1, 2)

    a = neurolang.add_tuple_set([(one_element,)], name='a')
    with neurolang.scope as e:
        e.q[e.x] = a[e.x]
        e.r[e.y] = a[e.w] & (e.y == e.w.x)
        res = neurolang.solve_all()

    q = res['q']
    r = res['r']
    assert len(q) == 1
    el = next(q.unwrapped_iter())[0]
    assert el == one_element
    assert r.unwrap() == {(one_element.x,)}


def test_neurolang_dl_set_destroy():
    neurolang = frontend.NeurolangDL()
    contains_ = neurolang.add_symbol(contains)

    a = neurolang.add_tuple_set([(frozenset((0, 1, 2)),)], name='a')
    with neurolang.scope as e:
        e.q[e.y] = a[e.x] & contains_(e.x, e.y)
        res = neurolang.solve_all()

    q = res['q'].unwrap()
    assert len(q) == 3
    assert set(q) == {(0,), (1,), (2,)}


def test_multiple_symbols_query():
    neurolang = frontend.RegionFrontend()
    r1 = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 10]]), np.eye(4))
    r2 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    neurolang.add_region(r1, name='r1')
    neurolang.add_region(r2, name='r2')

    central = ExplicitVBR(np.array([[0, 0, 1], [1, 1, 1]]), np.eye(4))
    neurolang.add_region(central, name='reference_region')

    x = neurolang.new_region_symbol(name='x')
    y = neurolang.new_region_symbol(name='y')
    pred = (
        neurolang.symbols.superior_of(x, neurolang.symbols.reference_region) &
        neurolang.symbols.inferior_of(y, neurolang.symbols.reference_region)
    )

    res = neurolang.query((x, y), pred).do()
    assert res.value == frozenset({(r1, r2)})


def test_tuple_symbol_multiple_types_query():
    neurolang = frontend.RegionFrontend()
    r1 = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 10]]), np.eye(4))
    r2 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    neurolang.add_region(r1, name='r1')
    neurolang.add_region(r2, name='r2')

    central = ExplicitVBR(np.array([[0, 0, 1], [1, 1, 1]]), np.eye(4))
    neurolang.add_region(central, name='reference_region')

    x = neurolang.new_region_symbol(name='x')
    z = neurolang.new_symbol(int, name='max_value')

    def norm_of_width(a: int, b: Region) -> bool:
        return bool(np.linalg.norm(b.width) < a)

    neurolang.add_tuple_set(range(10), int)

    neurolang.add_symbol(norm_of_width, 'norm_of_width_gt')

    pred = (
        neurolang.symbols.superior_of(x, neurolang.symbols.reference_region) &
        neurolang.symbols.norm_of_width_gt(
            z, neurolang.symbols.reference_region
        )
    )

    res = neurolang.query((x, z), pred).do()
    assert res.value != frozenset()


def test_quantifier_expressions():

    neurolang = frontend.RegionFrontend()

    i1 = ExplicitVBR(np.array([[0, 0, 2]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, 6]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, 10]]), np.eye(4))
    i4 = ExplicitVBR(np.array([[0, 0, 13], [0, 0, 15]]), np.eye(4))
    regions = {i1, i2, i3, i4}
    neurolang.add_tuple_set(regions, ExplicitVBR)

    central = ExplicitVBR(np.array([[0, 0, 15], [1, 1, 20]]), np.eye(4))
    neurolang.add_region(central, name='reference_region')

    x = neurolang.new_region_symbol(name='x')
    res = neurolang.all(
        x,
        ~neurolang.symbols.superior_of(x, neurolang.symbols.reference_region)
    )
    assert res.do().value

    res = neurolang.exists(
        x,
        neurolang.symbols.overlapping(x, neurolang.symbols.reference_region)
    )
    assert res.do().value


@patch(
    'neurolang.frontend.neurosynth_utils.'
    'NeuroSynthHandler.ns_region_set_from_term'
)
def test_neurosynth_region(mock_ns_regions):
    mock_ns_regions.return_value = {
        ExplicitVBR(np.array([[1, 0, 0], [1, 1, 0]]), np.eye(4))
    }
    neurolang = frontend.RegionFrontend()
    s = neurolang.load_neurosynth_term_regions(
        'gambling', 10, 'gambling_regions'
    )
    res = neurolang[s]
    mock_ns_regions.assert_called()

    assert res.type is AbstractSet[Tuple[ExplicitVBR]]
    assert res.value == frozenset((t,) for t in mock_ns_regions.return_value)


def test_translate_expression_to_fronted_expression():
    qr = frontend.NeurolangDL()
    tr = qre.TranslateExpressionToFrontEndExpression(qr)

    assert tr.walk(exp.Constant(1)) == 1

    symbol_exp = exp.Symbol('a')
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
        symbol_exp(exp.Symbol('x')),
        exp.Symbol('b')(exp.Symbol('x'))
    )
    imp_fe = tr.walk(imp_exp)
    assert imp_fe.expression == imp_exp
    assert imp_fe.consequent == tr.walk(imp_exp.consequent)
    assert imp_fe.antecedent == tr.walk(imp_exp.antecedent)
