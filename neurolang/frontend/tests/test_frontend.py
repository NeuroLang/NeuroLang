import numpy as np
from neurolang import frontend
from typing import AbstractSet, Tuple
from ..query_resolution_expressions import Symbol
from ...regions import Region, ExplicitVBR, SphericalVolume
from unittest.mock import patch


def test_new_symbol():
    neurolang = frontend.RegionFrontend()

    sym = neurolang.new_symbol(int)
    assert sym.expression.type is int

    sym_ = neurolang.new_symbol((float, int))
    assert sym_.expression.type is Tuple[float, int]
    assert sym.expression.name != sym_.expression.name

    sym = neurolang.new_symbol(int, name='a')
    assert sym.expression.name == 'a'


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


def test_add_regions_and_query():
    neurolang = frontend.RegionFrontend()

    inferior = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region(inferior, result_symbol_name='inferior_region')
    neurolang.add_region(superior, result_symbol_name='superior_region')
    assert neurolang.symbols.inferior_region.value == inferior
    assert neurolang.symbols.superior_region.value == superior

    result_symbol = neurolang.symbols.superior_of(
        superior, inferior).do(result_symbol_name='is_superior_test')
    assert result_symbol.value
    assert neurolang.get_symbol('is_superior_test').value

    x = neurolang.new_region_symbol(name='x')
    query = neurolang.query(
        x, neurolang.symbols.superior_of(x, neurolang.symbols.inferior_region)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')

    assert isinstance(query_result, Symbol)
    assert isinstance(query_result.value, frozenset)
    assert len(query_result.value) == 1
    assert superior == next(iter(query_result.value))


def test_query_regions_from_region_set():
    neurolang = frontend.RegionFrontend()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    neurolang.add_region(central, result_symbol_name='reference_region')

    i1 = ExplicitVBR(np.array([[0, 0, 2], [1, 1, 3]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))
    regions = {i1, i2, i3}
    neurolang.add_tuple_set(regions, ExplicitVBR)

    x = neurolang.new_region_symbol(name='x')
    query_result = neurolang.query(
        x, neurolang.symbols.inferior_of(x, neurolang.symbols.reference_region)
    ).do(result_symbol_name='result_of_test_query')

    assert len(query_result.value) == len(regions)
    assert query_result.value == {i1, i2, i3}


def test_query_new_predicate():
    neurolang = frontend.RegionFrontend()

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    reference_symbol = neurolang.add_region(
        central, result_symbol_name='reference_region')

    inferior_posterior = ExplicitVBR(
        np.array([[0, -10, -10], [1, -5, -5]]), np.eye(4))

    inferior_central = ExplicitVBR(
        np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    inferior_anterior = ExplicitVBR(
        np.array([[0, 2, 2], [1, 5, 3]]), np.eye(4))

    regions = {
        inferior_posterior, inferior_central,
        inferior_anterior
    }

    neurolang.add_tuple_set(regions, ExplicitVBR)

    def posterior_and_inferior(y, z):
        return (
            neurolang.symbols.anatomical_posterior_of(y, z) &
            neurolang.symbols.anatomical_inferior_of(y, z)
        )

    x = neurolang.new_region_symbol(name='x')
    query = neurolang.query(
        x, posterior_and_inferior(x, reference_symbol)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')
    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior_posterior


def test_load_spherical_volume():
    neurolang = frontend.RegionFrontend()

    inferior = ExplicitVBR(
        np.array([[0, 0, 0], [1, 1, 1]]), np.eye(4))

    neurolang.add_region(inferior, result_symbol_name='inferior_region')
    neurolang.sphere((0, 0, 0), 1, result_symbol_name='unit_sphere')

    assert (neurolang.symbols['unit_sphere'].value
            == SphericalVolume((0, 0, 0), 1))

    neurolang.make_implicit_regions_explicit(np.eye(4), (5,) * 3)
    for symbol_name in neurolang.region_names:
        assert isinstance(neurolang.symbols[symbol_name].value, ExplicitVBR)

    x = neurolang.new_region_symbol(name='x')
    query_result = neurolang.query(
        x, neurolang.symbols.overlapping(x, neurolang.symbols.unit_sphere)
    ).do()

    assert len(query_result.value) == 1
    assert next(iter(query_result.value)) == inferior


@patch('neurolang.frontend.neurosynth_utils.'
       'NeuroSynthHandler.ns_region_set_from_term')
def test_neurosynth_region(mock_ns_regions):
    mock_ns_regions.return_value = {
        ExplicitVBR(np.array([[1, 0, 0], [1, 1, 0]]), np.eye(4))
    }
    neurolang = frontend.RegionFrontend()
    s = neurolang.load_neurosynth_term_region(
        'gambling', 10, 'gambling_regions')
    res = neurolang[s]
    mock_ns_regions.assert_called()

    assert res.type is AbstractSet[ExplicitVBR]
    assert res.value == frozenset(mock_ns_regions.return_value)
