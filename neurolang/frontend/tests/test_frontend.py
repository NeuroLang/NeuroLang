from neurolang import frontend
from ... import region_solver_ds
from ..query_resolution_expressions import Symbol
from ...regions import Region, ExplicitVBR
from ... import neurolang as nl
import numpy as np
import pytest


def test_add_regions_and_query_included_predicate():

    neurolang = frontend.RegionFrontend(
        region_solver_ds.RegionSolver(nl.TypedSymbolTable())
    )

    inferior = Region((0, 0, 0), (1, 1, 1))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region(inferior, result_symbol_name='inferior_region')
    neurolang.add_region(superior, result_symbol_name='superior_region')
    assert neurolang.symbols.inferior_region.value == inferior
    assert neurolang.symbols.superior_region.value == superior

    result_symbol = neurolang.symbols.superior_of(superior, inferior).do(result_symbol_name='is_superior_test')
    result = result_symbol.expression.value
    result_symbol_from_table = neurolang.get_symbol('is_superior_test').expression.value
    assert result
    assert result_symbol_from_table

    x = neurolang.new_region_symbol(symbol_name='x')
    query = neurolang.query(
        x, neurolang.symbols.superior_of(x, neurolang.symbols.inferior_region)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')
    assert isinstance(query_result, Symbol)
    assert isinstance(query_result.value, frozenset)
    result_symbol = next(iter(query_result.value))
    assert neurolang.symbols[result_symbol].value == superior


def test_query_regions_from_region_set():

    neurolang = frontend.RegionFrontend(
        region_solver_ds.RegionSolver(nl.TypedSymbolTable())
    )

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    neurolang.add_region(central, result_symbol_name='reference_region')

    i1 = ExplicitVBR(np.array([[0, 0, 2], [1, 1, 3]]), np.eye(4))
    i2 = ExplicitVBR(np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    i3 = ExplicitVBR(np.array([[0, 0, -10], [1, 1, -5]]), np.eye(4))

    neurolang.add_region_set({i1, i2, i3}, result_symbol_name='inferiors')
    x = neurolang.new_region_symbol(symbol_name='x')
    query = neurolang.query(
        x, neurolang.symbols.inferior_of(x, neurolang.symbols.reference_region)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')
    assert len(query_result.value) is 3
    for symbol in query_result.value:
        assert neurolang.symbols[symbol].value in {i1, i2, i3}


def test_query_new_predicate():

    neurolang = frontend.RegionFrontend(
        region_solver_ds.RegionSolver(nl.TypedSymbolTable())
    )

    central = ExplicitVBR(np.array([[0, 0, 5], [1, 1, 8]]), np.eye(4))
    reference_symbol = neurolang.add_region(central, result_symbol_name='reference_region')

    inferior_posterior = ExplicitVBR(np.array([[0, -10, -10], [1, -5, -5]]), np.eye(4))
    inferior_central = ExplicitVBR(np.array([[0, 0, -1], [1, 1, 2]]), np.eye(4))
    inferior_anterior = ExplicitVBR(np.array([[0,  2, 2], [1, 5, 3]]), np.eye(4))

    neurolang.add_region_set({
        inferior_posterior, inferior_central, inferior_anterior},
        result_symbol_name='inferiors')

    def posterior_and_inferior_of(y, z):
        return neurolang.symbols.posterior_of(y, z) & neurolang.symbols.inferior_of(y, z)

    x = neurolang.new_region_symbol(symbol_name='x')
    query = neurolang.query(
        x, posterior_and_inferior_of(x, reference_symbol)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')
    for symbol in query_result.value:
        assert neurolang.symbols[symbol].value == inferior_posterior
