from neurolang import frontend
from ...regions import Region
import pytest


def test_add_regions_and_query():
    neurolang = frontend.RegionFrontend()

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region_set([inferior, superior])
    neurolang.add_region(central, result_symbol_name='CENTRAL')
    central_predicate = neurolang.define_predicate('superior_of', 'CENTRAL')
    result = neurolang.solve_query(central_predicate)
    assert result.value == frozenset([superior])


@pytest.mark.skip(reason="Need to fix neurosynth-based test")
def test_neurosynth_query():
    neurolang = frontend.RegionFrontend()
    result = neurolang.neurosynth_term_to_region_set('emotion')
    assert len(result.value) != 0


def test_anatomical_superior_of_query():
    neurolang = frontend.RegionFrontend()

    inferior = Region((2, 2, 2), (5, 5, 5))
    central = Region((2, 0, 0), (5, 3, 8))
    superior = Region((2, 2, 6), (5, 5, 8))

    neurolang.add_region(inferior, result_symbol_name='inferior_region')
    neurolang.add_region(central, result_symbol_name='central_region')
    neurolang.add_region(superior, result_symbol_name='superior_region')

    x = neurolang.new_region_symbol(symbol_name='x')
    query = neurolang.query(
        x, neurolang.symbols.superior_of(x, neurolang.symbols.inferior_region)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')

    assert len(query_result.value) == 2

    query = neurolang.query(
        x, neurolang.symbols.anatomical_superior_of(x, neurolang.symbols.inferior_region)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')

    assert len(query_result.value) == 1
    assert next(iter(query_result.value)).name == 'superior_region'
