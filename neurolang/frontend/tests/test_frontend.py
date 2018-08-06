from neurolang import frontend
from ...regions import Region


def test_add_regions_and_query():
    neurolang = frontend.RegionFrontend()

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    neurolang.add_region(inferior, result_symbol_name='inferior')
    neurolang.add_region(central, result_symbol_name='central')
    neurolang.add_region(superior, result_symbol_name='superior')

    x = neurolang.new_region_symbol(symbol_name='x')
    query = neurolang.query(
        x, neurolang.symbols.superior_of(x, neurolang.symbols.central)
    )
    query_result = query.do(result_symbol_name='result_of_test_query')
    result_symbol = next(iter(query_result.value))
    assert neurolang.symbols[result_symbol].value == superior


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
