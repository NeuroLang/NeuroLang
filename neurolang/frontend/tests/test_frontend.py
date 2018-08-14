from neurolang import frontend
from ...regions import Region

from typing import Tuple, AbstractSet


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
