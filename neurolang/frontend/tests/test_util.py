from neurolang import frontend
from ...region_solver import RegionsSetSolver
from ...symbols_and_types import TypedSymbolTable
from ...regions import Region
from typing import AbstractSet
from ... import neurolang as nl


def test_regions_names_from_table():
    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    center = Region((0, 0, 0), (1, 1, 1))
    l1 = Region((-5, -2, -2), (-2, 0, 0))
    l2 = Region((-10, 5, 5), (-8, 8, 7))
    l3 = Region((-12, 9, 8), (-11, 10, 10))

    l1_elem = frozenset([l1])
    l2_elem = frozenset([l2])
    l3_elem = frozenset([l3])
    center_elem = frozenset([center])

    solver.symbol_table[nl.Symbol[region_set_type]('L1')] = nl.Constant[region_set_type](l1_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('L2')] = nl.Constant[region_set_type](l2_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('L3')] = nl.Constant[region_set_type](l3_elem)
    solver.symbol_table[nl.Symbol[region_set_type]('CENTRAL')] = nl.Constant[region_set_type](center_elem)
    search_for = frozenset([l1, center])
    res = frontend.symbol_names_of_region_set(solver, search_for)
    assert res == ['L1', 'CENTRAL']


def test_query_symbols_from_table():

    region_set_type = AbstractSet[Region]
    solver = RegionsSetSolver(TypedSymbolTable())

    inferior = Region((0, 0, 0), (1, 1, 1))
    central = Region((0, 0, 2), (1, 1, 3))
    superior = Region((0, 0, 4), (1, 1, 5))

    all_elements = frozenset([inferior, central, superior])
    solver.symbol_table[nl.Symbol[region_set_type]('db')] = nl.Constant[region_set_type](all_elements)
    solver.symbol_table[nl.Symbol[region_set_type]('CENTRAL')] = nl.Constant[region_set_type](frozenset([central]))
    obtained = frontend.query_relation_region(solver, 'superior_of', 'CENTRAL')
    assert len(obtained) == 0

    solver.symbol_table[nl.Symbol[region_set_type]('BOTTOM')] = nl.Constant[region_set_type](frozenset([inferior]))
    solver.symbol_table[nl.Symbol[region_set_type]('TOP')] = nl.Constant[region_set_type](frozenset([superior]))
    obtained = frontend.query_relation_region(solver, 'superior_of', 'CENTRAL')
    assert obtained == ['TOP']

    frontend.query_relation_region(solver, 'superior_of', 'BOTTOM', store_symbol_name='not_bottom')
    assert solver.symbol_table['not_bottom'].value == frozenset([central, superior])


def test_ns_fr():
    solver = RegionsSetSolver(TypedSymbolTable())
    fs = frontend.neurosynth_term_to_region_set(solver, 'emotion')
    print(next(iter(fs)))