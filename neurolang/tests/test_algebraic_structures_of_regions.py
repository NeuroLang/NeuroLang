from ..region_solver import RegionsSetSolver
from ..symbols_and_types import TypedSymbolTable
from .. import neurolang as nl
import typing
from typing import AbstractSet, Callable
from ..regions import Region



def test_north_U_south():
    # solver = RegionsSetSolver()
    # solver.set_symbol_table(TypedSymbolTable())

    solver = RegionsSetSolver(TypedSymbolTable())

    db_elems = frozenset([
        Region((0, 5), (1, 6)), Region((0, -10), (1, -8))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('db')] = nl.Constant[typing.AbstractSet[Region]](db_elems)


    elem = frozenset([
        Region((0, 0), (1, 1))
    ])
    solver.symbol_table[nl.Symbol[AbstractSet[Region]]('e1')] = nl.Constant[typing.AbstractSet[Region]](elem)

    check_union_commutativity(AbstractSet[Region], solver, 'north_of', 'south_of', 'e1')

def check_union_commutativity(type, solver, relation1, relation2, element):
    p1 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation1),
        (nl.Symbol[type](element),)
    )

    p2 = nl.Predicate[type](
        nl.Symbol[Callable[[type], type]](relation2),
        (nl.Symbol[type](element),)
    )

    query_a = nl.Query[type](nl.Symbol[type]('a'), p1 | p2)
    query_b = nl.Query[type](nl.Symbol[type]('b'), p2 | p1)

    solver.walk(query_a)
    solver.walk(query_b)

    assert solver.symbol_table['a'] == solver.symbol_table['b']