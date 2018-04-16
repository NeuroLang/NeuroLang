from .. import neurolang as nl
from ..symbols_and_types import TypedSymbolTable
from ..region_solver import SetBasedSolver
from ..expressions import (
    Symbol, Predicate, ExistentialPredicate
)
from typing import Callable, AbstractSet


def test_existential_elem_in_set():
    type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[type]('element')] = nl.Constant[type](frozenset([1, 2, 3]))


    p1 = Predicate[type](
        Symbol('in'),
        (Symbol[type]('x'),)
    )

    exists = ExistentialPredicate[type](
         Symbol[type]('x'), p1
        )

    assert solver.walk(exists) == solver.symbol_table['element']