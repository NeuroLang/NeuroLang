from .. import neurolang as nl
from ..symbols_and_types import TypedSymbolTable
from ..solver import SetBasedSolver, FiniteDomainSet
from ..expressions import (
    Symbol, Predicate, ExistentialPredicate
)
from typing import Callable, AbstractSet


def test_existential_elem_in_set():
    type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[type]('elements')] = nl.Constant[type](frozenset([1, 2, 3]))

    p1 = Predicate[type](
        Symbol('in'),
        (Symbol[type]('x'),)
    )

    exists = ExistentialPredicate[type](
         Symbol[type]('x'), p1
        )

    assert solver.walk(exists) == solver.symbol_table['elements']


def test_existential_greater_than():
    type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

        def predicate_is_greater_than(self, reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)
    solver = SBS(TypedSymbolTable())
    values = range(0, 100)
    solver.symbol_table[nl.Symbol[type]('elements')] = nl.Constant[type](frozenset(values))

    pred_type = Callable[
        [type],
        type
    ]
    solver.symbol_table[nl.Symbol[pred_type]('is_greater_than')] = nl.Constant[pred_type](solver.predicate_is_greater_than)

    p1 = Predicate[type](
        Symbol('is_greater_than'),
        (Symbol[type]('x'),)
    )

    exists = ExistentialPredicate[type](
        Symbol[type]('x'), p1
    )
    res = solver.walk(exists)
    assert res == nl.Constant[type](frozenset(values[1:]))


def test_existential_bound_variable():
    type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

        def predicate_is_greater_than(self, reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[type]('elements')] = nl.Constant[type](frozenset(range(1, 100)))
    solver.symbol_table[nl.Symbol[type]('x')] = nl.Constant[type](frozenset([2]))

    pred_type = Callable[
        [type],
        type
    ]
    solver.symbol_table[nl.Symbol[pred_type]('is_greater_is_than')] = nl.Constant[pred_type](solver.predicate_is_greater_than)

    p1 = Predicate[type](
        Symbol('is_greater_than'),
        (Symbol[type]('x'),)
    )

    exists = ExistentialPredicate[type](
        Symbol[type]('x'), p1
    )

    assert solver.walk(exists) == solver.symbol_table['x']


def test_existential_negate_predicate():
    type = int
    set_type = AbstractSet[type]

    class SBS(SetBasedSolver):
        type = int

        def predicate_are_consecutives(self, reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if abs(elem - e) < 2 and e != elem:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[set_type](frozenset([1, 2, 10]))
    solver.symbol_table[nl.Symbol[type]('e1')] = nl.Constant[type](1)
    solver.symbol_table[nl.Symbol[type]('e2')] = nl.Constant[type](2)
    solver.symbol_table[nl.Symbol[type]('e3')] = nl.Constant[type](10)


    pred_type = Callable[
        [set_type],
        set_type
    ]
    solver.symbol_table[nl.Symbol[pred_type]('are_consecutives')] = nl.Constant[pred_type](
        solver.predicate_are_consecutives)

    p1 = Predicate[type](
        Symbol('are_consecutives'),
        (Symbol[set_type]('x'),)
    )

    exists = ExistentialPredicate[type](
        Symbol[set_type]('x'), p1
    )

    assert solver.walk(exists).value == frozenset([1, 2])

    exists = ExistentialPredicate[type](
        Symbol[set_type]('x'), ~p1
    )

    assert solver.walk(exists).value == frozenset([1, 2, 10])


def test_existential_unsat_predicate_returns_empty():
    type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

        def predicate_is_additive_inverse(self, reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem + e == 0:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[type]('elements')] = nl.Constant[type](frozenset([1, 2]))

    pred_type = Callable[
        [type],
        type
    ]
    solver.symbol_table[nl.Symbol[pred_type]('is_additive_inverse')] = nl.Constant[pred_type](
        solver.predicate_is_additive_inverse)

    p1 = Predicate[type](
        Symbol('is_additive_inverse'),
        (Symbol[type]('x'),)
    )

    exists = ExistentialPredicate[type](
        Symbol[type]('x'), p1
    )

    assert solver.walk(exists).value == frozenset()


def test_existential_predicates_conjuntion_and_disjunction():
    type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

        def predicate_greater_than(self, reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

        def predicate_sum_less_than_100(self, reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem + e < 100:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[type]('elements')] = nl.Constant[type](frozenset([1, 2, 100]))
    pred_type = Callable[
        [type],
        type
    ]
    solver.symbol_table[nl.Symbol[pred_type]('greater_than')] = nl.Constant[pred_type](solver.predicate_greater_than)
    solver.symbol_table[nl.Symbol[pred_type]('sum_less_than_100')] = nl.Constant[pred_type](solver.predicate_sum_less_than_100)

    p1 = Predicate[type](
        Symbol('greater_than'),
        (Symbol[type]('x'),)
    )

    p2 = Predicate[type](
        Symbol('sum_less_than_100'),
        (Symbol[type]('x'),)
    )

    exists = ExistentialPredicate[type](
        Symbol[type]('x'), (p1 & p2)
    )

    assert solver.walk(exists).value == frozenset([2])

    exists = ExistentialPredicate[type](
        Symbol[type]('x'), (p1 | p2)
    )

    assert solver.walk(exists).value == frozenset([1, 2, 100])
