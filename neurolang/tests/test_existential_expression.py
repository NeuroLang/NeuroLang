from .. import neurolang as nl
from ..symbols_and_types import TypedSymbolTable
from ..solver import SetBasedSolver, FiniteDomainSet
from ..expressions import (Symbol, Predicate, ExistentialPredicate)
from typing import Callable, AbstractSet


def test_existential_elem_in_set():
    set_type = AbstractSet[int]

    class SBS(SetBasedSolver[int]):
        type = int

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[
        set_type](frozenset(nl.Constant[int](i) for i in (1, 2, 3)))

    p1 = Predicate[set_type](Symbol('in'), (Symbol[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](Symbol[set_type]('x'), p1)

    assert solver.walk(exists) == solver.symbol_table['elements']


def test_existential_greater_than():
    set_type = AbstractSet[int]

    class SBS(SetBasedSolver[int]):
        def predicate_is_greater_than(
                self,
                reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                        AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    values = range(0, 100)
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[
        set_type](frozenset(nl.Constant[int](v) for v in values))

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[nl.Symbol[pred_type]('is_greater_than')] = nl.Constant[
        pred_type](solver.predicate_is_greater_than)

    p1 = Predicate[set_type](Symbol('is_greater_than'),
                             (Symbol[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](Symbol[set_type]('x'), p1)
    res = solver.walk(exists)
    assert res == nl.Constant[set_type](frozenset(values[:-1]))


def test_existential_bound_variable():
    set_type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

        def predicate_is_greater_than(
                self,
                reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                        AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[
        set_type](frozenset(range(1, 100)))
    solver.symbol_table[nl.Symbol[set_type]('x')] = nl.Constant[set_type](
        frozenset([2]))

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[nl.Symbol[pred_type](
        'is_greater_is_than')] = nl.Constant[pred_type](
            solver.predicate_is_greater_than)

    p1 = Predicate[set_type](Symbol('is_greater_than'),
                             (Symbol[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](Symbol[type]('x'), p1)

    assert solver.walk(exists) == solver.symbol_table['x']


def test_existential_negate_predicate():
    type = int
    set_type = AbstractSet[type]

    class SBS(SetBasedSolver):
        type = int

        def predicate_are_consecutives(
                self,
                reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                        AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if abs(elem - e) < 2 and e != elem:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[
        set_type](frozenset([1, 2, 10]))
    solver.symbol_table[nl.Symbol[type]('e1')] = nl.Constant[type](1)
    solver.symbol_table[nl.Symbol[type]('e2')] = nl.Constant[type](2)
    solver.symbol_table[nl.Symbol[type]('e3')] = nl.Constant[type](10)

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[nl.Symbol[pred_type](
        'are_consecutives')] = nl.Constant[pred_type](
            solver.predicate_are_consecutives)

    p1 = Predicate[set_type](Symbol('are_consecutives'),
                             (Symbol[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](Symbol[set_type]('x'), p1)

    assert solver.walk(exists).value == frozenset([1, 2])

    exists = ExistentialPredicate[type](Symbol[set_type]('x'), ~p1)

    assert solver.walk(exists).value == frozenset([1, 2, 10])


def test_existential_unsat_predicate_returns_empty():
    class SBS(SetBasedSolver):
        type = int

        def predicate_is_additive_inverse(
                self,
                reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                        AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem + e == 0:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    set_type = AbstractSet[int]
    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[
        set_type](frozenset(range(1, 500)))

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[nl.Symbol[pred_type](
        'is_additive_inverse')] = nl.Constant[pred_type](
            solver.predicate_is_additive_inverse)

    p1 = Predicate[set_type](Symbol('is_additive_inverse'),
                             (Symbol[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](Symbol[set_type]('x'), p1)

    assert solver.walk(exists).value == frozenset()


def test_existential_predicates_conjuntion_and_disjunction():
    class SBS(SetBasedSolver):
        type = int

        def predicate_greater_than(
                self,
                reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                        AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

        def predicate_sum_less_than_100(
                self,
                reference_elem_in_set: AbstractSet[int]) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                        AbstractSet[int]).values():
                    for elem in elem_set.value:
                        if elem + e < 100:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    set_type = AbstractSet[int]

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[nl.Symbol[set_type]('elements')] = nl.Constant[
        set_type](frozenset([1, 2, 100]))
    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[nl.Symbol[pred_type]('greater_than')] = nl.Constant[
        pred_type](solver.predicate_greater_than)
    solver.symbol_table[nl.Symbol[pred_type](
        'sum_less_than_100')] = nl.Constant[pred_type](
            solver.predicate_sum_less_than_100)

    p1 = Predicate[set_type](Symbol('greater_than'), (Symbol[set_type]('x'), ))

    p2 = Predicate[set_type](Symbol('sum_less_than_100'),
                             (Symbol[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](Symbol[set_type]('x'), p1 & p2)

    assert solver.walk(exists).value == frozenset([1])

    exists = ExistentialPredicate[set_type](Symbol[set_type]('x'), (p1 | p2))

    assert solver.walk(exists).value == frozenset([1, 2])
