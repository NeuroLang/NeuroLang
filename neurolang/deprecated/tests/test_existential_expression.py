import pytest

from typing import Callable, AbstractSet

from ...symbols_and_types import TypedSymbolTable
from .. import SetBasedSolver, FiniteDomainSet
from ... import expressions
from ...expressions import FunctionApplication, ExistentialPredicate

pytestmark = pytest.mark.skipif(..., reason='Deprecated semantics')
C_ = expressions.Constant
S_ = expressions.Symbol


def test_existential_elem_in_set():
    set_type = AbstractSet[int]

    class SBS(SetBasedSolver[int]):
        type = int

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[S_[set_type]('elements')] = C_(
        frozenset([1, 2, 3])
    ).cast(set_type)

    p1 = FunctionApplication[set_type](S_('in'), (S_[set_type]('x'), ))

    exists = ExistentialPredicate[set_type](S_[set_type]('x'), p1)

    assert solver.walk(exists) == solver.symbol_table['elements']


def test_existential_greater_than():
    set_type = AbstractSet[int]

    class SBS(SetBasedSolver[int]):
        def function_is_greater_than(
            self, reference_elem_in_set: AbstractSet[int]
        ) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                    AbstractSet[int]
                ).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    values = range(0, 100)
    solver.symbol_table[S_[set_type]('elements')] = C_(
        frozenset(values)
    ).cast(set_type)

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[S_[pred_type]('is_greater_than')] = C_[pred_type](
        solver.function_is_greater_than
    )

    p1 = FunctionApplication[set_type](
        S_('is_greater_than'), (S_[set_type]('x'), )
    )

    exists = ExistentialPredicate[set_type](S_[set_type]('x'), p1)
    res = solver.walk(exists)
    assert res == C_(frozenset(values[:-1])).cast(set_type)


def test_existential_bound_variable():
    set_type = AbstractSet[int]

    class SBS(SetBasedSolver):
        type = int

        def function_is_greater_than(
            self, reference_elem_in_set: AbstractSet[int]
        ) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                    AbstractSet[int]
                ).values():
                    for elem in elem_set.value:
                        if elem > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[S_[set_type]('elements')] = C_[set_type](
        frozenset(range(1, 100))
    )
    solver.symbol_table[S_[set_type]('x')] = C_[set_type](frozenset([2]))

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[S_[pred_type]('is_greater_is_than')] = C_[pred_type](
        solver.function_is_greater_than
    )

    p1 = FunctionApplication[set_type](
        S_('is_greater_than'), (S_[set_type]('x'), )
    )

    exists = ExistentialPredicate[set_type](S_[type]('x'), p1)

    assert solver.walk(exists) == solver.symbol_table['x']


def test_existential_negate_predicate():
    type = int
    set_type = AbstractSet[type]

    class SBS(SetBasedSolver):
        type = int

        def function_are_consecutives(
            self, reference_elem_in_set: AbstractSet[int]
        ) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                    AbstractSet[int]
                ).values():
                    for elem in elem_set.value:
                        if abs(elem - e) < 2 and e != elem:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[S_[set_type]('elements')] = C_[set_type](
        frozenset([C_(1), C_(2), C_(10)])
    )
    solver.symbol_table[S_[type]('e1')] = C_[type](1)
    solver.symbol_table[S_[type]('e2')] = C_[type](2)
    solver.symbol_table[S_[type]('e3')] = C_[type](10)

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[S_[pred_type]('are_consecutives')] = C_[pred_type](
        solver.function_are_consecutives
    )

    p1 = FunctionApplication[set_type](
        S_('are_consecutives'), (S_[set_type]('x'), )
    )

    exists = ExistentialPredicate[set_type](S_[set_type]('x'), p1)

    assert solver.walk(exists).value == frozenset([C_(1), C_(2)])

    exists = ExistentialPredicate[type](S_[set_type]('x'), ~p1)

    assert solver.walk(exists).value == frozenset([C_(1), C_(2), C_(10)])


def test_existential_unsat_predicate_returns_empty():
    class SBS(SetBasedSolver):
        type = int

        def function_is_additive_inverse(
            self, reference_elem_in_set: AbstractSet[int]
        ) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                    AbstractSet[int]
                ).values():
                    for elem in elem_set.value:
                        if elem + e == 0:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    set_type = AbstractSet[int]
    solver = SBS(TypedSymbolTable())
    solver.symbol_table[S_[set_type]('elements')] = C_[set_type](
        frozenset(C_(i) for i in range(1, 500))
    )

    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[S_[pred_type]('is_additive_inverse')] = C_[pred_type](
        solver.function_is_additive_inverse
    )

    p1 = FunctionApplication[set_type](
        S_('is_additive_inverse'), (S_[set_type]('x'), )
    )

    exists = ExistentialPredicate[set_type](S_[set_type]('x'), p1)

    assert solver.walk(exists).value == frozenset()


def test_existential_predicates_conjuntion_and_disjunction():
    class SBS(SetBasedSolver):
        type = int

        def function_greater_than(
            self, reference_elem_in_set: AbstractSet[int]
        ) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                    AbstractSet[int]
                ).values():
                    for elem in elem_set.value:
                        if elem.value > e:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

        def function_sum_less_than_100(
            self, reference_elem_in_set: AbstractSet[int]
        ) -> AbstractSet[int]:
            res = frozenset()
            for e in reference_elem_in_set:
                for elem_set in self.symbol_table.symbols_by_type(
                    AbstractSet[int]
                ).values():
                    for elem in elem_set.value:
                        if elem.value + e < 100:
                            res = res.union(frozenset([elem]))
            return FiniteDomainSet(res)

    set_type = AbstractSet[int]

    solver = SBS(TypedSymbolTable())
    solver.symbol_table[S_[set_type]('elements')] = C_[set_type](
        frozenset(C_(a) for a in [1, 2, 100])
    )
    pred_type = Callable[[set_type], set_type]
    solver.symbol_table[S_[pred_type]('greater_than')] = C_[pred_type](
        solver.function_greater_than
    )
    solver.symbol_table[S_[pred_type]('sum_less_than_100')] = C_[pred_type](
        solver.function_sum_less_than_100
    )

    p1 = FunctionApplication[set_type](
        S_('greater_than'), (S_[set_type]('x'), )
    )

    p2 = FunctionApplication[set_type](
        S_('sum_less_than_100'), (S_[set_type]('x'), )
    )

    exists = ExistentialPredicate[set_type](S_[set_type]('x'), p1 & p2)

    assert solver.walk(exists).value == frozenset([C_(1)])

    exists = ExistentialPredicate[set_type](S_[set_type]('x'), (p1 | p2))

    assert solver.walk(exists).value == frozenset([C_(1), C_(2)])
