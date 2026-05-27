"""
Tests for TypeResolutionMixin.

Validates that types from EDB predicates and builtin functions are
propagated to variable symbols in Datalog rules, and that rules with
no resolvable type information are passed through unchanged.
"""

from typing import Callable

import pytest

from ....datalog.basic_representation import DatalogProgram
from ....datalog.expressions import Implication
from ....expressions import Constant, Symbol
from ....frontend.type_resolution import TypeResolutionMixin
from ....logic import Conjunction
from ....type_system import Unknown


def _dummy_fn(a: int, b: float) -> bool:
    return a > b


class _TypeResolutionSolver(TypeResolutionMixin, DatalogProgram):
    """
    Combined solver so that self.walk(expression) flows through
    to DatalogProgram.statement_intensional after type resolution.
    """


@pytest.fixture
def solver():
    """Create a _TypeResolutionSolver with no EDB predicates loaded."""
    s = _TypeResolutionSolver()
    return s, s


class TestTypeResolutionMixin:
    def test_edb_predicate_body_resolves_variable_types(self, solver):
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(
            R, [(1, 2.5), (3, 4.5)]
        )

        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")

        assert x.type is Unknown
        assert y.type is Unknown

        rule = Implication(Q(x, y), R(x, y))
        mixin.walk(rule)

        assert x.type is not Unknown
        assert y.type is not Unknown
        assert x.type is int or issubclass(x.type, int)
        assert y.type is float or issubclass(y.type, float)

    def test_conjunctive_body_resolves_variables(self, solver):
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(
            R, [(10, "hello"), (20, "world")]
        )

        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")

        conj = Conjunction((R(x, y),))
        rule = Implication(Q(x, y), conj)
        mixin.walk(rule)

        assert x.type is not Unknown
        assert y.type is not Unknown

    def test_rule_passes_through_when_no_types_available(self, solver):
        mixin, _ = solver

        R = Symbol("R")
        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), R(x))
        result = mixin.walk(rule)

        assert x.type is Unknown
        assert isinstance(result, type(rule))

    def test_partial_resolution_mixed_predicates(self, solver):
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(R, [(1,)])

        S = Symbol("S")
        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")

        conj = Conjunction((R(x), S(y)))
        rule = Implication(Q(x, y), conj)
        mixin.walk(rule)

        assert x.type is not Unknown
        assert y.type is Unknown

    def test_builtin_callable_body_resolves_arg_types(self, solver):
        mixin, program = solver

        fn_sym = Symbol("fn")
        program.symbol_table[fn_sym] = Constant[Callable[[int, float], bool]](
            _dummy_fn
        )

        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")

        rule = Implication(Q(x, y), fn_sym(x, y))
        mixin.walk(rule)

        assert x.type is not Unknown
        assert y.type is not Unknown

    def test_constant_arguments_do_not_block_typed_predicates(self, solver):
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(
            R, [(1, 2)]
        )

        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), R(x, Constant[int](3)))
        mixin.walk(rule)

        assert x.type is not Unknown
