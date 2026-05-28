"""
Tests for TypeResolutionMixin.

Validates that types from EDB predicates and builtin functions are
propagated to variable symbols in Datalog rules, and that rules with
no resolvable type information are passed through unchanged.
"""

from typing import AbstractSet, Callable

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

        assert x.type is int
        assert y.type is float
        # Predicate symbol type propagated from EDB entry
        entry = program.symbol_table[R]
        assert entry.type is not Unknown

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

        assert x.type is int
        assert y.type is str
        entry = program.symbol_table[R]
        assert entry.type is not Unknown

    def test_rule_passes_through_when_no_types_available(self, solver):
        mixin, _ = solver

        R = Symbol("R")
        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), R(x))
        result = mixin.walk(rule)

        assert x.type is Unknown
        assert isinstance(result, type(rule))
        # Head FA type set to bool prevents infinite re-matching
        assert rule.consequent.type is bool

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

        assert x.type is int
        assert y.type is Unknown
        entry = program.symbol_table[R]
        assert entry.type is not Unknown

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

        assert x.type is int
        assert y.type is float

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

        assert x.type is int
        entry = program.symbol_table[R]
        assert entry.type is not Unknown


class TestTypeResolutionIntegration:
    """Integration tests using the full NeurolangDL solver stack."""

    def test_edb_propagates_types_through_full_solver(self):
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(10, "hello"), (20, "world")], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y]
            res = nl.query((e.x, e.y), e.Q[e.x, e.y])

        assert len(res) == 2
        assert set(res) == {(10, "hello"), (20, "world")}

    def test_partial_type_resolution_with_mixed_predicates(self):
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 2.5), (3, 4.5)], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y]
            res = nl.query((e.x, e.y), e.Q[e.x, e.y])

        assert len(res) == 2
        assert set(res) == {(1, 2.5), (3, 4.5)}

    def test_conjunctive_body_rule_in_full_solver(self):
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, "a"), (2, "b")], name="R")
        nl.add_tuple_set([(1, 10), (2, 20)], name="S")

        with nl.scope as e:
            e.Q[e.x, e.y, e.z] = e.R[e.x, e.y] & e.S[e.x, e.z]
            res = nl.query((e.x, e.y, e.z), e.Q[e.x, e.y, e.z])

        assert len(res) == 2
        assert set(res) == {(1, "a", 10), (2, "b", 20)}
