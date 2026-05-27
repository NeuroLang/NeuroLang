"""Tests for TypeResolutionMixin.

Validates that types from EDB predicates and builtin functions are
propagated to variable symbols in Datalog rules, and that rules with
no resolvable type information are passed through unchanged.
"""

from typing import AbstractSet, Callable, Tuple

import pytest

from .... import expressions as ir
from ....datalog.basic_representation import DatalogProgram
from ....datalog.expressions import Implication
from ....datalog.wrapped_collections import WrappedRelationalAlgebraFrozenSet
from ....expressions import Constant, FunctionApplication, Symbol
from ....frontend.type_resolution import TypeResolutionMixin
from ....logic import Conjunction
from ....type_system import Unknown
from ....utils import NamedRelationalAlgebraFrozenSet

def _dummy_fn(a: int, b: float) -> bool:
    return a > b


@pytest.fixture
def solver():
    """Create a TypeResolutionMixin wired to a fresh DatalogProgram's
    symbol table, with no EDB predicates loaded."""
    program = DatalogProgram()
    mixin = TypeResolutionMixin()
    mixin.symbol_table = program.symbol_table
    return mixin, program


def _typed_symbol(name, type_):
    """Helper to create a Symbol with a specific type annotation."""
    s = Symbol(name)
    s.type = type_
    return s


class TestTypeResolutionMixin:
    def test_edb_predicate_body_resolves_variable_types(self, solver):
        """A single EDB predicate in the body provides column types
        for the rule variables."""
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
        """When the body is a Conjunction with an EDB predicate,
        variable types are still resolved."""
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(
            R, [(10, "hello"), (20, "world")]
        )

        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")
        Q = Symbol("Q")

        conj = Conjunction((R(x, y),))
        rule = Implication(Q(x, y), conj)
        mixin.walk(rule)

        assert x.type is not Unknown
        assert y.type is not Unknown

    def test_skipped_when_no_types_available(self, solver):
        """When no body predicate has a known type in the symbol table,
        the implication is returned unchanged."""
        mixin, program = solver

        R = Symbol("R")
        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), R(x))
        result = mixin.walk(rule)

        assert x.type is Unknown
        assert result is rule

    def test_partial_resolution_mixed_predicates(self, solver):
        """When some body predicates have known types and others don't,
        variables that appear in the typed predicate get their types
        resolved."""
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
        """A body predicate whose functor is a builtin Callable
        provides parameter types."""
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
        """Constant arguments in body predicates don't block type
        inference from the predicate's column types."""
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
