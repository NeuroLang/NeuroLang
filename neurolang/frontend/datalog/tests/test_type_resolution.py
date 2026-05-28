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
    """
    Integration tests using the full solver stack.

    Type assertions inspect the stored Implication's body atoms directly
    (inside the scope block before scope exit discards symbols).
    """

    def _get_body_atoms(self, impl):
        from ....datalog.expression_processing import extract_logic_atoms
        return list(extract_logic_atoms(impl.antecedent))

    def _get_stored_implication(self, nl):
        """Retrieve the single Implication for head symbol Q from the
        program's IDB while still inside the active scope."""
        from .... import expressions as ir
        idb = nl.program_ir.intensional_database()
        q_sym = ir.Symbol("Q")
        q_union = idb[q_sym]
        return q_union.formulas[0]

    def test_edb_body_atom_types(self):
        """Simple EDB rule: body-predicate args have resolved types."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(10, "hello"), (20, "world")], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            assert atoms[0].args[1].type is str
            res = nl.query((e.x, e.y), e.Q[e.x, e.y])

        assert len(res) == 2
        assert set(res) == {(10, "hello"), (20, "world")}

    def test_multi_column_body_atom_types(self):
        """Three-column EDB: all column types resolved."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set(
            [(1, 2.5, "a"), (3, 4.5, "b")], name="R"
        )

        with nl.scope as e:
            e.Q[e.x, e.y, e.z] = e.R[e.x, e.y, e.z]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            assert atoms[0].args[1].type is float
            assert atoms[0].args[2].type is str
            res = nl.query((e.x, e.y, e.z), e.Q[e.x, e.y, e.z])

        assert len(res) == 2
        assert set(res) == {(1, 2.5, "a"), (3, 4.5, "b")}

    def test_type_unification_in_join_body_atoms(self):
        """Join variable in conjunctive body has unified type."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 2), (3, 4)], name="R")
        nl.add_tuple_set([(1, 10), (3, 30)], name="S")

        with nl.scope as e:
            e.Q[e.x, e.y, e.z] = e.R[e.x, e.y] & e.S[e.x, e.z]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            assert atoms[1].args[0].type is int
            res = nl.query((e.x, e.y, e.z), e.Q[e.x, e.y, e.z])

        assert len(res) == 2
        assert set(res) == {(1, 2, 10), (3, 4, 30)}

    def test_conjunctive_body_rule_query_result(self):
        """Conjunctive body query produces correct results."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, "a"), (2, "b")], name="R")
        nl.add_tuple_set([(1, 10), (2, 20)], name="S")

        with nl.scope as e:
            e.Q[e.x, e.y, e.z] = e.R[e.x, e.y] & e.S[e.x, e.z]
            res = nl.query((e.x, e.y, e.z), e.Q[e.x, e.y, e.z])

        assert len(res) == 2
        assert set(res) == {(1, "a", 10), (2, "b", 20)}

    def test_partial_type_resolution_body_atoms(self):
        """
        Mixed typed/untyped EDBs: only known column types resolved,
        untyped variables remain Unknown.
        """
        from ....frontend.deterministic_frontend import NeurolangDL
        from ....type_system import Unknown

        nl = NeurolangDL()
        nl.add_tuple_set([(1,)], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x] & e.S[e.y]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            assert atoms[1].args[0].type is Unknown

    def test_head_predicate_type_is_bool(self):
        """
        After resolution the head FunctionApplication type is bool,
        preventing re-matching by FunctionApplication[Unknown].
        """
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 2.5)], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y]
            impl = self._get_stored_implication(nl)
            assert impl.consequent.type is bool

    def test_constant_argument_body_atom_types(self):
        """EDB with constant argument: variable arg still resolved."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 3), (3, 4)], name="R")

        with nl.scope as e:
            e.Q[e.x] = e.R[e.x, 3]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            # The constant 3 is not a Symbol, so its type is irrelevant
            res = nl.query((e.x,), e.Q[e.x])

        assert len(res) == 1
        assert set(res) == {(1,)}

    def test_predicate_symbol_type_in_symbol_table(self):
        """
        Predicate symbol in the program's symbol table has its type
        set (from the EDB or builtin entry), not left as Unknown.
        """
        from ....frontend.deterministic_frontend import NeurolangDL
        from .... import expressions as ir
        from ....type_system import Unknown

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 2.5)], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y]

        r_entry = nl.program_ir.symbol_table.get(ir.Symbol("R"))
        assert r_entry is not None
        assert r_entry.type is not Unknown

    def test_region_frontend_solver_direct_type_resolution(self):
        """
        Using RegionFrontendDatalogSolver directly (the class where
        TypeResolutionMixin is wired into MRO) resolves IR Symbol types
        identically to the minimal _TypeResolutionSolver used in unit tests.
        """
        from ....frontend.deterministic_frontend import (
            RegionFrontendDatalogSolver,
        )
        from ....datalog.expressions import Implication
        from ....expressions import Symbol

        solver = RegionFrontendDatalogSolver()
        R, x, y, Q = Symbol("R"), Symbol("x"), Symbol("y"), Symbol("Q")
        solver.add_extensional_predicate_from_tuples(
            R, [(1, 2.5), (3, 4.5)]
        )

        rule = Implication(Q(x, y), R(x, y))
        result = solver.walk(rule)

        assert x.type is int
        assert y.type is float
        assert result.consequent.type is bool
