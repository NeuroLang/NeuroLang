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
from ....datalog.magic_sets import AdornedSymbol
from ....datalog.wrapped_collections import WrappedRelationalAlgebraSet
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

    # ------------------------------------------------------------------
    # Coverage gap: body atom with non-Symbol functor (line 76)
    # ------------------------------------------------------------------

    def test_constant_functor_body_atom(self, solver):
        """Body atom with Constant (non-Symbol) functor is skipped
        without error. Covers line 76 (not isinstance(pred.functor, Symbol))."""
        from operator import eq

        mixin, _ = solver

        eq_const = Constant(eq)
        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), eq_const(x, Constant[int](5)))
        result = mixin.walk(rule)

        # x stays Unknown — Constant functor has no symbol-table entry
        assert x.type is Unknown
        assert isinstance(result, Implication)
        assert result.consequent.type is bool

    # ------------------------------------------------------------------
    # Coverage gap: row_type is None for an EDB with no rows (line 103)
    # ------------------------------------------------------------------

    def test_edb_with_no_rows(self, solver):
        """An EDB registered with no tuples has no row_type,
        so _column_types_from_entry returns None and the predicate
        contributes no type information. Covers line 103."""
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(R, [])

        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), R(x))
        mixin.walk(rule)

        # No row type info means no type propagation
        assert x.type is Unknown
        assert rule.consequent.type is bool

    # ------------------------------------------------------------------
    # Coverage gap: column types can be Unknown (line 88)
    # ------------------------------------------------------------------

    def test_unknown_column_type_skipped(self, solver):
        """When a predicate has a column type of Unknown, the type
        resolution skips that column. Covers line 88
        (if inferred is Unknown: continue)."""
        mixin, program = solver

        from typing import Tuple

        R = Symbol("R")
        # Create a WRAS with explicit row_type where one column is Unknown
        wras = WrappedRelationalAlgebraSet(
            iterable=[(1, 2.5)],
            row_type=Tuple[int, Unknown],
            verify_row_type=False,
        )
        program.symbol_table[R] = Constant[AbstractSet[Tuple[int, Unknown]]](
            wras, auto_infer_type=False, verify_type=False
        )

        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")

        rule = Implication(Q(x, y), R(x, y))
        mixin.walk(rule)

        # First column (int) propagates to x
        assert x.type is int
        # Second column (Unknown) is skipped — y stays Unknown
        assert y.type is Unknown
        assert rule.consequent.type is bool

    # ------------------------------------------------------------------
    # Coverage gap: type unification with compatible types (lines 91-97)
    # ------------------------------------------------------------------

    def test_type_unification_compatible_types(self, solver):
        """When two EDBs share a join variable with compatible but
        different types (e.g. int vs int64), the types are unified.
        Covers lines 91-97 (unify_types branch)."""
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(R, [(1, 2.5)])
        S = Symbol("S")
        program.add_extensional_predicate_from_tuples(S, [(1, 10)])

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")
        Q = Symbol("Q")

        conj = Conjunction((R(x, y), S(x, z)))
        rule = Implication(Q(x, y, z), conj)
        mixin.walk(rule)

        # x is shared between R and S, both have int
        assert x.type is int
        assert y.type is float
        assert z.type is int

    # ------------------------------------------------------------------
    # Coverage gap: callable with Ellipsis / variadic args (line 109)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Coverage gap: type unification failure via exception (lines 92-95)
    # ------------------------------------------------------------------

    def test_type_unification_with_incompatible_types(self, solver):
        """When two EDBs share a join variable with incompatible types
        (here int vs str), unify_types raises NeuroLangTypeException.
        The exception is caught and the predicate is skipped.
        Covers lines 92-95 (except/continue in unification path)."""
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(R, [(1, "a")])
        S = Symbol("S")
        program.add_extensional_predicate_from_tuples(S, [("x", 2.5)])

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")
        Q = Symbol("Q")

        # R(x, y) sets x.type = int, y.type = str
        # S(x, z) has row_type Tuple[str, float], would infer x.type = str
        # int vs str are incompatible → unify_types raises → continue
        conj = Conjunction((R(x, y), S(x, z)))
        rule = Implication(Q(x, y, z), conj)
        mixin.walk(rule)

        # x still has its type from R (the continue only skips the
        # current arg in the for loop, not the whole predicate)
        assert x.type is int
        assert y.type is str
        # z gets float from S independently
        assert z.type is float

    def test_callable_with_variadic_args(self, solver):
        """A builtin with variadic Callable[..., bool] type has no
        parameter types to extract — _column_types_from_entry returns
        None via the Ellipsis check. Covers line 109."""
        mixin, program = solver

        fn_sym = Symbol("fn")
        program.symbol_table[fn_sym] = Constant[Callable[..., bool]](
            lambda x: True, auto_infer_type=False, verify_type=False
        )

        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), fn_sym(x))
        mixin.walk(rule)

        # Variadic callable has no resolved parameter types
        assert x.type is Unknown
        assert rule.consequent.type is bool

    # ------------------------------------------------------------------
    # Coverage gap: entry is neither EDB nor callable (line 112)
    # ------------------------------------------------------------------

    def test_non_edb_non_callable_entry(self, solver):
        """A symbol-table entry whose value is neither a
        WrappedRelationalAlgebraSet nor a callable causes
        _column_types_from_entry to return None.
        Covers lines 82 and 112."""
        mixin, program = solver

        R = Symbol("R")
        program.symbol_table[R] = Constant[int](42)

        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), R(x))
        mixin.walk(rule)

        # No type info from a plain int constant
        assert x.type is Unknown
        assert rule.consequent.type is bool



    # ------------------------------------------------------------------
    # User requested: several elements in conjunction
    # ------------------------------------------------------------------

    def test_several_conjunctive_elements(self, solver):
        """Rule with a 3-way conjunctive body resolves all variable
        types correctly."""
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(R, [(1, 2)])
        S = Symbol("S")
        program.add_extensional_predicate_from_tuples(S, [(2, 3)])
        T = Symbol("T")
        program.add_extensional_predicate_from_tuples(T, [(3, 4)])

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")
        w = Symbol("w")
        Q = Symbol("Q")

        conj = Conjunction((R(x, y), S(y, z), T(z, w)))
        rule = Implication(Q(x, y, z, w), conj)
        mixin.walk(rule)

        assert x.type is int
        assert y.type is int
        assert z.type is int
        assert w.type is int
        assert rule.consequent.type is bool

    # ------------------------------------------------------------------
    # User requested: predicates with equalities
    # ------------------------------------------------------------------

    def test_equality_predicate_in_body(self, solver):
        """Rule body with an equality atom (Constant functor like
        operator.eq from 'y == 5') does not block type resolution
        on shared variables that also appear in an EDB predicate."""
        mixin, program = solver

        from operator import eq

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(R, [(1, 2.5)])

        eq_sym = Symbol("eq")
        eq_const = Constant(eq)  # functor used by parsed "y == 5"

        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")

        body = Conjunction((R(x, y), eq_const(y, Constant[float](2.5))))
        rule = Implication(Q(x, y), body)
        mixin.walk(rule)

        # x and y get types from R (the EDB), not from the equality atom
        assert x.type is int
        assert y.type is float
        assert rule.consequent.type is bool

    # ------------------------------------------------------------------
    # IDB-to-IDB type propagation
    # ------------------------------------------------------------------

    def test_idb_predicate_type_propagation(self, solver):
        """Types propagate from an IDB predicate's resolved head args
        to body variables in a dependent rule."""
        mixin, program = solver

        R = Symbol("R")
        program.add_extensional_predicate_from_tuples(
            R, [(10, "hello")]
        )

        # Define Q in terms of R — the type resolver processes this
        # and DatalogProgram.statement_intensional registers Q in
        # the symbol table as a Union of Implications.
        x = Symbol("x")
        y = Symbol("y")
        Q = Symbol("Q")
        rule_q = Implication(Q(x, y), R(x, y))
        mixin.walk(rule_q)

        assert x.type is int
        assert y.type is str

        # Now define P in terms of Q — Q's head args have their types
        # set after rule_q was processed, so the IDB entry for Q
        # carries type information.
        a = Symbol("a")
        b = Symbol("b")
        P = Symbol("P")
        rule_p = Implication(P(a, b), Q(a, b))
        mixin.walk(rule_p)

        assert a.type is int
        assert b.type is str

    def test_warning_for_missing_symbol_table_entry(self, solver):
        """A warning is issued when a body predicate has no entry
        in the symbol table."""
        import warnings

        mixin, _ = solver

        Z = Symbol("Z")
        x = Symbol("x")
        Q = Symbol("Q")

        rule = Implication(Q(x), Z(x))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mixin.walk(rule)

        assert any(
            "no entry" in str(warning.message).lower()
            for warning in w
        ), "Expected a warning about missing symbol table entry"

    # ------------------------------------------------------------------
    # Magic-sets AdornedSymbol guards
    # ------------------------------------------------------------------

    def test_adorned_symbol_skips_type_resolution(self, solver):
        """When the head functor is an AdornedSymbol (as produced by
        magic-sets rewriting), type resolution is skipped entirely
        — no warnings for missing body predicates, no type
        propagation to body Symbols, and the head is still marked
        as bool to prevent re-matching."""
        import warnings

        mixin, _ = solver

        Z = Symbol("Z")
        x = Symbol("x")
        Q = Symbol("Q")

        # AdornedSymbol mimics what the magic-sets layer produces:
        # an adorned copy of the original rule's head functor.
        q_adorned = AdornedSymbol(Q, "bf", 0)
        rule = Implication(q_adorned(x), Z(x))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mixin.walk(rule)

        # No warning despite Z having no symbol-table entry
        assert len(w) == 0, (
            f"Expected no warnings for adorned rules, "
            f"got: {[str(msg.message) for msg in w]}"
        )

        # Body symbols were NOT resolved
        assert x.type is Unknown, "Body symbols should not be resolved"

        # Head FA type is set to bool (prevents re-matching on re-entry)
        assert rule.consequent.type is bool


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

    # ------------------------------------------------------------------
    # IDB-to-IDB type propagation via full solver
    # ------------------------------------------------------------------

    def test_idb_type_propagation_integration(self):
        """Types from an IDB predicate propagate to dependent rules
        through the full solver stack."""
        from ....frontend.deterministic_frontend import NeurolangDL
        from .... import expressions as ir

        nl = NeurolangDL()
        nl.add_tuple_set([(10, "hello")], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y]
            e.P[e.a, e.b] = e.Q[e.a, e.b]

            idb = nl.program_ir.intensional_database()
            p_union = idb[ir.Symbol("P")]
            p_impl = p_union.formulas[0]
            atoms = self._get_body_atoms(p_impl)
            assert atoms[0].args[0].type is int
            assert atoms[0].args[1].type is str

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
        Variable types across diverse column types are resolved.
        """
        from ....frontend.deterministic_frontend import NeurolangDL
        from .... import expressions as ir
        from ....type_system import Unknown

        nl = NeurolangDL()
        nl.add_tuple_set([(1, "hello", 2.5)], name="R")

        with nl.scope as e:
            e.Q[e.x, e.y, e.z] = e.R[e.x, e.y, e.z]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            assert atoms[0].args[1].type is str
            assert atoms[0].args[2].type is float

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
        R, x, y, z, Q = (
            Symbol("R"), Symbol("x"), Symbol("y"),
            Symbol("z"), Symbol("Q"),
        )
        solver.add_extensional_predicate_from_tuples(
            R, [(1, "hello", 2.5), (3, "world", 4.5)]
        )

        rule = Implication(Q(x, y, z), R(x, y, z))
        result = solver.walk(rule)

        assert x.type is int
        assert y.type is str
        assert z.type is float
        assert result.consequent.type is bool

    # ------------------------------------------------------------------
    # User requested: negation in body via full solver
    # ------------------------------------------------------------------

    def test_negation_in_body_type_propagation(self):
        """Negation in the rule body does not block type resolution
        on the positive EDB predicate. Uses the full solver with
        negation support."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 2), (2, 3)], name="R")
        nl.add_tuple_set([(1,)], name="S")

        with nl.scope as e:
            e.Q[e.x, e.y] = e.R[e.x, e.y] & ~e.S[e.x]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            # x and y get their types from R
            assert atoms[0].args[0].type is int
            assert atoms[0].args[1].type is int
            res = nl.query((e.x, e.y), e.Q[e.x, e.y])

        assert len(res) == 1
        assert set(res) == {(2, 3)}

    # ------------------------------------------------------------------
    # User requested: several elements in conjunction via full solver
    # ------------------------------------------------------------------

    def test_several_conjunctive_elements_type_propagation(self):
        """A 3-way conjunctive body resolves all shared variable
        types correctly in the full solver."""
        from ....frontend.deterministic_frontend import NeurolangDL

        nl = NeurolangDL()
        nl.add_tuple_set([(1, 2), (3, 4)], name="R")
        nl.add_tuple_set([(2, 3), (4, 5)], name="S")
        nl.add_tuple_set([(3, 10), (5, 20)], name="T")

        with nl.scope as e:
            e.Q[e.x, e.y, e.z, e.w] = e.R[e.x, e.y] & e.S[e.y, e.z] & e.T[e.z, e.w]
            impl = self._get_stored_implication(nl)
            atoms = self._get_body_atoms(impl)
            assert atoms[0].args[0].type is int
            assert atoms[0].args[1].type is int
            assert atoms[1].args[0].type is int
            assert atoms[1].args[1].type is int
            assert atoms[2].args[0].type is int
            assert atoms[2].args[1].type is int
            res = nl.query((e.x, e.y, e.z, e.w), e.Q[e.x, e.y, e.z, e.w])

        assert len(res) == 2
        assert set(res) == {(1, 2, 3, 10), (3, 4, 5, 20)}

    # ------------------------------------------------------------------
    # User requested: equality predicate in body via full solver
    # ------------------------------------------------------------------

    def test_equality_predicate_type_propagation(self):
        """An equality atom in the rule body (parsed from 'y == 5')
        does not interrupt type resolution on variables shared with
        EDB predicates. Uses text-based Datalog to express equality."""
        from ....frontend.deterministic_frontend import NeurolangDL
        from .... import expressions as ir

        nl = NeurolangDL()
        nl.execute_datalog_program(
            """
        R(1, 3)
        R(2, 5)
        R(3, 7)
        Q(x) :- R(x, y), (y == 5)
        """
        )

        idb = nl.program_ir.intensional_database()
        q_union = idb[ir.Symbol("Q")]
        impl = q_union.formulas[0]
        atoms = self._get_body_atoms(impl)
        # First body atom is R(x, y) — x and y get types from R
        r_atom = atoms[0]
        assert r_atom.args[0].type is int
        assert r_atom.args[1].type is int
        # Second body atom is y == 5 (Constant functor) — skipped
        assert impl.consequent.type is bool
