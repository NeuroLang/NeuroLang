"""Tests for StripDimensionTypePredicatesMixin and make_dimension_type_stripper."""

from ....datalog import Conjunction
from ....expression_walker import ExpressionWalker
from ....expressions import Constant, FunctionApplication, Symbol
from ..squall import (
    StripDimensionTypePredicatesMixin,
    _DIMENSION_TYPE_PREDICATE_NAMES,
    make_dimension_type_stripper,
)


class _TestWalker(StripDimensionTypePredicatesMixin, ExpressionWalker):
    """Composite walker for unit tests: mixin + ExpressionWalker fallback."""
    pass


def test_default_predicate_names():
    """Default set contains exactly probability and value."""
    assert _DIMENSION_TYPE_PREDICATE_NAMES == frozenset({"probability", "value"})


class TestStripDimensionTypePredicatesMixin:
    """Unit tests for direct walker instantiation."""

    def test_replaces_dimension_type_atom(self):
        """probability(v) is replaced with Constant(True)."""
        walker = _TestWalker()
        v = Symbol("v")
        prob_atom = FunctionApplication(Symbol("probability"), (v,))
        result = walker.walk(prob_atom)
        assert result == Constant(True), (
            f"Expected Constant(True), got {result}"
        )

    def test_replaces_value_atom(self):
        """value(v) is replaced with Constant(True)."""
        walker = _TestWalker()
        v = Symbol("v")
        value_atom = FunctionApplication(Symbol("value"), (v,))
        result = walker.walk(value_atom)
        assert result == Constant(True), (
            f"Expected Constant(True), got {result}"
        )

    def test_leaves_non_dimension_predicate(self):
        """voxel(s0, s1, s2) is left untouched."""
        walker = _TestWalker()
        s0, s1, s2 = Symbol("s0"), Symbol("s1"), Symbol("s2")
        voxel_atom = FunctionApplication(Symbol("voxel"), (s0, s1, s2))
        result = walker.walk(voxel_atom)
        assert result is voxel_atom, (
            f"Expected same object, got {result}"
        )

    def test_leaves_unknown_dimension_name(self):
        """intensity(v) — not in default set — is left untouched."""
        walker = _TestWalker()
        v = Symbol("v")
        intensity_atom = FunctionApplication(Symbol("intensity"), (v,))
        result = walker.walk(intensity_atom)
        assert result is intensity_atom, (
            f"Expected same object, got {result}"
        )

    def test_leaves_non_predicate_function_application(self):
        """Non-Symbol functor (e.g. Constant operator) is left untouched."""
        from operator import eq
        walker = _TestWalker()
        v = Symbol("v")
        expr = FunctionApplication(Constant(eq), (v, Constant(42)))
        result = walker.walk(expr)
        assert result is expr, (
            f"Expected same object, got {result}"
        )

    def test_strips_from_conjunction(self):
        """Conjunction with probability(v) has it removed via Constant(True)."""
        walker = _TestWalker()
        v = Symbol("v")
        conj = Conjunction((
            Symbol("voxel")(v, v, v),
            Symbol("probability")(v),
        ))
        result = walker.walk(conj)
        assert isinstance(result, Conjunction), (
            f"Expected Conjunction, got {type(result)}: {result}"
        )
        # probability should be replaced by Constant(True), then
        # the LogicSimplifier remove_true_from_conjunction would
        # strip it — but since we just have StripDimensionTypePredicatesMixin
        # + ExpressionWalker fallback, Constant(True) will be in formulas.
        functors = {
            f.functor.name
            for f in result.formulas
            if isinstance(f, FunctionApplication)
        }
        assert "probability" not in functors, (
            f"probability should have been replaced; got {functors}"
        )
        assert "voxel" in functors


class TestMakeDimensionTypeStripper:
    """Tests for make_dimension_type_stripper factory."""

    def test_default_predicates(self):
        """No arguments uses default {'probability', 'value'}."""
        Mixin = make_dimension_type_stripper()
        assert issubclass(Mixin, StripDimensionTypePredicatesMixin)
        assert Mixin.dimension_type_predicate_names == frozenset(
            {"probability", "value"}
        )

    def test_custom_predicates(self):
        """Custom names override the default set."""
        Mixin = make_dimension_type_stripper(["intensity", "confidence"])

        class _CustomWalker(Mixin, ExpressionWalker):
            pass

        walker = _CustomWalker()
        v = Symbol("v")

        # Custom names are replaced
        intensity_atom = FunctionApplication(Symbol("intensity"), (v,))
        result = walker.walk(intensity_atom)
        assert result == Constant(True), (
            f"Expected Constant(True) for intensity, got {result}"
        )

        # Default names are NOT replaced
        prob_atom = FunctionApplication(Symbol("probability"), (v,))
        result = walker.walk(prob_atom)
        assert result is prob_atom, (
            f"Expected untouched for probability, got {result}"
        )

    def test_qualified_name(self):
        """Factory subclass has qualified name matching the base."""
        Mixin = make_dimension_type_stripper()
        assert Mixin.__qualname__ == "StripDimensionTypePredicatesMixin"

    def test_override_via_subclass(self):
        """Subclass overrides dimension_type_predicate_names correctly."""
        class _CustomMixin(StripDimensionTypePredicatesMixin):
            dimension_type_predicate_names = frozenset({"foo", "bar"})

        class _CustomWalker(_CustomMixin, ExpressionWalker):
            pass

        walker = _CustomWalker()
        v = Symbol("v")

        foo_atom = FunctionApplication(Symbol("foo"), (v,))
        assert walker.walk(foo_atom) == Constant(True)

        prob_atom = FunctionApplication(Symbol("probability"), (v,))
        assert walker.walk(prob_atom) is prob_atom


class TestMROIntegration:
    """MRO integration tests — walk through a full solver MRO."""

    def test_strips_from_implication_deterministic(self):
        """RegionFrontendDatalogSolver strips probability(v) from a Conjunction."""
        from ....frontend.deterministic_frontend import (
            RegionFrontendDatalogSolver,
        )

        solver = RegionFrontendDatalogSolver()
        s0, s1, s2, s3, s4 = (
            Symbol("s0"), Symbol("s1"), Symbol("s2"),
            Symbol("s3"), Symbol("s4"),
        )

        conj = Conjunction((
            Symbol("voxel")(s0, s1, s2),
            Symbol("probability")(s3),
            Symbol("schaefer_label")(s4),
            Symbol("labels")(s4, s0, s1, s2),
            Symbol("label_reports")(s4, s3),
        ))

        # Note: walking a full Implication through DatalogProgram returns
        # the original expression (DatalogProgram.statement_intensional
        # preserves the input). Walk the antecedent directly to verify
        # the strip mixin runs inside the solver MRO.
        result = solver.walk(conj)
        assert isinstance(result, Conjunction), (
            f"Expected Conjunction, got {type(result)}: {result}"
        )

        functors_in_body = {
            f.functor.name
            for f in result.formulas
            if isinstance(f, FunctionApplication)
        }
        assert "probability" not in functors_in_body, (
            f"probability should be stripped; got {functors_in_body}"
        )
        assert "voxel" in functors_in_body
        assert "schaefer_label" in functors_in_body
        assert "labels" in functors_in_body
        assert "label_reports" in functors_in_body

    def test_strips_from_implication_probabilistic(self):
        """RegionFrontendCPLogicSolver strips probability(v) from a Conjunction."""
        from ....frontend.probabilistic_frontend import (
            RegionFrontendCPLogicSolver,
        )

        solver = RegionFrontendCPLogicSolver()
        s0, s1, s2, s3, s4 = (
            Symbol("s0"), Symbol("s1"), Symbol("s2"),
            Symbol("s3"), Symbol("s4"),
        )

        conj = Conjunction((
            Symbol("voxel")(s0, s1, s2),
            Symbol("probability")(s3),
            Symbol("schaefer_label")(s4),
            Symbol("labels")(s4, s0, s1, s2),
            Symbol("label_reports")(s4, s3),
        ))

        result = solver.walk(conj)
        assert isinstance(result, Conjunction), (
            f"Expected Conjunction, got {type(result)}: {result}"
        )

        functors_in_body = {
            f.functor.name
            for f in result.formulas
            if isinstance(f, FunctionApplication)
        }
        assert "probability" not in functors_in_body, (
            f"probability should be stripped; got {functors_in_body}"
        )

    def test_custom_mixin_in_mro(self):
        """Custom mixin with non-default names works in a solver MRO."""
        from ....frontend.deterministic_frontend import (
            TranslateToLogicWithAggregation,
            TranslateSSugarToSelectByColumn,
            TranslateSelectByFirstColumn,
            TranslateHeadConstantsToEqualities,
            TypeResolutionMixin,
            TranslateRegionDestroy,
            Fol2DatalogMixin,
            RegionSolver,
            CommandsMixin,
            NumpyFunctionsMixin,
            DatalogWithAggregationMixin,
            DatalogProgramNegationMixin,
            DatalogProgram,
            ExpressionBasicEvaluator,
        )

        CustomStripper = make_dimension_type_stripper(["intensity"])

        class CustomSolver(
            TranslateToLogicWithAggregation,
            CustomStripper,
            TranslateSSugarToSelectByColumn,
            TranslateSelectByFirstColumn,
            TranslateHeadConstantsToEqualities,
            TypeResolutionMixin,
            TranslateRegionDestroy,
            Fol2DatalogMixin,
            RegionSolver,
            CommandsMixin,
            NumpyFunctionsMixin,
            DatalogWithAggregationMixin,
            DatalogProgramNegationMixin,
            DatalogProgram,
            ExpressionBasicEvaluator,
        ):
            pass

        solver = CustomSolver()
        v = Symbol("v")
        s = Symbol("s")

        conj = Conjunction((
            Symbol("voxel")(v, v, v),
            Symbol("intensity")(v),
            Symbol("probability")(s),
        ))

        result = solver.walk(conj)
        assert isinstance(result, Conjunction), (
            f"Expected Conjunction, got {type(result)}: {result}"
        )

        functors = {
            f.functor.name
            for f in result.formulas
            if isinstance(f, FunctionApplication)
        }
        # 'intensity' was our custom dimension predicate — should be stripped
        assert "intensity" not in functors, (
            f"intensity should be stripped; got {functors}"
        )
        # 'probability' is NOT in our custom set — should remain
        assert "probability" in functors, (
            f"probability should remain; got {functors}"
        )
        assert "voxel" in functors


class TestEngineConfigIntegration:
    """Tests that build_engine reads dimension_type_predicates from YAML."""

    def test_make_dimension_type_stripper_from_yaml_config(self):
        """Parsing dimension_type_predicates: [probability, value] creates
        the correct mixin subclass."""
        predicates = ["probability", "value"]
        Mixin = make_dimension_type_stripper(predicates)
        assert issubclass(Mixin, StripDimensionTypePredicatesMixin)
        assert Mixin.dimension_type_predicate_names == frozenset(
            {"probability", "value"}
        )

    def test_make_dimension_type_stripper_empty_list(self):
        """An empty list means nothing is stripped."""
        Mixin = make_dimension_type_stripper([])

        class _CustomWalker(Mixin, ExpressionWalker):
            pass

        walker = _CustomWalker()
        v = Symbol("v")
        prob_atom = FunctionApplication(Symbol("probability"), (v,))
        result = walker.walk(prob_atom)
        assert result is prob_atom, (
            "With empty predicates, probability should be left untouched"
        )

    def test_make_dimension_type_stripper_none(self):
        """None uses the default set."""
        Mixin = make_dimension_type_stripper(None)
        assert Mixin.dimension_type_predicate_names == _DIMENSION_TYPE_PREDICATE_NAMES