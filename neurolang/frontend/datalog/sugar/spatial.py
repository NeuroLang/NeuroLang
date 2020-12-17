import operator
import typing

import numpy
import pandas
import scipy.spatial

from ....datalog.expression_processing import (
    UnifyVariableEqualitiesMixin,
    extract_logic_atoms,
)
from ....exceptions import UnsupportedProgramError
from ....expression_pattern_matching import add_match
from ....expression_walker import IdentityWalker, PatternWalker
from ....expressions import Constant, Expression, FunctionApplication, Symbol
from ....logic import Conjunction, Implication
from ....relational_algebra import RelationalAlgebraSet

EUCLIDEAN = Symbol("EUCLIDEAN")


class DetectEuclideanDistanceBoundMatrix(PatternWalker):
    """
    Detect a Euclidean spatial bound in the antecedent of a rule.

    This spatial bound is defined by 4 conjuncts in the antecedent of the rule:

        - a formula of the form `d = EUCLIDEAN(i1, j1, k1, i2, j2, k2)`, where
          EUCLIDEAN a special reserved symbol corresponding
        - a formula of the form `d < MAX_DIST`, where MAX_DIST is a constant
          value limiting the distance between 3D points (i1, j1, k1) and (i2,
          j2, k2)
        - a formula of the form `R1(x)` where x is a tuple containing `i1`,
          `j1` and `k1` providing a range for (i1, j1, k1) coordinates
        - a formula of the form `R2(x)` where x is a tuple containing `i2`,
          `j2` and `k2` providing a range for (i2, j2, k2) coordinates

    """

    def __init__(self):
        self._rule_normaliser = self._RuleNormaliser()

    class _RuleNormaliser(
        UnifyVariableEqualitiesMixin,
        IdentityWalker,
    ):
        pass

    @add_match(Implication(FunctionApplication, Conjunction))
    def implication(self, implication):
        implication = self._rule_normaliser.walk(implication)
        formulas = extract_logic_atoms(implication.antecedent)
        var_to_euclidean_equality_formula = self.get_var_to_euclidean_equality(
            formulas
        )
        distance_upper_bound_formula = self.get_distance_upper_bound(formulas)
        if (
            var_to_euclidean_equality_formula is None
            or distance_upper_bound_formula is None
        ):
            return False
        i1, j1, k1, i2, j2, k2 = var_to_euclidean_equality_formula.args[1].args
        if (
            self.get_range_pred_for_coord(formulas, (i1, j1, k1)) is None
            or self.get_range_pred_for_coord(formulas, (i2, j2, k2)) is None
        ):
            return False
        return True

    @add_match(Expression)
    def any_other_expression(self, expression):
        return False

    @staticmethod
    def get_range_pred_for_coord(formulas, coord_args):
        matching_formulas = list(
            formula
            for formula in formulas
            if isinstance(formula.functor, Symbol)
            and formula.functor != EUCLIDEAN
            and all(
                sum(arg == coord_arg for arg in formula.args) == 1
                for coord_arg in coord_args
            )
        )
        if len(matching_formulas) == 0:
            return None
        return matching_formulas[0]

    @staticmethod
    def get_var_to_euclidean_equality(formulas):
        matching_formulas = list(
            formula
            for formula in formulas
            if isinstance(formula.functor, Constant)
            and formula.functor.value == operator.eq
            and len(formula.args) == 2
            and isinstance(formula.args[0], Symbol)
            and isinstance(formula.args[1], FunctionApplication)
            and formula.args[1].functor == EUCLIDEAN
            and len(formula.args[1].args) == 6
        )
        if len(matching_formulas) != 1:
            return None
        return matching_formulas[0]

    @staticmethod
    def get_distance_upper_bound(formulas):
        matching_formulas = list(
            formula
            for formula in formulas
            if isinstance(formula.functor, Constant)
            and formula.functor.value == operator.lt
            and len(formula.args) == 2
            and isinstance(formula.args[0], Symbol)
            and isinstance(formula.args[1], Constant)
        )
        if len(matching_formulas) != 1:
            return None
        return matching_formulas[0]


class TranslateEuclideanDistanceBoundMatrixMixin(PatternWalker):
    @add_match(
        Implication(FunctionApplication, Conjunction),
        DetectEuclideanDistanceBoundMatrix().walk,
    )
    def euclidean_spatial_bound(self, implication):
        formulas = implication.antecedent.formulas
        var_to_euclidean_equality_formula = (
            DetectEuclideanDistanceBoundMatrix.get_var_to_euclidean_equality(
                formulas
            )
        )
        distance_upper_bound_formula = (
            DetectEuclideanDistanceBoundMatrix.get_distance_upper_bound(
                formulas
            )
        )
        d = var_to_euclidean_equality_formula.args[0]
        i1, j1, k1, i2, j2, k2 = var_to_euclidean_equality_formula.args[1].args
        upper_bound = distance_upper_bound_formula.args[1]
        first_range_pred = (
            DetectEuclideanDistanceBoundMatrix.get_range_pred_for_coord(
                formulas, (i1, j1, k1)
            )
        )
        second_range_pred = (
            DetectEuclideanDistanceBoundMatrix.get_range_pred_for_coord(
                formulas, (i2, j2, k2)
            )
        )
        first_coord_set = self.safe_range_pred_to_coord_set(
            first_range_pred, (i1, j1, k1)
        )
        second_coord_set = self.safe_range_pred_to_coord_set(
            second_range_pred, (i2, j2, k2)
        )
        max_dist = self.upper_bound_to_max_dist(upper_bound)
        spatial_bound_solution = self.solve_spatial_bound(
            first_coord_set.as_numpy_array(),
            second_coord_set.as_numpy_array(),
            max_dist,
        )
        new_pred_symb = Symbol.fresh()
        spatial_bound_solution_pred = new_pred_symb(i1, j1, k1, i2, j2, k2, d)
        self.add_extensional_predicate_from_tuples(
            new_pred_symb, spatial_bound_solution
        )
        removed_formulas = {
            var_to_euclidean_equality_formula,
            distance_upper_bound_formula,
        }
        new_formulas = (spatial_bound_solution_pred,) + tuple(
            formula for formula in formulas if formula not in removed_formulas
        )
        new_implication = Implication(
            implication.consequent, Conjunction(new_formulas)
        )
        return self.walk(new_implication)

    @staticmethod
    def upper_bound_to_max_dist(upper_bound):
        val = upper_bound.value
        if isinstance(val, (int, float)):
            return float(val)
        raise ValueError("Unsupported distance expression")

    @staticmethod
    def solve_spatial_bound(
        first_coord_array: numpy.array,
        second_coord_array: numpy.array,
        max_dist: float,
    ) -> numpy.array:
        first_ckd_tree = scipy.spatial.cKDTree(first_coord_array)
        second_ckd_tree = scipy.spatial.cKDTree(second_coord_array)
        dist_mat = first_ckd_tree.sparse_distance_matrix(
            second_ckd_tree,
            max_dist,
            output_type="ndarray",
            p=2,
        )
        dist_mat = pandas.DataFrame(dist_mat)
        solution_df = pandas.DataFrame(
            numpy.c_[
                first_coord_array[dist_mat.iloc[:, 0].values],
                second_coord_array[dist_mat.iloc[:, 1].values],
            ],
            dtype=int,
        )
        solution_df[len(solution_df.columns)] = numpy.atleast_2d(
            dist_mat.iloc[:, 2].values
        ).T
        return solution_df

    def safe_range_pred_to_coord_set(
        self,
        range_pred: FunctionApplication,
        coord_args: typing.Tuple[Symbol],
    ) -> RelationalAlgebraSet:
        pred_symb = range_pred.functor
        ra_set = self.symbol_table[pred_symb].value
        args_as_str = list(
            arg.name for arg in range_pred.args if isinstance(arg, Symbol)
        )
        proj_cols = (
            args_as_str.index(coord_arg.name) for coord_arg in coord_args
        )
        return ra_set.projection(*proj_cols)
