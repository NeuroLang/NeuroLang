import logging
import numbers
import operator
import typing
import warnings
from typing import AbstractSet

import pandas

from nl_core.datalog.expression_processing import UnifyVariableEqualitiesMixin
from nl_expr.exceptions import UnsupportedProgramError
from nl_expr.expression_pattern_matching import add_match
from . import delegate_to_next_match
from nl_expr.expression_walker import (
    IdentityWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    ReplaceSymbolWalker,
)
from nl_expr.expressions import (
    Constant,
    Expression,
    FunctionApplication,
    Symbol,
)
from nl_expr.logic import Conjunction, Implication
from nl_expr.utils.various import log_performance
from ...frontend_extensions import NumpyFunctionsMixin
from ....utils.euclidean_join import euclidean_distance, euclidean_join
from nl_expr.type_system import is_leq_informative
from ....regions import ExplicitVBR

#: Reserved symbol of the Euclidean distance spatial-bound sugar.
EUCLIDEAN = Symbol("EUCLIDEAN")
#: Name of the built-in Euclidean distance function.
EUCLIDEAN_FUNCTION = Symbol("euclidean")
#: Built-in Euclidean distance, registered by the frontends.
EUCLIDEAN_BUILTIN = NumpyFunctionsMixin.constant_euclidean

_EUCLIDEAN_NAMES = (EUCLIDEAN.name, EUCLIDEAN_FUNCTION.name)
_COMPARISONS = (operator.lt, operator.le, operator.gt, operator.ge)
_MIRRORED = {operator.gt: operator.lt, operator.ge: operator.le}

UNBOUNDED_EUCLIDEAN_MESSAGE = (
    "A maximum distance must be provided when defining a spatial prior "
    "using the Euclidean distance between two sets of points, in one of the "
    "forms `EUCLIDEAN(i1, j1, k1, i2, j2, k2) < r` (or `<=`, `r > ...`, "
    "`r >= ...`) or `d == EUCLIDEAN(i1, j1, k1, i2, j2, k2), d < r`, where r "
    "is a number and the coordinates are distinct variables, those of the "
    "first point followed by those of the second one; the built-in "
    "`euclidean` function accepts the same forms"
)

LOG = logging.getLogger(__name__)


class EuclideanBound(typing.NamedTuple):
    """
    Euclidean distance bound found in the antecedent of a rule.

    Attributes
    ----------
    application : FunctionApplication
        Application of the distance function to the coordinates.
    radius : float
        Upper bound of the distance.
    strict : bool
        Whether the distance must be lower than the radius rather than
        lower or equal.
    distance : Symbol
        Variable equal to the distance.
    formulas : tuple
        The equality defining the distance and its bounding comparison.
    """

    application: FunctionApplication
    radius: float
    strict: bool
    distance: Symbol
    formulas: tuple

    @property
    def points(self):
        """
        Return the coordinate variables of the two points.

        Returns
        -------
        tuple of tuple of Symbol
            The coordinates of the first point and of the second one.
        """
        args = self.application.args
        return args[:len(args) // 2], args[len(args) // 2:]


def _is_euclidean_functor(expression):
    """
    Return whether *expression* is a Euclidean distance function symbol.

    Parameters
    ----------
    expression : Expression
        Candidate functor.

    Returns
    -------
    bool
        Whether it is ``EUCLIDEAN`` or ``euclidean``.
    """
    return (
        isinstance(expression, Symbol)
        and expression.name in _EUCLIDEAN_NAMES
    )


def euclidean_application(expression):
    """
    Return *expression* if it applies a Euclidean distance to variables.

    Parameters
    ----------
    expression : Expression
        Candidate application.

    Returns
    -------
    FunctionApplication or None
        *expression* when it applies ``EUCLIDEAN`` or ``euclidean`` to an
        even, non-zero number of distinct variables, None otherwise.
    """
    if not (
        isinstance(expression, FunctionApplication)
        and _is_euclidean_functor(expression.functor)
    ):
        return None
    args = expression.args
    if (
        len(args) == 0 or len(args) % 2
        or not all(isinstance(arg, Symbol) for arg in args)
        or len(set(args)) != len(args)
    ):
        return None
    return expression


def constant_point_application(expression):
    """
    Return *expression* if it applies a Euclidean distance to a point of
    numbers and a point of distinct variables.

    Parameters
    ----------
    expression : Expression
        Candidate application, such as ``euclidean(x, y, z, -24, -4, -20)``.

    Returns
    -------
    FunctionApplication or None
        *expression* when one of its points is numbers and the other one
        distinct variables, in either order, None otherwise.
    """
    if not (
        isinstance(expression, FunctionApplication)
        and _is_euclidean_functor(expression.functor)
        and expression.args and len(expression.args) % 2 == 0
    ):
        return None
    half = len(expression.args) // 2
    points = expression.args[:half], expression.args[half:]
    for constants, variables in (points, points[::-1]):
        if (
            all(_is_number(arg) for arg in constants)
            and all(isinstance(arg, Symbol) for arg in variables)
            and len(set(variables)) == half
        ):
            return expression
    return None


def _with_variable_point(application):
    """
    Return *application* of :func:`constant_point_application` with its
    numbers replaced by fresh variables.
    """
    return application.functor(*(
        Symbol.fresh() if _is_number(arg) else arg
        for arg in application.args
    ))


def _constant_point_applications(implication):
    """
    Return the applications of :func:`constant_point_application` that
    the conjuncts of the antecedent of *implication* compare, when the
    rule bounds one of their distances, else ``[]``.

    The rule bounds the distance when, with the numbers of the points
    replaced by variables, :class:`DetectEuclideanDistanceBoundMatrix`
    finds the bound.

    Parameters
    ----------
    implication : Implication
        Rule whose antecedent is a conjunction.

    Returns
    -------
    list of FunctionApplication
        The applications, in the order of the conjuncts.
    """
    applications = [
        arg for formula in implication.antecedent.formulas
        if _comparison(formula, _COMPARISONS + (operator.eq,)) is not None
        for arg in formula.args
        if constant_point_application(arg) is not None
    ]
    if not applications:
        return []
    candidate = ReplaceExpressionWalker({
        application: _with_variable_point(application)
        for application in applications
    }).walk(implication)
    try:
        bounded = DetectEuclideanDistanceBoundMatrix().walk(candidate)
    except UnsupportedProgramError:
        bounded = False
    return applications if bounded else []


def _comparison(formula, operators):
    """
    Return the operator of *formula* if it compares two terms by one of them.

    Parameters
    ----------
    formula : Expression
        Candidate comparison.
    operators : tuple of callable
        Accepted comparison operators.

    Returns
    -------
    callable or None
        The comparison operator, or None.
    """
    if not (
        isinstance(formula, FunctionApplication)
        and isinstance(formula.functor, Constant)
        and len(formula.args) == 2
    ):
        return None
    operator_ = formula.functor.value
    return operator_ if operator_ in operators else None


def _is_number(expression):
    """
    Return whether *expression* is a real number constant.

    Parameters
    ----------
    expression : Expression
        Candidate constant.

    Returns
    -------
    bool
        Whether it is a non-boolean real number constant.
    """
    return (
        isinstance(expression, Constant)
        and isinstance(expression.value, numbers.Real)
        and not isinstance(expression.value, bool)
    )


def upper_bound(formula, target):
    """
    Return the upper bound that *formula* sets on *target*, if any.

    ``target < r`` and ``r > target`` give ``(r, True)``; ``target <= r``
    and ``r >= target`` give ``(r, False)``.

    Parameters
    ----------
    formula : Expression
        Candidate comparison.
    target : Symbol
        Bounded variable.

    Returns
    -------
    tuple or None
        The bound and whether it is strict, or None.
    """
    operator_ = _comparison(formula, _COMPARISONS)
    if operator_ is None:
        return None
    left, right = formula.args
    if operator_ in _MIRRORED:
        operator_, left, right = _MIRRORED[operator_], right, left
    if not (isinstance(left, Symbol) and left == target and _is_number(right)):
        return None
    return float(right.value), operator_ is operator.lt


def _distance_definition(formula):
    """
    Return the variable and the distance of ``d == euclidean(...)``.

    Parameters
    ----------
    formula : Expression
        Candidate equality, in either orientation.

    Returns
    -------
    tuple
        The variable and the application, or ``(None, None)``.
    """
    if _comparison(formula, (operator.eq,)) is None:
        return None, None
    left, right = formula.args
    if euclidean_application(left) is not None:
        left, right = right, left
    if isinstance(left, Symbol) and euclidean_application(right) is not None:
        return left, right
    return None, None


def find_euclidean_bound(formulas):
    """
    Return the first Euclidean distance bound among *formulas*.

    Only the ``d == euclidean(...)``, ``d < r`` form is recognised; see
    :func:`distance_variable_form` for the other forms.

    Parameters
    ----------
    formulas : tuple of Expression
        Conjuncts of a rule antecedent.

    Returns
    -------
    EuclideanBound or None
        The bound, or None.
    """
    for formula in formulas:
        distance, application = _distance_definition(formula)
        if distance is None:
            continue
        for other in formulas:
            bound = upper_bound(other, distance)
            if bound is not None:
                return EuclideanBound(
                    application, *bound, distance, (formula, other)
                )
    return None


def _as_distance_variable(formula):
    """
    Return *formula* with a compared distance replaced by a variable.

    ``euclidean(...) < r`` becomes ``d == euclidean(...)``, ``d < r`` with
    a fresh ``d``, and likewise for the other comparisons with a number.

    Parameters
    ----------
    formula : Expression
        Conjunct of a rule antecedent.

    Returns
    -------
    tuple of Expression
        The conjuncts replacing *formula*.
    """
    if _comparison(formula, _COMPARISONS) is None:
        return (formula,)
    args = formula.args
    for position, arg in enumerate(args):
        other = args[1 - position]
        if euclidean_application(arg) is not None and _is_number(other):
            distance = Symbol.fresh()
            new_args = (other, distance) if position else (distance, other)
            return (
                Constant(operator.eq)(distance, arg),
                formula.functor(*new_args),
            )
    return (formula,)


def distance_variable_form(implication):
    """
    Return *implication* with its compared distances named by variables.

    This makes ``euclidean(...) < r`` and ``d == euclidean(...), d < r``
    the same rule, so that both forms are processed alike.

    Parameters
    ----------
    implication : Implication
        Rule whose antecedent is a conjunction.

    Returns
    -------
    Implication
        The rule, unchanged when it compares no distance with a number.
    """
    formulas = tuple(
        new_formula
        for formula in implication.antecedent.formulas
        for new_formula in _as_distance_variable(formula)
    )
    if len(formulas) == len(implication.antecedent.formulas):
        return implication
    return Implication(implication.consequent, Conjunction(formulas))


def _mentions_euclidean(formula, names=_EUCLIDEAN_NAMES):
    """
    Return whether *formula* applies a Euclidean distance at its top level.

    Parameters
    ----------
    formula : Expression
        Conjunct of a rule antecedent.
    names : tuple of str, optional
        Names of the distance functions looked for.

    Returns
    -------
    bool
        Whether an argument of *formula* applies one of the functions.
    """
    return isinstance(formula, FunctionApplication) and any(
        isinstance(arg, FunctionApplication)
        and isinstance(arg.functor, Symbol)
        and arg.functor.name in names
        for arg in formula.args
    )


class DetectEuclideanDistanceBoundMatrix(PatternWalker):
    """
    Detect a Euclidean spatial bound in the antecedent of a rule.

    The bound is given by a comparison between a number ``r`` and the
    Euclidean distance between two points, either directly, as in
    ``euclidean(i1, j1, k1, i2, j2, k2) < r``, or through a variable, as in
    ``d == euclidean(i1, j1, k1, i2, j2, k2), d < r``. Both forms are the
    same rule (see :func:`distance_variable_form`). ``<``, ``<=`` and the
    mirrored ``r > ...`` and ``r >= ...`` are recognised, strictly as
    written. The distance function is either the reserved ``EUCLIDEAN``
    symbol or the built-in ``euclidean`` function, applied to the distinct
    variable coordinates of the two points, those of the first one followed
    by those of the second one (3 each in a 3-dimensional space).

    The coordinates of each point are expected to be ranged by a relation
    of the antecedent, ``R1(..., i1, j1, k1, ...)`` and
    ``R2(..., i2, j2, k2, ...)``; this is checked when translating the rule
    (see :class:`TranslateEuclideanDistanceBoundMatrixMixin`).

    Walking a rule returns whether it has such a bound. A rule using the
    reserved ``EUCLIDEAN`` symbol without a bound raises an
    :class:`~neurolang.exceptions.UnsupportedProgramError`.
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
        if not any(
            _mentions_euclidean(formula)
            for formula in implication.antecedent.formulas
        ):
            return False
        return self.detect(self.normalise(implication)) is not None

    @add_match(Expression)
    def any_other_expression(self, expression):
        return False

    def normalise(self, implication):
        """
        Return *implication* normalised for the detection of the bound.

        Parameters
        ----------
        implication : Implication
            Rule whose antecedent is a conjunction.

        Returns
        -------
        Implication
            The rule with its variable equalities unified and its compared
            distances named by variables.
        """
        normalised = self._rule_normaliser.walk(implication)
        if not isinstance(normalised.antecedent, Conjunction):
            normalised = Implication(
                normalised.consequent, Conjunction((normalised.antecedent,))
            )
        return distance_variable_form(normalised)

    @staticmethod
    def detect(implication):
        """
        Return the Euclidean distance bound of a normalised rule.

        Parameters
        ----------
        implication : Implication
            Rule normalised by :meth:`normalise`.

        Returns
        -------
        EuclideanBound or None
            The bound, or None.

        Raises
        ------
        UnsupportedProgramError
            If the rule uses the reserved ``EUCLIDEAN`` symbol without
            bounding the distance.
        """
        formulas = implication.antecedent.formulas
        bound = find_euclidean_bound(formulas)
        if bound is None and any(
            _mentions_euclidean(formula, (EUCLIDEAN.name,))
            for formula in formulas
        ):
            raise UnsupportedProgramError(UNBOUNDED_EUCLIDEAN_MESSAGE)
        return bound


def _ranges_point(formula, point):
    """
    Return whether the atom *formula* has each coordinate of *point* once.

    Parameters
    ----------
    formula : Expression
        Conjunct of a rule antecedent.
    point : tuple of Symbol
        Coordinate variables.

    Returns
    -------
    bool
        Whether *formula* is a relation atom ranging the coordinates.
    """
    if not (
        isinstance(formula, FunctionApplication)
        and isinstance(formula.functor, Symbol)
        and not _is_euclidean_functor(formula.functor)
    ):
        return False
    return all(
        sum(isinstance(arg, Symbol) and arg == coordinate
            for arg in formula.args) == 1
        for coordinate in point
    )


def _symbol_names(expressions):
    """
    Return the names of the symbols *expressions*, comma-separated.

    Parameters
    ----------
    expressions : iterable of Symbol
        Symbols to name.

    Returns
    -------
    str
        Their names.
    """
    return ", ".join(expression.name for expression in expressions)


class TranslateEuclideanDistanceBoundMatrixMixin(PatternWalker):
    """
    Precompute the pairs of points of a rule's Euclidean distance bound.

    A rule bounding the Euclidean distance between two points whose
    coordinates are ranged by stored relations (see
    :class:`DetectEuclideanDistanceBoundMatrix`), such as

    ``close(i, j, k, s) :- voxel(i, j, k), peak(i1, j1, k1, s),
    euclidean(i, j, k, i1, j1, k1) < 2``

    is rewritten to use a fresh relation holding the pairs of points within
    the distance, computed by
    :func:`~neurolang.utils.euclidean_join.euclidean_join`. A ranging atom
    whose arguments are distinct variables, used by no other ranging atom,
    is absorbed into the pairs, which are then projected on the variables
    the rest of the rule uses, the distance included; here both are,

    ``close(i, j, k, s) :- fresh(i, j, k, s)``

    Any other ranging atom stays in the rule, joined with the pairs on the
    point's coordinates.

    When a point is not ranged by a stored deterministic relation (it is
    derived, probabilistic or missing), the rule is kept and the distance
    computed as a function, with a warning. The reserved ``EUCLIDEAN``
    symbol is then replaced by the built-in ``euclidean`` function, unless
    the program defines it. A program defining its own ``euclidean``
    function keeps it: its rules are not rewritten.
    """

    @add_match(
        Implication(FunctionApplication, Conjunction),
        _constant_point_applications,
    )
    def euclidean_constant_point(self, implication):
        """
        Range the point of numbers of a Euclidean distance by a relation.

        ``near(s) :- peak(x, y, z, s), euclidean(x, y, z, -24, -4, -20)
        <= 3`` becomes ``near(s) :- peak(x, y, z, s), fresh(c1, c2, c3),
        euclidean(x, y, z, c1, c2, c3) <= 3``, ``fresh`` a stored
        relation of the one point ``(-24, -4, -20)``: the bound is then
        precomputed (see :meth:`euclidean_spatial_bound`), instead of the
        distance being computed for every row of ``peak``.
        """
        applications = _constant_point_applications(implication)
        if self._euclidean_is_user_defined(applications[0].functor):
            return delegate_to_next_match(
                self, implication, type(self).euclidean_constant_point
            )
        replacements, atoms = {}, []
        for application in applications:
            if application in replacements:
                continue
            replaced = _with_variable_point(application)
            point = [
                (fresh, arg.value)
                for fresh, arg in zip(replaced.args, application.args)
                if _is_number(arg)
            ]
            frame = pandas.DataFrame([[value for _, value in point]])
            atoms.append(self._add_pairs_relation(frame)(
                *(fresh for fresh, _ in point)
            ))
            replacements[application] = replaced
        antecedent = ReplaceExpressionWalker(replacements).walk(
            implication.antecedent
        )
        return self.walk(Implication(
            implication.consequent,
            Conjunction(tuple(atoms) + antecedent.formulas),
        ))

    @add_match(
        Implication(FunctionApplication, Conjunction),
        DetectEuclideanDistanceBoundMatrix().walk,
    )
    def euclidean_spatial_bound(self, implication):
        detector = DetectEuclideanDistanceBoundMatrix()
        rule = detector.normalise(implication)
        bound = detector.detect(rule)
        if self._euclidean_is_user_defined(bound.application.functor):
            return delegate_to_next_match(
                self, implication, type(self).euclidean_spatial_bound
            )
        formulas = rule.antecedent.formulas
        ranges = tuple(
            self._coordinate_range(formulas, point) for point in bound.points
        )
        if any(range_ is None for range_ in ranges):
            warnings.warn(self._fallback_message(rule, bound, ranges))
            return self._euclidean_bound_fallback(implication, bound)
        return self.walk(self._rewrite_with_pairs(rule, bound, ranges))

    def _euclidean_is_user_defined(self, functor):
        """
        Return whether the program defines its own ``euclidean`` function.

        Parameters
        ----------
        functor : Symbol
            Distance function symbol of the rule.

        Returns
        -------
        bool
            Whether *functor* is ``euclidean`` and the program defines it
            otherwise than as the built-in function.
        """
        if functor.name != EUCLIDEAN_FUNCTION.name:
            return False
        if functor not in self.symbol_table:
            return False
        value = getattr(self.symbol_table[functor], "value", None)
        return value is not euclidean_distance

    def _coordinate_range(self, formulas, point):
        """
        Return the stored relation atom ranging the coordinates of *point*.

        Parameters
        ----------
        formulas : tuple of Expression
            Conjuncts of the rule antecedent.
        point : tuple of Symbol
            Coordinate variables.

        Returns
        -------
        FunctionApplication or None
            The first such atom, or None.
        """
        for formula in formulas:
            if _ranges_point(formula, point) and self._is_stored_relation(
                formula
            ):
                return formula
        return None

    def _is_stored_relation(self, atom):
        """
        Return whether *atom* is over a stored deterministic relation.

        Parameters
        ----------
        atom : FunctionApplication
            Relation atom.

        Returns
        -------
        bool
            Whether its predicate is an extensional, non-probabilistic
            relation of the atom's arity.
        """
        symbol = atom.functor
        if symbol not in self.symbol_table:
            return False
        value = self.symbol_table[symbol]
        probabilistic = set(getattr(self, "pfact_pred_symbs", ())) | set(
            getattr(self, "pchoice_pred_symbs", ())
        )
        return (
            isinstance(value, Constant)
            and is_leq_informative(value.type, AbstractSet)
            and symbol not in probabilistic
            and getattr(value.value, "arity", None) == len(atom.args)
        )

    def _euclidean_bound_fallback(self, rule, bound):
        """
        Keep the rule as written, computing the distance as a function.

        The reserved ``EUCLIDEAN`` symbol, unless the program defines it,
        is replaced by the built-in ``euclidean`` function.

        Parameters
        ----------
        rule : Implication
            Rule as written.
        bound : EuclideanBound
            Its distance bound.

        Returns
        -------
        Expression
            The result of processing the rule.
        """
        functor = bound.application.functor
        if functor.name == EUCLIDEAN.name and functor not in self.symbol_table:
            rule = ReplaceSymbolWalker(
                {EUCLIDEAN.name: self._builtin_euclidean()}
            ).walk(rule)
        return delegate_to_next_match(
            self, rule, type(self).euclidean_spatial_bound
        )

    def _builtin_euclidean(self):
        """
        Return the expression of the built-in ``euclidean`` function.

        Returns
        -------
        Expression
            The ``euclidean`` symbol when it names the built-in function in
            the program, the function itself otherwise.
        """
        if EUCLIDEAN_FUNCTION in self.symbol_table and not (
            self._euclidean_is_user_defined(EUCLIDEAN_FUNCTION)
        ):
            return EUCLIDEAN_FUNCTION
        return EUCLIDEAN_BUILTIN

    @staticmethod
    def _fallback_message(rule, bound, ranges):
        """
        Return the warning given when the pairs are not precomputed.

        Parameters
        ----------
        rule : Implication
            Normalised rule.
        bound : EuclideanBound
            Its distance bound.
        ranges : tuple
            The atom ranging each point, None for those not ranged by a
            stored relation.

        Returns
        -------
        str
            The warning message.
        """
        reasons = []
        for point, range_ in zip(bound.points, ranges):
            if range_ is not None:
                continue
            atoms = [
                formula.functor for formula in rule.antecedent.formulas
                if _ranges_point(formula, point)
            ]
            ranged = "not ranged by a relation"
            if atoms:
                ranged = (
                    f"ranged by {_symbol_names(atoms)}, which is not a "
                    "stored deterministic relation"
                )
            reasons.append(f"({_symbol_names(point)}) is {ranged}")
        return (
            "The euclidean distance bound of the rule for "
            f"{getattr(rule.consequent.functor, 'name', '')} cannot be "
            f"precomputed: "
            f"{'; '.join(reasons)}. The distance is computed for every "
            "combination of the points instead, which can be slow on large "
            "relations."
        )

    def _rewrite_with_pairs(self, rule, bound, ranges):
        """
        Return *rule* joining a fresh relation of the pairs within bound.

        Parameters
        ----------
        rule : Implication
            Normalised rule.
        bound : EuclideanBound
            Its distance bound.
        ranges : tuple of FunctionApplication
            The stored relation atom ranging each point.

        Returns
        -------
        Implication
            The rewritten rule.
        """
        absorbed = _absorbed(bound, ranges)
        removed = {id(formula) for formula in bound.formulas} | {
            id(atom) for atom, absorb in zip(ranges, absorbed) if absorb
        }
        formulas = tuple(
            formula for formula in rule.antecedent.formulas
            if id(formula) not in removed
        )
        variables = _pair_variables(rule, bound, formulas, ranges, absorbed)
        tables = [
            self._range_table(atom, point, absorb)
            for atom, point, absorb in zip(ranges, bound.points, absorbed)
        ]
        pairs = euclidean_join(
            *tables, bound.radius, bound.strict,
            [variable.name for variable in variables], bound.distance.name,
        )
        return Implication(
            rule.consequent,
            Conjunction(
                (self._add_pairs_relation(pairs)(*variables),) + formulas
            ),
        )

    def _range_table(self, atom, point, absorbed):
        """
        Return the rows of the relation ranging a point, as a table.

        Parameters
        ----------
        atom : FunctionApplication
            Stored relation atom ranging the point.
        point : tuple of Symbol
            Coordinate variables.
        absorbed : bool
            Whether *atom* is absorbed into the pairs relation, its rows
            being then all needed rather than only the point coordinates.

        Returns
        -------
        tuple
            The rows, as a pandas.DataFrame whose columns are named by the
            variables, and the names of the coordinate columns.
        """
        relation = self.symbol_table[atom.functor].value
        variables = atom.args
        if not absorbed:
            columns = (atom.args.index(coordinate) for coordinate in point)
            relation, variables = relation.projection(*columns), point
        with log_performance(LOG, "reading %s", (atom.functor.name,)):
            frame = relation.as_pandas_dataframe()
        names = [variable.name for variable in variables]
        return frame.set_axis(names, axis=1), [arg.name for arg in point]

    def _add_pairs_relation(self, frame):
        """
        Store the rows of *frame* as a fresh relation without duplicates.

        Parameters
        ----------
        frame : pandas.DataFrame
            Rows of the relation, without duplicates.

        Returns
        -------
        Symbol
            The predicate of the relation.
        """
        symbol = Symbol.fresh()
        with log_performance(
            LOG, "storing %s euclidean pairs", (len(frame),)
        ):
            self.add_extensional_predicate_from_tuples(
                symbol, frame.set_axis(range(frame.shape[1]), axis=1)
            )
        relation = self.symbol_table[symbol].value
        if hasattr(relation, "might_have_duplicates"):
            relation.might_have_duplicates = False
        return symbol


def _absorbed(bound, ranges):
    """
    Return whether the atom ranging each point is absorbed into the pairs.

    The atom ranging a point is absorbed into the pairs relation when its
    arguments are distinct variables, of which those other than the
    point's coordinates appear in no other absorbed atom, in no coordinate
    of the other point and are not the distance.

    Parameters
    ----------
    bound : EuclideanBound
        Distance bound of a rule.
    ranges : tuple of FunctionApplication
        The stored relation atom ranging each point.

    Returns
    -------
    list of bool
        Whether each atom is absorbed.
    """
    excluded = set(bound.application.args) | {bound.distance}
    absorbed = []
    for atom, point in zip(ranges, bound.points):
        args = set(atom.args)
        absorbed.append(
            all(isinstance(arg, Symbol) for arg in atom.args)
            and len(args) == len(atom.args)
            and not (args - set(point)) & excluded
        )
        if absorbed[-1]:
            excluded |= args
    return absorbed


def _pair_variables(rule, bound, formulas, ranges, absorbed):
    """
    Return the variables of the pairs relation.

    Parameters
    ----------
    rule : Implication
        Normalised rule.
    bound : EuclideanBound
        Its distance bound.
    formulas : tuple of Expression
        The conjuncts left in the rule besides the pairs relation.
    ranges : tuple of FunctionApplication
        The stored relation atom ranging each point.
    absorbed : list of bool
        Whether each atom is absorbed into the pairs relation.

    Returns
    -------
    list of Symbol
        The variables of the points, of the absorbed atoms and the distance
        that the head or the other conjuncts use; the coordinates of the
        first point if they use none.
    """
    available = [
        variable
        for atom, point, absorb in zip(ranges, bound.points, absorbed)
        for variable in (atom.args if absorb else point)
    ] + [bound.distance]
    needed = set(rule.consequent._symbols).union(
        *(formula._symbols for formula in formulas)
    )
    selected = [variable for variable in available if variable in needed]
    return selected or list(bound.points[0])


def _is_contains_atom(atom):
    is_contains = (
        isinstance(atom.functor, Constant)
        and atom.functor.value is operator.contains
        or isinstance(atom.functor, Symbol)
        and atom.functor.name == "contains"
    )
    if not (is_contains and len(atom.args) >= 1 and isinstance(atom.args[0], Symbol)):
        return False
    return is_leq_informative(atom.args[0].type, ExplicitVBR)


def region_voxels(region: ExplicitVBR) -> typing.Tuple:
    """
    Return the voxels of *region* as a tuple of coordinate tuples.

    Values without voxels, such as the elements of a column of unknown
    type which are not regions, are returned unchanged.

    Parameters
    ----------
    region : ExplicitVBR
        Region whose voxels are returned.

    Returns
    -------
    tuple
        The voxel coordinates of the region, each a tuple of ints.
    """
    voxels = getattr(region, "voxels", None)
    if voxels is None:
        return region
    return tuple(map(tuple, voxels.tolist()))


#: Built-in computing the voxels of a region during evaluation.
REGION_VOXELS = Constant(region_voxels)


def _is_region_contains(formula):
    """
    Return whether *formula* is a ``contains`` atom over a region.

    Parameters
    ----------
    formula : Expression
        Conjunct of a rule antecedent.

    Returns
    -------
    bool
        Whether *formula* is ``contains(r, ...)`` with ``r`` a region.
    """
    return isinstance(formula, FunctionApplication) and _is_contains_atom(
        formula
    )


def _region_voxels_formulas(formula):
    """
    Return the conjuncts replacing a ``contains`` atom over a region.

    ``contains(r, x)`` becomes ``v == region_voxels(r), contains(v, x)``
    with a fresh ``v``, other formulas are kept.

    Parameters
    ----------
    formula : Expression
        Conjunct of a rule antecedent.

    Returns
    -------
    tuple of Expression
        The conjuncts replacing *formula*.
    """
    if not _is_region_contains(formula):
        return (formula,)
    region, *elements = formula.args
    voxels = Symbol[typing.Tuple].fresh()
    return (
        Constant(operator.eq)(voxels, REGION_VOXELS(region)),
        formula.functor(voxels, *elements),
    )


class TranslateRegionDestroy(PatternWalker):
    """
    Syntactic sugar for rules using contains/Destroy with
    ExplicitVBR region columns.

    ExplicitVBR values aren't iterable (no ``__iter__`` to avoid breaking
    Constant type inference), so the contains/Destroy pipeline can't
    explode them. This walker rewrites ``contains(r, x)``, with ``r`` a
    region, as ``v == region_voxels(r), contains(v, x)``: the voxels of
    each region are computed during evaluation and the relation storing
    the regions is left unchanged.
    """

    @add_match(
        Implication(FunctionApplication, Conjunction),
        lambda implication: any(
            _is_region_contains(formula)
            for formula in implication.antecedent.formulas
        ),
    )
    def region_destroy(self, implication):
        """
        Compute the voxels of the destructured regions during evaluation.
        """
        formulas = tuple(
            new_formula
            for formula in implication.antecedent.formulas
            for new_formula in _region_voxels_formulas(formula)
        )
        return self.walk(
            Implication(implication.consequent, Conjunction(formulas))
        )
