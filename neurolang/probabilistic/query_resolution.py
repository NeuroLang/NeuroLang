import logging
import operator
import typing
from typing import AbstractSet

from ..datalog.aggregation import is_builtin_aggregation_functor
from ..datalog.expression_processing import (
    EQ,
    conjunct_formulas,
    extract_logic_free_variables
)
from ..datalog.instance import MapInstance, WrappedRelationalAlgebraFrozenSet
from ..expression_pattern_matching import add_match
from ..expression_walker import ChainedWalker, ExpressionWalker, PatternWalker
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..logic import TRUE, Conjunction, Implication, Union
from ..relational_algebra import (
    EliminateTrivialProjections,
    ExtendedProjection,
    FunctionApplicationListMember,
    Projection,
    RelationalAlgebraOperation,
    PushInSelections,
    RelationalAlgebraSolver,
    RenameOptimizations,
    SimplifyExtendedProjectionsWithConstants,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import NaturalJoinInverse
from ..relational_algebra.optimisers import PushUnnamedSelectionsUp
from .cplogic.program import CPLogicProgram
from .exceptions import RepeatedTuplesInProbabilisticRelationError
from .expression_processing import (
    construct_within_language_succ_result,
    is_query_based_probfact,
    is_within_language_prob_query,
    within_language_succ_query_to_intensional_rule
)
from .expressions import Condition, ProbabilisticPredicate
from .probabilistic_semiring_solver import (
    ProbSemiringToRelationalAlgebraSolver
)
from .antishattering import ProjectionSelectionByPChoiceConstant


def _qbased_probfact_needs_translation(formula: Implication) -> bool:
    if isinstance(formula.antecedent, FunctionApplication):
        antecedent_pred = formula.antecedent
    elif (
        isinstance(formula.antecedent, Conjunction)
        and len(formula.antecedent.formulas) == 1
    ):
        antecedent_pred = formula.antecedent.formulas[0]
    else:
        return True
    return not (
        isinstance(antecedent_pred.functor, Symbol)
        and antecedent_pred.functor.is_fresh
    )


LOG = logging.getLogger(__name__)


class QueryBasedProbFactToDetRule(PatternWalker):
    """
    Translate a query-based probabilistic fact to two rules. A deterministic
    rule that takes care of inferring the set of probabilistic tuples and their
    probabilities and a probabilistic rule that transforms the result from the
    deterministic resolution into a proper probabilistic table by re-attaching
    the deterministically-obtained probabilities.

    For example, the rule `P(x) : f(x) :- Q(x)` is translated into the rules

        _f1_(_f2_, x) :- Q(x), _f2_ = f(x)
        P(x) : _f2_ :- _f1_(_f2_, x)

    where `_f1_` and `_f2_` are fresh symbols. Note: we make sure not to
    expose the resulting `_f1_` relation to the user, as it is not part of
    her program.

    """

    @add_match(
        Union,
        lambda union: any(
            is_query_based_probfact(formula)
            and _qbased_probfact_needs_translation(formula)
            for formula in union.formulas
        ),
    )
    def union_with_query_based_pfact(self, union):
        new_formulas = list()
        for formula in union.formulas:
            if is_query_based_probfact(
                formula
            ) and _qbased_probfact_needs_translation(formula):
                (
                    det_rule,
                    prob_rule,
                ) = self._query_based_probabilistic_fact_to_det_and_prob_rules(
                    formula
                )
                new_formulas.append(det_rule)
                new_formulas.append(prob_rule)
            else:
                new_formulas.append(formula)
        return self.walk(Union(tuple(new_formulas)))

    @add_match(
        Implication,
        lambda implication: is_query_based_probfact(implication)
        and _qbased_probfact_needs_translation(implication),
    )
    def query_based_probafact(self, impl):
        (
            det_rule,
            prob_rule,
        ) = self._query_based_probabilistic_fact_to_det_and_prob_rules(impl)
        return self.walk(Union((det_rule, prob_rule)))

    @staticmethod
    def _query_based_probabilistic_fact_to_det_and_prob_rules(impl):
        prob_symb = Symbol.fresh()
        det_pred_symb = Symbol.fresh()
        agg_functor = QueryBasedProbFactToDetRule._get_agg_functor(impl)
        if agg_functor is None:
            eq_formula = EQ(prob_symb, impl.consequent.probability)
            det_consequent = det_pred_symb(
                prob_symb, *impl.consequent.body.args
            )
            prob_antecedent = det_consequent
        else:
            assert len(impl.consequent.probability.args) == 1
            eq_formula = EQ(prob_symb, impl.consequent.probability.args[0])
            det_consequent = det_pred_symb(
                agg_functor(prob_symb), *impl.consequent.body.args
            )
            prob_antecedent = det_pred_symb(
                prob_symb, *impl.consequent.body.args
            )
        det_antecedent = conjunct_formulas(impl.antecedent, eq_formula)
        det_rule = Implication(det_consequent, det_antecedent)
        prob_consequent = ProbabilisticPredicate(
            prob_symb, impl.consequent.body
        )
        prob_rule = Implication(prob_consequent, prob_antecedent)
        return det_rule, prob_rule

    @staticmethod
    def _get_agg_functor(
        impl: Implication,
    ) -> typing.Union[None, typing.Callable]:
        if not isinstance(
            impl.consequent.probability, FunctionApplication
        ) or not is_builtin_aggregation_functor(
            impl.consequent.probability.functor
        ):
            return
        return impl.consequent.probability.functor


def _solve_within_language_prob_query(
    cpl: CPLogicProgram,
    rule: Implication,
    succ_prob_solver: typing.Callable,
    marg_prob_solver: typing.Callable,
) -> Constant[AbstractSet]:
    query = within_language_succ_query_to_intensional_rule(rule)
    if isinstance(rule.antecedent, Condition):
        provset = marg_prob_solver(query, cpl)
    else:
        provset = succ_prob_solver(query, cpl)
    relation = construct_within_language_succ_result(provset, rule)
    return relation


def _solve_for_probabilistic_rule(
    cpl: CPLogicProgram,
    rule: Implication,
    succ_prob_solver: typing.Callable,
):
    provset = succ_prob_solver(rule, cpl)
    return provset.relation


def compute_probabilistic_solution(
    det_edb,
    pfact_db,
    pchoice_edb,
    prob_idb,
    succ_prob_solver,
    marg_prob_solver,
    check_qbased_pfact_tuple_unicity=False,
):
    solution = MapInstance()
    cpl, prob_idb = _build_probabilistic_program(
        det_edb,
        pfact_db,
        pchoice_edb,
        prob_idb,
        check_qbased_pfact_tuple_unicity,
    )
    for rule in prob_idb.formulas:
        if is_within_language_prob_query(rule):
            relation = _solve_within_language_prob_query(
                cpl, rule, succ_prob_solver, marg_prob_solver
            )
            solution[rule.consequent.functor] = Constant[AbstractSet](
                relation.value.to_unnamed()
            )
    return solution


def lift_solve_marg_query(rule, cpl, succ_solver):
    """
    Solve a MARG query on a CP-Logic program.

    Parameters
    ----------
    query : Implication
        Consequent must be of type `Condition`.
        MARG query of the form `ans(x) :- P(x)`.
    cpl_program : CPLogicProgram
        CP-Logic program on which the query should be solved.

    Returns
    -------
    ProvenanceAlgebraSet
        Provenance set labelled with probabilities for each tuple in the result
        set.

    """
    res_args = tuple(s for s in rule.consequent.args if isinstance(s, Symbol))

    joint_antecedent = Conjunction(
        (
            rule.antecedent.conditioned,
            rule.antecedent.conditioning
        )
    )
    joint_logic_variables = set(res_args)
    joint_rule = Implication(
        Symbol.fresh()(*joint_logic_variables), joint_antecedent
    )
    joint_provset = succ_solver(
        joint_rule, cpl, run_relational_algebra_solver=False
    )

    denominator_antecedent = rule.antecedent.conditioning
    denominator_logic_variables = (
        extract_logic_free_variables(denominator_antecedent) & res_args
    )
    denominator_rule = Implication(
        Symbol.fresh()(*denominator_logic_variables), denominator_antecedent
    )
    denominator_provset = succ_solver(
        denominator_rule, cpl, run_relational_algebra_solver=False
    )
    query_solver = generate_provenance_query_solver({}, True)
    provset = query_solver.walk(
        Projection(
            NaturalJoinInverse(joint_provset, denominator_provset),
            tuple(str2columnstr_constant(s.name) for s in res_args),
        )
    )
    return provset


def _discard_query_based_probfacts(prob_idb):
    return Union(
        tuple(
            formula
            for formula in prob_idb.formulas
            if not (
                isinstance(formula.consequent, ProbabilisticPredicate)
                and formula.antecedent != TRUE
            )
        )
    )


def _add_to_probabilistic_program(
    add_fun,
    pred_symb,
    expr,
    det_edb,
    check_qbased_pfact_tuple_unicity=False,
):
    # handle set-based probabilistic tables
    if isinstance(expr, Constant[typing.AbstractSet]):
        ra_set = expr.value
    # handle query-based probabilistic facts
    elif isinstance(expr, Union):
        impl = expr.formulas[0]
        # we know the rule is of the form
        # P(x_1, ..., x_n) : y :- Q(y, x_1, ..., x_n)
        # where Q is an extensional relation symbol
        # so the values can be retrieved from the EDB
        if impl.antecedent.functor in det_edb:
            ra_set = det_edb[impl.antecedent.functor].value
            if check_qbased_pfact_tuple_unicity:
                _check_tuple_prob_unicity(ra_set)
        else:
            ra_set = WrappedRelationalAlgebraFrozenSet.dum()
    add_fun(pred_symb, ra_set.unwrap())


def _check_tuple_prob_unicity(ra_set: Constant[AbstractSet]) -> None:
    length = len(ra_set)
    proj_cols = list(ra_set.columns)[1:]
    length_without_probs = len(ra_set.projection(*proj_cols))
    if length_without_probs != length:
        n_repeated_tuples = length - length_without_probs
        raise RepeatedTuplesInProbabilisticRelationError(
            n_repeated_tuples,
            length,
            "Some tuples have multiple probability labels. "
            f"Found {n_repeated_tuples} tuple repetitions, out of "
            f"{length} total tuples. If your query-based probabilistic fact "
            "leads to multiple probabilities for the same tuple, you might "
            "want to consider aggregating these probabilities by taking their "
            "maximum or average.",
        )


def _build_probabilistic_program(
    det_edb,
    pfact_db,
    pchoice_edb,
    prob_idb,
    check_qbased_pfact_tuple_unicity=False,
):
    cpl = CPLogicProgram()
    db_to_add_fun = [
        (det_edb, cpl.add_extensional_predicate_from_tuples),
        (pfact_db, cpl.add_probabilistic_facts_from_tuples),
        (pchoice_edb, cpl.add_probabilistic_choice_from_tuples),
    ]
    for database, add_fun in db_to_add_fun:
        for pred_symb, expr in database.items():
            _add_to_probabilistic_program(
                add_fun,
                pred_symb,
                expr,
                det_edb,
                check_qbased_pfact_tuple_unicity,
            )
    # remove query-based probabilistic facts that have already been processed
    # and transformed into probabilistic tables based on the deterministic
    # solution of their probability and antecedent
    prob_idb = _discard_query_based_probfacts(prob_idb)
    cpl.walk(prob_idb)
    return cpl, prob_idb


class FloatArithmeticSimplifier(PatternWalker):
    @add_match(
        FunctionApplication(
            Constant[typing.Any](operator.mul),
            (Constant[float](1.0), Expression[typing.Any]))
        )
    def simplify_mul_left(self, expression):
        return self.walk(expression.args[1])

    @add_match(
        FunctionApplication(
            Constant[typing.Any](operator.mul),
            (Expression[typing.Any], Constant[float](1.0))
        )
    )
    def simplify_mul_right(self, expression):
        return self.walk(expression.args[0])


class RAQueryOptimiser(
    EliminateTrivialProjections,
    PushInSelections,
    RenameOptimizations,
    SimplifyExtendedProjectionsWithConstants,
    FloatArithmeticSimplifier,
    PushUnnamedSelectionsUp,
    ExpressionWalker,
):
    pass


def generate_provenance_query_solver(
    symbol_table, run_relational_algebra_solver, constants_by_formula={},
    solver_class=ProbSemiringToRelationalAlgebraSolver
):
    """
    Generate a walker that solves a RAP query.

    Parameters
    ----------
    symbol_table : Mapping
        Mapping from symbols to probabilistic or deterministic sets to
        solve the query.
    run_relational_algebra_solver : bool
        if `true` the walker will return a ProvenanceAlgebraSet containing
        a NamedAlgebraSet as `relation` attribute. If `false` the walker will
        produce a relational algebra expression as `relation` attribute.
    solver_class: PatternWalker
        class to translate a provenance RA sets program into a RA program.
        Default is `ProbSemiringToRelationalAlgebraSolver`.
    """

    class LogExpression(PatternWalker):
        def __init__(self, logger, message, level):
            self.logger = logger
            self.message = message
            self.level = level

        @add_match(...)
        def log_exp(self, expression):
            LOG.log(self.level, self.message, expression)
            return expression

    steps = [
        RAQueryOptimiser(),
        solver_class(symbol_table=symbol_table),
        ProjectionSelectionByPChoiceConstant(constants_by_formula),
        LogExpression(LOG, "About to optimise RA query %s", logging.INFO),
        RAQueryOptimiser(),
        LogExpression(LOG, "Optimised RA query %s", logging.INFO)
    ]

    if run_relational_algebra_solver:
        steps.append(RelationalAlgebraSolver())

    query_compiler = ChainedWalker(*steps)
    return query_compiler


def reintroduce_unified_head_terms(
    ra_query: RelationalAlgebraOperation,
    flat_query: Implication,
    unified_query: Implication,
) -> RelationalAlgebraOperation:
    """
    Reintroduce terms that have been removed through unification of the query.

    There are two such cases:

    1. A head variable was repeated, such as in `ans(y, y)` and the extensional
    plan computes all possible values for `y` but will output only one column.
    This function makes sure that a second column for `y` is outputed, and that
    the resulting solution set will be a binary relation, as required.

    2. The query's head contains a constant, such as in `ans(2, y)`, in which
    case a constant column is added to the solution set.

    """
    proj_list = list()
    for old, new in zip(
        flat_query.consequent.args, unified_query.consequent.args
    ):
        dst_column = str2columnstr_constant(old.name)
        fun_exp = dst_column
        if new != old:
            if isinstance(new, Symbol):
                fun_exp = str2columnstr_constant(new.name)
            elif isinstance(new, Constant):
                fun_exp = new
            else:
                raise ValueError(
                    f"Unexpected argument {new}. "
                    "Expected symbol or constant"
                )
        member = FunctionApplicationListMember(fun_exp, dst_column)
        proj_list.append(member)
    return ExtendedProjection(ra_query, tuple(proj_list))
