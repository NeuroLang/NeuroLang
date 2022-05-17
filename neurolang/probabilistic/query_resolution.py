import logging
import operator
import typing
from itertools import chain
from typing import AbstractSet, Tuple

from ..datalog.aggregation import is_builtin_aggregation_functor
from ..datalog.expression_processing import (
    EQ,
    conjunct_formulas,
    extract_logic_free_variables
)
from ..datalog.instance import MapInstance, WrappedRelationalAlgebraFrozenSet
from ..expression_pattern_matching import add_match
from ..expression_walker import (
    ChainedWalker,
    ExpressionWalker,
    PatternWalker,
    ReplaceExpressionWalker,
    expression_iterator
)
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..logic import TRUE, Conjunction, Disjunction, Implication, Union
from ..relational_algebra import (
    BinaryRelationalAlgebraOperation,
    ColumnStr,
    Difference,
    EliminateTrivialProjections,
    ExtendedProjection,
    FunctionApplicationListMember,
    NameColumns,
    Projection,
    PushInSelections,
    RelationalAlgebraOperation,
    RelationalAlgebraSolver,
    RenameOptimizations,
    Selection,
    SimplifyExtendedProjectionsWithConstants,
    UnaryRelationalAlgebraOperation,
    Union as RAUnion,
    str2columnstr_constant
)
from ..relational_algebra.optimisers import PushUnnamedSelectionsUp
from ..relational_algebra.relational_algebra import (
    FullOuterNaturalJoin,
    GroupByAggregation,
    LeftNaturalJoin,
    RenameColumn,
    RenameColumns,
    int2columnint_constant
)
from ..relational_algebra_provenance import (
    Complement,
    ComplementSolverMixin,
    DisjointProjection,
    IndependentProjection,
    NaturalJoinInverse,
    ProvenanceAlgebraSet
)
from ..utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
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
from .shattering import HeadVar

AND = Constant(operator.and_)
NE = Constant(operator.ne)

ZERO = Constant[float](0.)
GT = Constant(operator.gt)


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


def selection_columnstr_args(selection):
    return set(
        exp
        for _, exp in expression_iterator(selection.formula)
        if isinstance(exp, Constant[ColumnStr])
    )


def selection_headvar_args(selection):
    return set(
        exp
        for _, exp in expression_iterator(selection.formula)
        if isinstance(exp, Constant[HeadVar])
    )


def conditions_headvar_args(selection):
    return set(
        exp
        for _, exp in expression_iterator(selection.formula)
        if (
            isinstance(exp, FunctionApplication) and
            any(isinstance(arg, Constant[HeadVar]) for arg in exp.args)
        )
    )


def selection_headvar_dst(selection):
    return set(
        exp.args[0] if isinstance(exp.args[1], Constant[HeadVar])
        else exp.args[1]
        for _, exp in expression_iterator(selection.formula)
        if (
            isinstance(exp, FunctionApplication) and
            any(isinstance(arg, Constant[HeadVar]) for arg in exp.args)
        )
    )


class PushSelectionsWithHeadVarUp(PatternWalker):
    @add_match(ProvenanceAlgebraSet(Selection, ...))
    def no_push_up_rap(self, expression):
        return ProvenanceAlgebraSet(
            self.walk(expression.relation),
            expression.provenance_column
        )

    @add_match(
        Selection(Selection, ...),
        lambda expression: (
            selection_headvar_args(expression.relation) and
            not selection_headvar_args(expression)
        )
    )
    def switch_selection_order(self, expression):
        return Selection(
            Selection(
                expression.relation.relation,
                expression.formula
            ),
            expression.relation.formula
        )

    @add_match(
        IndependentProjection(Selection, ...),
        lambda expression: ((
            selection_headvar_args(expression.relation) -
            expression.columns()
        ))

    )
    def push_up_independent_projection(self, expression):
        return self.push_up_projection(expression)

    @add_match(
        DisjointProjection(Selection, ...),
        lambda expression: ((
            selection_headvar_args(expression.relation) -
            expression.columns()
        ))

    )
    def push_up_disjoint_projection(self, expression):
        return self.push_up_projection(expression)

    @add_match(
        RAUnion(
            Selection(
                ...,
                FunctionApplication(
                    NE,
                    (Constant[ColumnStr], Constant[HeadVar]))
                ),
            ...
        )
    )
    def push_up_union_with_ne_left(self, expression):
        relation_left, relation_right = expression.unapply()
        return self.walk(Complement(
            FullOuterNaturalJoin(
                Complement(relation_left),
                Complement(relation_right)
            )
        ))

    @add_match(
        RAUnion(
            ...,
            Selection(
                ...,
                FunctionApplication(
                    NE,
                    (Constant[ColumnStr], Constant[HeadVar]))
                ),
        )
    )
    def push_up_union_with_ne_right(self, expression):
        relation, formula = expression.relation_right.unapply()
        new_formula = FunctionApplication(EQ, formula.args)
        new_relation = Complement(relation)

        res = Selection(
            RAUnion(
                expression.relation_left,
                new_relation
            ),
            new_formula
        )

        return self.walk(res)

    @add_match(
        Difference(..., Selection),
        lambda expression: ((
            selection_headvar_args(expression.relation_right) -
            expression.columns()
        ))
    )
    def push_up_complement(self, expression):
        complement_replace = {
            EQ: NE,
            NE: EQ
        }
        rew = ReplaceExpressionWalker(complement_replace)

        return_formulae = []
        relation = expression.relation_right
        while (
            isinstance(relation, Selection) and
            selection_headvar_args(relation)
        ):
            return_formulae.append(relation.formula)
            relation = relation.relation

        return_diff = Difference(
            expression.relation_left,
            relation
        )
        for formula in return_formulae:
            return_diff = Selection(return_diff, formula)

        returns = []
        formulas_out = []
        left_columns = expression.relation_left.columns()
        relation_in = False
        for f in return_formulae:
            f = rew.walk(f)
            if f.args[0] in left_columns:
                returns.append(
                    Selection(expression.relation_left, rew.walk(f))
                )
            else:
                formulas_out.append(f)
                if f.functor == NE and not relation_in:
                    returns.append(expression.relation_left)
                    relation_in = True

        res = return_diff
        for ret in returns:
            res = RAUnion(ret, res)

        for f in formulas_out:
            res = Selection(res, f)

        return res

    @add_match(
        GroupByAggregation(Selection, ..., ...),
        lambda expression: ((
            selection_headvar_args(expression.relation) -
            expression.columns()
        ))
    )
    def push_up_groupby(self, expression):
        relation, groupby, aggs = expression.unapply()
        missing = set(
            cond.args[0]
            for cond in conditions_headvar_args(relation)
        ) - expression.columns()
        groupby += tuple(missing)

        return Selection(
            self.walk(expression.apply(relation.relation, groupby, aggs)),
            relation.formula
        )

    @add_match(
        ExtendedProjection(Selection, ...),
        lambda expression: ((
            selection_headvar_args(expression.relation) -
            expression.columns()
        ))
    )
    def push_up_extended_projection(self, expression):
        relation, args = expression.unapply()

        head_cond_columns = set(
            cond.args[0]
            for cond in conditions_headvar_args(relation)
        )
        missing = head_cond_columns - expression.columns()
        for var in missing:
            args += (FunctionApplicationListMember(var, var),)

        return Selection(
            self.walk(expression.apply(relation.relation, args)),
            relation.formula
        )

    @add_match(
        Projection(Selection, ...),
        lambda expression: ((
            selection_headvar_args(expression.relation) -
            expression.columns()
        ))
    )
    def push_up_projection(self, expression):
        relation, args = expression.unapply()
        head_cond_columns = set(
            cond.args[0]
            for cond in conditions_headvar_args(relation)
        )
        missing = head_cond_columns - expression.columns()
        args += tuple(missing)

        return Selection(
            self.walk(expression.apply(relation.relation, args)),
            relation.formula
        )

    @add_match(
        NameColumns(Selection, ...),
        lambda expression: (
            (
                selection_headvar_args(expression.relation) -
                expression.columns()
            )
            and
            len(expression.columns()) == len(expression.relation.columns())
        )
    )
    def push_up_name_columns_all_in(self, expression):
        relation, args = expression.unapply()
        rew = ReplaceExpressionWalker({
            int2columnint_constant(i): arg
            for i, arg in enumerate(args)
        })
        new_formula = rew.walk(relation.formula)
        return Selection(
            self.walk(NameColumns(relation.relation, args)),
            new_formula
        )

    @add_match(
        Selection(Selection, ...),
        lambda expression: (
            selection_headvar_args(expression.relation) and
            not selection_headvar_args(expression)
        )
    )
    def push_up_nested_selection(self, expression):
        exp_args = expression.unapply()
        relation = exp_args[0]

        return Selection(
            self.walk(expression.apply(relation.relation, *exp_args[1:])),
            relation.formula
        )

    @add_match(
        RenameColumns(Selection, ...),
        lambda expression: (
            selection_headvar_args(expression.relation) and
            (
                selection_headvar_dst(expression.relation) &
                set(src for src, _ in expression.renames)
            )
        )
    )
    def push_rename_columns_up(self, expression):
        replacements = {src: dst for src, dst in expression.renames}
        new_formula = ReplaceExpressionWalker(replacements).walk(
            expression.relation.formula
        )

        return Selection(
            RenameColumns(
                self.walk(expression.relation.relation),
                expression.renames,
            ),
            new_formula
        )

    @add_match(
        RenameColumn(Selection, ..., ...),
        lambda expression: (
            selection_headvar_args(expression.relation) and
            expression.src in selection_headvar_dst(expression.relation)
        )
    )
    def push_rename_column_up(self, expression):
        replacements = {expression.src: expression.dst}
        new_formula = ReplaceExpressionWalker(replacements).walk(
            expression.relation.formula
        )

        return Selection(
            RenameColumn(
                self.walk(expression.relation.relation),
                expression.src,
                expression.dst
            ),
            new_formula
        )

    @add_match(
        UnaryRelationalAlgebraOperation,
        lambda expression: (
            not isinstance(expression, Selection) and
            isinstance(expression.relation, Selection) and
            (
                selection_headvar_args(expression.relation) -
                expression.columns()
            )
        )
    )
    def push_unary_up(self, expression):
        exp_args = expression.unapply()
        relation = exp_args[0]

        return Selection(
            self.walk(expression.apply(relation.relation, *exp_args[1:])),
            relation.formula
        )

    @add_match(
        BinaryRelationalAlgebraOperation(
            Selection(..., FunctionApplication(EQ, ...)),
            Selection(..., FunctionApplication(EQ, ...))
        ),
        lambda expression: (
            (
                selection_headvar_args(expression.relation_left) -
                expression.columns()
            ) and (
                selection_headvar_args(expression.relation_right) -
                expression.columns()
            )
        )
    )
    def push_binary_both_up(self, expression):
        relation_left, relation_right = expression.unapply()

        left_headvar_args = selection_headvar_args(relation_left)
        right_headvar_args = selection_headvar_args(relation_right)
        both_head_var_args = left_headvar_args & right_headvar_args
        left_conditions = {var: set() for var in both_head_var_args}
        right_renames = []
        for condition in conditions_headvar_args(relation_left):
            if condition.args[1] in left_conditions:
                left_conditions[condition.args[1]].add(condition)
        for condition in conditions_headvar_args(relation_right):
            src, var = condition.args
            if var in left_conditions:
                new_src = next(iter(left_conditions[var])).args[0]
                right_renames.append((src, new_src))

        if right_renames:
            relation_right = self.walk(
                RenameColumns(relation_right, tuple(right_renames))
            )
        binary_op = self._binary_op_fix(
            expression.apply(
                relation_left.relation,
                relation_right
            )
        )

        return Selection(
            binary_op,
            relation_left.formula
        )

    @add_match(
        BinaryRelationalAlgebraOperation(Selection, ...),
        lambda expression: (
            selection_headvar_args(expression.relation_left) -
            expression.columns()
        )
    )
    def push_binary_left_up(self, expression):
        relation_left, relation_right = expression.unapply()

        binary_op = self._binary_op_fix(
            expression.apply(
                relation_left.relation, relation_right
            )
        )

        return Selection(
            binary_op,
            relation_left.formula
        )

    @add_match(
        BinaryRelationalAlgebraOperation,
        lambda expression: (
            isinstance(expression.relation_right, Selection) and
            (
                selection_headvar_args(expression.relation_right) -
                expression.columns()
            )
        )
    )
    def push_binary_right_up(self, expression):
        relation_left, relation_right = expression.unapply()

        binary_op = self._binary_op_fix(
            expression.apply(
                relation_left, relation_right.relation
            )
        )

        return Selection(
            binary_op,
            relation_right.formula
        )

    def _binary_op_fix(self, binary_op):
        left = binary_op.relation_left
        right = binary_op.relation_right

        if isinstance(binary_op, RAUnion):
            return binary_op
            if (left.columns() != right.columns()):
                return RAUnion(
                    LeftNaturalJoin(left, right),
                    LeftNaturalJoin(right, left)
                )

        return binary_op


class SplitSelectionConjunctions(PatternWalker):
    @add_match(Selection(..., FunctionApplication(AND, ...)))
    def split_selection_conjunction(self, expression):
        new_expression = expression.relation
        for arg in expression.formula.args:
            new_expression = Selection(new_expression, arg)
        return self.walk(new_expression)


class SplitSelectionConjunctionsWalker(
    SplitSelectionConjunctions,
    ExpressionWalker
):
    pass


def _repeated_selections(expression):
    new_expression = expression
    formulas = set()
    number_of_selections = 0
    while isinstance(new_expression, Selection):
        formulas.add(new_expression.formula)
        number_of_selections += 1
        new_expression = new_expression.relation

    return number_of_selections != len(formulas)


class MoveSelectNonEqualsUp(PatternWalker):
    @add_match(
        Selection(Selection, ...),
        _repeated_selections
    )
    def nested_selection(self, expression):
        new_expression = expression
        formulas = set()
        while isinstance(new_expression, Selection):
            formulas.add(new_expression.formula)
            new_expression = new_expression.relation

        for formula in formulas:
            new_expression = Selection(new_expression, formula)

        return new_expression

    @add_match(
        Selection(
            Selection(
                ...,
                FunctionApplication(NE, ...)
            ),
            FunctionApplication(EQ, ...)
        )
    )
    def move_ne_up(self, expression):
        return Selection(
            self.walk(Selection(
                expression.relation.relation,
                expression.formula
            )),
            expression.relation.formula
        )


class SelectionsToRenames(ExpressionWalker):
    @add_match(
        Selection(
            ...,
            FunctionApplication(EQ, (Constant[ColumnStr], Constant[HeadVar]))
        ),
        lambda expression: (
            expression.formula.args[1] not in expression.relation.columns()
        )
    )
    def selection_to_rename(self, expression):
        relation = self.walk(expression.relation)
        if expression.formula.args[1] not in relation.columns():
            res = self.walk(RenameColumn(
                relation,
                expression.formula.args[0],
                expression.formula.args[1]
            ))
        else:
            res = Selection(
                relation,
                expression.formula
            )
        return res

    @add_match(
        Projection(
            Selection(
                ...,
                FunctionApplication(
                    ...,
                    (Constant[ColumnStr], Constant[HeadVar])
                )
            ),
            ...
        ),
        lambda expression: (
            expression.relation.formula.functor in (EQ, NE) and
            (
                expression.relation.formula.args[1] in
                expression.relation.relation.columns()
            ) and
            expression.relation.formula.args[0] not in expression.columns()
        )
    )
    def dont_project_out(self, expression):
        return expression

    @add_match(
        Selection(
            ...,
            FunctionApplication(..., (Constant[ColumnStr], Constant[HeadVar]))
        ),
        lambda expression: (
            expression.formula.functor in (EQ, NE) and
            expression.formula.args[1] in expression.relation.columns()
        )
    )
    def selection_project_out(self, expression):
        new_columns = expression.columns() - {expression.formula.args[0]}
        new_relation = Projection(expression, tuple(new_columns))
        return new_relation


class PushSelectionsWithHeadVarUpWalker(
    PushSelectionsWithHeadVarUp,
    ExpressionWalker
):
    pass


class PostProcessPushUps(
    MoveSelectNonEqualsUp,
    ExpressionWalker
):
    pass


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


class AddNeededProjections(PatternWalker):
    def __init__(self, needed_projections):
        self.needed_projections = needed_projections

    @add_match(ProvenanceAlgebraSet)
    def add_projection(self, expression):
        if self.needed_projections:
            projections = (
                self.needed_projections +
                (FunctionApplicationListMember(
                    expression.provenance_column,
                    expression.provenance_column
                ),)
            )
            expression = ProvenanceAlgebraSet(
                ExtendedProjection(
                    expression.relation,
                    projections
                ),
                expression.provenance_column
            )
        return expression


class PushUnnamedSelectionsUpWalker(
    PushUnnamedSelectionsUp,
    ExpressionWalker
):
    pass


class ComplementSolverWalker(
    ComplementSolverMixin,
    ExpressionWalker
):
    pass
class FilterZeroProbability(ExpressionWalker):
    @add_match(ProvenanceAlgebraSet)
    def add_zero_filter(self, expression):
        return ProvenanceAlgebraSet(
            Selection(
                expression.relation,
                GT(expression.provenance_column, ZERO)
            ),
            expression.provenance_column
        )


def generate_provenance_query_solver(
    symbol_table, run_relational_algebra_solver,
    needed_projections=None,
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

    if needed_projections is None:
        needed_projections = tuple()

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
        # RAQueryOptimiser(),
        ComplementSolverWalker(),
        SplitSelectionConjunctionsWalker(),
        PushUnnamedSelectionsUpWalker(),
        PushSelectionsWithHeadVarUpWalker(),
        solver_class(symbol_table=symbol_table),
        PostProcessPushUps(),
        SelectionsToRenames(),
        LogExpression(LOG, "About to optimise RA query %s", logging.INFO),
        # RAQueryOptimiser(),
        AddNeededProjections(needed_projections),
        FilterZeroProbability(),
        RAQueryOptimiser(),
        LogExpression(LOG, "Optimised RA query %s", logging.INFO)
    ]

    if run_relational_algebra_solver:
        steps.append(RelationalAlgebraSolver())

    query_compiler = ChainedWalker(*steps)
    return query_compiler


def compute_projections_needed_to_reintroduce_head_terms(
    ra_query: RelationalAlgebraOperation,
    flat_query: Implication,
    unified_query: Implication,
) -> Tuple[FunctionApplicationListMember, ...]:
    """
    Produce the extended projection list for the terms that have been
    removed through unification of the query.

    There are two such cases:

    1. A head variable was repeated, such as in `ans(y, y)` and the extensional
    plan computes all possible values for `y` but will output only one column.
    This function makes sure that a second column for `y` is outputed, and that
    the resulting solution set will be a binary relation, as required.

    2. The query's head contains a constant, such as in `ans(2, y)`, in which
    case a constant column is added to the solution set.

    """
    proj_list = list()
    needed = False
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
            needed = True
        member = FunctionApplicationListMember(fun_exp, dst_column)
        proj_list.append(member)
    if not needed:
        proj_list = []
    return tuple(proj_list)
