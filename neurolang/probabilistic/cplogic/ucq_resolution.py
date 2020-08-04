import collections
import operator
import typing

from ...datalog.expression_processing import enforce_conjunction, flatten_query
from ...exceptions import ForbiddenDisjunctionError, ForbiddenExpressionError
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction, Implication, Union
from ...relational_algebra import (
    ColumnStr,
    ConcatenateConstantColumn,
    ExtendedProjection,
    ExtendedProjectionListMember,
    NaturalJoin,
    Projection,
    RelationalAlgebraSolver,
    RelationalAlgebraStringExpression,
    RenameColumn,
    RenameColumns,
    Selection,
    ra_binary_to_nary,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
    RelationalAlgebraProvenanceExpressionSemringSolver,
)
from ..expression_processing import (
    get_probchoice_variable_equalities,
    group_preds_by_pred_symb,
)
from ..expressions import ProbabilisticChoiceGrounding, ProbabilisticPredicate
from .grounding import (
    get_grounding_pred_symb,
    get_grounding_predicate,
    ground_cplogic_program,
)

EQUAL = Constant(operator.eq)


def rename_columns_for_args_to_match(relation, src_args, dst_args):
    """
    Rename the columns of a relation so that they match the targeted args.

    Parameters
    ----------
    relation : ProvenanceAlgebraSet or RelationalAlgebraOperation
        The relation on which the renaming of the columns should happen.
    src_args : tuple of Symbols
        The predicate's arguments currently matching the columns.
    dst_args : tuple of Symbols
        New args that the naming of the columns should match.

    Returns
    -------
    RelationalAlgebraOperation
        The unsolved nested operations that apply the renaming scheme.

    """
    src_cols = list(str2columnstr_constant(arg.name) for arg in src_args)
    dst_cols = list(str2columnstr_constant(arg.name) for arg in dst_args)
    result = relation
    for dst_col in set(dst_cols):
        idxs = [i for i, c in enumerate(dst_cols) if c == dst_col]
        result = RenameColumn(result, src_cols[idxs[0]], dst_col)
        for idx in idxs[1:]:
            result = Selection(result, EQUAL(src_cols[idx], dst_col))
    return result


def grounding_to_provset(grounding, dst_pred):
    src_pred = get_grounding_predicate(grounding.expression)
    relation = grounding.relation
    if isinstance(grounding, ProbabilisticChoiceGrounding) or isinstance(
        grounding.expression.consequent, ProbabilisticPredicate
    ):
        # /!\ this assumes the first column contains the probability
        prov_col = ColumnStr(grounding.relation.value.columns[0])
    else:
        prov_col = ColumnStr(Symbol.fresh().name)
        # add a new probability column with name `prob_col` and ones everywhere
        cst_one_probability = Constant[float](
            1.0, auto_infer_type=False, verify_type=False
        )
        relation = ConcatenateConstantColumn(
            relation, str2columnstr_constant(prov_col), cst_one_probability
        )
    relation = rename_columns_for_args_to_match(
        relation, src_pred.args, dst_pred.args
    )
    solver = RelationalAlgebraSolver()
    relation = solver.walk(relation)
    provset = ProvenanceAlgebraSet(relation.value, prov_col)
    return provset


def add_fresh_symbol_column(relation, dst_col):
    lst_members = [
        ExtendedProjectionListMember(
            Constant[RelationalAlgebraStringExpression](
                RelationalAlgebraStringExpression(c.value), verify_type=False
            ),
            c,
        )
        for c in relation.non_provenance_columns
    ] + [
        ExtendedProjectionListMember(
            FunctionApplication(
                Constant[typing.Callable](lambda _: Symbol.fresh()), tuple(),
            ),
            dst_col,
        )
    ]
    op = ExtendedProjection(relation, lst_members)
    return RelationalAlgebraProvenanceCountingSolver().walk(op)


def build_label_provset_from_probset(probset, pred_symb):
    # add label column with fresh symbols for each tuple
    label_col = str2columnstr_constant(Symbol.fresh().name)
    probset = add_fresh_symbol_column(probset, label_col)
    without_prob = Projection(
        Constant[typing.AbstractSet](probset.value),
        probset.non_provenance_columns,
    )
    without_prob = RelationalAlgebraProvenanceCountingSolver().walk(
        without_prob
    )
    return ProvenanceAlgebraSet(without_prob, label_col)


def solve_probfact_self_joins(predicates, probset):
    pred_symb = next(iter(predicates)).functor
    provset, label_to_prob = build_label_provset_from_probset(
        probset, pred_symb
    )


def solve_probfact_polynomial_dependencies(predicates, pfact_pred_symbs):
    grouped_pfact_preds = group_preds_by_pred_symb(
        predicates, pfact_pred_symbs
    )


def solve_succ_query(query, cpl):
    query = enforce_conjunction(query)
    conj_query = flatten_query(query, cpl)
    grounded = ground_cplogic_program(cpl)
    pred_symb_to_grounding = {
        get_grounding_pred_symb(grounding.expression): grounding
        for grounding in grounded.expressions
    }
    relations = []
    for predicate in conj_query.formulas:
        grounding = pred_symb_to_grounding[predicate.functor]
        relation = grounding_to_provset(grounding, predicate)
        relations.append(relation)
    relation = ra_binary_to_nary(NaturalJoin)(relations)
    proj_cols = tuple(
        str2columnstr_constant(arg.name)
        for arg in set.union(*[set(pred.args) for pred in query.formulas])
    )
    relation = Projection(relation, proj_cols)
    for x, y in get_probchoice_variable_equalities(
        conj_query.formulas, cpl.pchoice_pred_symbs
    ):
        relation = Selection(
            relation,
            EQUAL(
                str2columnstr_constant(x.name), str2columnstr_constant(y.name)
            ),
        )
    solver = RelationalAlgebraProvenanceCountingSolver()
    return solver.walk(relation)
