from ...datalog.expression_processing import flatten_query, enforce_conjunction
from ...exceptions import ForbiddenDisjunctionError, ForbiddenExpressionError
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Conjunction, Implication, Union
from ...relational_algebra import (
    ConcatenateConstantColumn,
    NaturalJoin,
    Projection,
    RelationalAlgebraSolver,
    RenameColumns,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
    ra_binary_to_nary,
)
from ..expressions import ProbabilisticChoiceGrounding, ProbabilisticPredicate
from .grounding import (
    get_grounding_pred_symb,
    get_grounding_predicate,
    ground_cplogic_program,
)


def grounding_to_provset(grounding, dst_pred):
    src_pred = get_grounding_predicate(grounding.expression)
    relation = grounding.relation
    if isinstance(grounding, ProbabilisticChoiceGrounding) or isinstance(
        grounding.expression.consequent, ProbabilisticPredicate
    ):
        # /!\ this assumes the first column contains the probability
        prov_col = str2columnstr_constant(grounding.relation.value.columns[0])
    else:
        prov_col = str2columnstr_constant(Symbol.fresh().name)
        # add a new probability column with name `prob_col` and ones everywhere
        cst_one_probability = Constant[float](
            1.0, auto_infer_type=False, verify_type=False
        )
        relation = ConcatenateConstantColumn(
            relation, prov_col, cst_one_probability
        )
    renames = tuple(
        (str2columnstr_constant(src.name), str2columnstr_constant(dst.name))
        for src, dst in zip(src_pred.args, dst_pred.args)
    )
    relation = RenameColumns(relation, renames)
    solver = RelationalAlgebraSolver()
    relation = solver.walk(relation)
    prov_set = ProvenanceAlgebraSet(relation.value, prov_col)
    return prov_set


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
    solver = RelationalAlgebraProvenanceCountingSolver()
    return solver.walk(relation)
