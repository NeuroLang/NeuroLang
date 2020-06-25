from ...datalog.expression_processing import (
    conjunct_formulas,
    extract_logic_free_variables,
)
from ...exceptions import ForbiddenDisjunctionError, ForbiddenExpressionError
from ...expressions import FunctionApplication, Symbol
from ...logic import Conjunction, Implication, Union
from ...relational_algebra import (
    RelationalAlgebraSolver,
    RenameColumns,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .grounding import (
    get_grounding_pred_symb,
    get_grounding_predicate,
    ground_cplogic_program,
)


def enforce_conjunction(expression):
    if isinstance(expression, Conjunction):
        return expression
    elif isinstance(expression, FunctionApplication):
        return Conjunction((expression,))
    raise ForbiddenExpressionError(
        "Cannot conjunct expression of type {}".format(type(expression))
    )


def get_rule_for_predicate(predicate, idb):
    if predicate.functor not in idb:
        return None
    expression = idb[predicate.functor]
    if isinstance(expression, Implication):
        return expression
    elif isinstance(expression, Union) and len(expression.formulas) == 1:
        return expression.formulas[0]
    raise ForbiddenDisjunctionError()


def freshen_free_variables(conjunction, free_variables):
    new_preds = []
    for pred in conjunction.formulas:
        new_args = tuple(
            Symbol.fresh() if arg in free_variables else arg
            for arg in pred.args
        )
        new_pred = FunctionApplication[pred.type](pred.functor, new_args)
        new_preds.append(new_pred)
    return Conjunction(tuple(new_preds))


def rename_to_match(conjunction, src_predicate, dst_predicate):
    renames = {
        src: dst
        for src, dst in zip(src_predicate.args, dst_predicate.args)
        if src != dst
    }
    new_preds = []
    for pred in conjunction.formulas:
        new_args = tuple(renames.get(arg, arg) for arg in pred.args)
        new_pred = FunctionApplication[pred.type](pred.functor, new_args)
        new_preds.append(new_pred)
    return Conjunction(tuple(new_preds))


def flatten_query(query, idb):
    query = enforce_conjunction(query)
    conj_query = Conjunction(tuple())
    for predicate in query.formulas:
        rule = get_rule_for_predicate(predicate, idb)
        if rule is None:
            formula = predicate
        else:
            formula = enforce_conjunction(rule.antecedent)
            free_variables = extract_logic_free_variables(rule)
            formula = freshen_free_variables(formula, free_variables)
            formula = flatten_query(formula, idb)
            formula = rename_to_match(formula, rule.consequent, predicate)
        conj_query = conjunct_formulas(conj_query, formula)
    return conj_query


def prob_grounding_to_provset(grounding, dst_pred):
    src_pred = get_grounding_predicate(grounding.expression)
    # /!\ this assumes the first column contains the probability
    prov_col = str2columnstr_constant(grounding.relation.value.columns[0])
    renames = {
        str2columnstr_constant(src): str2columnstr_constant(dst)
        for src, dst in zip(src_pred.args, dst_pred.args)
    }
    relation = RenameColumns(grounding.relation, renames)
    solver = RelationalAlgebraSolver()
    relation = solver.walk(relation)
    prov_set = ProvenanceAlgebraSet(grounding.relation, prov_col)
    return prov_set


def solve_succ_query(query, cpl):
    conj_query = flatten_query(query, cpl.intensional_database())
    pred_symb_to_grounding = {
        get_grounding_pred_symb(grounding): grounding
        for grounding in ground_cplogic_program(cpl)
    }
