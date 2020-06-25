from ...datalog.expression_processing import (
    conjunct_formulas,
    extract_logic_free_variables,
)
from ...exceptions import ForbiddenDisjunctionError, ForbiddenExpressionError
from ...expressions import FunctionApplication, Symbol
from ...logic import Conjunction, Implication, Union


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
    return Conjunction(
        tuple(
            FunctionApplication[pred.type](
                pred.functor, tuple(renames.get(arg, arg) for arg in pred.args)
            )
            for pred in conjunction.formulas
        )
    )


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
