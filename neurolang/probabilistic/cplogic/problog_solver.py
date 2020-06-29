import collections

import problog.core
import problog.logic
import problog.program
import problog.sdd_formula

from ...expressions import Constant, FunctionApplication, Symbol
from ...relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    str2columnstr_constant,
)
from ...relational_algebra_provenance import ProvenanceAlgebraSet


def nl_pred_to_pl_pred(pred):
    args = (
        problog.logic.Constant(arg.value)
        if isinstance(arg, Constant)
        else problog.logic.Var(arg.name)
        for arg in pred.args
    )
    return problog.logic.Term(pred.functor.name, *args)


def pl_preds_to_prov_set(pl_preds, columns):
    tuples = set()
    for pl_pred, prob in pl_preds.items():
        if prob > 0:
            tupl = (prob,) + tuple(arg.value for arg in pl_pred.args)
            tuples.add(tupl)
    prob_col = str2columnstr_constant(Symbol.fresh().name)
    return ProvenanceAlgebraSet(
        NamedRelationalAlgebraFrozenSet(
            columns=(prob_col.value,) + tuple(c.value for c in columns),
            iterable=tuples,
        ),
        prob_col,
    )


def pl_solution_to_nl_solution(pl_solution, query_preds):
    pred_symb_to_tuples = collections.defaultdict(set)
    for pl_pred, prob in pl_solution.items():
        tupl = (prob,) + tuple(arg.value for arg in pl_pred.args)
        pred_symb = Symbol(pl_pred.functor)
        pred_symb_to_tuples[pred_symb].add(tupl)
    prob_col = str2columnstr_constant(Symbol.fresh().name)
    return {
        qpred.functor: ProvenanceAlgebraSet(
            NamedRelationalAlgebraFrozenSet(
                columns=(prob_col.value,)
                + tuple(arg.name for arg in qpred.args),
                iterable=pred_symb_to_tuples[qpred.functor],
            ),
            prob_col,
        )
        for qpred in query_preds
    }


def pl_pred_from_tuple(pred_symb, tupl):
    probability = problog.logic.Constant(tupl[0])
    args = (problog.logic.Constant(arg) for arg in tupl[1:])
    return problog.logic.Term(pred_symb, *args, p=probability)


def add_facts_to_problog(pred_symb, relation, pl):
    pred_symb = pred_symb.name
    if relation.is_dee():
        pl += problog.logic.Term(pred_symb)
    for tupl in relation:
        args = (problog.logic.Constant(arg) for arg in tupl)
        fact = problog.logic.Term(pred_symb, *args)
        pl += fact


def add_probchoice_to_problog(pred_symb, relation, pl):
    pred_symb = pred_symb.name
    heads = []
    for tupl in relation:
        heads.append(pl_pred_from_tuple(pred_symb, tupl))
    pl += problog.logic.Or.from_list(heads)


def add_probfacts_to_problog(pred_symb, relation, pl):
    pred_symb = pred_symb.name
    for tupl in relation:
        pl += pl_pred_from_tuple(pred_symb, tupl)


def add_rule_to_problog(rule, pl):
    csqt = nl_pred_to_pl_pred(rule.consequent)
    if isinstance(rule.antecedent, FunctionApplication):
        antecedent = nl_pred_to_pl_pred(rule.antecedent)
    else:
        it = iter(rule.antecedent.formulas)
        antecedent = nl_pred_to_pl_pred(next(it))
        for pred in it:
            antecedent &= nl_pred_to_pl_pred(pred)
    pl_rule = csqt << antecedent
    pl += pl_rule


def cplogic_to_problog(cpl):
    pl = problog.program.SimpleProgram()
    for pred_symb, relation in cpl.extensional_database().items():
        add_facts_to_problog(pred_symb, relation.value.unwrap(), pl)
    for pred_symb in cpl.pfact_pred_symbs:
        add_probfacts_to_problog(
            pred_symb, cpl.symbol_table[pred_symb].value.unwrap(), pl
        )
    for pred_symb in cpl.pchoice_pred_symbs:
        add_probchoice_to_problog(
            pred_symb, cpl.symbol_table[pred_symb].value.unwrap(), pl
        )
    for union in cpl.intensional_database().values():
        for rule in union.formulas:
            add_rule_to_problog(rule, pl)
    return pl


def solve_succ_query(query_pred, cpl):
    pl = cplogic_to_problog(cpl)
    query = problog.logic.Term("query")
    pl += query(nl_pred_to_pl_pred(query_pred))
    res = problog.core.ProbLog.convert(pl, problog.sdd_formula.SDD).evaluate()
    columns = tuple(
        str2columnstr_constant(arg.name)
        if isinstance(arg, Symbol)
        else str2columnstr_constant(Symbol.fresh().name)
        for arg in query_pred.args
    )
    return pl_preds_to_prov_set(res, columns)
