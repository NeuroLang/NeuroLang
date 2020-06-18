import problog.logic
import problog.program

from ...expressions import Constant, FunctionApplication
from ..expression_processing import is_probabilistic_fact
from ..expressions import ProbabilisticChoiceGrounding
from .grounding import get_grounding_pred_symb, ground_cplogic_program


def nl_pred_to_pl_pred(pred):
    pred_symb = problog.logic.Term(pred.functor.name)
    args = (
        problog.logic.Constant(arg.value)
        if isinstance(arg, Constant)
        else problog.logic.Var(arg.name)
        for arg in pred.args
    )
    return pred_symb(*args)


def pl_pred_from_tuple(pred_symb, tupl):
    probability = problog.logic.Constant(tupl[0])
    args = (problog.logic.Constant(arg) for arg in tupl[1:])
    return problog.logic.Term(pred_symb, *args, p=probability)


def add_facts_to_problog(pred_symb, relation, pl):
    pred_symb = pred_symb.name
    for tupl in relation.value.itervalues():
        args = (problog.logic.Constant(arg) for arg in tupl)
        fact = problog.logic.Term(pred_symb, *args)
        pl += fact


def add_probchoice_to_problog(pred_symb, relation, pl):
    pred_symb = pred_symb.name
    heads = []
    for tupl in relation.value.itervalues():
        heads.append(pl_pred_from_tuple(pred_symb, tupl))
    pl += problog.logic.Or.from_list(heads)


def add_probfacts_to_problog(pred_symb, relation, pl):
    pred_symb = pred_symb.name
    for tupl in relation.value.itervalues():
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
    for pred_symb, relation in cpl.extensional_database():
        add_facts_to_problog(pred_symb, relation, pl)
    for pred_symb in cpl.pfact_pred_symbs:
        add_probfacts_to_problog(pred_symb, cpl.symbol_table[pred_symb], pl)
    for pred_symb in cpl.pchoice_pred_symbs:
        add_probchoice_to_problog(pred_symb, cpl.symbol_table[pred_symb], pl)
    for union in cpl.intensional_database().values():
        add_rule_to_problog(union.formulas[0], pl)
    return pl
