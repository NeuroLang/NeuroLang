from typing import AbstractSet

from ..datalog.instance import MapInstance
from ..expressions import Constant
from .cplogic.program import CPLogicProgram
from .expression_processing import (
    construct_within_language_succ_result,
    is_within_language_succ_query,
    within_language_succ_query_to_intensional_rule,
)


def compute_probabilistic_solution(
    det_edb, pfact_edb, pchoice_edb, prob_idb, prob_solver
):
    solution = MapInstance()
    cpl = _build_probabilistic_program(
        det_edb, pfact_edb, pchoice_edb, prob_idb
    )
    for rule in prob_idb.formulas:
        if is_within_language_succ_query(rule):
            pred = within_language_succ_query_to_intensional_rule(
                rule
            ).consequent
            provset = prob_solver(pred, cpl)
            relation = construct_within_language_succ_result(provset, rule)
        else:
            pred = rule.consequent
            provset = prob_solver(pred, cpl)
            relation = Constant[AbstractSet](
                provset.value, auto_infer_type=False, verify_type=False,
            )
        solution[pred.functor] = Constant[AbstractSet](
            relation.value.to_unnamed()
        )
    return solution


def _build_probabilistic_program(det_edb, pfact_edb, pchoice_edb, prob_idb):
    cpl = CPLogicProgram()
    db_to_add_fun = [
        (det_edb, cpl.add_extensional_predicate_from_tuples),
        (pfact_edb, cpl.add_probabilistic_facts_from_tuples),
        (pchoice_edb, cpl.add_probabilistic_choice_from_tuples),
    ]
    for database, add_fun in db_to_add_fun:
        for pred_symb, ra_set in database.items():
            add_fun(pred_symb, ra_set.value.unwrap())
    cpl.walk(prob_idb)
    return cpl
