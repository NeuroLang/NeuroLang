from typing import AbstractSet

from ..datalog.instance import MapInstance
from ..expressions import Constant
from .cplogic.program import CPLogicProgram
from .expression_processing import (
    construct_within_language_succ_result,
    is_within_language_prob_query,
    within_language_succ_query_to_intensional_rule
)
from .expressions import Condition


def compute_probabilistic_solution(
    det_edb, pfact_edb, pchoice_edb, prob_idb,
    succ_prob_solver, marg_prob_solver
):
    solution = MapInstance()
    cpl = _build_probabilistic_program(
        det_edb, pfact_edb, pchoice_edb, prob_idb
    )
    for rule in prob_idb.formulas:
        is_wlq_pq = is_within_language_prob_query(rule)
        if is_wlq_pq and isinstance(rule.antecedent, Condition):
            query = within_language_succ_query_to_intensional_rule(rule)
            if isinstance(rule.antecedent, Condition):
                provset = marg_prob_solver(query, cpl)
            else:
                provset = succ_prob_solver(query, cpl)
            relation = construct_within_language_succ_result(provset, rule)
        else:
            query = rule
            provset = succ_prob_solver(query, cpl)
            relation = Constant[AbstractSet](
                provset.value,
                auto_infer_type=False,
                verify_type=False,
            )
        solution[query.consequent.functor] = Constant[AbstractSet](
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
