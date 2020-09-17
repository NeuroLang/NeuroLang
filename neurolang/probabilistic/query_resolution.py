from typing import AbstractSet

from ..expressions import Constant
from .cplogic.program import CPLogicProgram
from .expression_processing import (
    construct_within_language_succ_result,
    is_within_language_succ_query,
    within_language_succ_query_to_intensional_rule,
)


def compute_probabilistic_solution(
    program, solution, prob_idb, probabilistic_solver
):
    cpl = _make_probabilistic_program_from_deterministic_solution(
        program, solution, prob_idb
    )
    for rule in prob_idb.formulas:
        if is_within_language_succ_query(rule):
            pred = within_language_succ_query_to_intensional_rule(
                rule
            ).consequent
            provset = probabilistic_solver(pred, cpl)
            relation = construct_within_language_succ_result(provset, rule)
        else:
            pred = rule.consequent
            provset = probabilistic_solver(pred, cpl)
            relation = Constant[AbstractSet](
                provset.value, auto_infer_type=False, verify_type=False,
            )
        solution[pred.functor] = Constant[AbstractSet](
            relation.value.to_unnamed()
        )
    return solution


def _make_probabilistic_program_from_deterministic_solution(
    program, deterministic_solution, probabilistic_idb
):
    cpl = CPLogicProgram()
    for pred_symb, ra_set in deterministic_solution.items():
        cpl.add_extensional_predicate_from_tuples(
            pred_symb, ra_set.value.unwrap()
        )
    for pred_symb in program.pfact_pred_symbs:
        cpl.add_probabilistic_facts_from_tuples(
            pred_symb, program.symbol_table[pred_symb].value.unwrap()
        )
    for pred_symb in program.pchoice_pred_symbs:
        cpl.add_probabilistic_choice_from_tuples(
            pred_symb, program.symbol_table[pred_symb].value.unwrap()
        )
    cpl.walk(probabilistic_idb)
    return cpl
