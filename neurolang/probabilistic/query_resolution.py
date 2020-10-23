import typing
from typing import AbstractSet

from ..datalog.instance import MapInstance
from ..expressions import Constant
from ..logic import Implication
from .cplogic.program import CPLogicProgram
from .expression_processing import (
    construct_within_language_succ_result,
    is_within_language_prob_query,
    within_language_succ_query_to_intensional_rule,
)
from .expressions import Condition


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
    relation = Constant[AbstractSet](
        provset.value,
        auto_infer_type=False,
        verify_type=False,
    )
    return relation


def compute_probabilistic_solution(
    det_edb,
    pfact_edb,
    pchoice_edb,
    prob_idb,
    succ_prob_solver,
    marg_prob_solver,
):
    solution = MapInstance()
    cpl = _build_probabilistic_program(
        det_edb, pfact_edb, pchoice_edb, prob_idb
    )
    for rule in prob_idb.formulas:
        if is_within_language_prob_query(rule):
            relation = _solve_within_language_prob_query(
                cpl, rule, succ_prob_solver, marg_prob_solver
            )
        else:
            relation = _solve_for_probabilistic_rule(
                cpl, rule, succ_prob_solver
            )
        solution[rule.consequent.functor] = Constant[AbstractSet](
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
