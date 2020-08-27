import numpy as np

from ..datalog.expression_processing import (
    dependency_matrix,
    extract_logic_predicates,
    reachable_code,
)
from ..exceptions import UnsupportedProgramError
from ..logic import Implication, Union


def reachable_code_from_query(query, program):
    """
    Find the part of the intensional database of the program that is needed to
    answer a rule-based query Head :- Body such that Head is not necesarrily
    part of the program.

    This is useful if the query is formulated as a rule instead of simply a
    query predicate (e.g. `ans(x, y) :- P(x, z), P(y, z)`).

    """
    predicates = [query.consequent] + list(
        extract_logic_predicates(query.antecedent)
    )
    reachable = set()
    for pred in predicates:
        rule = program.intensional_database().get(pred.functor, None)
        if isinstance(rule, Implication):
            reachable |= set(reachable_code(rule, program).formulas)
        elif isinstance(rule, Union):
            for formula in rule.formulas:
                reachable |= set(reachable_code(formula, program).formulas)
    return Union(tuple(reachable))


def stratify_program(query, program):
    """
    Statically analyse the program to isolate its deterministic strat, its
    probabilistic strats and its post-probabilistic-query deterministic strat.

    A query can be solved through stratification if the probabilistic and
    deterministic parts are well separated. In case there exists one
    within-language probabilistic query dependency, no probabilistic predicate
    should appear in the strat that depends on the query.

    """
    reachable_idb = list(reachable_code_from_query(query, program).formulas)
    idb_symbs, dep_mat = dependency_matrix(program, reachable_idb)
    wlq_symbs = set(program.within_language_succ_queries())
    # limit to reachable within-language queries
    wlq_symbs = wlq_symbs.intersection(idb_symbs)
    _check_for_dependencies_between_wlqs(dep_mat, idb_symbs, wlq_symbs)
    det_symbs = set(program.extensional_database()) | set(program.builtins())
    prob_symbs = program.pfact_pred_symbs | program.pchoice_pred_symbs
    ppq_det_symbs = set()
    det_idb = list()
    prob_idb = list()
    ppq_det_idb = list()
    rules = reachable_idb
    while rules:
        rule = rules.pop(0)
        symb = rule.consequent.functor
        dep_symbs = set(
            pred.functor for pred in extract_logic_predicates(rule.antecedent)
        )
        if det_symbs.issuperset(dep_symbs):
            det_symbs.add(symb)
            det_idb.append(rule)
        elif (det_symbs | wlq_symbs).issuperset(dep_symbs):
            ppq_det_symbs.add(symb)
            ppq_det_idb.append(rule)
        elif symb in wlq_symbs or not prob_symbs.isdisjoint(dep_symbs):
            prob_symbs.add(symb)
            prob_idb.append(rule)
        else:
            rules.append(rule)
    return (
        Union(tuple(det_idb)),
        Union(tuple(prob_idb)),
        Union(tuple(ppq_det_idb)),
    )


def _check_for_dependencies_between_wlqs(dep_mat, idb_symbs, wlq_symbs):
    """
    Raise an exception if a WLQ (Within-Language Query) depends on another WLQ.
    """
    wlq_symbol_idxs = {idb_symbs.index(wlq_symb) for wlq_symb in wlq_symbs}
    for wlq_symb_idx in wlq_symbol_idxs:
        stack = [wlq_symb_idx]
        while stack:
            dep_idxs = np.argwhere(dep_mat[stack.pop()].astype(bool)).flatten()
            if not wlq_symbol_idxs.isdisjoint(dep_idxs):
                raise UnsupportedProgramError(
                    "Unsupported dependency between within-language queries"
                )
            stack += list(dep_idxs)
