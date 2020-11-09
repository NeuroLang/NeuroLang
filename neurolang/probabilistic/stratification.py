import collections

import numpy as np

from neurolang.logic.expression_processing import extract_logic_atoms

from ..datalog.expression_processing import (
    dependency_matrix,
    extract_logic_predicates,
    reachable_code,
)
from ..exceptions import ForbiddenRecursivityError, UnsupportedProgramError
from ..expressions import Symbol
from ..logic import Implication, Union


def _iter_implication_or_union_of_implications(expression):
    if expression is None:
        return
    if isinstance(expression, Implication):
        yield expression
    elif isinstance(expression, Union):
        for formula in expression.formulas:
            yield formula


def reachable_code_from_query(query, program):
    """
    Find the part of the intensional database of the program that is needed to
    answer a rule-based query Head :- Body such that Head is not necesarrily
    part of the program.

    This is useful if the query is formulated as a rule instead of simply a
    query predicate (e.g. `ans(x, y) :- P(x, z), P(y, z)`).

    """
    if query is None:
        return Union(tuple(_get_list_of_intensional_rules(program)))
    predicates = [query.consequent] + list(
        extract_logic_predicates(query.antecedent)
    )
    reachable = set()
    for pred in predicates:
        for rule in _iter_implication_or_union_of_implications(
            program.intensional_database().get(pred.functor, None)
        ):
            reachable |= set(reachable_code(rule, program).formulas)
    return Union(tuple(reachable))


def stratify_program(query, program):
    """
    Statically analyse the program to isolate its deterministic strat, its
    probabilistic strats and its post-probabilistic-query deterministic strat.

    A query can be solved through stratification if the probabilistic and
    deterministic parts are well separated. In case there exists one
    within-language probabilistic query dependency, no probabilistic predicate
    should appear in the strat that depends on the query.

    Parameters
    ----------
    query : Implication
        Query defining the part of the program that needs to be stratified.
    program : CPLogicProgram
        Program that will be stratified.

    Returns
    -------
    mapping from idb type to Union
        Deterministic, probabilistic and post-probabilistic IDBs.

    Raises
    ------
    UnsupportedProgramError
        When a WLQ (within-language query) depends on another WLQ.

    """
    idb = list(reachable_code_from_query(query, program).formulas)
    idb_symbs, dep_mat = dependency_matrix(program, idb)
    wlq_symbs = set(program.within_language_succ_queries()).intersection(
        idb_symbs
    )
    _check_for_dependencies_between_wlqs(dep_mat, idb_symbs, wlq_symbs)
    grpd_symbs = collections.defaultdict(set)
    grpd_symbs["deterministic"] = _get_program_deterministic_symbols(program)
    grpd_symbs["probabilistic"] = program.probabilistic_predicate_symbols
    grpd_idbs = collections.defaultdict(list)
    count = len(idb)
    while idb:
        rule = idb.pop(0)
        idb_type = _get_rule_idb_type(rule, grpd_symbs, wlq_symbs)
        if idb_type is None:
            idb.append(rule)
            count -= 1
            if count < 0:
                raise ForbiddenRecursivityError(
                    "Unstratifiable recursive program"
                )
        else:
            grpd_symbs[idb_type].add(rule.consequent.functor)
            grpd_idbs[idb_type].append(rule)
            count = len(idb)
    return {
        idb_type: Union(tuple(idb_rules))
        for idb_type, idb_rules in grpd_idbs.items()
    }


def _get_list_of_intensional_rules(program):
    idb = [
        rule
        for exp in program.intensional_database().values()
        for rule in _iter_implication_or_union_of_implications(exp)
    ]
    return idb


def _get_program_deterministic_symbols(program):
    det_symbs = set(program.extensional_database().keys())
    det_symbs |= set(program.builtins())
    if hasattr(program, "constraints"):
        det_symbs |= set(
            formula.consequent.functor
            for formula in program.constraints().formulas
        )
    return det_symbs


def _get_rule_idb_type(rule, grpd_symbs, wlq_symbs):
    dep_symbs = set(
        pred.functor
        for pred in extract_logic_atoms(rule.antecedent)
        if isinstance(pred.functor, Symbol)
    )
    idb_type = None
    # handle the case of a WLQ with deterministic-only dependencies
    if rule.consequent.functor in wlq_symbs:
        idb_type = "probabilistic"
    elif grpd_symbs["deterministic"].issuperset(dep_symbs):
        idb_type = "deterministic"
    elif (
        grpd_symbs["deterministic"]
        | wlq_symbs
        | grpd_symbs["post_probabilistic"]
    ).issuperset(dep_symbs):
        idb_type = "post_probabilistic"
    elif not grpd_symbs["probabilistic"].isdisjoint(dep_symbs):
        idb_type = "probabilistic"
    return idb_type


def _check_for_dependencies_between_wlqs(dep_mat, idb_symbs, wlq_symbs):
    """
    Raise an exception if a WLQ (Within-Language Query) depends on another WLQ.
    """
    wlq_symb_idxs = {idb_symbs.index(wlq_symb) for wlq_symb in wlq_symbs}
    for wlq_symb_idx in wlq_symb_idxs:
        if _wlq_depends_on_other_wlq(wlq_symb_idx, dep_mat, wlq_symb_idxs):
            raise UnsupportedProgramError(
                "Unsupported dependency between within-language queries"
            )


def _wlq_depends_on_other_wlq(wlq_symb_idx, dep_mat, wlq_symb_idxs):
    stack = [wlq_symb_idx]
    while stack:
        dep_idxs = np.argwhere(dep_mat[stack.pop()].astype(bool)).flatten()
        if not wlq_symb_idxs.isdisjoint(dep_idxs):
            return True
        stack += list(dep_idxs)
    return False
