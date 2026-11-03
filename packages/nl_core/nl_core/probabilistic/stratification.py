import collections
import typing

import numpy as np
from scipy.sparse.csgraph import connected_components

from ..datalog.expression_processing import (
    dependency_matrix,
    extract_logic_atoms,
)
from ..datalog.constraints_representation import (
    RightImplication,
    get_list_of_intensional_rules,
    iter_implications,
    reachable_code,
)
from nl_expr.exceptions import (
    ForbiddenRecursivityError,
    UnsupportedProgramError,
)
from nl_expr.expressions import Symbol
from nl_expr.logic import TRUE, Implication, Union
from .expressions import ProbabilisticFact


def reachable_code_from_query(query, program):
    """
    Find the part of the intentional database of the program that is needed to
    answer a rule-based query Head :- Body such that Head is not necessarily
    part of the program.

    This is useful if the query is formulated as a rule instead of simply a
    query predicate (e.g. `ans(x, y) :- P(x, z), P(y, z)`).

    """
    if query is None:
        return Union(tuple(get_list_of_intensional_rules(program)))

    predicates = [query.consequent] + list(
        extract_logic_atoms(query.antecedent)
    )
    reachable = set()
    for pred in predicates:
        for rule in iter_implications(
            program.intensional_database().get(pred.functor, None)
        ):
            if not isinstance(rule, RightImplication):
                reachable |= set(reachable_code(rule, program).formulas)
    return Union(tuple(reachable))


def stratify_program(query, program):
    """
    Statically analyse the program to isolate its deterministic strata, its
    probabilistic strats and its post-probabilistic-query deterministic strata.

    A query can be solved through stratification if the probabilistic and
    deterministic parts are well separated. In case there exists one
    within-language probabilistic query dependency, no probabilistic predicate
    should appear in the strata that depends on the query.

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
    wlq_symbs = set(program.within_language_prob_queries()).intersection(
        idb_symbs
    )
    _check_for_dependencies_between_wlqs(dep_mat, idb_symbs, wlq_symbs)
    grpd_symbs = collections.defaultdict(set)
    grpd_symbs["deterministic"] = _get_program_deterministic_symbols(program)
    grpd_symbs["probabilistic"] = program.probabilistic_predicate_symbols
    grpd_idbs = collections.defaultdict(list)
    # mutually recursive predicates are typed together, from all their rules
    _, component_of = connected_components(dep_mat, connection="strong")
    components = collections.defaultdict(list)
    for rule in idb:
        symb_ix = idb_symbs.index(rule.consequent.functor)
        components[component_of[symb_ix]].append(rule)
    idb = []
    for rules in components.values():
        # the rules starting the recursion first, as they are solved
        symbs = {rule.consequent.functor for rule in rules}
        idb.append(sorted(rules, key=lambda r: _depends_on(r, symbs)))
    count = len(idb)
    while idb:
        rules = idb.pop(0)
        idb_type = _get_component_idb_type(rules, grpd_symbs, wlq_symbs)
        if idb_type is None:
            idb.append(rules)
            count -= 1
            if count < 0:
                raise ForbiddenRecursivityError(
                    "Unstratifiable recursive program"
                )
        else:
            grpd_symbs[idb_type] |= {rule.consequent.functor for rule in rules}
            grpd_idbs[idb_type] += rules
            count = len(idb)
    _check_for_query_based_probfact_dependency_on_prob_relation(
        grpd_idbs["probabilistic"], grpd_symbs
    )
    return {
        idb_type: Union(tuple(idb_rules))
        for idb_type, idb_rules in grpd_idbs.items()
        if idb_rules
    }


def _get_program_deterministic_symbols(program):
    det_symbs = set(program.extensional_database().keys())
    det_symbs |= set(program.builtins())
    if hasattr(program, "constraints"):
        det_symbs |= set(
            formula.consequent.functor
            for formula in program.constraints().formulas
        )

        det_symbs |= set(
            rule.functor for rule in program.existential_rules.keys()
        )
    return det_symbs


def _get_idb_type_already_classified(pred_symb, grpd_symbs):
    for typ, symbs in grpd_symbs.items():
        if pred_symb in symbs:
            return typ
    return None


def _get_component_idb_type(rules, grpd_symbs, wlq_symbs):
    """
    Return the type of the mutually recursive predicates defined by
    *rules*, those starting the recursion first: the most probabilistic
    type of the rules, or None while a rule depends on an untyped
    predicate, or if no rule starts the recursion.
    """
    symbs = {rule.consequent.functor for rule in rules}
    if _depends_on(rules[0], symbs):
        return None
    idb_types = [
        _get_rule_idb_type(rule, grpd_symbs, wlq_symbs, symbs)
        for rule in rules
    ]
    if None in idb_types:
        return None
    return max(idb_types, key=(
        "deterministic", "post_probabilistic", "probabilistic"
    ).index)


def _depends_on(rule, symbs):
    return any(
        atom.functor in symbs for atom in extract_logic_atoms(rule.antecedent)
    )


def _get_rule_idb_type(rule, grpd_symbs, wlq_symbs, component_symbs=()):
    pred_symb = rule.consequent.functor
    idb_type = _get_idb_type_already_classified(pred_symb, grpd_symbs)
    if idb_type is not None:
        return idb_type
    dep_symbs = set(
        pred.functor
        for pred in extract_logic_atoms(rule.antecedent)
        if isinstance(pred.functor, Symbol)
    ) - set(component_symbs)
    idb_type = None
    # handle the case of a WLQ with deterministic-only dependencies
    if pred_symb in wlq_symbs:
        idb_type = "probabilistic"
    elif grpd_symbs["deterministic"].issuperset(dep_symbs):
        idb_type = "deterministic"
    elif (
        grpd_symbs["deterministic"]
        | wlq_symbs
        | grpd_symbs["post_probabilistic"]
    ).issuperset(dep_symbs):
        idb_type = "post_probabilistic"
    elif not (grpd_symbs["probabilistic"] - wlq_symbs).isdisjoint(dep_symbs):
        idb_type = "probabilistic"
    return idb_type


def _check_for_query_based_probfact_dependency_on_prob_relation(
    prob_idb: typing.Iterable[Implication],
    grpd_symbs: typing.Mapping[Symbol, str],
) -> None:
    if any(
        any(
            atom.functor in grpd_symbs["probabilistic"]
            for atom in extract_logic_atoms(rule.antecedent)
        )
        for rule in prob_idb
        if isinstance(rule.consequent, ProbabilisticFact)
        and rule.antecedent != TRUE
    ):
        raise UnsupportedProgramError(
            "Query-based probabilistic facts cannot depend on probabilistic "
            "predicate"
        )


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
    seen = set()
    while stack:
        dep_idxs = set(np.argwhere(dep_mat[stack.pop()].astype(bool)).flatten())
        if not wlq_symb_idxs.isdisjoint(dep_idxs):
            return True
        stack += list(dep_idxs - seen)
        seen.update(dep_idxs)
    return False
