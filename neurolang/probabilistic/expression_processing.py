import collections
from typing import AbstractSet, Iterable

import numpy
from neurolang.datalog.constraints_representation import RightImplication

from neurolang.relational_algebra import str2columnstr_constant

from ..datalog import WrappedRelationalAlgebraSet
from ..datalog.expression_processing import (
    EQ,
    UnifyVariableEqualities,
    conjunct_formulas,
    extract_logic_atoms,
    extract_logic_predicates,
    reachable_code
)
from ..exceptions import NeuroLangFrontendException, UnexpectedExpressionError
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..logic import TRUE, Conjunction, Implication, Negation, Union
from ..logic.transformations import GuaranteeConjunction
from ..relational_algebra import Projection, RelationalAlgebraSolver
from ..utils import OrderedSet
from .exceptions import DistributionDoesNotSumToOneError
from .expressions import PROB, ProbabilisticPredicate, ProbabilisticFact, ProbabilisticQuery


def is_probabilistic_fact(expression):
    r"""
    Whether the expression is a probabilistic fact.

    Notes
    -----
    In CP-Logic [1]_, a probabilistic fact is seen as a CP-event

    .. math:: \left( \alpha \text{P}(x_1, \dots, x_n)  \right) \gets \top

    with a single atom in its head which is true with probability
    :math:`\alpha`.

    .. [1] Vennekens, Joost, Marc Denecker, and Maurice Bruynooghe. "CP-Logic:
       A Language of Causal Probabilistic Events and Its Relation to Logic
       Programming." Theory and Practice of Logic Programming 9, no. 3 (May
       2009): 245–308. https://doi.org/10.1017/S1471068409003767.

    """
    return (
        isinstance(expression, Implication)
        and isinstance(expression.consequent, ProbabilisticFact)
        and isinstance(expression.consequent.body, FunctionApplication)
        and expression.antecedent == TRUE
    )


def is_query_based_probpredicate(expression):
    return (
        isinstance(expression, Implication)
        and isinstance(expression.consequent, ProbabilisticPredicate)
        and expression.antecedent != TRUE
    )


def group_probabilistic_facts_by_pred_symb(union):
    probfacts = collections.defaultdict(list)
    non_probfacts = list()
    for expression in union.formulas:
        if is_probabilistic_fact(expression):
            probfacts[expression.consequent.body.functor].append(expression)
        else:
            non_probfacts.append(expression)
    return probfacts, non_probfacts


def const_or_symb_as_python_type(exp):
    if isinstance(exp, Constant):
        return exp.value
    else:
        return exp.name


def build_probabilistic_fact_set(pred_symb, pfacts):
    iterable = [
        (const_or_symb_as_python_type(pf.consequent.probability),)
        + tuple(
            const_or_symb_as_python_type(arg)
            for arg in pf.consequent.body.args
        )
        for pf in pfacts
    ]
    return Constant[AbstractSet](WrappedRelationalAlgebraSet(iterable))


def check_probabilistic_choice_set_probabilities_sum_to_one(ra_set):
    probs_sum = sum(v.value[0].value for v in ra_set.value)
    if not numpy.isclose(probs_sum, 1.0):
        raise DistributionDoesNotSumToOneError(
            "Probability labels of a probabilistic choice should sum to 1. "
            f"Got {probs_sum} instead."
        )


def add_to_union(union, to_add):
    """
    Extend `Union` with another `Union` or an iterable of `Expression`.

    Parameters
    ----------
    union: Union
        The initial `Union` to which expressions will be added.
    to_add: Unino or Expression iterable
        `Expression`s to be added to the `Union`.

    Returns
    -------
    new_union: Union
        A new `Union` containing the new expressions.

    """
    if isinstance(to_add, Union):
        return Union(union.formulas + to_add.formulas)
    if isinstance(to_add, Iterable):
        if not all(isinstance(item, Expression) for item in to_add):
            raise UnexpectedExpressionError("Expected Expression")
        return Union(union.formulas + tuple(to_add))
    raise UnexpectedExpressionError("Expected Union or Expression iterable")


def union_contains_probabilistic_facts(union):
    return any(is_probabilistic_fact(exp) for exp in union.formulas)


def separate_deterministic_probabilistic_code(
    program, query_pred=None, det_symbols=None, prob_symbols=None
):
    if det_symbols is None:
        det_symbols = set()
    if prob_symbols is None:
        prob_symbols = set()
    if query_pred is None:
        formulas = tuple()
        for union in program.intensional_database().values():
            formulas += union.formulas
        query_reachable_code = Union(formulas)
    else:
        query_reachable_code = reachable_code(query_pred, program)

    if hasattr(program, "constraints"):
        constraints_symbols = set(
            [ri.consequent.functor for ri in program.constraints().formulas]
        )
    else:
        constraints_symbols = set()

    deterministic_symbols = (
        set(program.extensional_database().keys())
        | set(det_symbols)
        | set(program.builtins().keys())
        | constraints_symbols
    )
    deterministic_program = list()

    probabilistic_symbols = (
        program.pfact_pred_symbs
        | program.pchoice_pred_symbs
        | set(prob_symbols)
    )

    probabilistic_program = list()
    unclassified_code = list(query_reachable_code.formulas)
    unclassified = 0
    initial_unclassified_length = len(unclassified_code) + 1
    while (
        len(unclassified_code) > 0
        and unclassified <= initial_unclassified_length
    ):
        pred = unclassified_code.pop(0)
        initial_unclassified_length = len(unclassified_code)
        preds_antecedent = set(
            p.functor
            for p in extract_logic_atoms(pred.antecedent)
            if p.functor != pred.consequent.functor
            and not is_builtin(p, program.builtins())
        )

        if is_within_language_prob_query(
            pred
        ) or not probabilistic_symbols.isdisjoint(preds_antecedent):
            probabilistic_symbols.add(pred.consequent.functor)
            probabilistic_program.append(pred)
            unclassified = 0
        elif deterministic_symbols.issuperset(preds_antecedent):
            deterministic_symbols.add(pred.consequent.functor)
            deterministic_program.append(pred)
            unclassified = 0
        else:
            unclassified_code.append(pred)
            unclassified += 1
    if not probabilistic_symbols.isdisjoint(deterministic_symbols):
        raise NeuroLangFrontendException(
            "An atom was defined as both deterministic and probabilistic"
        )
    if len(unclassified_code) > 0:
        raise NeuroLangFrontendException("There are unclassified atoms")

    return Union(deterministic_program), Union(probabilistic_program)


def is_builtin(pred, known_builtins=None):
    if known_builtins is None:
        known_builtins = set()

    return isinstance(pred.functor, Constant) or pred.functor in known_builtins


def is_within_language_prob_query(implication):
    try:
        get_within_language_prob_query_prob_term(implication)
        return True
    except ValueError:
        return False


def get_within_language_prob_query_prob_term(implication):
    try:
        prob_term = next(
            arg
            for arg in implication.consequent.args
            if isinstance(arg, ProbabilisticQuery) and arg.functor == PROB
        )
        return prob_term
    except StopIteration:
        raise ValueError("Expression does not have a SUCC probabilistic term")


def within_language_succ_query_to_intensional_rule(rule):
    csqt_without_prob = rule.consequent.functor(
        *(
            arg
            for arg in rule.consequent.args
            if isinstance(arg, (Constant, Symbol))
        )
    )
    return Implication(csqt_without_prob, rule.antecedent)


def construct_within_language_succ_result(provset, rule):
    proj_cols = list()
    for arg in rule.consequent.args:
        if isinstance(arg, Symbol):
            proj_cols.append(str2columnstr_constant(arg.name))
        elif isinstance(arg, ProbabilisticQuery) and arg.functor == PROB:
            proj_cols.append(provset.provenance_column)
    return RelationalAlgebraSolver().walk(
        Projection(provset.relation, proj_cols)
    )


def group_preds_by_functor(predicates, filter_set=None):
    """
    Group predicates by their functor.

    An optional filter set of functors can be passed to only return the ones in
    the set.

    Parameters
    ----------
    predicates : iterable of predicates
        Predicates that should be grouped.
    filter_set : set of functors (optional)
        Functors to consider.

    Returns
    -------
    dict of functors to set of predicates

    """
    grouped = collections.defaultdict(OrderedSet)
    for pred in predicates:
        if filter_set is None or pred.functor in filter_set:
            grouped[pred.functor].add(pred)
    return dict(grouped)


def get_probchoice_variable_equalities(predicates, pchoice_pred_symbs):
    """
    Infer variable equalities from repeated probabilistic choice predicates.

    Parameters
    ----------
    predicates : iterable of predicates
        Predicates that are part of a conjunction.
    pchoice_pred_symbs : iterable of predicate symbols
        Predicate symbols associated with probabilistic choices.

    Returns
    -------
    set of pairs of symbol variables
        Each pair in the set represents the equality between two variables.
        Variables within the pair are sorted in lexicographical order.

    Notes
    -----
    A probabilistic choice encodes mutually exclusive random events. Let `P` be
    the predicate symbol of a probabilistic choice. The conjunction `P(x),
    P(y)` can only be true if `x == y`.

    """
    grouped_pchoice_preds = group_preds_by_functor(
        predicates, pchoice_pred_symbs
    )
    eq_set = OrderedSet()
    for predicates in grouped_pchoice_preds.values():
        predicates = list(predicates)
        arity = len(predicates[0].args)
        for var_idx in range(arity):
            for pred_idx in range(1, len(predicates)):
                x = predicates[pred_idx - 1].args[var_idx]
                y = predicates[pred_idx].args[var_idx]
                if x == y:
                    continue
                eq_set.add(
                    (
                        min(x, y, key=lambda symb: symb.name),
                        max(x, y, key=lambda symb: symb.name),
                    )
                )
    return eq_set


def lift_optimization_for_choice_predicates(query, program):
    """Replace multiple instances of choice predicates by
    single instances enforncing the definition that the probability
    that two different grounded choice predicates are mutually exclusive.

    Parameters
    ----------
    query : predicate or conjunction of predicates
        The query for which the conjunction is constructed.
    program : a program with a probabilistic database.
        Program with logical rules that will be used to construct the
        conjunction corresponding to the given query.

    Returns
    -------
    Conjunctive query
        conjunctive query rewritten for choice predicate implementation.

    """
    if len(program.pchoice_pred_symbs) == 0:
        return query
    positive_predicates = OrderedSet()
    negative_predicates = OrderedSet()
    for pred in extract_logic_predicates(query):
        if isinstance(pred, Negation):
            negative_predicates.add(pred)
        else:
            positive_predicates.add(pred)
    pchoice_eqs = get_probchoice_variable_equalities(
        positive_predicates,
        program.pchoice_pred_symbs
    )
    if len(pchoice_eqs) == 0:
        return query
    eq_conj = Conjunction(tuple(EQ(x, y) for x, y in pchoice_eqs))
    grpd_preds = group_preds_by_functor(positive_predicates)
    new_formulas = OrderedSet(eq_conj.formulas) | OrderedSet(negative_predicates)
    for functor, preds in grpd_preds.items():
        if functor not in program.pchoice_pred_symbs:
            new_formulas |= OrderedSet(preds)
        else:
            conj = conjunct_formulas(Conjunction(tuple(preds)), eq_conj)
            unifier = UnifyVariableEqualities()
            rule = Implication(Symbol.fresh()(tuple()), conj)
            unified_antecedent = unifier.walk(rule).antecedent
            unified_conj = GuaranteeConjunction().walk(unified_antecedent)
            new_formulas |= OrderedSet(unified_conj.formulas)
    new_query = Conjunction(tuple(new_formulas))
    return new_query


def is_probabilistic_predicate_symbol(pred_symb, program):
    wlq_symbs = set(program.within_language_prob_queries())
    prob_symbs = program.pfact_pred_symbs | program.pchoice_pred_symbs
    stack = [pred_symb]
    while stack:
        pred_symb = stack.pop()
        if pred_symb in prob_symbs:
            return True
        if (
            pred_symb in wlq_symbs
            or pred_symb not in program.intensional_database()
        ):
            continue

        if isinstance(program.symbol_table[pred_symb], Constant):
            continue

        for rule in program.symbol_table[pred_symb].formulas:
            if isinstance(rule, RightImplication):
                continue
            stack += [
                apred.functor
                for apred in extract_logic_atoms(rule.antecedent)
                if apred.functor not in wlq_symbs
            ]
    return False
