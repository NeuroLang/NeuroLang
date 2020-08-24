import collections
import operator as op
from typing import AbstractSet, Iterable

import numpy

from ..datalog import WrappedRelationalAlgebraSet
from ..datalog.expression_processing import (
    conjunct_if_needed,
    extract_logic_predicates,
    reachable_code,
)
from ..exceptions import NeuroLangFrontendException, UnexpectedExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..expressions import Constant, Expression, FunctionApplication, Symbol
from ..logic import Conjunction, Implication, Union
from .exceptions import DistributionDoesNotSumToOneError
from .expressions import PROB, ProbabilisticPredicate, ProbabilisticQuery


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
        and isinstance(expression.consequent, ProbabilisticPredicate)
        and isinstance(expression.consequent.body, FunctionApplication)
        and expression.antecedent == Constant[bool](True)
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
            for p in extract_logic_predicates(pred.antecedent)
            if p.functor != pred.consequent.functor
            and not is_builtin(p, program.builtins())
        )

        if is_within_language_succ_query(
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


def is_within_language_succ_query(implication):
    try:
        get_within_language_succ_query_prob_term(implication)
        return True
    except ValueError:
        return False


def get_within_language_succ_query_prob_term(implication):
    try:
        prob_term = next(
            arg
            for arg in implication.consequent.args
            if isinstance(arg, ProbabilisticQuery) and arg.functor == PROB
        )
        return prob_term
    except StopIteration:
        raise ValueError("Expression does not have a SUCC probabilistic term")


def group_preds_by_pred_symb(predicates, filter_set=None):
    """
    Group predicates by their predicate symbol.

    An optional filter set of predicate symbols can be passed to only return
    the ones in the set.

    Parameters
    ----------
    predicates : iterable of predicates
        Predicates that should be grouped.
    filter_set : set of predicate symbols (optional)
        Predicate symbols to consider.

    Returns
    -------
    dict of predicate symbol to set of predicates

    """
    grouped = collections.defaultdict(set)
    for pred in predicates:
        if filter_set is not None and pred.functor in filter_set:
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
    grouped_pchoice_preds = group_preds_by_pred_symb(
        predicates, pchoice_pred_symbs
    )
    eq_set = set()
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
    if len(program.pchoice_pred_symbs) > 0:
        eq = Constant(op.eq)
        added_equalities = []
        for x, y in get_probchoice_variable_equalities(
            query.formulas, program.pchoice_pred_symbs
        ):
            added_equalities.append(eq(x, y))
        if len(added_equalities) > 0:
            query = Conjunction(query.formulas + tuple(added_equalities))
    return query


def iter_conjunctive_query_predicates(query):
    if isinstance(query, FunctionApplication):
        yield query
    elif isinstance(query, Conjunction):
        for predicate in query.formulas:
            yield predicate
    else:
        raise UnexpectedExpressionError(
            "Expected a predicate or conjunction of predicates, got {}".format(
                type(query)
            )
        )


def is_easily_shatterable(predicates):
    """
    Examples
    --------
    The following conjunctive queries can be shattered easily:
        - P(a, x), P(b, x)
    The following conjunctive queries cannot be shattered easily:
        - P(x), P(y)
        - P(a, x), P(a, y)
        - P(a, x), P(y, b)
        - P(a), P(x)

    """
    idx_to_const = dict()
    idx_to_symb = dict()
    for predicate in predicates:
        for idx, arg in enumerate(predicate.args):
            if isinstance(arg, Constant):
                if idx in idx_to_symb or (
                    idx in idx_to_const and idx_to_const[idx] == arg
                ):
                    return False
                idx_to_const[idx] = arg
            elif isinstance(arg, Symbol):
                if idx in idx_to_const or (
                    idx in idx_to_symb and idx_to_symb[idx] != arg
                ):
                    return False
                idx_to_symb[idx] = arg
    return True


class Shatter(FunctionApplication):
    pass


class ShatterProbfact(Shatter):
    pass


class QueryEasyShatteringTagger(PatternWalker):
    def __init__(self, program):
        self.program = program

    @add_match(FunctionApplication, lambda fa: not isinstance(fa, Shatter))
    def predicate(self, predicate):
        if predicate.functor in self.program.pfact_pred_symbs:
            return ShatterProbfact(*predicate.unapply())
        return predicate

    @add_match(Conjunction)
    def conjunction(self, conjunction):
        return Conjunction(
            (self.walk(formula) for formula in conjunction.formulas)
        )


class EasyQueryShatterer(PatternWalker):
    def __init__(self, program):
        self.program = program

    @add_match(ShatterProbfact)
    def easy_shatter_probfact(self, shatter):
        const_idxs = list(
            i
            for i, arg in enumerate(shatter.args)
            if isinstance(arg, Constant)
        )
        if const_idxs:
            new_relation = self.program.symbol_table[shatter.functor].value
            new_relation = new_relation.selection(
                {
                    new_relation.columns[i + 1]: shatter.args[i].value
                    for i in const_idxs
                }
            )
            proj_cols = (0,) + tuple(
                i + 1
                for i, arg in enumerate(shatter.args)
                if not isinstance(arg, Constant)
            )
            new_relation = new_relation.projection(*proj_cols)
            new_pred_symb = Symbol.fresh()
            self.program.add_probabilistic_facts_from_tuples(
                new_pred_symb, new_relation
            )
            non_const_args = (
                arg for arg in shatter.args if not isinstance(arg, Constant)
            )
            new_predicate = new_pred_symb(*non_const_args)
            return new_predicate
        else:
            return shatter

    @add_match(FunctionApplication)
    def function_application(self, function_application):
        return function_application

    @add_match(Conjunction)
    def conjunction(self, conjunction):
        return Conjunction(
            (self.walk(formula) for formula in conjunction.formulas)
        )


def shatter_easy_probfacts(query, program):
    """
    Remove constants occurring in a given query, possibly removing self-joins.

    If there is a self-join, the self-joined relation is split into multiple
    relations. These relations are added in-place to the program and the
    returned equivalent query makes use of these relations.

    Parameters
    ----------
    query : conjunctive query (can be a single predicate)
        A query that contains constants.
    program : probabilistic program
        Program containing probabilistic relations associated with the query.

    Returns
    -------
    Conjunctive query
        An equivalent conjunctive query without constants.

    Notes
    -----
    TODO: handle queries like Q = R(a, y), R(x, b) (see example 4.2 in [1]_)

    .. [1] Van den Broeck, G., and Suciu, D. (2017). Query Processing on
       Probabilistic Data: A Survey. FNT in Databases 7, 197–341.

    """
    grouped_pfact_preds = group_preds_by_pred_symb(
        list(iter_conjunctive_query_predicates(query)),
        program.pfact_pred_symbs,
    )
    for pred_symb, predicates in grouped_pfact_preds.items():
        if not is_easily_shatterable(predicates):
            raise UnexpectedExpressionError(
                f"Cannot easily shatter {pred_symb}-predicates"
            )
    tagger = QueryEasyShatteringTagger(program)
    tagged_query = tagger.walk(query)
    shatterer = EasyQueryShatterer(program)
    shattered_query = shatterer.walk(tagged_query)
    if isinstance(shattered_query, Shatter) or (
        isinstance(shattered_query, Conjunction)
        and any(
            isinstance(formula, Shatter)
            for formula in shattered_query.formulas
        )
    ):
        raise UnexpectedExpressionError("Cannot easily shatter query")
    return shattered_query
