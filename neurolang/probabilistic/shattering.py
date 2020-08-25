import collections
from typing import AbstractSet

from ..exceptions import UnexpectedExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction
from .expression_processing import (
    group_preds_by_pred_symb,
    iter_conjunctive_query_predicates,
)
from .probabilistic_ra_utils import ProbabilisticFactSet


def group_terms_by_index(predicates):
    idx_to_terms = collections.defaultdict(list)
    for predicate in predicates:
        for idx, term in enumerate(predicate.args):
            idx_to_terms[idx].append(term)
    return idx_to_terms


def has_repeated_constant(list_of_terms):
    return any(
        count > 1
        for term, count in collections.Counter(list_of_terms).items()
        if isinstance(term, Constant)
    )


def has_both_symbol_and_constant(list_of_terms):
    has_symbol = False
    has_constant = False
    for term in list_of_terms:
        has_symbol |= isinstance(term, Symbol)
        has_constant |= isinstance(term, Constant)
        if has_symbol and has_constant:
            return True
    return False


def has_multiple_symbols(list_of_terms):
    return (
        len(set(term for term in list_of_terms if isinstance(term, Symbol)))
        > 1
    )


def is_easily_shatterable_self_join(predicates):
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
    idx_to_terms = group_terms_by_index(predicates)
    return not any(
        has_repeated_constant(terms)
        or has_both_symbol_and_constant(terms)
        or has_multiple_symbols(terms)
        for terms in idx_to_terms.values()
    )


class Shatter(FunctionApplication):
    pass


class QueryEasyShatteringTagger(ExpressionWalker):
    @add_match(
        FunctionApplication(ProbabilisticFactSet, ...),
        lambda fa: not isinstance(fa, Shatter)
        and any(isinstance(arg, Constant) for arg in fa.args),
    )
    def shatter_probfact_predicates(self, function_application):
        return Shatter(*function_application.unapply())


class EasyQueryShatterer(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Shatter(ProbabilisticFactSet, ...))
    def easy_shatter_probfact(self, shatter):
        const_idxs = list(
            i
            for i, arg in enumerate(shatter.args)
            if isinstance(arg, Constant)
        )
        new_relation = self.symbol_table[shatter.functor.relation].value
        new_relation = new_relation.selection(
            {
                new_relation.columns[i + 1]: shatter.args[i].value
                for i in const_idxs
            }
        )
        non_prob_columns = tuple(
            c
            for c in new_relation.columns
            if c != shatter.functor.probability_column.value
        )
        proj_cols = (shatter.functor.probability_column.value,) + tuple(
            non_prob_columns[i]
            for i, arg in enumerate(shatter.args)
            if not isinstance(arg, Constant)
        )
        new_relation = new_relation.projection(*proj_cols)
        new_pred_symb = Symbol.fresh()
        self.symbol_table[new_pred_symb] = Constant[AbstractSet](new_relation)
        non_const_args = tuple(
            arg for arg in shatter.args if not isinstance(arg, Constant)
        )
        new_tagged = ProbabilisticFactSet(
            new_pred_symb, shatter.functor.probability_column
        )
        return FunctionApplication(new_tagged, non_const_args)


def query_to_tagged_set_representation(query, symbol_table):
    new_predicates = list()
    for predicate in iter_conjunctive_query_predicates(query):
        new_predicate = FunctionApplication(
            symbol_table[predicate.functor], predicate.args
        )
        new_predicates.append(new_predicate)
    return Conjunction(tuple(new_predicates))


def shatter_easy_probfacts(query, symbol_table):
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
    ws_query = query_to_tagged_set_representation(query, symbol_table)
    pfact_pred_symbs = set(
        pred_symb
        for pred_symb, tagged_set in symbol_table.items()
        if isinstance(tagged_set, ProbabilisticFactSet)
    )
    grouped_pfact_preds = group_preds_by_pred_symb(
        list(iter_conjunctive_query_predicates(query)), pfact_pred_symbs
    )
    for pred_symb, predicates in grouped_pfact_preds.items():
        if not is_easily_shatterable_self_join(predicates):
            raise UnexpectedExpressionError(
                f"Cannot easily shatter {pred_symb}-predicates"
            )
    tagger = QueryEasyShatteringTagger()
    tagged_query = tagger.walk(ws_query)
    shatterer = EasyQueryShatterer(symbol_table)
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
