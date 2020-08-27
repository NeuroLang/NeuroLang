import collections
from typing import AbstractSet

from ..exceptions import UnexpectedExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction
from .expression_processing import iter_conjunctive_query_predicates
from .probabilistic_ra_utils import ProbabilisticFactSet


def group_terms_by_index(list_of_tuple_of_terms):
    idx_to_terms = collections.defaultdict(list)
    for terms in list_of_tuple_of_terms:
        for idx, term in enumerate(terms):
            idx_to_terms[idx].append(term)
    return idx_to_terms


def group_indexes_by_symbol(list_of_tuple_of_terms):
    symbol_to_indexes = collections.defaultdict(list)
    for terms in list_of_tuple_of_terms:
        for idx, term in enumerate(terms):
            if isinstance(term, Symbol):
                symbol_to_indexes[term].append(idx)
    return symbol_to_indexes


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


def any_symbol_occurs_in_different_locations(symbol_to_indexes):
    return any(
        len(set(indexes)) > 1 for symbol, indexes in symbol_to_indexes.items()
    )


def is_easily_shatterable_self_join(list_of_tuple_of_terms):
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
    idx_to_terms = group_terms_by_index(list_of_tuple_of_terms)
    symbol_to_indexes = group_indexes_by_symbol(list_of_tuple_of_terms)
    return not any(
        has_repeated_constant(terms)
        or has_both_symbol_and_constant(terms)
        or has_multiple_symbols(terms)
        for terms in idx_to_terms.values()
    ) and not any_symbol_occurs_in_different_locations(symbol_to_indexes)


class Shatter(FunctionApplication):
    pass


class QueryEasyShatteringTagger(ExpressionWalker):
    def __init__(self):
        self._cached_args = collections.defaultdict(set)

    @add_match(
        FunctionApplication(ProbabilisticFactSet, ...),
        lambda fa: not isinstance(fa, Shatter)
        and any(isinstance(arg, Constant) for arg in fa.args),
    )
    def shatter_probfact_predicates(self, function_application):
        self._check_can_shatter(function_application)
        self._cached_args[function_application.functor.relation].add(
            function_application.args
        )
        return Shatter(*function_application.unapply())

    def _check_can_shatter(self, function_application):
        pred_symb = function_application.functor.relation
        args = function_application.args
        list_of_tuple_of_terms = self._cached_args.get(pred_symb, set()).union(
            {args}
        )
        if not is_easily_shatterable_self_join(list_of_tuple_of_terms):
            raise UnexpectedExpressionError(
                f"Cannot easily shatter {pred_symb}-predicates"
            )


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


def _check_shatter_fully_solved(shattered_query):
    if isinstance(shattered_query, Shatter) or (
        isinstance(shattered_query, Conjunction)
        and any(
            isinstance(formula, Shatter)
            for formula in shattered_query.formulas
        )
    ):
        raise UnexpectedExpressionError("Cannot easily shatter query")


def shatter_easy_probfacts(query, symbol_table):
    """
    Remove constants occurring in a given query, possibly removing self-joins.

    If there is a self-join, the self-joined relation is split into multiple
    relations. These relations are added in-place to the symbol table. The
    returned equivalent query makes use of these relations.

    Parameters
    ----------
    query : conjunctive query (can be a single predicate)
        A query that contains constants.
    symbol_table : mapping, mutable
        Symbol table containing relations associated with the query. This
        `symbol_table` can be modified in-place by this function to add newly
        generated symbols and relations by the shattering process.

    Returns
    -------
    Conjunctive query
        An equivalent conjunctive query without constants.

    Notes
    -----
    TODO: shatter queries like `Q = R(a, y), R(x, b)` (see example 4.2 in [1]_)

    .. [1] Van den Broeck, G., and Suciu, D. (2017). Query Processing on
       Probabilistic Data: A Survey. FNT in Databases 7, 197–341.

    """
    ws_query = query_to_tagged_set_representation(query, symbol_table)
    tagger = QueryEasyShatteringTagger()
    tagged_query = tagger.walk(ws_query)
    shatterer = EasyQueryShatterer(symbol_table)
    shattered_query = shatterer.walk(tagged_query)
    _check_shatter_fully_solved(shattered_query)
    return shattered_query
