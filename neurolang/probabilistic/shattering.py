import collections

from ..exceptions import UnexpectedExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker, PatternWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction
from .expression_processing import (
    group_preds_by_pred_symb,
    iter_conjunctive_query_predicates,
)


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


class ShatterProbfact(Shatter):
    pass


class QueryEasyShatteringTagger(PatternWalker):
    def __init__(self, program):
        self.program = program

    @add_match(FunctionApplication)
    def shatter_probfact_predicates(self, predicate):
        if predicate.functor not in self.program.pfact_pred_symbs:
            return predicate
        const_idxs = list(
            i
            for i, arg in enumerate(predicate.args)
            if isinstance(arg, Constant)
        )
        if const_idxs:
            return ShatterProbfact(*predicate.unapply())
        else:
            return predicate

    @add_match(Conjunction)
    def conjunction(self, conjunction):
        return Conjunction(
            (self.walk(formula) for formula in conjunction.formulas)
        )


class EasyQueryShatterer(ExpressionWalker):
    def __init__(self, program):
        self.program = program

    @add_match(ShatterProbfact)
    def easy_shatter_probfact(self, shatter):
        const_idxs = list(
            i
            for i, arg in enumerate(shatter.args)
            if isinstance(arg, Constant)
        )
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
        if not is_easily_shatterable_self_join(predicates):
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
