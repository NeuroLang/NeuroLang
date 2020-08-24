from ..exceptions import UnexpectedExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker, PatternWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction
from .expression_processing import (
    group_preds_by_pred_symb,
    iter_conjunctive_query_predicates,
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
