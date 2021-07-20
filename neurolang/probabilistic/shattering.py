import collections
import itertools
import operator
from typing import AbstractSet

from ..datalog.expression_processing import (
    enforce_conjunctive_antecedent,
    extract_logic_predicates,
    is_conjunctive_expression,
    maybe_deconjunct_single_pred,
)
from ..exceptions import ForbiddenExpressionError
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import Conjunction, Disjunction, Implication, Negation
from ..logic.transformations import (
    RemoveTrivialOperations,
    RemoveDuplicatedConjunctsDisjuncts,
)
from .exceptions import NotEasilyShatterableError
from .probabilistic_ra_utils import ProbabilisticFactSet
from .transforms import convert_to_dnf_ucq

EQ = Constant(operator.eq)


def query_to_tagged_set_representation(query, symbol_table):
    new_antecedent = ReplaceExpressionWalker(symbol_table).walk(
        query.antecedent
    )
    return query.apply(query.consequent, new_antecedent)


def terms_differ_by_constant_term(terms_a, terms_b):
    return any(
        isinstance(term_a, Constant)
        and isinstance(term_b, Constant)
        and term_a != term_b
        for term_a, term_b in zip(terms_a, terms_b)
    )


def all_terms_differ_by_constant_term(list_of_tuple_of_terms):
    return all(
        terms_differ_by_constant_term(terms_a, terms_b)
        for terms_a, terms_b in itertools.combinations(
            list_of_tuple_of_terms, 2
        )
    )


def constant_terms_are_constant_in_all_tuples(list_of_tuple_of_terms):
    arity = len(list_of_tuple_of_terms[0])
    for i in range(arity):
        nb_of_constant_terms = sum(
            isinstance(terms[i], Constant) for terms in list_of_tuple_of_terms
        )
        if 0 < nb_of_constant_terms < len(list_of_tuple_of_terms):
            return False
    return True


def is_easily_shatterable_self_join(list_of_tuple_of_terms):
    """
    A self-join of `m` predicates is easily shatterable if the following two
    conditions are met.

    Firstly, if the `i`th term of one of the predicates is a constant, then the
    `i`th terms of all the other self-joined predicates must also be constants.
    For example, the self-join `P(x, a), P(x, y)` does not meet this condition
    because the second term appears both as a constant (in the first predicate)
    and as a variable (in the second predicate).

    Secondly, all predicates in the self-join must differ by at least one
    constant term. In other words, for two predicates P1 and P2 in the
    self-join, there must be a constant term at some position `i` in P1 whose
    value is different than the constant `i`th term in P2 (we know it's
    constant from the first condition). For example, the self-join `P(x, a),
    P(y, a)` does not meet this condition because the only constant term is `a`
    in both predicates. However, the self-join `P(x, a, b), P(x, a, c)` does
    meet this condition because the predicates differ in their constant value
    of the third term (`b` for the first predicate and `c` for the second
    predicate).

    """
    return constant_terms_are_constant_in_all_tuples(
        list_of_tuple_of_terms
    ) and all_terms_differ_by_constant_term(list_of_tuple_of_terms)


class Shatter(FunctionApplication):
    pass


class QueryEasyShatteringTagger(ExpressionWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    @add_match(
        FunctionApplication(ProbabilisticFactSet, ...),
        lambda fa: not isinstance(fa, Shatter),
    )
    def cache_non_constant_args(self, function_application):
        self._check_can_shatter(function_application)
        self._cached_args[function_application.functor.relation].add(
            function_application.args
        )
        return function_application

    def _check_can_shatter(self, function_application):
        pred_symb = function_application.functor.relation
        args = function_application.args
        list_of_tuple_of_terms = list(
            self._cached_args.get(pred_symb, set()).union({args})
        )
        if not is_easily_shatterable_self_join(list_of_tuple_of_terms):
            raise NotEasilyShatterableError(
                f"Cannot easily shatter {pred_symb}-predicates"
            )


class EasyQueryShatterer(ExpressionWalker):
    def __init__(self, symbol_table, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol_table = symbol_table
        self._cached = dict()

    @add_match(Implication(..., FunctionApplication))
    def implication(self, implication):
        conjunctive_query = enforce_conjunctive_antecedent(implication)
        return self.walk(conjunctive_query)

    @add_match(Implication(..., Conjunction))
    def conjunctive_query(self, conjunctive_query):
        return Implication(
            conjunctive_query.consequent,
            self._shatter_conjunction(conjunctive_query.antecedent),
        )

    @add_match(Implication(..., Disjunction))
    def disjunctive_query(self, disjunctive_query):
        disjuncts = list()
        for conjunctive_query in disjunctive_query.antecedent.formulas:
            disjuncts.append(self._shatter_conjunction(conjunctive_query))
        return Implication(
            disjunctive_query.consequent,
            Disjunction(tuple(disjuncts)),
        )

    def _shatter_conjunction(self, conjunction):
        tagger = QueryEasyShatteringTagger()
        tagged_conjunction = tagger.walk(conjunction)
        return self.walk(tagged_conjunction)

    @add_match(Shatter(ProbabilisticFactSet, ...))
    def easy_shatter_probfact(self, shatter):
        pred_symb = shatter.functor.relation
        cache_key = (pred_symb,)
        for i, arg in enumerate(shatter.args):
            if isinstance(arg, Constant):
                cache_key += (arg,)
            else:
                cache_key += (i,)
        if cache_key in self._cached:
            new_pred_symb = self._cached[cache_key]
        else:
            new_pred_symb = Symbol.fresh()
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
                if str(c) != str(shatter.functor.probability_column.value)
            )
            proj_cols = (shatter.functor.probability_column.value,) + tuple(
                non_prob_columns[i]
                for i, arg in enumerate(shatter.args)
                if not isinstance(arg, Constant)
            )
            new_relation = new_relation.projection(*proj_cols)
            self.symbol_table[new_pred_symb] = Constant[AbstractSet](
                new_relation
            )
            self._cached[cache_key] = new_pred_symb
        new_tagged = ProbabilisticFactSet(
            new_pred_symb, shatter.functor.probability_column
        )
        non_const_args = tuple(
            arg for arg in shatter.args if not isinstance(arg, Constant)
        )
        return FunctionApplication(new_tagged, non_const_args)


def shatter_easy_probfacts(query, symbol_table):
    """
    Remove constants occurring in a given query, possibly removing self-joins.

    A query containing self-joins can be "easily" shattered whenever the
    predicates in the self-joins do not have more than one variable occurring
    in the same term in multiple predicates (e.g. `P(x), P(y)`, both `x` and
    `y` occurr in the same term in both predicates) or the same constant
    occurring in the same term in multiple predicates (e.g. `P(a, x), P(a, b)`,
    `a` occurrs in the same term in both predicates).

    If there is a self-join, the self-joined relation is split into multiple
    relations. These relations are added in-place to the symbol table. The
    returned equivalent query makes use of these relations.

    Parameters
    ----------
    query : Implication
        A conjunctive query (the body can be a single predicate).
    symbol_table : mapping, mutable
        Symbol table containing relations associated with the query's
        relational symbols. This `symbol_table` can be modified in-place by
        this function to add newly generated relational symbols and their
        associated relations.

    Returns
    -------
    Implication
        An equivalent conjunctive query without constants.

    """
    dnf_query_antecedent = convert_to_dnf_ucq(query.antecedent)
    dnf_query_antecedent = RemoveDuplicatedConjunctsDisjuncts().walk(
        dnf_query_antecedent
    )
    dnf_query = Implication(query.consequent, dnf_query_antecedent)
    tagged_set_query = query_to_tagged_set_representation(
        dnf_query, symbol_table
    )
    shatterer = EasyQueryShatterer(symbol_table)
    shattered = shatterer.walk(tagged_set_query)
    shattered = RemoveTrivialOperations().walk(shattered)
    return shattered
