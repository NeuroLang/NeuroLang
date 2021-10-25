import collections
import itertools
import operator

from ..datalog.expression_processing import (
    extract_logic_free_variables,
    extract_logic_predicates,
)
from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker, ReplaceExpressionWalker
from ..expressions import Constant, Definition, FunctionApplication, Symbol
from ..logic import Conjunction, ExistentialPredicate, Implication, Negation
from ..logic.transformations import (
    RemoveDuplicatedConjunctsDisjuncts,
    RemoveTrivialOperations,
)
from ..relational_algebra import (
    NameColumns,
    NaturalJoin,
    Projection,
    Selection,
    int2columnint_constant,
    str2columnstr_constant,
)
from ..relational_algebra_provenance import (
    IndependentProjection,
    ProvenanceAlgebraSet,
)
from .exceptions import NotEasilyShatterableError
from .probabilistic_ra_utils import ProbabilisticFactSet
from .transforms import convert_to_dnf_ucq

EQ = Constant(operator.eq)
NE = Constant(operator.ne)


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


def _extract_shatterable_negexist_selfjoin_ixs(conjunction):
    (
        pfact_pred_ixs,
        pfact_preds,
    ) = _extract_pfact_positive_literal_conjuncts(conjunction)
    shatterable = list()
    for conjunct_ix, conjunct in enumerate(conjunction.formulas):
        tmp = _extract_neg_existential_with_single_pfact_and_not_equal(
            conjunct
        )
        if tmp is None:
            continue
        (
            pfact_pred,
            var2var_noteq,
            _,
        ) = tmp
        matching_pred_ixs = [
            pred_ix
            for pred_ix, pred in zip(pfact_pred_ixs, pfact_preds)
            if pred.functor == pfact_pred.functor
        ]
        if len(matching_pred_ixs) != 1:
            continue
        matching_pred_ix = matching_pred_ixs[0]
        matching_pred = conjunction.formulas[matching_pred_ix]
        eqvar_ix = list(pfact_pred.args).index(var2var_noteq[0])
        extract_logic_free_variables
        if matching_pred.args[eqvar_ix] != var2var_noteq[1]:
            continue
        if not _pred_only_occurs_in_shatterable_selfjoin(
            matching_pred_ix, conjunct_ix, conjunction
        ):
            continue
        shatterable.append((matching_pred_ix, conjunct_ix))
    return shatterable


def _pred_only_occurs_in_shatterable_selfjoin(
    pred_ix, negexist_ix, conjunction
):
    for ix, conjunct in enumerate(conjunction.formulas):
        if ix in (pred_ix, negexist_ix):
            continue
        if any(
            pred.functor == conjunction.formulas[pred_ix].functor
            for pred in extract_logic_predicates(conjunct)
        ):
            return False
    return True


def _extract_neg_existential_with_single_pfact_and_not_equal(expression):
    if not isinstance(expression, Negation):
        return
    expression = expression.formula
    neg_existential_vars = set()
    while isinstance(expression, ExistentialPredicate):
        neg_existential_vars.add(expression.head)
        expression = expression.body
    if not isinstance(expression, Conjunction) and all(
        isinstance(conjunct, FunctionApplication)
        for conjunct in expression.formulas
    ):
        return
    _, pfact_preds = _extract_pfact_positive_literal_conjuncts(expression)
    if len(pfact_preds) != 1:
        return
    pfact_pred = next(iter(pfact_preds))
    var2var_noteqs = set(
        tuple(fa.args) for fa in expression.formulas if fa.functor == NE
    )
    if len(var2var_noteqs) != 1:
        return
    var2var_noteq = next(iter(var2var_noteqs))
    if (
        var2var_noteq[0] not in neg_existential_vars
        and var2var_noteq[1] in neg_existential_vars
    ):
        var2var_noteq = tuple(reversed(var2var_noteq))
    if var2var_noteq[0] not in neg_existential_vars:
        return
    return pfact_pred, var2var_noteq, neg_existential_vars


def _extract_pfact_positive_literal_conjuncts(conjunction):
    ixs = list()
    preds = list()
    for ix, conjunct in enumerate(conjunction.formulas):
        if isinstance(conjunct, FunctionApplication) and isinstance(
            conjunct.functor, ProbabilisticFactSet
        ):
            ixs.append(ix)
            preds.append(conjunct)
    return ixs, preds


class ShatterNegExistentialProbFactSelfJoin(Definition):
    def __init__(self, pfact_literal, negexist_conjunct):
        self.pfact_literal = pfact_literal
        self.negexist_conjunct = negexist_conjunct


class Shatterer(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table
        self._cached_args = collections.defaultdict(set)
        self._cached = dict()

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
            columns = [
                int2columnint_constant(c)
                for c in self.symbol_table[
                    shatter.functor.relation
                ].value.columns
            ]
            new_relation = shatter.functor.relation
            for i in const_idxs:
                new_relation = Selection(
                    new_relation, EQ(columns[i + 1], shatter.args[i])
                )
            non_prob_columns = tuple(
                c for c in columns if c != shatter.functor.probability_column
            )
            proj_cols = (shatter.functor.probability_column,) + tuple(
                non_prob_columns[i]
                for i, arg in enumerate(shatter.args)
                if not isinstance(arg, Constant)
            )
            new_relation = Projection(new_relation, proj_cols)
            self.symbol_table[new_pred_symb] = new_relation
            self._cached[cache_key] = new_pred_symb
        new_tagged = ProbabilisticFactSet(
            new_pred_symb, shatter.functor.probability_column
        )
        non_const_args = tuple(
            arg for arg in shatter.args if not isinstance(arg, Constant)
        )
        return FunctionApplication(new_tagged, non_const_args)

    @add_match(Conjunction, _extract_shatterable_negexist_selfjoin_ixs)
    def tag_shatterable_negexist_selfjoin(self, conjunction):
        shatterable = _extract_shatterable_negexist_selfjoin_ixs(conjunction)
        new_conjuncts = list(
            conjunction.formulas[ix]
            for ix in range(len(conjunction.formulas))
            if ix not in set.union(*(set(s) for s in shatterable))
        )
        for pred_ix, negexist_ix in shatterable:
            new_conjuncts.append(
                ShatterNegExistentialProbFactSelfJoin(
                    conjunction.formulas[pred_ix],
                    conjunction.formulas[negexist_ix],
                )
            )
        return self.walk(Conjunction(tuple(new_conjuncts)))

    @add_match(ShatterNegExistentialProbFactSelfJoin)
    def shatterable_negexistential_probfact_selfjoin(self, shatter):
        pfact_literal = shatter.pfact_literal
        (
            negexist_pfact_literal,
            var2var_noteq,
            _,
        ) = _extract_neg_existential_with_single_pfact_and_not_equal(
            shatter.negexist_conjunct
        )
        relation = pfact_literal.functor.relation
        prob_col = str2columnstr_constant(Symbol.fresh())
        cols_a = (prob_col,) + tuple(
            str2columnstr_constant(arg) for arg in pfact_literal.args
        )
        relation_a = NameColumns(relation, cols_a)
        proj_cols = tuple(
            str2columnstr_constant(arg) for arg in pfact_literal.args
        )
        relation_a = Projection(relation_a, proj_cols)
        cols_b = (prob_col,) + tuple(
            str2columnstr_constant(arg) for arg in negexist_pfact_literal.args
        )
        relation_b = NameColumns(relation, cols_b)
        new_relation = NaturalJoin(relation_a, relation_b)
        selection_criteria = NE(
            str2columnstr_constant(var2var_noteq[0]),
            str2columnstr_constant(var2var_noteq[1]),
        )
        new_relation = Selection(new_relation, selection_criteria)
        new_pred_symb = Symbol.fresh()
        self.symbol_table[new_pred_symb] = new_relation
        new_tagged = ProbabilisticFactSet(new_pred_symb, prob_col)
        new_args = tuple(
            sorted(set(pfact_literal.args) | set(negexist_pfact_literal.args))
        )
        return FunctionApplication(new_tagged, new_args)

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
        return self.walk(Shatter(*function_application.unapply()))

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
    tagged_query = query_to_tagged_set_representation(dnf_query, symbol_table)
    shatterer = Shatterer(symbol_table)
    shattered = shatterer.walk(tagged_query)
    shattered = RemoveTrivialOperations().walk(shattered)
    return shattered
