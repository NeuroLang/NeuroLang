import typing
from operator import floordiv

from ...datalog import DatalogProgram
from ...datalog.basic_representation import UnionOfConjunctiveQueries
from ...exceptions import ForbiddenDisjunctionError, ForbiddenExpressionError
from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker, PatternWalker
from ...expressions import Constant, FunctionApplication, Symbol
from ...logic import Implication, Union
from ...type_system import get_generic_type
from ..exceptions import (
    ForbiddenConditionalQueryNoProb,
    MalformedProbabilisticTupleError,
    UnsupportedProbabilisticQueryError,
)
from ..expression_processing import (
    add_to_union,
    build_probabilistic_fact_set,
    check_probabilistic_choice_set_probabilities_sum_to_one,
    get_within_language_prob_query_prob_term,
    group_probabilistic_facts_by_pred_symb,
    is_probabilistic_fact,
    is_within_language_prob_query,
    union_contains_probabilistic_facts,
)
from ..expressions import PROB, Condition, ProbabilisticQuery


def is_within_language_prob_query_wannabe(expression):
    return (
        isinstance(expression, FunctionApplication)
        and get_generic_type(type(expression)) is FunctionApplication
        and expression.functor == PROB
    )


class TranslateProbabilisticQueryMixin(PatternWalker):
    @add_match(
        Implication(
            ..., FunctionApplication(Constant[typing.Any](floordiv), ...)
        )
    )
    def conditional_query(self, implication):
        new_implication = Implication(
            implication.consequent, Condition(*implication.antecedent.args)
        )
        if not (
            any(
                isinstance(arg, ProbabilisticQuery)
                or is_within_language_prob_query_wannabe(arg)
                for arg in new_implication.consequent.args
            )
        ):
            raise ForbiddenExpressionError(
                "Missing probabilistic term in consequent's head"
            )
        if (
            sum(
                isinstance(arg, ProbabilisticQuery)
                for arg in new_implication.consequent.args
            )
            > 1
        ):
            raise ForbiddenExpressionError(
                "Can only contain one probabilistic term in consequent's head"
            )
        return self.walk(new_implication)

    @add_match(
        Implication,
        lambda implication: any(
            is_within_language_prob_query_wannabe(arg)
            for arg in implication.consequent.args
        ),
    )
    def within_language_prob_query(self, implication):
        csqt_args = tuple()
        for arg in implication.consequent.args:
            if is_within_language_prob_query_wannabe(arg):
                arg = ProbabilisticQuery(*arg.unapply())
            csqt_args += (arg,)
        consequent = implication.consequent.functor(*csqt_args)
        return self.walk(Implication(consequent, implication.antecedent))


class CPLogicMixin(PatternWalker):
    """
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    """

    pfact_pred_symb_set_symb = Symbol("__pfact_pred_symb_set_symb__")
    pchoice_pred_symb_set_symb = Symbol("__pchoice_pred_symb_set_symb__")
    protected_keywords = {"PROB"}

    @property
    def predicate_symbols(self):
        return (
            set(self.intensional_database())
            | set(self.extensional_database())
            | set(self.pfact_pred_symbs)
            | set(self.pchoice_pred_symbs)
        )

    @property
    def probabilistic_predicate_symbols(self):
        return (
            self.pfact_pred_symbs
            | self.pchoice_pred_symbs
            | set(self.within_language_succ_queries())
        )

    @property
    def pfact_pred_symbs(self):
        return self._get_pred_symbs(self.pfact_pred_symb_set_symb)

    @property
    def pchoice_pred_symbs(self):
        return self._get_pred_symbs(self.pchoice_pred_symb_set_symb)

    def within_language_succ_queries(self):
        return {
            pred_symb: union.formulas[0]
            for pred_symb, union in self.intensional_database().items()
            if len(union.formulas) == 1
            and is_within_language_prob_query(union.formulas[0])
        }

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if k in self.pfact_pred_symbs
        }

    def probabilistic_choices(self):
        """Return probabilistic choices of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if k in self.pchoice_pred_symbs
        }

    def _get_pred_symbs(self, set_symb):
        return self.symbol_table.get(
            set_symb, Constant[typing.AbstractSet](set())
        ).value

    def extensional_database(self):
        exclude = (
            self.protected_keywords
            | self.pfact_pred_symbs
            | self.pchoice_pred_symbs
        )
        ret = self.symbol_table.symbols_by_type(typing.AbstractSet)
        for keyword in exclude:
            if keyword in ret:
                del ret[keyword]
        return ret

    def add_probabilistic_facts_from_tuples(self, symbol, iterable):
        """
        Add probabilistic facts from tuples whose first element
        contains the probability label attached to that tuple.

        Examples
        --------
        The following

        >>> P = Symbol('P')
        >>> tuples = {(0.9, 'a'), (0.8, 'b')}
        >>> program.add_probabilistic_facts_from_tuples(P, tuples)

        adds the probabilistic facts

            P(a) : 0.9  <-  T
            P(b) : 0.8  <-  T

        to the program.

        """
        self._register_prob_pred_symb_set_symb(
            symbol, self.pfact_pred_symb_set_symb
        )
        type_, iterable = self.infer_iterable_type(iterable)
        self._check_iterable_prob_type(type_)
        constant = Constant[typing.AbstractSet[type_]](
            self.new_set(iterable), auto_infer_type=False, verify_type=False
        )
        symbol = symbol.cast(constant.type)
        self.symbol_table[symbol] = constant

    def add_probabilistic_choice_from_tuples(self, symbol, iterable):
        """
        Add a probabilistic choice from a predicate symbol and a set of
        tuples where the first element is the probability label
        attached to a head predicate and the other elements are the
        constant terms of the head predicate.

        Examples
        --------
        The following

        >>> P = Symbol('P')
        >>> tuples = {(0.2, 'a'), (0.8, 'b')}
        >>> program.add_probabilistic_choice_from_tuples(P, tuples)

        adds the probabilistic choice

            P(a) : 0.2  v  P(b) : 0.8  <-  T

        to the program.

        """
        self._register_prob_pred_symb_set_symb(
            symbol, self.pchoice_pred_symb_set_symb
        )
        type_, iterable = self.infer_iterable_type(iterable)
        self._check_iterable_prob_type(type_)
        if symbol in self.symbol_table:
            raise ForbiddenDisjunctionError(
                "Cannot define multiple probabilistic choices with the same "
                f"predicate symbol. Predicate symbol was: {symbol}"
            )
        ra_set = Constant[typing.AbstractSet[type_]](
            self.new_set(iterable), auto_infer_type=False, verify_type=False
        )
        check_probabilistic_choice_set_probabilities_sum_to_one(ra_set)
        self.symbol_table[symbol] = ra_set

    @staticmethod
    def _check_iterable_prob_type(iterable_type):
        if not (
            issubclass(iterable_type.__origin__, typing.Tuple)
            and iterable_type.__args__[0] is float
        ):
            raise MalformedProbabilisticTupleError(
                "Expected tuples to have a probability as their first element"
            )

    @add_match(Union, union_contains_probabilistic_facts)
    def union_with_probabilistic_facts(self, code):
        pfacts, other_expressions = group_probabilistic_facts_by_pred_symb(
            code
        )
        for pred_symb, pfacts in pfacts.items():
            self._register_prob_pred_symb_set_symb(
                pred_symb, self.pfact_pred_symb_set_symb
            )
            if len(pfacts) > 1:
                self.symbol_table[pred_symb] = build_probabilistic_fact_set(
                    pred_symb, pfacts
                )
            else:
                self.walk(list(pfacts)[0])
        self.walk(Union(other_expressions))

    def _register_prob_pred_symb_set_symb(self, pred_symb, set_symb):
        if set_symb.name not in self.protected_keywords:
            self.protected_keywords.add(set_symb.name)
        if set_symb not in self.symbol_table:
            self.symbol_table[set_symb] = Constant[typing.AbstractSet](set())
        self.symbol_table[set_symb] = Constant[typing.AbstractSet](
            self.symbol_table[set_symb].value | {pred_symb}
        )

    @add_match(Implication, is_probabilistic_fact)
    def probabilistic_fact(self, expression):
        pred_symb = expression.consequent.body.functor
        if pred_symb not in self.symbol_table:
            self.symbol_table[pred_symb] = Union(tuple())
        self.symbol_table[pred_symb] = add_to_union(
            self.symbol_table[pred_symb], [expression]
        )
        return expression

    @add_match(Implication(..., Condition), is_within_language_prob_query)
    def within_language_marg_query(self, implication):
        self._validate_within_language_marg_query(implication)
        pred_symb = implication.consequent.functor.cast(
            UnionOfConjunctiveQueries
        )
        if pred_symb in self.symbol_table:
            raise ForbiddenDisjunctionError(
                "Disjunctive within-language probabilistic queries "
                "are not allowed"
            )
        self.symbol_table[pred_symb] = Union((implication,))
        return implication

    @add_match(Implication(..., Condition))
    def marg_implication(self, implication):
        raise ForbiddenConditionalQueryNoProb(
            "Conditional queries which don't have PROB in "
            "the head (within-language probabilistic) "
            "are forbidden"
        )

    @add_match(Implication, is_within_language_prob_query)
    def within_language_succ_query(self, implication):
        self._validate_within_language_succ_query(implication)
        pred_symb = implication.consequent.functor.cast(
            UnionOfConjunctiveQueries
        )
        if pred_symb in self.symbol_table:
            raise ForbiddenDisjunctionError(
                "Disjunctive within-language probabilistic queries "
                "are not allowed"
            )
        self.symbol_table[pred_symb] = Union((implication,))
        return implication

    @staticmethod
    def _validate_within_language_succ_query(implication):
        csqt_vars = set(
            arg
            for arg in implication.consequent.args
            if isinstance(arg, Symbol)
        )
        prob_term = get_within_language_prob_query_prob_term(implication)
        if not prob_term.args:
            raise UnsupportedProbabilisticQueryError(
                "Probabilistic boolean queries are not currently supported"
            )
        if not all(isinstance(arg, Symbol) for arg in prob_term.args):
            bad_vars = (
                repr(arg)
                for arg in prob_term.args
                if not isinstance(arg, Symbol)
            )
            raise ForbiddenExpressionError(
                "All terms in PROB(...) should be variables. "
                "Found these terms: {}".format(", ".join(bad_vars))
            )
        prob_vars = set(prob_term.args)
        if csqt_vars != prob_vars:
            raise ForbiddenExpressionError(
                "Variables of the set-based query and variables in the "
                "PROB(...) term should be the same variables"
            )

    @staticmethod
    def _validate_within_language_marg_query(implication):
        csqt_vars = set(
            arg
            for arg in implication.consequent.args
            if isinstance(arg, Symbol)
        )
        prob_term = get_within_language_prob_query_prob_term(implication)
        if not prob_term.args:
            raise UnsupportedProbabilisticQueryError(
                "Probabilistic boolean queries are not currently supported"
            )
        if not all(isinstance(arg, Symbol) for arg in prob_term.args):
            bad_vars = (
                repr(arg)
                for arg in prob_term.args
                if not isinstance(arg, Symbol)
            )
            raise ForbiddenExpressionError(
                "All terms in PROB(...) should be variables. "
                "Found these terms: {}".format(", ".join(bad_vars))
            )
        prob_vars = set(prob_term.args)
        if csqt_vars != prob_vars:
            raise ForbiddenExpressionError(
                "Variables of the set-based query and variables in the "
                "PROB(...) term should be the same variables"
            )


class CPLogicProgram(CPLogicMixin, DatalogProgram, ExpressionWalker):
    pass
