import typing

from ...datalog import DatalogProgram
from ...datalog.basic_representation import UnionOfConjunctiveQueries
from ...datalog.negation import DatalogProgramNegationMixin
from ...exceptions import ForbiddenDisjunctionError, ForbiddenExpressionError
from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker, PatternWalker
from ...expressions import Constant, Symbol
from ...logic import TRUE, Implication, Union
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
    is_within_language_prob_query,
    union_contains_probabilistic_facts,
)
from ..expressions import Condition, ProbabilisticFact, ProbabilisticChoice, ProbabilisticPredicate


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
            | set(self.within_language_prob_queries())
        )

    @property
    def pfact_pred_symbs(self):
        return self._get_pred_symbs(self.pfact_pred_symb_set_symb)

    @property
    def pchoice_pred_symbs(self):
        return self._get_pred_symbs(self.pchoice_pred_symb_set_symb)

    def within_language_prob_queries(self):
        return {
            pred_symb: union.formulas[0]
            for pred_symb, union in self.intensional_database().items()
            if not isinstance(union, Constant) and len(union.formulas) == 1
            and is_within_language_prob_query(union.formulas[0])
        }

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: self.symbol_table[k]
            for k in self.pfact_pred_symbs
        }

    def probabilistic_choices(self):
        """Return probabilistic choices of the symbol table."""
        return {
            k: self.symbol_table[k]
            for k in self.pchoice_pred_symbs
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
        if not hasattr(iterable, '__len__') or len(iterable) > 0:
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
        self.symbol_table[symbol.cast(typing.AbstractSet[ra_set.value.row_type])] = ra_set

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

    @add_match(Implication(ProbabilisticFact, TRUE))
    def probabilistic_fact(self, expression):
        pred_symb = expression.consequent.body.functor
        self._register_prob_pred_symb_set_symb(
            pred_symb, self.pfact_pred_symb_set_symb
        )
        if pred_symb not in self.symbol_table:
            self.symbol_table[pred_symb] = Union(tuple())
        elif isinstance(
            self.symbol_table[pred_symb], Constant[typing.AbstractSet]
        ):
            raise ForbiddenDisjunctionError(
                "Probabilistic facts cannot be defined both from sets and "
                "from rules"
            )
        self.symbol_table[pred_symb] = add_to_union(
            self.symbol_table[pred_symb], [expression]
        )
        return expression

    @add_match(Implication(ProbabilisticChoice, ...))
    def query_based_probabilistic_choice(self, implication):
        """
        Construct probabilistic choices from deterministic queries.

        This extends the syntax with rules such as

            Or_x P(x) : f(x) :- Q(x)

        where x is a set of variables, f(x) is an arithmetic expression
        yielding a probability between [0, 1] that may use built-ins, and where
        Q(x) is a conjunction of predicates. The sum of f(x) for all x must be
        lower than 1.

        Only deterministic antecedents are allowed. Declarativity makes it
        impossible to enforce that at declaration time. Thus, if a query-based
        probabilistic fact has a dependency on a probabilistic predicate, this
        will be discovered at query-resolution time, after the program has been
        fully declared.

        """
        pred_symb = implication.consequent.body.functor
        pred_symb = pred_symb.cast(UnionOfConjunctiveQueries)
        self._register_prob_pred_symb_set_symb(
            pred_symb, self.pchoice_pred_symb_set_symb
        )
        if pred_symb in self.symbol_table:
            raise ForbiddenDisjunctionError(
                "Probabilistic choice {} already defined".format(pred_symb)
            )
        self.symbol_table[pred_symb] = Union((implication,))
        return implication

    @add_match(Implication(ProbabilisticFact, ...))
    def query_based_probabilistic_fact(self, implication):
        """
        Construct probabilistic facts from deterministic queries.

        This extends the syntax with rules such as

            P(x) : f(x) :- Q(x)

        where x is a set of variables, f(x) is an arithmetic expression
        yielding a probability between [0, 1] that may use built-ins, and where
        Q(x) is a conjunction of predicates.

        Only deterministic antecedents are allowed. Declarativity makes it
        impossible to enforce that at declaration time. Thus, if a query-based
        probabilistic fact has a dependency on a probabilistic predicate, this
        will be discovered at query-resolution time, after the program has been
        fully declared.

        """
        pred_symb = implication.consequent.body.functor
        pred_symb = pred_symb.cast(UnionOfConjunctiveQueries)
        self._register_prob_pred_symb_set_symb(
            pred_symb, self.pfact_pred_symb_set_symb
        )
        if pred_symb in self.symbol_table:
            raise ForbiddenDisjunctionError(
                "Probabilistic predicate {} already defined".format(pred_symb)
            )
        self.symbol_table[pred_symb] = Union((implication,))
        return implication

    @add_match(Implication(ProbabilisticPredicate, ...))
    def query_based_probabilistic_predicate(self, implication):
        """
            Left for backward compatibility. Should be referred to the
            ProbabilisticFact case

        """
        return self.query_based_probabilistic_fact(implication)

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


class CPLogicProgram(
    CPLogicMixin, DatalogProgramNegationMixin,
    DatalogProgram, ExpressionWalker
):
    pass
