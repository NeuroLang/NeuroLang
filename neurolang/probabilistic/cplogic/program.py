import typing

from ...exceptions import NeuroLangException
from ...expression_walker import ExpressionWalker
from ...expression_pattern_matching import add_match
from ...expressions import Symbol, Constant, Unknown, ExpressionBlock
from ...logic import Implication
from ...datalog import DatalogProgram, WrappedRelationalAlgebraSet
from ..expression_processing import (
    is_probabilistic_fact,
    is_existential_probabilistic_fact,
    check_probchoice_probs_sum_to_one,
    check_existential_probfact_validity,
    extract_probfact_or_eprobfact_pred_symb,
    concatenate_to_expression_block,
    group_probfacts_by_pred_symb,
    build_pfact_set,
)


class CPLogicProgram(DatalogProgram, ExpressionWalker):
    """
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    """

    pfact_pred_symbs_symb = Symbol("__pfact_pred_symbs__")
    pchoice_pred_symbs_symb = Symbol("__pchoice_pred_symbs__")

    @property
    def predicate_symbols(self):
        return (
            set(self.intensional_database())
            | set(self.extensional_database())
            | set(self.pfact_pred_symbs)
            | set(self.pchoice_pred_symbs)
        )

    @property
    def pfact_pred_symbs(self):
        return self.symbol_table.get(
            self.pfact_pred_symbs_symb, Constant[typing.AbstractSet](set())
        ).value

    @property
    def pchoice_pred_symbs(self):
        return self.symbol_table.get(
            self.pchoice_pred_symbs_symb, Constant[typing.AbstractSet](set())
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

    def add_probfacts_from_tuples(self, symbol, iterable, type_=Unknown):
        self._register_probfact_pred_symb(symbol)
        if type_ is Unknown:
            type_, iterable = self.infer_iterable_type(iterable)
        self._check_iterable_prob_type(type_)
        constant = Constant[typing.AbstractSet[type_]](
            self.new_probability_set(list(iterable)),
            auto_infer_type=False,
            verify_type=False,
        )
        symbol = symbol.cast(constant.type)
        self.symbol_table[symbol] = constant

    def add_probchoice_from_tuples(self, symbol, iterable, type_=Unknown):
        """
        Add a probabilistic choice to the symbol table.

        """
        self._register_probchoice_pred_symb(symbol)
        if type_ is Unknown:
            type_, iterable = self.infer_iterable_type(iterable)
        self._check_iterable_prob_type(type_)
        if symbol in self.symbol_table:
            raise NeuroLangException("Symbol already used")
        ra_set = Constant[typing.AbstractSet](
            self.new_probability_set(list(iterable)),
            auto_infer_type=False,
            verify_type=False,
        )
        check_probchoice_probs_sum_to_one(ra_set)
        self.symbol_table[symbol] = ra_set

    @staticmethod
    def new_probability_set(iterable=None):
        return WrappedRelationalAlgebraSet(iterable=iterable)

    @staticmethod
    def _check_iterable_prob_type(iterable_type):
        if not (
            issubclass(iterable_type.__origin__, typing.Tuple)
            and iterable_type.__args__[0] is float
        ):
            raise NeuroLangException(
                "Expected tuples to have a probability as their first element"
            )

    @add_match(ExpressionBlock)
    def program_code(self, code):
        probfacts, other_expressions = group_probfacts_by_pred_symb(code)
        for pred_symb, pfacts in probfacts.items():
            self._register_probfact_pred_symb(pred_symb)
            if pred_symb in self.symbol_table:
                raise NeuroLangException(
                    "Probabilistic fact predicate symbol already seen"
                )
            if len(pfacts) > 1:
                self.symbol_table[pred_symb] = build_pfact_set(
                    pred_symb, pfacts
                )
            else:
                self.walk(list(pfacts)[0])
        super().process_expression(ExpressionBlock(other_expressions))

    def _register_probfact_pred_symb(self, pred_symb):
        self.protected_keywords.add(self.pfact_pred_symbs_symb.name)
        self.symbol_table[self.pfact_pred_symbs_symb] = Constant[
            typing.AbstractSet
        ](self.pfact_pred_symbs | {pred_symb})

    def _register_probchoice_pred_symb(self, pred_symb):
        self.protected_keywords.add(self.pchoice_pred_symbs_symb.name)
        self.symbol_table[self.pchoice_pred_symbs_symb] = Constant[
            typing.AbstractSet
        ](self.pchoice_pred_symbs | {pred_symb})

    @add_match(
        Implication,
        lambda exp: is_probabilistic_fact(exp)
        or is_existential_probabilistic_fact(exp),
    )
    def probfact_or_existential_probfact(self, expression):
        if is_existential_probabilistic_fact(expression):
            check_existential_probfact_validity(expression)
        pred_symb = extract_probfact_or_eprobfact_pred_symb(expression)
        if pred_symb not in self.symbol_table:
            self.symbol_table[pred_symb] = ExpressionBlock(tuple())
        self.symbol_table[pred_symb] = concatenate_to_expression_block(
            self.symbol_table[pred_symb], [expression]
        )
        return expression

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if k in self.pfact_pred_symbs
        }
