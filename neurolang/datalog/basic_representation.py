"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets.
"""

from typing import AbstractSet, Any, Callable, Tuple

from ..expression_walker import PatternWalker, add_match
from ..expressions import (Constant, Expression, FunctionApplication,
                           NeuroLangException, Symbol, TypedSymbolTableMixin,
                           is_leq_informative)
from ..type_system import Unknown, infer_type
from .expression_processing import (
    extract_datalog_free_variables, is_conjunctive_expression,
    is_conjunctive_expression_with_nested_predicates)
from .expressions import (NULL, UNDEFINED, Disjunction, Fact, Implication,
                          NullConstant, Undefined)
from .wrapped_collections import WrappedRelationalAlgebraSet
from itertools import tee


__all__ = [
    "Implication", "Fact", "Undefined", "NullConstant",
    "UNDEFINED", "NULL", "WrappedRelationalAlgebraSet",
    "DatalogProgram"
]


class DatalogProgram(TypedSymbolTableMixin, PatternWalker):
    '''
    Implementation of Datalog grammar in terms of
    Intermediate Representations. No query resolution implemented.
    In the symbol table the value for a symbol `S` is implemented as:

    * If `S` is part of the extensional database, then the value of the symbol
    is a set of tuples `a` representing `S(*a)` as facts

    * If `S` is part of the intensional database then its value is an
    `Disjunction` of `Implications`. For instance
    `Q(x) :- R(x, x)` and `Q(x) :- T(x)` is represented as a symbol `Q`
     with value
     `Disjunction((Implication(Q(x), R(x, x)), Implication(Q(x), T(x))))`
    '''

    protected_keywords = set()

    def function_equals(self, a: Any, b: Any) -> bool:
        return a == b

    @add_match(Symbol)
    def symbol(self, expression):
        new_expression = self.symbol_table.get(expression, expression)
        if new_expression is expression:
            return expression
        elif isinstance(new_expression, (Disjunction, Constant[AbstractSet])):
            return expression
        else:
            return new_expression

    @add_match(Fact(FunctionApplication[bool](Symbol, ...)))
    def fact(self, expression):
        fact = expression.fact
        if fact.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if any(
            not isinstance(a, Constant)
            for a in fact.args
        ):
            raise NeuroLangException(
                'Facts can only have constants as arguments'
            )

        self._initialize_fact_set_if_needed(fact)
        fact_set = self.symbol_table[fact.functor]

        if isinstance(fact_set, Disjunction):
            raise NeuroLangException(
                f'{fact.functor} has been previously '
                'define as intensional predicate.'
            )

        fact_set.value.add(fact.args)

        return expression

    def _initialize_fact_set_if_needed(self, fact):
        if fact.functor not in self.symbol_table:
            if fact.functor.type is Unknown:
                c = Constant(fact.args)
                set_type = c.type
            elif isinstance(fact.functor.type, Callable):
                set_type = Tuple[fact.functor.type.__args__[:-1]]
            else:
                raise NeuroLangException('Fact functor type incorrect')

            self.symbol_table[fact.functor] = \
                Constant[AbstractSet[set_type]](
                    WrappedRelationalAlgebraSet(),
                    verify_type=False
                )

    @add_match(Implication(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._validate_implication_syntax(consequent, antecedent)

        if consequent.functor in self.symbol_table:
            disj = self._new_intensional_internal_representation(consequent)
        else:
            disj = tuple()

        if expression not in disj:
            disj += (expression,)

        self.symbol_table[consequent.functor] = Disjunction(disj)

        return expression

    def _new_intensional_internal_representation(self, consequent):
        value = self.symbol_table[consequent.functor]
        if (
            isinstance(value, Constant) and
            is_leq_informative(value.type, AbstractSet)
        ):
            raise NeuroLangException(
                f'{consequent.functor.name} has been previously '
                'defined as Fact or extensional database.'
            )
        disj = self.symbol_table[consequent.functor].formulas

        if (
            not isinstance(disj[0].consequent, FunctionApplication) or
            len(extract_datalog_free_variables(disj[0].consequent.args)) !=
            len(consequent.args)
        ):
            raise NeuroLangException(
                f"{disj[0].consequent} is already in the IDB "
                f"with different signature."
            )

        return disj

    def _validate_implication_syntax(self, consequent, antecedent):
        if consequent.functor in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if any(
            not isinstance(arg, (Constant, Symbol))
            for arg in consequent.args
        ):
            raise NeuroLangException(
                f'The consequent {consequent} can only be '
                'constants or symbols'
            )

        consequent_symbols = consequent._symbols - consequent.functor._symbols

        if not consequent_symbols.issubset(antecedent._symbols):
            raise NeuroLangException(
                "All variables on the consequent need to be on the antecedent"
            )

        if not is_conjunctive_expression(consequent):
            raise NeuroLangException(
                f'Expression {consequent} is not conjunctive'
            )

        if not is_conjunctive_expression_with_nested_predicates(antecedent):
            raise NeuroLangException(
                f'Expression {antecedent} is not conjunctive'
            )

    @staticmethod
    def new_set(iterable=None):
        return WrappedRelationalAlgebraSet(iterable=iterable)

    def intensional_database(self):
        return {
            k: v for k, v in self.symbol_table.items()
            if (
                k not in self.protected_keywords and
                isinstance(v, Disjunction)
            )
        }

    def extensional_database(self):
        ret = self.symbol_table.symbols_by_type(AbstractSet)
        for keyword in self.protected_keywords:
            if keyword in ret:
                del ret[keyword]
        return ret

    def builtins(self):
        return self.symbol_table.symbols_by_type(Callable)

    def add_extensional_predicate_from_tuples(
        self, symbol, iterable, type_=Unknown
    ):
        if type_ is Unknown:
            type_ = self.infer_iterable_type(iterable)

        constant = Constant[AbstractSet[type_]](
            self.new_set(list(iterable)),
            auto_infer_type=False,
            verify_type=False
        )
        symbol = symbol.cast(constant.type)
        self.symbol_table[symbol] = constant

    @staticmethod
    def infer_iterable_type(iterable):
        type_ = Unknown
        try:
            iterable_, _ = tee(iterable)
            first = next(iterable_)
            if isinstance(first, Expression):
                type_ = first.type
            else:
                type_ = infer_type(first)
        except StopIteration:
            pass
        return type_
