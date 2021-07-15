"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets.
"""

from itertools import tee
from typing import AbstractSet, Any, Callable, Tuple
from warnings import warn

from ..expression_walker import PatternWalker, add_match
from ..exceptions import ProtectedKeywordError
from ..expressions import (Constant, Expression, FunctionApplication,
                           NeuroLangException, Symbol, TypedSymbolTableMixin,
                           is_leq_informative)
from ..type_system import Unknown, get_args, infer_type
from .expression_processing import (
    extract_logic_free_variables, is_conjunctive_expression,
    is_conjunctive_expression_with_nested_predicates)
from .expressions import (NULL, UNDEFINED, Fact, Implication, NullConstant,
                          Undefined, Union)
from .wrapped_collections import WrappedRelationalAlgebraSet

__all__ = [
    "Implication", "Fact", "Undefined", "NullConstant",
    "UNDEFINED", "NULL", "WrappedRelationalAlgebraSet",
    "DatalogProgram", "UnionOfConjunctiveQueries"
]


class UnionOfConjunctiveQueries:
    pass


class DatalogProgram(TypedSymbolTableMixin, PatternWalker):
    '''
    Implementation of Datalog grammar in terms of
    Intermediate Representations. No query resolution implemented.
    In the symbol table the value for a symbol `S` is implemented as:

    * If `S` is part of the extensional database, then the value of the symbol
    is a set of tuples `a` representing `S(*a)` as facts. The type of the
    symbol must be more informative than `AbstractSet[Tuple]`

    * If `S` is part of the intensional database then its value is an
    `Union` of `Implications`. For instance
    `Q(x) :- R(x, x)` and `Q(x) :- T(x)` is represented as a symbol `Q`
     with value
     `Union((Implication(Q(x), R(x, x)), Implication(Q(x), T(x))))`.
     The type of the symbol `Q` in the symbol_table will be
     `UnionOfConjunctiveQueries`.
    '''

    def __init_subclass__(cls, **kwargs):
        for c in cls.mro()[1:-1]:
            if not hasattr(c, 'protected_keywords'):
                continue
            cls.protected_keywords = (
                cls.protected_keywords |
                c.protected_keywords
            )
        super().__init_subclass__(**kwargs)

    protected_keywords = set()

    def function_equals(self, a: Any, b: Any) -> bool:
        return a == b

    @add_match(Symbol)
    def symbol(self, expression):
        new_expression = self.symbol_table.get(expression, expression)
        if new_expression is expression:
            return expression
        elif isinstance(new_expression, (Union, Constant[AbstractSet])):
            return expression
        else:
            return new_expression

    @add_match(Fact(FunctionApplication(Symbol, ...)))
    def fact(self, expression):
        fact = expression.fact
        if fact.functor.name in self.protected_keywords:
            raise ProtectedKeywordError(
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

        if isinstance(fact_set, Union):
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

        symbol = consequent.functor.cast(UnionOfConjunctiveQueries)

        if symbol in self.symbol_table:
            disj = self._new_intensional_internal_representation(consequent)
        else:
            disj = tuple()

        if expression not in disj:
            disj += (expression,)

        self.symbol_table[symbol] = Union(disj)

        return expression

    def _new_intensional_internal_representation(self, consequent):
        symbol = consequent.functor.cast(UnionOfConjunctiveQueries)
        value = self.symbol_table[symbol]
        if (
            isinstance(value, Constant) and
            is_leq_informative(value.type, AbstractSet)
        ):
            raise NeuroLangException(
                f'{consequent.functor.name} has been previously '
                'defined as Fact or extensional database.'
            )
        disj = self.symbol_table[symbol].formulas

        if (
            not isinstance(disj[0].consequent, FunctionApplication) or
            len(extract_logic_free_variables(disj[0].consequent.args)) !=
            len(consequent.args)
        ):
            raise NeuroLangException(
                f"{disj[0].consequent} is already in the IDB "
                f"with different signature."
            )

        return disj

    def _validate_implication_syntax(self, consequent, antecedent):
        if consequent.functor.name in self.protected_keywords:
            raise ProtectedKeywordError(
                f'symbol {consequent.functor} is protected'
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
    def new_set(iterable=None, row_type=None, verify_row_type=False):
        return WrappedRelationalAlgebraSet(
            iterable=iterable, row_type=row_type,
            verify_row_type=verify_row_type
        )

    def intensional_database(self):
        return {
            k: v for k, v
            in self.symbol_table.items()
            if (
                k not in self.protected_keywords
                and k.type is UnionOfConjunctiveQueries
            )
        }

    def predicate_terms(self, predicate):
        try:
            pred_repr = self.symbol_table[predicate]
            if isinstance(pred_repr, Union):
                head_args = self._predicate_terms_intensional(pred_repr)
                return head_args
            elif is_leq_informative(pred_repr.type, AbstractSet):
                row_type = get_args(pred_repr.type)[0]
                row_len = len(get_args(row_type))
                return tuple(
                    Symbol(str(i))
                    for i in range(row_len)
                )
            else:
                raise NeuroLangException(f'Predicate {predicate} not found')
        except KeyError:
            raise NeuroLangException(f'Predicate {predicate} not found')

    def _predicate_terms_intensional(self, pred_repr):
        head_args = None
        for formula in pred_repr.formulas:
            if head_args is None:
                head_args = formula.consequent.args
            elif head_args != formula.consequent.args:
                warn(
                    'Several argument names found in the rules '
                    f'defining {formula.consequent.functor.name}, keeping one'
                )
                break
        return head_args

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
            new_set = self.new_set(iterable)
            type_ = new_set.row_type
        else:
            new_set = self.new_set(iterable=iterable, row_type=type_)

        constant = Constant[AbstractSet[type_]](
            new_set,
            auto_infer_type=False,
            verify_type=False
        )
        symbol = symbol.cast(constant.type)
        self.symbol_table[symbol] = constant

    @staticmethod
    def infer_iterable_type(iterable):
        """Infer the type of iterable elements
        without modifying the iterable.

        Parameters
        ----------
        iterable : Iterable
            iterable from which to infer
            element's type

        Returns
        -------
        type, iterable
            the inferred type and the iterable.
        """
        type_ = Unknown
        if hasattr(iterable, 'fetch_one'):
            if iterable.is_empty():
                first = None
            else:
                first = iterable.fetch_one()
                if first == tuple():
                    first = None
        else:
            iterable_, iterable = tee(iterable)
            try:
                first = next(iterable_)
            except StopIteration:
                first = None
        if isinstance(first, Expression):
            type_ = first.type
        elif first is not None:
            type_ = infer_type(first)
        return type_, iterable
