"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets.
"""

from itertools import tee
from typing import AbstractSet, Any, Callable, Tuple

from ..expression_walker import (PatternWalker, add_match)
from ..expressions import (Constant, Expression, ExpressionBlock,
                           FunctionApplication, NeuroLangException, Symbol,
                           is_leq_informative)
from ..type_system import Unknown, infer_type
from ..utils import RelationalAlgebraSet
from .expression_processing import (
    extract_datalog_free_variables, is_conjunctive_expression,
    is_conjunctive_expression_with_nested_predicates)
from .expressions import (NULL, UNDEFINED, Fact, Implication, NullConstant,
                          Undefined)

__all__ = [
    "Implication", "Fact", "Undefined", "NullConstant",
    "UNDEFINED", "NULL", "WrappedRelationalAlgebraSet",
    "DatalogProgram"
]


class WrappedExpressionIterable:
    def __init__(self, iterable=None):
        self.__row_type = None
        if iterable is not None:
            if isinstance(iterable, type(self)):
                iterable = super().__iter__()
            else:
                it1, it2 = tee(iterable)
                try:
                    if isinstance(next(it1), Constant[Tuple]):
                        iterable = list(
                            tuple(a.value for a in e.value)
                            for e in it2
                        )
                except StopIteration:
                    pass

        super().__init__(iterable)

    def __iter__(self):
        type_ = self.row_type
        return (
            Constant[type_](
                tuple(
                    Constant[e_t](e, verify_type=False)
                    for e_t, e in zip(type_.__args__, t)
                ),
                verify_type=False
            )
            for t in super().__iter__()
        )

    def add(self, element):
        if isinstance(element, Constant[Tuple]):
            element = element.value
        element_ = tuple()
        for e in element:
            if isinstance(e, Constant):
                e = e.value
            element_ += (e,)
        super().add(element_)

    @property
    def row_type(self):
        if len(self) == 0:
            return None

        if self.__row_type is None:
            self.__row_type = infer_type(next(super().__iter__()))

        return self.__row_type


class WrappedRelationalAlgebraSet(
    WrappedExpressionIterable, RelationalAlgebraSet
):
    def __contains__(self, element):
        if not isinstance(element, Constant):
            element = self._normalise_element(element)
        return (
            self._container is not None and
            hash(element) in self._container.index
        )


class DatalogProgram(PatternWalker):
    '''
    Implementation of Datalog grammar in terms of
    Intermediate Representations. No query resolution implemented.
    In the symbol table the value for a symbol `S` is implemented as:

    * If `S` is part of the extensional database, then the value of the symbol
    is a set of tuples `a` representing `S(*a)` as facts

    * If `S` is part of the intensional database then its value is an
    `ExpressionBlock` such that each expression is a case of the symbol
    each expression is a `Lambda` instance where the `function_expression` is
    the query and the `args` are the needed projection. For instance
    `Q(x) :- R(x, x)` and `Q(x) :- T(x)` is represented as a symbol `Q`
     with value `ExpressionBlock((Lambda(R(x, x), (x,)), Lambda(T(x), (x,))))`
    '''

    protected_keywords = set()

    def function_equals(self, a: Any, b: Any) -> bool:
        return a == b

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

        if isinstance(fact_set, ExpressionBlock):
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
        Constant[bool](True)
    ))
    def statement_extensional(self, expression):
        return self.walk(Fact(expression.consequent))

    @add_match(Implication(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._validate_implication_syntax(consequent, antecedent)

        if consequent.functor in self.symbol_table:
            eb = self._new_intensional_internal_representation(consequent)
        else:
            eb = tuple()

        eb = eb + (expression,)

        self.symbol_table[consequent.functor] = ExpressionBlock(eb)

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
        eb = self.symbol_table[consequent.functor].expressions

        if (
            not isinstance(eb[0].consequent, FunctionApplication) or
            len(extract_datalog_free_variables(eb[0].consequent.args)) !=
            len(consequent.args)
        ):
            raise NeuroLangException(
                f"{eb[0].consequent} is already in the IDB "
                f"with different signature."
            )

        return eb

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
                isinstance(v, ExpressionBlock)
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
            iterable_ = iter(iterable)
            first = next(iterable_)
            if isinstance(first, Expression):
                type_ = first.type
            else:
                type_ = infer_type(first)
        except StopIteration:
            pass
        return type_
