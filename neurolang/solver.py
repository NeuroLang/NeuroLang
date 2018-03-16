import logging
import inspect
import typing
from copy import copy

from .ast import ASTWalker
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Expression, Symbol, Constant, Predicate, FunctionApplication,
    type_validation_value,
    NeuroLangTypeException,
    is_subtype, get_type_and_value, replace_type_variable
)
from operator import invert
from .expression_walker import (
    add_match, ExpressionWalker, ExpressionBasicEvaluator
)


T = typing.TypeVar('T')


class NeuroLangPredicateException(NeuroLangException):
    pass


class FiniteDomain(object):
    pass


class FiniteDomainSet(set):
    def __init__(self, *args, type_=None, typed_symbol_table=None):
        super().__init__(*args)
        self.type = type_
        self.symbol_table = typed_symbol_table

    def __invert__(self):
        result = FiniteDomainSet(
            (
                v.value.value for v in
                self.symbol_table.symbols_by_type(self.type).values()
                if v.value.value not in self
            ),
            type_=self.type,
            typed_symbol_table=self.symbol_table
        )
        return result


class GenericSolver(ExpressionBasicEvaluator):
    @property
    def plural_type_name(self):
        return self.type_name + 's'

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Predicate)
    def predicate(self, expression):
        logging.debug(str(self.__class__.__name__) + " evaluating predicate")

        if isinstance(expression.function, Symbol):
            identifier = expression.function

        predicate_method = 'predicate_' + identifier.name
        if hasattr(self, predicate_method):
            method = getattr(self, predicate_method)
            signature = inspect.signature(method)
            type_hints = typing.get_type_hints(method)
        else:
            raise NeuroLangException(
                "Predicate %s not implemented" % identifier
            )

        argument = self.walk(expression.args[0])
        if len(signature.parameters) != 1:
            raise NeuroLangPredicateException(
                "Predicates take exactly one parameter"
            )
        else:
            parameter_type = type_hints[
                next(iter(signature.parameters.keys()))
            ]

        type_, value = get_type_and_value(
            argument, symbol_table=self.symbol_table
        )

        if not type_validation_value(
            value,
            replace_type_variable(
                self.type,
                parameter_type,
                type_var=T
             ),
            symbol_table=self.symbol_table
        ):
            raise NeuroLangTypeException("argument of wrong type")

        # type_, value = get_type_and_value(
        #    result, symbol_table=self.symbol_table
        # )

        return_type = type_hints['return']
        return_type = replace_type_variable(
            self.type,
            return_type,
            type_var=T
         )
        result = method(value)
        if not isinstance(result, Expression):
            result = Constant(method(value), type_=return_type)

        # if not is_subtype(type_, return_type):
        #    raise NeuroLangTypeException(
        #        "Value returned doesn't have the right type"
        #    )

        # result = Expression(
        #    value,
        #    type_=return_type,
        #    symbol_table=self.symbol_table
        # )

        return result


class SetBasedSolver(GenericSolver):
    def predicate_in(
        self, argument: typing.AbstractSet[T]
    )->typing.AbstractSet[T]:
        return argument

    @add_match(
        FunctionApplication(Constant(invert), ...),
        lambda expression: isinstance(expression.args[0], FiniteDomainSet)
    )
    def rewrite_finite_domain_inversion(self, expression):
        print("Invert set")
        raise ValueError
        return FunctionApplication(Symbol('negate'), expression.args)
