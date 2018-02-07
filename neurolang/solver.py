import logging
import inspect
import typing
from copy import copy

from .ast import ASTWalker
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Symbol, Identifier, type_validation_value, NeuroLangTypeException,
    is_subtype, get_type_and_value, replace_type_variable
)


T = typing.TypeVar('T')


class NeuroLangPredicateException(NeuroLangException):
    pass


class FiniteDomain(object):
    pass


class GenericSolver(ASTWalker):
    comparison_names = {
        '==': 'eq',
        '!=': 'ne',
        '>': 'gt',
        '>=': 'ge',
        '<': 'lt',
        '<=': 'le'
    }
    is_plural_evaluation = False

    @property
    def plural_type_name(self):
        return self.type_name + 's'

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    def predicate(self, ast):
        logging.debug(str(self.__class__.__name__) + " evaluating predicate")
        identifier = ast['identifier']

        predicate_method = 'predicate_' + identifier.value
        if not hasattr(self, predicate_method):
            raise NeuroLangException(
                "Predicate %s not implemented" % identifier.value
            )

        method = getattr(self, predicate_method)
        signature = inspect.signature(method)
        type_hints = typing.get_type_hints(method)

        argument = ast['argument']
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

        if (
            isinstance(argument, Identifier) and
            argument == self.identifier and
            issubclass(type_, FiniteDomain)
        ):
            all_ = self.symbol_table.symbols_by_type[type_]
            result = set(
                s for s in all_
                if method(s)
            )
        else:
            result = method(value)

        type_, value = get_type_and_value(
            result, symbol_table=self.symbol_table
        )

        return_type = type_hints['return']
        return_type = replace_type_variable(
            self.type,
            return_type,
            type_var=T
         )

        if not is_subtype(type_, return_type):
            raise NeuroLangTypeException(
                "Value returned doesn't have the right type"
            )

        result = Symbol(
            return_type,
            value,
            symbol_table=self.symbol_table
        )

        return result

    def comparison(self, ast):
        logging.debug(str(self.__class__.__name__) + " evaluating comparison")
        comparison_name = (
            'comparison_%s' % self.comparison_names[ast['operator']]
        )

        if hasattr(self, comparison_name):
            method = getattr(self, comparison_name)
            operand_values = []
        elif hasattr(self, 'comparison_default'):
            method = self.comparison_default
            operand_values = [self.comparison_names[ast['operator']]]
        else:
            raise NeuroLangException(
                "Comparison %s not implemented" % ast['operator']
            )

        signature = inspect.signature(method)
        operands = ast['operand']
        if len(signature.parameters) != 2:
            raise NeuroLangPredicateException(
                "Comparisons take two parameters"
            )

        parameters = signature.parameters.values()

        operand_values += [
            get_type_and_value(operand, symbol_table=self.symbol_table)[1]
            for operand in operands
        ]

        if any(
            not type_validation_value(
                operand_value, parameter.annotation
            )
            for operand_value, parameter
            in zip(operand_values, parameters)
        ):
            raise NeuroLangTypeException("operand of wrong type")

        result = method(*operand_values)

        type_, value = get_type_and_value(
            result, symbol_table=self.symbol_table
        )

        if not is_subtype(type_, signature.return_annotation):
            raise

        return Symbol(
            type_,
            value
        )

    def execute(self, ast, plural=False, identifier=None):
        self.set_symbol_table(self.symbol_table.create_scope())
        self.is_plural_evaluation = plural
        self.query_identifier = identifier
        self.symbol_table[self.query_identifier] = None
        result = self.specialized_execute(ast)
        self.set_symbol_table(self.symbol_table.enclosing_scope)
        return result

    def specialized_execute(self, ast):
        return self.evaluate(ast)

    def statement(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        _, solution = get_type_and_value(
            arguments[0], symbol_table=self.symbol_table
        )

        for argument in arguments[1:]:
            _, argument = get_type_and_value(
                        argument, symbol_table=self.symbol_table
                    )
            solution = solution | argument

        return solution

    def and_test(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        _, solution = get_type_and_value(
            arguments[0], symbol_table=self.symbol_table
        )

        for argument in arguments[1:]:
            _, argument = get_type_and_value(
                        argument, symbol_table=self.symbol_table
                    )
            solution = solution & argument

        return solution

    def negated_argument(self, ast):
        argument = ast['argument']
        type_, value = get_type_and_value(
                    argument, symbol_table=self.symbol_table
                )

        if issubclass(type_, typing.AbstractSet):
            type_ = type_.__args__[0]

        if issubclass(type_, FiniteDomain):
            all_elements = set(
                self.symbol_table.symbols_by_type(self.type).keys()
            )
            difference = all_elements - value

            return difference
        else:
            raise NeuroLangException("Type %s is not a finite type" % type_)


class SetBasedSolver(GenericSolver):
    def __new__(cls):
        return super().__new__(cls)

    def predicate_in(
        self, argument: typing.AbstractSet[T]
    )->typing.AbstractSet[T]:
        return argument

        argument = copy(argument)
        value = argument.pop()
        for next_value in argument:
            value = value.union(next_value)
        return value

    def specialized_execute(self, ast):
        value = self.evaluate(ast)
        if (
            isinstance(value, typing.AbstractSet) and
            not self.is_plural_evaluation
        ):
            value_set = copy(value)
            value = self.symbol_table[value_set.pop()].value
            for other_value in value_set:
                other_value = self.symbol_table[other_value].value
                value = value.union(other_value)
        return value
