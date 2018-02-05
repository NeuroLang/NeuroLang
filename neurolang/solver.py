import logging
import inspect
import typing
from copy import copy

from .ast import ASTWalker
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Symbol, type_validation_value, NeuroLangTypeException,
    is_subtype, resolve_forward_references, get_type_and_value
)


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
        argument = ast['argument']
        if len(signature.parameters) != 1:
            raise NeuroLangPredicateException(
                "Predicates take exactly one parameter"
            )
        else:
            parameter = next(iter(signature.parameters.values()))

        type_, value = get_type_and_value(
            argument, value_mapping=self.symbol_table
        )

        if not type_validation_value(
            value,
            resolve_forward_references(
                self.type,
                parameter.annotation,
                type_name='type'
            ),
            value_mapping=self.symbol_table
        ):
            raise NeuroLangTypeException("argument of wrong type")

        result = method(value)

        type_, value = get_type_and_value(
            result, value_mapping=self.symbol_table
        )

        return_type = resolve_forward_references(
            self.type,
            signature.return_annotation,
            type_name='type'
        )

        result = Symbol(
            return_type,
            value,
            value_mapping=self.symbol_table
        )

        return result

    def comparison(self, ast):
        logging.debug(str(self.__class__.__name__) + " evaluating comparison")
        comparison_name = (
            'comparison_%s' % self.comparison_names[ast['operator']]
        )

        try:
            if hasattr(self, comparison_name):
                method = getattr(self, comparison_name)
                operand_values = []
            else:
                method = self.comparison_default
                operand_values = [self.comparison_names[ast['operator']]]

            signature = inspect.signature(method)
            operands = ast['operand']
            if len(signature.parameters) != 2:
                raise NeuroLangPredicateException(
                    "Comparisons take two parameters"
                )

            parameters = signature.parameters.values()

            operand_values += [
                get_type_and_value(operand, mapping=self.symbol_table)[1]
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
                result, value_mapping=self.symbol_table
            )

            if not is_subtype(type_, signature.return_annotation):
                raise

            return Symbol(
                signature.return_annotation,
                value
            )

        except AttributeError:
            raise NeuroLangException(
                "Condition %s not implemented" % ast['operator']
            )

    def execute(self, ast, plural=False):
        self.set_symbol_table(self.symbol_table.create_scope())
        result = self.evaluate(ast)
        self.set_symbol_table(self.symbol_table.enclosing_scope)
        return result

    def statement(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        _, solution = get_type_and_value(
            arguments[0], value_mapping=self.symbol_table
        )

        for argument in arguments[1:]:
            _, argument = get_type_and_value(
                        argument, value_mapping=self.symbol_table
                    )
            solution = solution | argument

        return solution

    def and_test(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        _, solution = get_type_and_value(
            arguments[0], value_mapping=self.symbol_table
        )

        for argument in arguments[1:]:
            _, argument = get_type_and_value(
                        argument, value_mapping=self.symbol_table
                    )
            solution = solution & argument

        return solution

    def negated_argument(self, ast):
        argument = ast['argument']
        type_, value = get_type_and_value(
                    argument, value_mapping=self.symbol_table
                )

        if issubclass(type_, typing.Set):
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
        self, argument: typing.Set['type']
    )->typing.Set['type']:
        return argument

        argument = copy(argument)
        value = argument.pop()
        for next_value in argument:
            value = value.union(next_value)
        return value

    def execute(self, ast, plural=False):
        value = self.evaluate(ast)
        if isinstance(value, typing.Set) and not plural:
            value_set = copy(value)
            value = self.symbol_table[value_set.pop()].value
            for other_value in value_set:
                other_value = self.symbol_table[other_value].value
                value = value.union(other_value)
        return value
