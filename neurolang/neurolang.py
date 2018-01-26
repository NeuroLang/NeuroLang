from __future__ import absolute_import, division, print_function
import collections
import typing
import inspect
from copy import copy
import logging

import tatsu
from .ast import TatsuASTConverter, ASTWalker, ASTNode

# import numpy as np
# from .due import due, Doi

# __all__ = ["Model", "Fit", "opt_err_func", "transform_data", "cumgauss"]


# due.cite(Doi("10.1167/13.9.30"),
#         description="Template project for small scientific Python projects",
#         tags=["reference-implementation"],
#         path='neurolang')
grammar_EBNF = r'''
    @@whitespace :: /[\s\t\n\r\\ ]/

    start =  { @+:simple_statement [';'] ~ } $;
    simple_statement = import_statement
                     | query
                     | assignment
                     | value;

    import_statement = "import" ~ module:dotted_identifier;
    query = identifier:dotted_identifier "is" "a"
        category:identifier statement:statement;
    assignment = identifier:dotted_identifier "=" argument:value;

    statement = argument+:and_test { OR ~ argument:and_test };
    and_test = argument+:not_test { AND ~ argument:not_test };
    not_test = negated_argument
             | argument;
    negated_argument = NOT argument:argument;

    argument = '('~ @:statement ')'
             | predicate;

    predicate = identifier:dotted_identifier argument:value
              | WHERE comparison:comparison;

    comparison = operand+:value [operator:comparison_operator ~ operand:value];

    function_application = identifier:dotted_identifier
        "("~ [argument+:function_argument
        {"," ~ argument:function_argument}] ")";
    function_argument = value | statement;

    value = value:function_application
          | value:dotted_identifier
          | value:literal;

    literal = string | number;

    dotted_identifier = root:identifier { '.' ~ children:identifier };
    identifier = /[a-zA-Z_][a-zA-Z0-9_]*/;


    OR = "or";
    AND = "and";
    NOT = "not";
    WHERE = "where";

    preposition = "to" | "in";

    comparison_operator = "<" | ">" | "<=" | ">=" | "!=" | "==";

    number = point_float
           | integer;

    integer = value:/[0-9]+/;
    point_float = value:/[0-9]*/ '.' /[0-9]+/
                | value:/[0-9]+/ '.';

    string = '"'value:/(\\(\w+|\S+)|[^\r\n\f"])*/'"'
           | "'"value:/(\\(\w+|\S+)|[^\r\n\f"])*/"'";
    newline = {['\u000C'] ['\r'] '\n'}+;
    SPACE = /[\s\t\n]+/;
'''


def typing_callable_from_annotated_function(function):
    signature = inspect.signature(function)
    parameter_types = [
        v.annotation for v in signature.parameters.values()
    ]
    return typing.Callable[
        parameter_types,
        signature.return_annotation
    ]


def is_subtype(left, right):
    if (
        issubclass(right, typing.Callable) and
        issubclass(left, typing.Callable)
    ):
        left_args = left.__args__
        right_args = right.__args__

        if len(left_args) != len(right_args):
            False

        return all((
            is_subtype(left_arg, right_arg)
            for left_arg, right_arg in zip(left_args, right_args)
        ))
    else:
        if right == int:
            right = typing.SupportsInt
        elif right == float:
            right = typing.SupportsFloat
        elif right == complex:
            right = typing.SupportsComplex
        elif right == str:
            right = typing.Text

        return issubclass(left, right)


def get_type(value):
    if isinstance(value, typing.Callable):
        return typing_callable_from_annotated_function(value)
    else:
        return type(value)


def type_validation_value(value, type_, value_mapping=None):
    if issubclass(type_, typing.Callable):
        symbol_type = typing_callable_from_annotated_function(value)
        return is_subtype(symbol_type, type_)
    elif issubclass(type_, typing.Mapping):
        return (
            issubclass(type(value), type_.__base__) and
            all((
                type_validation_value(
                    k, type_.__args__[0], value_mapping=value_mapping
                ) and
                type_validation_value(
                    v, type_.__args__[1], value_mapping=value_mapping
                )
                for k, v in value.items()
            ))
        )
    elif issubclass(type_, typing.Iterable):
        return (
            issubclass(type(value), type_.__base__) and
            all((
                type_validation_value(
                    i, type_.__args__[0], value_mapping=value_mapping
                )
                for i in value
            ))
        )
    else:
        if value_mapping is None:
            if isinstance(value, Symbol):
                value = value.value
            return isinstance(value, type_)
        else:
            value = value_mapping[value]
            if isinstance(value, Symbol):
                value = value.value
            return isinstance(value, type_)


class Symbol(object):
    def __init__(self, type_, value, value_mapping=None):
        if not type_validation_value(
            value, type_, value_mapping=value_mapping
        ):
            raise ValueError(
                "The value %s does not correspond to the type %s" %
                (value, type_)
            )
        self.type = type_
        self.value = value

    def __repr__(self):
        return '%s: %s' % (self.value, self.type)


class SymbolTable(collections.MutableMapping):
    def __init__(self):
        self._symbols = collections.OrderedDict()
        self._symbols_by_type = dict()

    def __len__(self):
        return len(self._symbols)

    def __getitem__(self, key):
        return self._symbols[key]

    def __setitem__(self, key, value):
        if isinstance(value, Symbol) and isinstance(key, str):
            self._symbols[key] = value
            if value.type not in self._symbols_by_type:
                self._symbols_by_type[value.type] = dict()
            self._symbols_by_type[value.type][key] = value
        else:
            raise ValueError("Wrong assignement %s" % str(value))

    def __delitem__(self, key):
        value = self._symbols[key]
        del self._symbols_by_type[value.type][key]
        del self._symbols[key]

    def __iter__(self):
        return iter(self._labels)

    def __repr__(self):
        return '{%s}' % (
            ', '.join([
                '%s: (%s)' % (k, v)
                for k, v in self._symbols.items()
            ])
        )

    def types(self):
        return self._symbols_by_type.keys()

    def symbols_by_type(self, type_):
        return dict(self._symbols_by_type[type_])


def type_validation(symbol, type_):
    if isinstance(type_, typing.Callable):
        if isinstance(symbol, Symbol):
            return issubclass(symbol.type, type_)
        else:
            symbol_type = typing_callable_from_annotated_function(symbol)
            return issubclass(symbol_type, type_)
    else:
        if isinstance(symbol, Symbol):
            return isinstance(symbol.value, type_)
        else:
            return isinstance(symbol, type_)
    raise


class GenericSolver(ASTWalker):

    @property
    def plural_type_name(self):
        return self.type_name + 's'

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    def predicate(self, ast):
        logging.debug(str(self.__class__.__name__) + " evaluating predicate")

        identifier = ast['identifier']
        try:
            method = getattr(self, 'predicate_' + identifier)
            signature = inspect.signature(method)
            argument = ast['argument']
            if len(signature.parameters) != 1:
                raise ValueError("Predicates take exactly one parameter")
            else:
                parameter = next(iter(signature.parameters.values()))

            if isinstance(argument, Symbol):
                argument_value = argument.value
            else:
                argument_value = argument

            if not isinstance(argument, parameter.annotation):
                raise ValueError

            value = method(argument_value)

            return Symbol(
                signature.return_annotation,
                value
            )
        except AttributeError:
            raise

    def resolve(self, ast, plural=False):
        return self.evaluate(ast)


class NeuroLangInterpreter(ASTWalker):
    def __init__(self, category_solvers=None, functions=None, symbols=None):
        self.category_solvers = dict()
        for category_solver in category_solvers:
            self.category_solvers[category_solver.type_name] = category_solver
            self.category_solvers[
                category_solver.plural_type_name
            ] = category_solver
        self.symbols = SymbolTable()
        if symbols is not None:
            for k, v in symbols.items():
                self.symbols[k] = v
        self.functions = dict()
        for f in functions:
            if isinstance(f, tuple):
                func = f[0]
                name = f[1]
            else:
                func = f
                name = f.__name__
            self.symbols[name] = Symbol(
                typing_callable_from_annotated_function(func),
                func
            )

        for solver in self.category_solvers.values():
            solver.set_symbol_table(self.symbols)

    def query(self, ast):
        category_solver = self.category_solvers[ast['category']]
        is_plural = category_solver.plural_type_name == ast['category']
        symbol_type = category_solver.type

        if is_plural:
            symbol_type = typing.Set[category_solver.type]
            value_mapping = self.symbols
        else:
            value_mapping = None

        value = category_solver.resolve(
            ast['statement'], is_plural
        )

        if isinstance(value, Symbol):
            if not is_subtype(value.type, symbol_type):
                raise ValueError()
        else:
            value = Symbol(
                symbol_type, value,
                value_mapping=value_mapping
            )

        self.symbols[ast['identifier']] = value
        return ast

    def assignment(self, ast):
        self.symbols[ast['identifier']] = ast['argument']
        logging.debug(self.symbols[ast['identifier']])
        return ast['argument']

    def category(self, ast):
        self.category = ast['category']
        return ast['category']

    def predicate(self, ast):
        return ast

    def value(self, ast):
        ast = ast['value']
        if isinstance(ast, ASTNode):
            if ast.name == 'identifier':
                identifier = ast['root']
                if ast['children'] is not None:
                    identifier += '.' + '.'.join(ast['children'])
                return self.symbols[identifier]
            elif ast.name == 'string':
                return str(ast['value'])
            else:
                raise ValueError(str(ast))
        elif isinstance(ast, str):
            return self.symbols[ast]
        else:
            return ast

    def statement(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]
        for argument in arguments:
            if not isinstance(argument, bool):
                if isinstance(argument, ASTNode):
                    return ast
                else:
                    raise ValueError
                if argument:
                    return True
        return False

    def and_test(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]
        for argument in arguments:
            if not isinstance(argument, bool):
                if isinstance(argument, ASTNode):
                    return ast
                else:
                    raise ValueError
                if not argument:
                    return False
        return True

    def dotted_identifier(self, ast):
        identifier = ast['root']
        if 'children' in ast and ast['children'] is not None:
            identifier += '.' + '.'.join(ast['children'])
        return identifier

    def function_application(self, ast):
        function_symbol = self.symbols[ast['identifier']]
        function = function_symbol.value

        if not isinstance(function_symbol.type, typing.Callable):
            raise

        function_type_arguments, function_type_return = \
            get_Callable_arguments_and_return(
                function_symbol.type
            )

        arguments = []
        for i, a in enumerate(ast['argument']):
            if isinstance(a, Symbol):
                arguments.append(a.value)
                argument_type = a.type
            else:
                arguments.append(a)
                argument_type = type(a)

            if not isinstance(argument_type, function_type_arguments[i]):
                raise

        result = function(*arguments)
        if not isinstance(result, function_type_return):
            raise

        return Symbol(
            function_type_return,
            result,
        )

    def point_float(self, ast):
        return float(ast['value'])

    def integer(self, ast):
        return int(ast['value'])


class SetBasedSolver(ASTWalker):
    def __init__(self):
        pass

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    def statement(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        solution = arguments[0]
        for argument in arguments[1:]:
            solution = solution.union(argument)

        return solution

    def and_test(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        solution = arguments[0]
        for argument in arguments[1:]:
            solution = solution.intersection(argument)

        return solution

    def negated_argument(self, ast):
        raise

    def predicate(self, ast):
        logging.debug(str(self.__class__.__name__) + " evaluating predicate")
        if ast['identifier'] == "in":
            argument = ast['argument']
            if isinstance(argument.type, typing.Set):
                argument = copy(argument)
                value = argument.pop()
                for next_value in argument:
                    value = value.union(next_value)
                return value
            else:
                raise
        else:
            return None

    def resolve(self, ast, plural=False):
        value = self.evaluate(ast)
        if isinstance(value, typing.Set) and not plural:
            value_set = copy(value)
            value = self.symbol_table[value_set.pop()].value
            for other_value in value_set:
                other_value = self.symbol_table[other_value].value
                value = value.union(other_value)
        return value


def get_Callable_arguments_and_return(callable):
    return callable.__args__[:-1], callable.__args__[-1]


def parser(code, **kwargs):
    parser_tatsu = tatsu.compile(grammar_EBNF)
    ast = parser_tatsu.parse(code, parseinfo=True, trace=False, colorize=True,
                             semantics=TatsuASTConverter())

    return ast
