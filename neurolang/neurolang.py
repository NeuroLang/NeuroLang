from __future__ import absolute_import, division, print_function
import typing
import logging
import inspect

import tatsu

from .ast import TatsuASTConverter, ASTWalker, ASTNode
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Identifier, Symbol, SymbolTable, typing_callable_from_annotated_function,
    NeuroLangTypeException, is_subtype,
    get_type_and_value
)


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
    query = identifier:dotted_identifier link:("is" "a" | "are")
        category:identifier statement:statement;
    assignment = identifier:dotted_identifier "=" argument:value;

    statement = argument+:and_test { OR ~ argument:and_test };
    and_test = argument+:not_test { AND ~ argument:not_test };
    not_test = negated_argument
             | argument;
    negated_argument = NOT argument:argument;

    argument = '('~ @:statement ')'
             | WHERE @:comparison
             | @:predicate;

    comparison = operand+:sum operator:comparison_operator ~ operand:sum;
    predicate = identifier:dotted_identifier argument:sum;

    sum = term+:product { op+:('+' | '-') ~ term:product };
    product = factor+:power { op+:('*' | '//' | '/') ~ factor:power};
    power = base:value ['**' exponent:value];

    value = value:function_application
          | value:projection
          | value:dotted_identifier
          | value:literal
          | "(" value:sum ")";

    function_application = identifier:dotted_identifier
        "("~ [argument+:function_argument
        {"," ~ argument:function_argument}] ")";
    function_argument = value | statement;

    projection = identifier:dotted_identifier"["item:integer"]";

    literal = string | number | tuple;

    tuple = '(' element+:sum ({',' element:sum}+ | ',') ')';

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

    integer = value:/-{0,1}[0-9]+/;
    point_float = value:/-{0,1}[0-9]*/ '.' /[0-9]+/
                | value:/-{0,1}[0-9]+/ '.';

    string = '"'value:/(\\(\w+|\S+)|[^\r\n\f"])*/'"'
           | "'"value:/(\\(\w+|\S+)|[^\r\n\f"])*/"'";
    newline = {['\u000C'] ['\r'] '\n'}+;
    SPACE = /[\s\t\n]+/;
'''


class NeuroLangInterpreter(ASTWalker):
    def __init__(
        self, category_solvers=None, functions=None,
        types=None, symbols=None
    ):
        self.symbol_table = SymbolTable()

        self.category_solvers = dict()

        if types is None:
            types = []

        for category_solver in category_solvers:
            self.category_solvers[category_solver.type_name] = category_solver
            self.category_solvers[
                category_solver.plural_type_name
            ] = category_solver

            types.append((category_solver.type, category_solver.type_name))

        for type_, type_name in types:
            for name, member in inspect.getmembers(type_):
                if not inspect.isfunction(member) or name.startswith('_'):
                    continue
                signature = inspect.signature(member)
                parameters_items = iter(signature.parameters.items())

                next(parameters_items)
                if (
                    signature.return_annotation == inspect._empty or
                    any(
                        v == inspect._empty for k, v in parameters_items
                    )
                ):
                    continue

                argument_types = iter(signature.parameters.values())
                next(argument_types)

                member.__annotations__['self'] = type_
                for k, v in typing.get_type_hints(member).items():
                    member.__annotations__[k] = v
                functions = functions + [
                    (member, type_name + '_' + name)
                ]

        if symbols is not None:
            for k, v in symbols.items():
                self.symbol_table[Identifier(k)] = v

        for f in functions:
            if isinstance(f, tuple):
                func = f[0]
                name = f[1]
            else:
                func = f
                name = f.__name__
            self.symbol_table[Identifier(name)] = Symbol(
                typing_callable_from_annotated_function(func),
                func
            )

        for solver in self.category_solvers.values():
            solver.set_symbol_table(self.symbol_table)

    def query(self, ast):
        identifier = ast['identifier']
        category_solver = self.category_solvers[ast['category']]
        is_plural = category_solver.plural_type_name == ast['category']

        if not (is_plural or ast['link'] != 'are'):
            raise NeuroLangException(
                "Singular queries must be specified with 'is a' and "
                "plural queries with 'are'"
            )

        symbol_type = category_solver.type

        if is_plural:
            symbol_type = typing.AbstractSet[category_solver.type]

        query_result = category_solver.execute(
            ast['statement'], plural=is_plural, identifier=identifier
        )

        value_type, value = get_type_and_value(
            query_result, symbol_table=self.symbol_table
        )
        if not is_subtype(value_type, symbol_type):
            raise NeuroLangTypeException(
                "%s doesn't have type %s" % (value, symbol_type)
            )

        value = Symbol(
            value_type, value,
            symbol_table=self.symbol_table
        )

        self.symbol_table[ast['identifier']] = value
        return ast

    def assignment(self, ast):
        self.symbol_table[ast['identifier']] = ast['argument']
        logging.debug(self.symbol_table[ast['identifier']])
        return ast['argument']

    def tuple(self, ast):
        types_ = []
        values = []
        for element in ast['element']:
            type_, value = get_type_and_value(
                element, symbol_table=self.symbol_table
            )
            types_.append(type_)
            values.append(value)

        return Symbol(
            typing.Tuple[tuple(types_)],
            tuple(values)
        )

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
                identifier = Identifier(identifier)
                return self.symbol_table[identifier]
            elif ast.name == 'string':
                return str(ast['value'])
            else:
                raise NeuroLangTypeException(
                    "Value %s not recognised" % str(ast)
                )
        elif isinstance(ast, Identifier):
            return self.symbol_table[ast]
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
                    raise ValueError("Argument is not boolean")
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
                    raise ValueError("Argument is not boolean")
                if not argument:
                    return False
        return True

    def negated_argument(self, ast):
        argument = ast['argument']
        if not isinstance(argument, bool):
            if isinstance(argument, ASTNode):
                return ast
            else:
                raise ValueError("Argument is not boolean")
        return not argument

    def sum(self, ast):
        arguments = ast['term']
        result = arguments[0]
        if 'op' in ast:
            for op, argument in zip(ast['op'], arguments[1:]):
                if op == '+':
                    result = result + argument
                else:
                    result = result - argument
        return result

    def product(self, ast):
        arguments = ast['factor']
        result = arguments[0]
        if 'op' in ast:
            for op, argument in zip(ast['op'], arguments[1:]):
                if op == '*':
                    result = result * argument
                elif op == '/':
                    result = result / argument
                else:
                    result = result // argument
        return result

    def power(self, ast):
        result = ast['base']

        if 'exponent' in ast:
            result = result ** ast['exponent']

        return result

    def dotted_identifier(self, ast):
        identifier = ast['root']
        if 'children' in ast and ast['children'] is not None:
            identifier += '.' + '.'.join(ast['children'])
        return Identifier(identifier)

    def function_application(self, ast):
        function_symbol = self.symbol_table[ast['identifier']]
        function = function_symbol.value

        if not isinstance(function_symbol.type, typing.Callable):
            raise NeuroLangTypeException()

        function_type_arguments, function_type_return = \
            get_Callable_arguments_and_return(
                function_symbol.type
            )

        arguments = []
        for i, a in enumerate(ast['argument']):
            argument_type, value = get_type_and_value(
                a, symbol_table=self.symbol_table
            )
            arguments.append(value)

            if not is_subtype(argument_type, function_type_arguments[i]):
                raise NeuroLangTypeException()

        result_type, result = get_type_and_value(
            function(*arguments)
        )
        if not is_subtype(result_type, function_type_return):
            raise NeuroLangTypeException()

        return Symbol(
            result_type,
            result,
        )

    def projection(self, ast):
        identifier = self.symbol_table[ast['identifier']]
        item, item_type = get_type_and_value(
            ast['item'],
            symbol_table=self.symbol_table
        )
        if (
            isinstance(identifier, Symbol) and
            issubclass(identifier.type, typing.Tuple)
        ):
            if not is_subtype(item_type, typing.SupportsInt):
                raise NeuroLangTypeException(
                    "Tuple projection argument should be an int"
                )
            item = int(item)
            if len(identifier.value) > item:
                return Symbol(
                    identifier.type.__args__[item],
                    identifier.value[item]
                )
            else:
                raise NeuroLangTypeException(
                    "Tuple doesn't have %d items" % item
                )
        elif (
            isinstance(identifier, Symbol) and
            issubclass(identifier.type, typing.Mapping)
        ):
            key_type = identifier.type.__args__[0]
            if not is_subtype(item_type, key_type):
                raise NeuroLangTypeException(
                    "key type does not agree with Mapping key %s" % key_type
                )

            return Symbol(
                identifier.type.__args__[1],
                identifier.value[item]
            )
        else:
            raise NeuroLangTypeException("%s is not a tuple" % identifier)

    def point_float(self, ast):
        return float(ast['value'])

    def integer(self, ast):
        return int(ast['value'])


def get_Callable_arguments_and_return(callable):
    return callable.__args__[:-1], callable.__args__[-1]


def parser(code, **kwargs):
    kwargs['semantics'] = kwargs.get('semantics', TatsuASTConverter())
    kwargs['parseinfo'] = True
    kwargs['trace'] = kwargs.get('trace', False)
    kwargs['colorize'] = True

    parser_tatsu = tatsu.compile(grammar_EBNF)
    ast = parser_tatsu.parse(code, **kwargs)

    return ast
