from __future__ import absolute_import, division, print_function
import typing
import logging
import inspect

import tatsu

from .ast import ASTWalker, ASTNode
from .ast_tatsu import TatsuASTConverter
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Symbol, Constant, Expression, Function, Definition, Query,
    TypedSymbolTable, unify_types,
    NeuroLangTypeException, is_subtype, get_Callable_arguments_and_return,
    get_type_and_value, evaluate
)


__all__ = [
    'NeuroLangInterpreter', 'NeuroLangIntermediateRepresentation',
    'grammar_EBNF', 'parser'
]


# import numpy as np
# from .due import due, Doi

# __all__ = []


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
    query = identifier:dotted_identifier link:("is" ("a" | "an") | "are")
        category:identifier statement:statement;
    assignment = identifier:dotted_identifier "=" argument:sum;

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
    point_float = value:(/-{0,1}[0-9]*/ '.' /[0-9]+/)
                | value:(/-{0,1}[0-9]+/ '.');

    string = '"'value:/(\\(\w+|\S+)|[^\r\n\f"])*/'"'
           | "'"value:/(\\(\w+|\S+)|[^\r\n\f"])*/"'";
    newline = {['\u000C'] ['\r'] '\n'}+;
    SPACE = /[\s\t\n]+/;
'''


class NeuroLangIntermediateRepresentation(ASTWalker):
    def __init__(self, category_solvers=None):
        self.symbol_table = TypedSymbolTable()

        types = []
        self.category_solvers = dict()

        if category_solvers is not None:
            for category_solver in category_solvers:
                self.category_solvers[
                    category_solver.type_name
                ] = category_solver
                self.category_solvers[
                    category_solver.plural_type_name
                ] = category_solver

                types.append((category_solver.type, category_solver.type_name))

        for solver in self.category_solvers.values():
            solver.set_symbol_table(self.symbol_table)

    def query(self, ast):
        identifier = ast['identifier']
        category_solver = self.category_solvers[ast['category']]
        is_plural = category_solver.plural_type_name == ast['category']

        if not (
            (is_plural and ast['link'] == 'are') or
            (not is_plural and (ast['link'] not in ('is a', 'is an')))
        ):
            raise NeuroLangException(
                "Singular queries must be specified with 'is a' and "
                "plural queries with 'are'"
            )

        symbol_type = category_solver.type

        if is_plural:
            symbol_type = typing.AbstractSet[category_solver.type]

        value = Query(
            symbol_type, identifier, ast['statement'],
            symbol_table=self.symbol_table
        )

        self.symbol_table[identifier] = value
        return value

    def assignment(self, ast):
        identifier = ast['identifier']
        type_, value = get_type_and_value(ast['argument'])
        identifier = Symbol(identifier.name, type_)
        result = Definition(
            type_, identifier, ast['argument']
        )
        self.symbol_table[identifier] = result
        logging.debug(self.symbol_table[identifier])
        return result

    def tuple(self, ast):
        types_ = []
        values = []
        for element in ast['element']:
            type_, value = get_type_and_value(
                element, symbol_table=self.symbol_table
            )
            types_.append(type_)
            values.append(value)

        return Expression(
            tuple(values),
            type_=typing.Tuple[tuple(types_)]
        )

    def predicate(self, ast):
        return Function(ast['identifier'], args=[ast['argument']])

    def value(self, ast):
        if (
            isinstance(ast['value'], Symbol) and
            ast['value'] in self.symbol_table
        ):
            symbol = self.symbol_table[ast['value']]
            if isinstance(symbol, Definition):
                symbol = symbol.symbol
            return symbol
        return ast['value']

    def statement(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]
        return Function(Symbol('or'))(arguments)

    def and_test(self, ast):
        arguments = ast['argument']
        if len(arguments) == 1:
            return arguments[0]

        return Function(Symbol('and'))(arguments)

    def negated_argument(self, ast):
        argument = ast['argument']
        return Function(Symbol('not'))(argument)

    def sum(self, ast):
        arguments = ast['term']
        result_type, _ = get_type_and_value(arguments[0])
        result = arguments[0]
        if 'op' in ast:
            for op, argument in zip(ast['op'], arguments[1:]):
                argument_type, _ = get_type_and_value(argument)
                result_type = unify_types(result_type, argument_type)
                if op == '+':
                    result = result + argument
                else:
                    result = result - argument
                result.type = result_type
        return result

    def product(self, ast):
        arguments = ast['factor']
        result_type, _ = get_type_and_value(arguments[0])
        result = arguments[0]
        if 'op' in ast:
            for op, argument in zip(ast['op'], arguments[1:]):
                argument_type, _ = get_type_and_value(argument)
                result_type = unify_types(result_type, argument_type)
                if op == '*':
                    result = result * argument
                elif op == '/':
                    result = result / argument
                elif op == '//':
                    result = result // argument
                    result_type = int
                result.type = result_type
        return result

    def power(self, ast):
        result = ast['base']

        if 'exponent' in ast:
            exponent = ast['exponent']
            result_type, _ = get_type_and_value(result)
            exponent_type, _ = get_type_and_value(exponent)
            result = (
                result ** exponent
            )
            result.type = unify_types(result_type, exponent_type)
        return result

    def comparison(self, ast):
        if len(ast['operand']) == 1:
            return ast['operand']
        else:
            return Function(Symbol(ast['operator']))(ast['operand'])

    def dotted_identifier(self, ast):
        identifier = ast['root']
        if 'children' in ast and ast['children'] is not None:
            identifier += '.' + '.'.join(ast['children'])
        return Symbol(identifier)

    def function_application(self, ast):
        function = ast['identifier']

        arguments = []
        argument_types = []
        for i, a in enumerate(ast['argument']):
            argument_type, value = get_type_and_value(
                a, symbol_table=self.symbol_table
            )
            if isinstance(value, Definition):
                value = value.symbol

            arguments.append(a)
            argument_types.append(argument_type)

        function = Function(
            function,
            args=arguments,
            type_=typing.Callable[argument_types, typing.Any]
        )

        return function

    def projection(self, ast):
        symbol = self.symbol_table[ast['identifier']]
        item_type, item = get_type_and_value(
            ast['item'],
            symbol_table=self.symbol_table
        )
        if (
            isinstance(symbol, Expression) and
            issubclass(symbol.type, typing.Tuple)
        ):
            if not is_subtype(item_type, typing.SupportsInt):
                raise NeuroLangTypeException(
                    "Tuple projection argument should be an int"
                )
            item = int(item)
            if len(symbol.value) > item:
                return Expression(
                    symbol.value[item],
                    type_=symbol.type.__args__[item]
                )
            else:
                raise NeuroLangTypeException(
                    "Tuple doesn't have %d items" % item
                )
        elif (
            isinstance(symbol, Expression) and
            issubclass(symbol.type, typing.Mapping)
        ):
            key_type = symbol.type.__args__[0]
            if not is_subtype(item_type, key_type):
                raise NeuroLangTypeException(
                    "key type does not agree with Mapping key %s" % key_type
                )

            return Expression(
                symbol.name[item],
                type_=symbol.type.__args__[1]
            )
        else:
            raise NeuroLangTypeException(
                "%s is not a tuple" % ast['identifier']
            )

    def string(self, ast):
        return Constant(str(ast['value']), type_=str)

    def point_float(self, ast):
        return Constant(float(''.join(ast['value'])), type_=float)

    def integer(self, ast):
        return Constant(int(ast['value']), type_=int)

    def compile(self, ast=None):
        if ast is not None:
            self.evaluate(ast)
        for k, v in self.symbol_table.items():
            if isinstance(v, Expression):
                self.symbol_table[k] = Expression(
                    evaluate(v.value),
                    type_=v.type
                )


class NeuroLangInterpreter(ASTWalker):
    def __init__(
        self, category_solvers=None, functions=None,
        types=None, symbols=None
    ):
        self.symbol_table = TypedSymbolTable()

        self.category_solvers = dict()

        if types is None:
            types = []

        if category_solvers is not None:
            for category_solver in category_solvers:
                self.category_solvers[
                    category_solver.type_name
                ] = category_solver
                self.category_solvers[
                    category_solver.plural_type_name
                ] = category_solver

                types.append((category_solver.type, category_solver.type_name))

        if types is None:
            types = []
        types += [(int, 'int'), (str, 'str'), (float, 'float')]
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
                self.symbol_table[Symbol(k)] = v

        if functions is not None:
            for f in functions:
                if isinstance(f, tuple):
                    func = f[0]
                    name = f[1]
                else:
                    func = f
                    name = f.__name__

                signature = inspect.signature(func)
                parameters_items = iter(signature.parameters.items())

                argument_types = iter(signature.parameters.values())
                next(argument_types)

                for k, v in typing.get_type_hints(func).items():
                    func.__annotations__[k] = v
                self.symbol_table[Symbol(name)] = Function(func)

        for solver in self.category_solvers.values():
            solver.set_symbol_table(self.symbol_table)

    def query(self, ast):
        identifier = ast['identifier']
        category_solver = self.category_solvers[ast['category']]
        is_plural = category_solver.plural_type_name == ast['category']

        if not (
            (is_plural and ast['link'] == 'are') or
            (not is_plural and (ast['link'] not in ('is a', 'is an')))
        ):
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

        identifier = Symbol(ast['identifier'].name, symbol_type)

        value = Definition(
            value_type, identifier, value,
            symbol_table=self.symbol_table
        )

        self.symbol_table[identifier] = value
        return identifier

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

        return Expression(
            tuple(values),
            type_=typing.Tuple[tuple(types_)]
        )

    def predicate(self, ast):
        return ast

    def value(self, ast):
        ast = ast['value']
        if isinstance(ast, Symbol):
                return self.symbol_table[ast]
            # raise NeuroLangException(
            #    "Value %s not recognised" % str(ast)
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
        result_type, result = get_type_and_value(arguments[0])
        if 'op' in ast:
            for op, argument in zip(ast['op'], arguments[1:]):
                argument_type, argument = get_type_and_value(argument)
                if (
                    (argument_type != result_type) and
                    is_subtype(result_type, argument_type)
                ):
                    result_type = argument_type
                if op == '+':
                    result = result + argument
                else:
                    result = result - argument
            return Expression(result, type_=result_type)
        else:
            return arguments[0]

    def product(self, ast):
        arguments = ast['factor']
        result_type, result = get_type_and_value(arguments[0])
        if 'op' in ast:
            for op, argument in zip(ast['op'], arguments[1:]):
                argument_type, argument = get_type_and_value(argument)
                if (
                    (argument_type != result_type) and
                    is_subtype(result_type, argument_type)
                ):
                    result_type = argument_type
                if op == '*':
                    result = result * argument
                elif op == '/':
                    result = result / argument
                elif op == '//':
                    result = result // argument
                    result_type = int
            return Expression(result, type_=result_type)
        else:
            return arguments[0]

    def power(self, ast):
        result = ast['base']

        if 'exponent' in ast:
            result_type, result_value = get_type_and_value(result)
            result_value = (
                result_value **
                get_type_and_value(ast['exponent'])[1]
            )
            result = Expression(result_value, type_=result_type)

        return result

    def dotted_identifier(self, ast):
        identifier = ast['root']
        if 'children' in ast and ast['children'] is not None:
            identifier += '.' + '.'.join(ast['children'])
        return Symbol(identifier)

    def function_application(self, ast):
        function = self.symbol_table[ast['identifier']]

        if not is_subtype(function.type, typing.Callable):
            raise NeuroLangTypeException()

        function_type_arguments, function_type_return = \
            get_Callable_arguments_and_return(
                function.type
            )

        arguments = []
        for i, a in enumerate(ast['argument']):
            argument_type, value = get_type_and_value(
                a, symbol_table=self.symbol_table
            )
            arguments.append(value)

            if not is_subtype(argument_type, function_type_arguments[i]):
                raise NeuroLangTypeException()

        function = Function(function, args=arguments)
        result_type, result = get_type_and_value(function)

        if not is_subtype(result_type, function_type_return):
            raise NeuroLangTypeException()

        return Expression(
            result,
            type_=result_type
        )

    def projection(self, ast):
        symbol = self.symbol_table[ast['identifier']]
        item_type, item = get_type_and_value(
            ast['item'],
            symbol_table=self.symbol_table
        )
        if (
            isinstance(symbol, Expression) and
            issubclass(symbol.type, typing.Tuple)
        ):
            if not is_subtype(item_type, typing.SupportsInt):
                raise NeuroLangTypeException(
                    "Tuple projection argument should be an int"
                )
            item = int(item)
            if len(symbol.value) > item:
                return Expression(
                    symbol.value[item],
                    type_=symbol.type.__args__[item]
                )
            else:
                raise NeuroLangTypeException(
                    "Tuple doesn't have %d items" % item
                )
        elif (
            isinstance(symbol, Expression) and
            issubclass(symbol.type, typing.Mapping)
        ):
            key_type = symbol.type.__args__[0]
            if not is_subtype(item_type, key_type):
                raise NeuroLangTypeException(
                    "key type does not agree with Mapping key %s" % key_type
                )

            return Expression(
                symbol.name[item],
                type_=symbol.type.__args__[1]
            )
        else:
            raise NeuroLangTypeException(
                "%s is not a tuple" % ast['identifier']
            )

    def string(self, ast):
        return Constant(str(ast['value']), type_=str)

    def point_float(self, ast):
        return Constant(float(''.join(ast['value'])), type_=float)

    def integer(self, ast):
        return Constant(int(ast['value']), type_=int)
        return ast

    def compile(self, ast=None):
        if ast is not None:
            self.evaluate(ast)
        for k, v in self.symbol_table.items():
            if (
                isinstance(v, Expression) and
                not isinstance(v, Function)
            ):
                self.symbol_table[k] = Expression(
                    evaluate(v.value),
                    type_=v.type
                )


def parser(code, **kwargs):
    kwargs['semantics'] = kwargs.get('semantics', TatsuASTConverter())
    kwargs['parseinfo'] = True
    kwargs['trace'] = kwargs.get('trace', False)
    kwargs['colorize'] = True

    parser_tatsu = tatsu.compile(grammar_EBNF)
    ast = parser_tatsu.parse(code, **kwargs)

    return ast
