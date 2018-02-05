from __future__ import absolute_import, division, print_function
import typing
import logging
import inspect

import tatsu

from .ast import TatsuASTConverter, ASTWalker, ASTNode
from .symbols_and_types import (
    Identifier, Symbol, SymbolTable, typing_callable_from_annotated_function,
    NeuroLangTypeException, is_subtype, resolve_forward_references
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
    query = identifier:dotted_identifier ("is" "a" | "are")
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
    raise NeuroLangTypeException(
        "Can't validate type between symbol %s and type %s" (symbol, type_)
    )


class NeuroLangInterpreter(ASTWalker):
    def __init__(self, category_solvers=None, functions=None, symbols=None):
        self.symbols = SymbolTable()

        self.category_solvers = dict()
        for category_solver in category_solvers:
            self.category_solvers[category_solver.type_name] = category_solver
            self.category_solvers[
                category_solver.plural_type_name
            ] = category_solver

            for name, member in inspect.getmembers(category_solver.type):
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

                member.__annotations__['self'] = category_solver.type
                for k, v in member.__annotations__.items():
                    member.__annotations__[k] = resolve_forward_references(
                        category_solver.type,
                        v
                    )
                functions = functions + [
                    (member, category_solver.type_name + '_' + name)
                ]

        if symbols is not None:
            for k, v in symbols.items():
                self.symbols[Identifier(k)] = v
        self.functions = dict()
        for f in functions:
            if isinstance(f, tuple):
                func = f[0]
                name = f[1]
            else:
                func = f
                name = f.__name__
            self.symbols[Identifier(name)] = Symbol(
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

        value = category_solver.execute(
            ast['statement'], is_plural
        )

        if isinstance(value, Symbol):
            if not is_subtype(value.type, symbol_type):
                raise NeuroLangTypeException(
                    "%s doesn't have type %s" % (value, symbol_type)
                )
        else:
            value = Symbol(
                symbol_type, value,
                value_mapping=value_mapping
            )

        self.symbols[Identifier(ast['identifier'])] = value
        return ast

    def assignment(self, ast):
        self.symbols[Identifier(ast['identifier'])] = ast['argument']
        logging.debug(self.symbols[Identifier(ast['identifier'])])
        return ast['argument']

    def tuple(self, ast):
        types_ = []
        values = []
        for element in ast['element']:
            if isinstance(element, Symbol):
                types_.append(element.type)
                values.append(element.value)
            else:
                types_.append(type(element))
                values.append(element)

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
                return self.symbols[Identifier(identifier)]
            elif ast.name == 'string':
                return str(ast['value'])
            else:
                raise NeuroLangTypeException(
                    "Value %s not recognised" % str(ast)
                )
        elif isinstance(ast, str):
            return self.symbols[Identifier(ast)]
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
        return identifier

    def function_application(self, ast):
        function_symbol = self.symbols[Identifier(ast['identifier'])]
        function = function_symbol.value

        if not isinstance(function_symbol.type, typing.Callable):
            raise NeuroLangTypeException()

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

            if not is_subtype(argument_type, function_type_arguments[i]):
                raise NeuroLangTypeException()

        result = function(*arguments)
        if not is_subtype(type(result), function_type_return):
            raise NeuroLangTypeException()

        return Symbol(
            function_type_return,
            result,
        )

    def projection(self, ast):
        identifier = self.symbols[Identifier(ast['identifier'])]
        item = ast['item']
        if (
            isinstance(identifier, Symbol) and
            issubclass(identifier.type, typing.Tuple)
        ):
            if len(identifier.value) > item:
                return Symbol(
                    identifier.type.__args__[item],
                    identifier.value[item]
                )
            else:
                raise NeuroLangTypeException(
                    "Tuple doesn't have %d items" % item
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
