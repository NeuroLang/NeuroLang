from __future__ import absolute_import, division, print_function
import typing
import inspect

import tatsu

from .ast import ASTWalker
from .ast_tatsu import TatsuASTConverter
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Symbol, Constant, Expression, FunctionApplication, Definition, Query,
    Projection, Predicate,
    TypedSymbolTable, unify_types, ToBeInferred,
    NeuroLangTypeException, is_subtype,
    get_type_and_value
)

from .expression_walker import (
    add_match,
    ExpressionBasicEvaluator, ExpressionReplacement
)


__all__ = [
    'NeuroLangIntermediateRepresentation',
    'ExpressionReplacement', 'NeuroLangException',
    'NeuroLangIntermediateRepresentationCompiler',
    'grammar_EBNF', 'parser',
    'Constant', 'Symbol', 'FunctionApplication', 'Definition', 'Query'
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
    def __init__(self):
        pass

    def query(self, ast):
        identifier = ast['identifier']
        category = ast['category']

        value = Query(
            identifier, ast['statement'],
            type_=category
        )
        return value

    def assignment(self, ast):
        identifier = ast['identifier']
        type_, value = get_type_and_value(ast['argument'])
        identifier = Symbol(identifier.name, type_)
        result = Definition(
            identifier, ast['argument'], type_=type_
        )
        return result

    def tuple(self, ast):
        types_ = []
        values = []
        for element in ast['element']:
            type_, value = get_type_and_value(
                element
            )
            types_.append(type_)
            values.append(element)

        return Constant(
            tuple(values),
            type_=typing.Tuple[tuple(types_)]
        )

    def predicate(self, ast):
        return Predicate(ast['identifier'], args=[ast['argument']])

    def value(self, ast):
        return ast['value']

    def statement(self, ast):
        arguments = ast['argument']
        result = arguments[0]
        for argument in arguments[1:]:
            result = result | argument
        return result

    def and_test(self, ast):
        arguments = ast['argument']
        result = arguments[0]
        for argument in arguments[1:]:
            result = result & argument
        return result

    def negated_argument(self, ast):
        argument = ast['argument']
        return ~argument

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
            return FunctionApplication(Symbol(ast['operator']))(ast['operand'])

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
            argument_type, value = get_type_and_value(a)
            if isinstance(value, Definition):
                value = value.symbol

            arguments.append(a)
            argument_types.append(argument_type)

        function = FunctionApplication(
            function,
            args=arguments,
            type_=typing.Callable[argument_types, typing.Any]
        )

        return function

    def projection(self, ast):
        symbol = ast['identifier']
        item = ast['item']
        if symbol.type == ToBeInferred:
            return Projection(symbol, item)
        elif is_subtype(symbol.type, typing.Tuple):
            item_type, item = get_type_and_value(item)
            if not is_subtype(item_type, typing.SupportsInt):
                raise NeuroLangTypeException(
                    "Tuple projection argument should be an int"
                )
            item = Constant(int(item), type_=int)
            if len(symbol.type.__args__) > item:
                return Projection(
                    symbol, item, type_=symbol.type.__args__[item]
                )
            else:
                raise NeuroLangTypeException(
                    "Tuple doesn't have %d items" % item
                )
        elif is_subtype(symbol.type, typing.Mapping):
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


class NeuroLangIntermediateRepresentationCompiler(ExpressionBasicEvaluator):
    def __init__(
        self, category_solvers=None, functions=None,
        types=None, symbols=None
    ):
        self.symbol_table = TypedSymbolTable()

        self.category_solvers = dict()

        if types is None:
            types = []
        if functions is None:
            functions = []

        if category_solvers is not None:
            for category_solver in category_solvers:
                self.category_solvers[
                    category_solver.type_name
                ] = category_solver
                self.category_solvers[
                    category_solver.type
                ] = category_solver

                self.category_solvers[
                    category_solver.plural_type_name
                ] = category_solver
                self.category_solvers[
                    typing.AbstractSet[category_solver.type]
                ] = category_solver

                types.append((category_solver.type, category_solver.type_name))
                types.append((
                    typing.AbstractSet[category_solver.type],
                    category_solver.plural_type_name
                ))

        else:
            category_solvers = dict()

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
                if not isinstance(v, Constant):
                    t, v = get_type_and_value(v)
                    v = Constant(v, type_=t)
                self.symbol_table[Symbol(k, type_=v.type)] = v

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

                t, func = get_type_and_value(func)
                self.symbol_table[Symbol(name, type_=t)] = Constant(
                    func, type_=t
                )

        for solver in self.category_solvers.values():
            solver.set_symbol_table(self.symbol_table)

    @add_match(Query)
    def query(self, expression):
        solver = self.category_solvers[expression.type]
        is_plural = solver.plural_type_name == expression.type

        if is_plural:
            symbol_type = typing.AbstractSet[solver.type]
        else:
            symbol_type = solver.type

        query_result = solver.walk(
            expression.value,  # plural=is_plural,
            # identifier=expression.symbol
        )

        value_type, value = get_type_and_value(query_result)

        if not is_subtype(value_type, symbol_type):
            raise NeuroLangTypeException(
                "%s doesn't have type %s" % (value, symbol_type)
            )

        result = Query(expression.symbol, query_result, type_=symbol_type)
        self.symbol_table[Symbol(expression.symbol.name, symbol_type)] = result
        return result

    def compile(self, ast):
        nli = NeuroLangIntermediateRepresentation()
        return self.walk(nli.evaluate(ast))


def parser(code, **kwargs):
    kwargs['semantics'] = kwargs.get('semantics', TatsuASTConverter())
    kwargs['parseinfo'] = True
    kwargs['trace'] = kwargs.get('trace', False)
    kwargs['colorize'] = True

    parser_tatsu = tatsu.compile(grammar_EBNF)
    ast = parser_tatsu.parse(code, **kwargs)

    return ast
