from __future__ import absolute_import, division, print_function
import typing
import inspect
from collections import namedtuple, Iterable, Mapping
import logging

import tatsu

from .ast import ASTWalker
from .ast_tatsu import TatsuASTConverter
from .exceptions import NeuroLangException
from .symbols_and_types import (
    Symbol, Constant, Expression, FunctionApplication, Statement, Query,
    Projection, Predicate, ExistentialPredicate,
    TypedSymbolTable, unify_types, ToBeInferred,
    NeuroLangTypeException, is_subtype,
    get_type_and_value
)


from .expression_walker import (
    add_match,
    ExpressionBasicEvaluator,
    PatternMatcher
)


__all__ = [
    'NeuroLangIntermediateRepresentation',
    'NeuroLangException',
    'NeuroLangIntermediateRepresentationCompiler',
    'PatternMatcher',
    'grammar_EBNF', 'parser', 'add_match',
    'Constant', 'Symbol', 'FunctionApplication',
    'Statement', 'Query', 'ExistentialPredicate'
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


Category = namedtuple('Category', 'type_name type_name_plural type')


class NeuroLangIntermediateRepresentation(ASTWalker):
    def __init__(self, type_name_map=None):
        if isinstance(type_name_map, Mapping):
            self.type_name_map = type_name_map
        elif isinstance(type_name_map, Iterable):
            self.type_name_map = dict()
            for c in type_name_map:
                self.type_name_map[c.type_name] = c.type
                self.type_name_map[c.type_name_plural] = \
                    typing.AbstractSet(c.type)
        elif type_name_map is not None:
            raise ValueError(
                'type_name_map should be a map or iterable'
            )

    def query(self, ast):
        identifier = ast['identifier']
        category = ast['category']
        link = ast['link']

        if category in self.type_name_map:
            category = self.type_name_map[category]

            if (
                hasattr(category, '__origin__') and
                category.__origin__ is typing.AbstractSet
            ):
                if 'are' not in link:
                    raise NeuroLangException(
                        'Plural type queries need to be linked with "are"'
                    )
            else:
                if 'is' not in link:
                    raise NeuroLangException(
                        'Singular type queries need to be linked with "are"'
                    )
        identifier = identifier.cast(category)
        value = ast['statement'].cast(category)

        logging.debug('Evaluating query {} {} {}'.format(
            identifier, link, value
        ))

        result = Query[category](
            identifier, value
        )
        return result

    def assignment(self, ast):
        identifier = ast['identifier']
        type_, value = get_type_and_value(ast['argument'])
        identifier = Symbol[type_](identifier.name)
        result = Statement[type_](
            identifier, ast['argument']
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

        return Constant[typing.Tuple[tuple(types_)]](
            tuple(values)
        )

    def predicate(self, ast):
        return Predicate(ast['identifier'], args=(ast['argument'],))

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
            return FunctionApplication(
                Symbol(ast['operator']), tuple(ast['operand'],)
            )

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
            if isinstance(value, Statement):
                value = value.symbol
            elif isinstance(value, Query):
                value = value.head

            arguments.append(a)
            argument_types.append(argument_type)

        function = FunctionApplication[
            typing.Any
        ](
            function,
            args=tuple(arguments)
        )

        return function

    def projection(self, ast):
        symbol = ast['identifier']
        item = ast['item']
        if symbol.type is ToBeInferred:
            return Projection(symbol, item)
        elif is_subtype(symbol.type, typing.Tuple):
            item_type, item = get_type_and_value(item)
            if not is_subtype(item_type, typing.SupportsInt):
                raise NeuroLangTypeException(
                    "Tuple projection argument should be an int"
                )
            item = Constant[int](int(item))
            if len(symbol.type.__args__) > item:
                return Projection[symbol.type.__args__[item]](
                    symbol, item
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

            return Expression[symbol.type.__args__[1]](
                symbol.name[item]
            )
        else:
            raise NeuroLangTypeException(
                "%s is not a tuple" % ast['identifier']
            )

    def string(self, ast):
        return Constant[str](str(ast['value']))

    def point_float(self, ast):
        return Constant[float](float(''.join(ast['value'])))

    def integer(self, ast):
        return Constant[int](int(ast['value']))


class NeuroLangIntermediateRepresentationCompiler(ExpressionBasicEvaluator):
    def __init__(
        self, functions=None, type_name_map=None,
        types=None, symbols=None
    ):
        super().__init__()

        if functions is None:
            functions = []
        if type_name_map is None:
            self.type_name_map = dict()
        else:
            self.type_name_map = type_name_map

        self.type_name_map.update({'int': int, 'str': str, 'float': float})

        for mixin_class in self.__class__.mro():
            if (
                hasattr(mixin_class, 'type') and
                hasattr(mixin_class, 'type_name')
            ):
                self.type_name_map[mixin_class.type_name] = mixin_class.type

                if hasattr(mixin_class, 'type_name_plural'):
                    type_name_plural = mixin_class.type_name_plural
                else:
                    type_name_plural = mixin_class.type_name + 's'

                self.type_name_map[type_name_plural] = \
                    typing.AbstractSet[mixin_class.type]

        for type_name, type_ in self.type_name_map.items():
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
                    v = Constant[t](v)
                self.symbol_table[Symbol[v.type](k)] = v

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
                self.symbol_table[Symbol[t](name)] = Constant[t](
                    func
                )

        self.nli = NeuroLangIntermediateRepresentation(
            type_name_map=self.type_name_map
        )

    def get_intermediate_representation(self, ast, **kwargs):
        if isinstance(ast, str):
            ast = parser(ast, **kwargs)
        return self.nli.evaluate(ast)

    def compile(self, ast, **kwargs):
        return self.walk(self.get_intermediate_representation(ast, **kwargs))


def parser(code, **kwargs):
    kwargs['semantics'] = kwargs.get('semantics', TatsuASTConverter())
    kwargs['parseinfo'] = True
    kwargs['trace'] = kwargs.get('trace', False)
    kwargs['colorize'] = True

    parser_tatsu = tatsu.compile(grammar_EBNF)
    ast = parser_tatsu.parse(code, **kwargs)

    return ast
