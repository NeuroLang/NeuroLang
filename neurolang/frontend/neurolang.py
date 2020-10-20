r"""
Neurolang datalog grammar definition
and translation to intermediate representation
=============================================

1- defines the neurolang datalog syntax
2- code written using this grammar can be parsed
into an Abstract Syntax Tree (AST)
3- the obtained AST can ba walked through to translate it
to the intermediate representation used in the backend
"""

from __future__ import absolute_import, division, print_function

import inspect
import logging
import typing
from collections import Iterable, Mapping
from typing import Optional, Union

import tatsu

from ..exceptions import NeuroLangException
from ..expression_walker import (
    ExpressionBasicEvaluator,
    PatternMatcher,
    add_match,
)
from ..expressions import Constant as IRConstant
from ..expressions import Expression as IRExpression
from ..expressions import ExpressionBlock as IRExpressionBlock
from ..expressions import FunctionApplication as IRFunctionApplication
from ..expressions import Lambda as IRLambda
from ..expressions import NeuroLangTypeException as IRNeuroLangTypeException
from ..expressions import Projection as IRProjection
from ..expressions import Query as IRQuery
from ..expressions import Statement as IRStatement
from ..expressions import Symbol as IRSymbol
from ..expressions import Unknown as IRUnknown
from ..expressions import infer_type, is_leq_informative, unify_types
from ..logic import ExistentialPredicate
from .ast import ASTWalker
from .ast_tatsu import TatsuASTConverter

__all__ = [
    "NeuroLangIntermediateRepresentation",
    "NeuroLangException",
    "NeuroLangIntermediateRepresentationCompiler",
    "PatternMatcher",
    "grammar_EBNF",
    "parser",
    "add_match",
    "IRConstant",
    "IRSymbol",
    "IRFunctionApplication",
    "IRLambda",
    "IRStatement",
    "IRQuery",
    "ExistentialPredicate",
]

# Extended Backus–Naur Form (EBNF) grammar describing the datalog syntax
# used to write programs:
grammar_EBNF = r"""
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
          | value:formula
          | "(" value:sum ")";

    function_application = identifier:dotted_identifier
        "("~ [argument+:function_argument
        {"," ~ argument:function_argument}] ")";
    function_argument = value | statement;

    projection = identifier:dotted_identifier"["item:integer"]";

    formula = string | number | tuple;

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
"""


class NeuroLangIntermediateRepresentation(ASTWalker):
    """Abstract Syntax Tree walker class implementing
    translation from an ASTNode to the corresponding
    Neurolang Intermediate Representation Expression
    (IRExpression)"""

    def __init__(
        self, type_name_map: Optional[Union[Iterable, Mapping]] = None
    ):
        if isinstance(type_name_map, Mapping):
            self.type_name_map = type_name_map
        elif isinstance(type_name_map, Iterable):
            self.type_name_map = dict()
            for c in type_name_map:
                self.type_name_map[c.type_name] = c.type
                self.type_name_map[c.type_name_plural] = typing.AbstractSet[
                    c.type
                ]
        elif type_name_map is not None:
            raise ValueError("type_name_map should be a map or iterable")

    def query(self, ast):
        identifier = ast["identifier"]
        category = ast["category"]
        link = ast["link"]

        if category in self.type_name_map:
            category = self.type_name_map[category]
            self._verify_query_spelling_arity(category, link)

        identifier = identifier.cast(category)
        value = ast["statement"].cast(category)

        logging.debug(
            "Evaluating query {} {} {}".format(identifier, link, value)
        )

        result = IRQuery[category](identifier, value)
        return result

    @staticmethod
    def _verify_query_spelling_arity(category, link):
        if is_leq_informative(category, typing.AbstractSet):
            if "are" not in link:
                raise NeuroLangException(
                    'Plural type queries need to be linked with "are"'
                )
        else:
            if "is" not in link:
                raise NeuroLangException(
                    'Singular type queries need to be linked with "are"'
                )

    def assignment(self, ast):
        identifier = ast["identifier"]
        type_ = infer_type(ast["argument"])
        identifier = IRSymbol[type_](identifier.name)
        result = IRStatement[type_](identifier, ast["argument"])
        return result

    def tuple(self, ast):
        types_ = []
        values = []
        for element in ast["element"]:
            type_ = infer_type(element)
            types_.append(type_)
            values.append(element)

        return IRConstant[typing.Tuple[tuple(types_)]](tuple(values))

    def predicate(self, ast):
        return IRFunctionApplication(
            ast["identifier"], args=(ast["argument"],)
        )

    def value(self, ast):
        return ast["value"]

    def statement(self, ast):
        arguments = ast["argument"]
        result = arguments[0]
        for argument in arguments[1:]:
            result = result | argument
        return result

    def and_test(self, ast):
        arguments = ast["argument"]
        result = arguments[0]
        for argument in arguments[1:]:
            result = result & argument
        return result

    def negated_argument(self, ast):
        argument = ast["argument"]
        return ~argument

    def sum(self, ast):
        arguments = ast["term"]
        result_type = infer_type(arguments[0])
        result = arguments[0]
        if "op" in ast:
            for op, argument in zip(ast["op"], arguments[1:]):
                argument_type = infer_type(argument)
                result_type = unify_types(result_type, argument_type)
                if op == "+":
                    result = result + argument
                else:
                    result = result - argument
                result.type = result_type
        return result

    def product(self, ast):
        arguments = ast["factor"]
        result_type = infer_type(arguments[0])
        result = arguments[0]
        if "op" in ast:
            for op, argument in zip(ast["op"], arguments[1:]):
                argument_type = infer_type(argument)
                result_type = unify_types(result_type, argument_type)
                if op == "*":
                    result = result * argument
                elif op == "/":
                    result = result / argument
                elif op == "//":
                    result = result // argument
                    result_type = int
                result.type = result_type
        return result

    def power(self, ast):
        result = ast["base"]

        if "exponent" in ast:
            exponent = ast["exponent"]
            result_type = infer_type(result)
            exponent_type = infer_type(exponent)
            result = result ** exponent
            result.type = unify_types(result_type, exponent_type)
        return result

    def comparison(self, ast):
        if len(ast["operand"]) == 1:
            return ast["operand"]
        else:
            return IRFunctionApplication(
                IRSymbol(ast["operator"]),
                tuple(
                    ast["operand"],
                ),
            )

    def dotted_identifier(self, ast):
        identifier = ast["root"]
        if "children" in ast and ast["children"] is not None:
            identifier += "." + ".".join(ast["children"])
        return IRSymbol(identifier)

    def function_application(self, ast):
        function = ast["identifier"]

        arguments = []
        argument_types = []
        for a in ast["argument"]:
            argument_type = infer_type(a)
            value = a
            if isinstance(value, IRStatement):
                value = value.lhs
            elif isinstance(value, IRQuery):
                value = value.head

            arguments.append(a)
            argument_types.append(argument_type)

        function = IRFunctionApplication[typing.Any](
            function, args=tuple(arguments)
        )

        return function

    def projection(self, ast):
        symbol = ast["identifier"]
        item = ast["item"]
        if symbol.type is IRUnknown:
            return IRProjection(symbol, item)
        elif is_leq_informative(symbol.type, typing.Tuple):
            item_type = infer_type(item)
            if not is_leq_informative(item_type, typing.SupportsInt):
                raise IRNeuroLangTypeException(
                    "Tuple projection argument should be an int"
                )
            item = IRConstant[int](int(item))
            if len(symbol.type.__args__) > item:
                return IRProjection[symbol.type.__args__[item]](symbol, item)
            else:
                raise IRNeuroLangTypeException(
                    "Tuple doesn't have %d items" % item
                )
        elif is_leq_informative(symbol.type, typing.Mapping):
            key_type = symbol.type.__args__[0]
            if not is_leq_informative(item_type, key_type):
                raise IRNeuroLangTypeException(
                    "key type does not agree with Mapping key %s" % key_type
                )

            return IRExpression[symbol.type.__args__[1]](symbol.name[item])
        else:
            raise IRNeuroLangTypeException(
                "%s is not a tuple" % ast["identifier"]
            )

    def string(self, ast):
        return IRConstant[str](str(ast["value"]))

    def point_float(self, ast):
        return IRConstant[float](float("".join(ast["value"])))

    def integer(self, ast):
        return IRConstant[int](int(ast["value"]))


class NeuroLangIntermediateRepresentationCompiler(ExpressionBasicEvaluator):
    def __init__(
        self, functions=None, type_name_map=None, types=None, symbols=None
    ):
        super().__init__()

        if functions is None:
            functions = []
        if type_name_map is None:
            self.type_name_map = dict()
        else:
            self.type_name_map = type_name_map

        self.type_name_map.update({"int": int, "str": str, "float": float})

        self._init_plural_type_names()

        functions = self._init_functions(functions)

        if symbols is not None:
            for k, v in symbols.items():
                if not isinstance(v, IRConstant):
                    t = infer_type(v)
                    v = IRConstant[t](v)
                self.symbol_table[IRSymbol[v.type](k)] = v

        self._init_function_symbols(functions)

        self.nli = NeuroLangIntermediateRepresentation(
            type_name_map=self.type_name_map
        )

    def _init_function_symbols(self, functions):
        if functions is None:
            return

        for f in functions:
            if isinstance(f, tuple):
                func = f[0]
                name = f[1]
            else:
                func = f
                name = f.__name__

            signature = inspect.signature(func)
            argument_types = iter(signature.parameters.values())
            next(argument_types)

            for k, v in typing.get_type_hints(func).items():
                func.__annotations__[k] = v

            t = infer_type(func)
            self.symbol_table[IRSymbol[t](name)] = IRConstant[t](func)

    def _init_functions(self, functions):
        for type_name, type_ in self.type_name_map.items():
            for name, member in inspect.getmembers(type_):
                if (
                    not inspect.isfunction(member) or name.startswith("_")
                ) and self._update_member_signature(member, type_):
                    functions += [(member, type_name + "_" + name)]

        return functions

    def _update_member_signature(self, member, type_):
        signature = inspect.signature(member)
        parameters_items = iter(signature.parameters.items())

        next(parameters_items)
        if signature.return_annotation == inspect._empty or any(
            v == inspect._empty for k, v in parameters_items
        ):
            return False
        argument_types = iter(signature.parameters.values())
        next(argument_types)

        member.__annotations__["self"] = type_
        for k, v in typing.get_type_hints(member).items():
            member.__annotations__[k] = v
        return True

    def _init_plural_type_names(self):
        for mixin_class in self.__class__.mro():
            if not (
                hasattr(mixin_class, "type")
                and hasattr(mixin_class, "type_name")
            ):
                continue

            self.type_name_map[mixin_class.type_name] = mixin_class.type

            if hasattr(mixin_class, "type_name_plural"):
                type_name_plural = mixin_class.type_name_plural
            else:
                type_name_plural = mixin_class.type_name + "s"

            self.type_name_map[type_name_plural] = typing.AbstractSet[
                mixin_class.type
            ]

    def get_intermediate_representation(self, ast, **kwargs):
        if isinstance(ast, str):
            ast = parser(ast, **kwargs)
        return self.nli.evaluate(ast)

    def compile(self, ast, **kwargs):
        return self.walk(
            IRExpressionBlock(
                self.get_intermediate_representation(ast, **kwargs)
            )
        )


def parser(code: str, **kwargs):
    """Parses datalog code into an Abstract Syntax Tree (AST)

    Parameters
    ----------
    code : str
        code written in datalog, as described by it's EBNF syntax
    **kwargs
        completed and passed to the tatsu parser

    Returns
    -------
        AST
        Abstract Syntax Tree resulting from code parsing
    """
    kwargs["semantics"] = kwargs.get("semantics", TatsuASTConverter())
    kwargs["parseinfo"] = True
    kwargs["trace"] = kwargs.get("trace", False)
    kwargs["colorize"] = True

    parser_tatsu = tatsu.compile(grammar_EBNF)
    ast = parser_tatsu.parse(code, **kwargs)

    return ast
