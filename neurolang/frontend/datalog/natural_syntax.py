from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu

from ...probabilistic.expressions import ProbabilisticPredicate
from ...datalog import Implication
from ...datalog.aggregation import AggregationApplication
from ...expressions import Expression, FunctionApplication
from .standard_syntax import (
    DatalogSemantics as DatalogClassicSemantics
)


GRAMMAR = u"""
    @@grammar::Datalog
    @@parseinfo :: True
    @@whitespace :: /[\t ]+/
    @@eol_comments :: /#([^\n]*?)$/

    start = expressions $ ;

    expressions = ( newline ).{ probabilistic_expression | expression };

    probabilistic_expression = probability:probability '::' \
                               expression:expression
                             | 'with' 'probability'\
                               probability:probability [','] \
                               expression:expression ;

    probability = (float | int_ext_identifier );

    expression = ['or'] @:rule | constraint | fact ;
    fact = constant_predicate ;
    rule = head implication body ;
    constraint = body right_implication head ;
    head = head_predicate ;
    body = ( conjunction ).{ predicate } ;

    conjunction = ',' | '&' | '\N{LOGICAL AND}' | 'and';
    implication = ':-' | '\N{LEFTWARDS ARROW}' | 'if' ;
    right_implication = '-:' | '\N{RIGHTWARDS ARROW}' | 'implies' ;
    head_predicate = predicate:identifier'(' arguments:[ arguments ] ')'
                   | arguments:argument 'is' arguments:argument"'s"\
                        predicate:identifier
                   | arguments:argument 'is'  predicate:identifier\
                        preposition arguments:argument
                   | arguments:argument 'has' arguments:argument\
                        predicate:identifier
                   | arguments+:argument 'is' ['a'] predicate:identifier ;

    predicate = predicate: int_ext_identifier'(' arguments:[ arguments ] ')'
              | arguments:argument 'is' arguments:argument"'s"\
                   predicate:int_ext_identifier
              | arguments:argument 'is'  predicate:int_ext_identifier\
                   preposition arguments:argument
              | arguments:argument 'has' arguments:argument\
                   [preposition] predicate:int_ext_identifier
              | arguments+:argument 'is' ['a'] predicate:int_ext_identifier
              | negated_predicate
              | comparison
              | logical_constant ;

    constant_predicate = predicate:identifier'(' ','.{ arguments+:literal } ')'
                       | arguments:literal 'is' arguments:literal"'s"\
                            predicate:identifier
                       | arguments:literal 'is' ['a'] predicate:identifier\
                            preposition arguments:literal
                       | arguments:literal 'has' ['a'] arguments:literal\
                            predicate:identifier
                       | arguments+:literal ('is' | 'are') ['a']\
                         predicate:identifier ;

    preposition = 'to' | 'from' | 'of' | 'than' | 'the' ;

    negated_predicate = ('~' | '\u00AC' | 'not' ) predicate ;

    comparison = argument comparison_operator argument ;

    arguments = ','.{ argument }+ ;
    argument = arithmetic_operation
             | function_application
             | '...' ;

    int_ext_identifier = identifier | ext_identifier ;
    ext_identifier = '@'identifier;

    function_application = int_ext_identifier'(' [ arguments ] ')';

    arithmetic_operation = term [ ('+' | '-') term ] ;

    term = factor [ ( '*' | '/' ) factor ] ;

    factor =  exponent [ '**' exponential ];

    exponential = exponent ;

    exponent = literal
             | function_application
             | int_ext_identifier
             | '(' @:argument ')' ;

    literal = number
            | text
            | ext_identifier ;

    identifier = /[a-zA-Z_][a-zA-Z0-9_]*/
               | '`'@:?"[0-9a-zA-Z/#%._:-]+"'`';

    comparison_operator = '==' | '<' | '<=' | '>=' | '>' | '!=' ;

    text = '"' /[a-zA-Z0-9 ]*/ '"'
          | "'" /[a-zA-Z0-9 ]*/ "'" ;

    number = float | integer ;
    integer = [ '+' | '-' ] /[0-9]+/ ;
    float = /[0-9]*/'.'/[0-9]+/
          | /[0-9]+/'.'/[0-9]*/ ;

    logical_constant = TRUE | FALSE ;
    TRUE = 'True' | '\u22A4' ;
    FALSE = 'False' | '\u22A5' ;

    newline = {['\\u000C'] ['\\r'] '\\n'}+ ;
"""


OPERATOR = {
    "+": add,
    "-": sub,
    "==": eq,
    ">=": ge,
    ">": gt,
    "<=": le,
    "<": lt,
    "*": mul,
    "!=": ne,
    "**": pow,
    "/": truediv,
}


COMPILED_GRAMMAR = tatsu.compile(GRAMMAR)


class DatalogSemantics(DatalogClassicSemantics):
    def constant_predicate(self, ast):
        return ast["predicate"](*ast["arguments"])

    def head_predicate(self, ast):
        if not isinstance(ast, Expression):
            if ast["arguments"] is not None and len(ast["arguments"]) > 0:
                arguments = []
                for arg in ast["arguments"]:
                    if isinstance(arg, FunctionApplication):
                        arg = AggregationApplication(arg.functor, arg.args)
                    arguments.append(arg)
                ast = ast["predicate"](*arguments)
            else:
                ast = ast["predicate"]()
        return ast

    def predicate(self, ast):
        if not isinstance(ast, Expression):
            ast = ast["predicate"](*ast["arguments"])
        return ast

    def probabilistic_expression(self, ast):
        probability = ast["probability"]
        expression = ast["expression"]

        if isinstance(expression, Implication):
            return Implication(
                ProbabilisticPredicate(probability, expression.consequent),
                expression.antecedent,
            )
        else:
            raise ValueError("Invalid rule")


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR,
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
    )
