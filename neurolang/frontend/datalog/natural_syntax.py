from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu

from ...datalog import Implication
from ...datalog.aggregation import AggregationApplication
from ...expressions import Expression, FunctionApplication, Symbol
from ...logic import ExistentialPredicate
from ...probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticPredicate
)
from .standard_syntax import DatalogSemantics as DatalogClassicSemantics

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

    probability = argument;

    expression = pir_rule | ['or'] @:rule | constraint | fact ;
    fact = constant_predicate ;
    rule = head implication body ;
    pir_rule = \
        probability_of head:head implication body:body \
        [given condition:body];
    constraint = body right_implication head ;
    head = head_predicate ;
    body = ( conjunction ).{ predicate } ;

    conjunction = ',' | '&' | '\N{LOGICAL AND}' | 'and' | 'AND';
    implication = ':-' | '\N{LEFTWARDS ARROW}' | 'if' | 'IF';
    right_implication = '-:' | '\N{RIGHTWARDS ARROW}' | 'implies' | 'IMPLIES' ;
    negation = '~' | '\u00AC' | 'not' | 'NOT';
    exists = 'exists' | '\u2204' | 'EXISTS';
    where = 'where' | 'WHERE';
    given = 'given' | 'GIVEN';
    reserved_words = conjunction
                   | implication
                   | right_implication
                   | negation
                   | exists
                   | where
                   | given;
    probability_of = 'compute probability of' | 'COMPUTE PROBABILITY OF';

    head_predicate = predicate:identifier'(' arguments:[ arguments ] ')'
                   | arguments:argument 'is' arguments:argument"'s"\
                        predicate:identifier
                   | arguments:argument 'is'  predicate:identifier\
                        preposition arguments:argument
                   | arguments:argument 'has' arguments:argument\
                        predicate:identifier
                   | arguments+:argument 'is' ['a'] predicate:identifier
                   | [ ','.{'?'arguments+:argument}+ ] {\
                        { predicate:identifier }+ \
                        ','.{ '?'arguments+:argument }+ \
                     }+ [ { predicate:identifier }+ ];

    predicate = >head_predicate
              | negated_predicate
              | existential_predicate
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

    negated_predicate = negation predicate [';'] ;
    existential_predicate = \
        exists ','.{ argument+:argument }+ where body:body [';'] ;

    comparison = argument comparison_operator argument ;

    arguments = ','.{ argument }+ ;
    argument = arithmetic_operation
             | function_application
             | '...' ;

    int_ext_identifier = identifier | ext_identifier ;
    ext_identifier = '@'identifier;

    function_application = int_ext_identifier'(' [ arguments ] ')';

    arithmetic_operation = [('+' | '-')] term [ ('+' | '-') term ] ;

    term = factor [ ( '*' | '/' ) factor ] ;

    factor =  exponent [ '**' exponential ];

    exponential = exponent ;

    exponent = literal
             | function_application
             | ['?']@:int_ext_identifier
             | '(' @:argument ')' ;

    literal = number
            | text
            | ext_identifier ;

    identifier = !reserved_words identifier_pure
               | '`'@:?"[0-9a-zA-Z/#%._:-]+"'`';

    identifier_qm = '?'identifier_pure;
    identifier_pure = /[a-zA-Z_][a-zA-Z0-9_]*/;

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
            predicate = self.normalise_predicate(ast["predicate"])
            if ast["arguments"] is not None and len(ast["arguments"]) > 0:
                arguments = []
                for arg in ast["arguments"]:
                    if isinstance(arg, FunctionApplication):
                        arg = AggregationApplication(arg.functor, arg.args)
                    arguments.append(arg)
                ast = predicate(*arguments)
            else:
                ast = predicate()
        return ast

    def predicate(self, ast):
        if not isinstance(ast, Expression):
            predicate = self.normalise_predicate(ast["predicate"])
            ast = predicate(*ast["arguments"])
        return ast

    def normalise_predicate(self, predicate):
        if isinstance(predicate, list) and not isinstance(predicate, str):
            predicate = Symbol('_'.join([p.name for p in predicate]))
        return predicate

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

    def pir_rule(self, ast):
        head = ast['head']
        head = FunctionApplication(
            head.functor,
            head.args + (PROB(*head.args),)
        )
        body = ast['body']
        if 'condition' in ast:
            body = Condition(
                body, ast['condition']
            )
        return Implication(head, body)

    def existential_predicate(self, ast):
        exp = ast['body']
        for arg in ast['argument']:
            exp = ExistentialPredicate(arg, exp)
        return exp


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR,
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
    )
