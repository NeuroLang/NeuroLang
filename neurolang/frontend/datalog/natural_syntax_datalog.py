from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...expressions import Constant, Expression, Symbol

GRAMMAR = u"""
    @@grammar::Datalog
    @@parseinfo :: True
    @@whitespace :: /[\t ]+/
    @@eol_comments :: /#([^\n]*?)$/

    start = expressions $ ;

    expressions = ( newline ).{ expression };


    probabilistic_expression = (number | int_ext_identifier ) '::' expression ;
    expression = ['or'] @:rule | constraint | fact;
    fact = constant_predicate ;
    rule = head implication body ;
    constraint = body right_implication head ;
    head = head_predicate ;
    body = ( conjunction ).{ predicate } ;

    conjunction = ',' | '&' | '\N{LOGICAL AND}' | 'and';
    implication = ':-' | '\N{LEFTWARDS ARROW}' | 'if' ;
    right_implication = '-:' | '\N{RIGHTWARDS ARROW}' | 'implies' ;
    head_predicate = predicate:identifier'(' arguments:[ arguments ] ')'
                   | arguments:argument 'is' arguments:argument"'s" predicate:identifier
                   | arguments:argument 'is'  predicate:identifier preposition arguments:argument
                   | arguments:argument 'has' arguments:argument predicate:identifier
                   | arguments+:argument 'is' predicate:identifier ;

    predicate = predicate: int_ext_identifier'(' arguments:[ arguments ] ')'
              | arguments:argument 'is' arguments:argument"'s" predicate:int_ext_identifier
              | arguments:argument 'is'  predicate:int_ext_identifier preposition arguments:argument
              | arguments:argument 'has' arguments:argument predicate:int_ext_identifier
              | arguments+:argument 'is'  predicate:int_ext_identifier
              | negated_predicate
              | comparison
              | logical_constant ;

    constant_predicate = predicate:identifier'(' ','.{ arguments+:literal } ')'
                       | arguments:literal 'is' arguments:literal"'s" predicate:identifier
                       | arguments:literal 'is'  predicate:identifier preposition arguments:literal
                       | arguments:literal 'has' arguments:literal predicate:identifier
                       | arguments+:literal 'is' predicate:identifier ;

    preposition = 'to' | 'from' | 'of' | 'than' | 'the' ;

    negated_predicate = ('~' | '\u00AC' | 'not' ) predicate ;

    comparison = argument comparison_operator argument ;

    arguments = ','.{ argument }+ ;
    argument = arithmetic_operation
             | function_application ;

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

    identifier = /[a-zA-Z_][a-zA-Z0-9_]*/ ;

    comparison_operator = '==' | '<' | '<=' | '>=' | '>' | '!=' ;

    text = '"' /[a-zA-Z0-9]*/ '"'
          | "'" /[a-zA-Z0-9]*/ "'" ;

    number = [ '+' | '-' ] /[0-9]+/ ;
    logical_constant = TRUE | FALSE ;
    TRUE = 'True' | '\u22A4' ;
    FALSE = 'False' | '\u22A5' ;

    newline = {['\\u000C'] ['\\r'] '\\n'}+ ;
"""


OPERATOR = {
    '+': add,
    '-': sub,
    '==': eq,
    '>=': ge,
    '>': gt,
    '<=': le,
    '<': lt,
    '*': mul,
    '!=': ne,
    '**': pow,
    '/': truediv
}


COMPILED_GRAMMAR = tatsu.compile(GRAMMAR)


class ExternalSymbol(Symbol):
    def __repr__(self):
        return "@S{{{}: {}}}".format(self.name, self.__type_repr__)


class DatalogSemantics:
    def __init__(self, locals=None, globals=None):
        super().__init__()

        if locals is None:
            locals = {}
        if globals is None:
            globals = {}

        self.locals = locals
        self.globals = globals

    def start(self, ast):
        return ast

    def expressions(self, ast):
        if isinstance(ast, Expression):
            ast = (ast,)
        return Union(ast)

    def fact(self, ast):
        return Fact(ast)

    def constant_predicate(self, ast):
        return ast['predicate'](*ast['arguments'])

    def rule(self, ast):
        return Implication(ast[0], ast[2])

    def body(self, ast):
        return Conjunction(ast)

    def head_predicate(self, ast):
        if not isinstance(ast, Expression):
            if len(ast['arguments']) > 0:
                ast = ast['predicate'](*ast['arguments'])
            else:
                ast = ast['predicate']()
        return ast

    def predicate(self, ast):
        if not isinstance(ast, Expression):
            ast = ast['predicate'](*ast['arguments'])
        return ast

    def negated_predicate(self, ast):
        return Negation(ast[1])

    def comparison(self, ast):
        operator = Constant(OPERATOR[ast[1]])
        return operator(ast[0], ast[2])

    def arguments(self, ast):
        if isinstance(ast, Expression):
            return (ast,)
        return tuple(ast)

    def ext_identifier(self, ast):
        ast = ast[1]
        return ExternalSymbol[ast.type](ast.name)

    def arithmetic_operation(self, ast):
        if isinstance(ast, Expression):
            return ast

        if len(ast) == 1:
            return ast[0]

        op = Constant(OPERATOR[ast[1]])

        return op(*ast[0::2])

    def term(self, ast):
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            return ast[0]

        op = Constant(OPERATOR[ast[1]])

        return op(ast[0], ast[2])

    def factor(self, ast):
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            return ast[0]
        else:
            return Constant(pow)(ast[0], ast[2])

    def function_application(self, ast):
        return Symbol(ast[0])(*ast[2])

    def identifier(self, ast):
        return Symbol(ast)

    def text(self, ast):
        return Constant(ast[1])

    def number(self, ast):
        return Constant(float("".join(ast)))

    def _default(self, ast):
        return ast


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR, code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals)
    )
