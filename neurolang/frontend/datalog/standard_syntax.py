from neurolang.logic import ExistentialPredicate
from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...datalog.constraints_representation import RightImplication
from ...expressions import Constant, Expression, FunctionApplication, Query, Symbol
from ...probabilistic.expressions import Condition, ProbabilisticPredicate, PROB


GRAMMAR = u"""
    @@grammar::Datalog
    @@parseinfo :: True
    @@whitespace :: /[\t ]+/
    @@eol_comments :: /#([^\n]*?)$/

    start = expressions $ ;

    expressions = ( newline ).{ expression };


    expression = rule | constraint | fact | probabilistic_rule | probabilistic_fact ;
    fact = constant_predicate ;
    probabilistic_fact = ( arithmetic_operation | int_ext_identifier ) '::' constant_predicate ;
    rule = (head | query) implication (condition | body) ;
    probabilistic_rule = head '::' ( arithmetic_operation | int_ext_identifier ) implication (condition | body) ;
    constraint = body right_implication head ;
    head = head_predicate ;
    body = conjunction ;
    condition = composite_predicate '//' composite_predicate ;
    conjunction = ( conjunction_symbol ).{ predicate } ;
    composite_predicate = '(' @:conjunction ')'
                        | predicate ;
    exists = 'exists' | '\u2203' | 'EXISTS';
    such_that = 'st' | ';' ;
    reserved_words = exists 
                   | 'st' 
                   | 'ans' ;

    conjunction_symbol = ',' | '&' | '\N{LOGICAL AND}' ;
    implication = ':-' | '\N{LEFTWARDS ARROW}' ;
    right_implication = '-:' | '\N{RIGHTWARDS ARROW}' ;
    head_predicate = identifier'(' [ arguments ] ')' ;
    query = 'ans(' [ arguments ] ')' ;
    predicate = int_ext_identifier'(' [ arguments ] ')'
              | negated_predicate
              | existential_predicate
              | comparison
              | logical_constant
              | '(' @:predicate ')';

    constant_predicate = identifier'(' ','.{ literal } ')' ;

    negated_predicate = ('~' | '\u00AC' ) predicate ;
    existential_body = arguments such_that composite_predicate;
    existential_predicate = \
        exists '(' @:existential_body ')' ;

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

    identifier = !reserved_words /[a-zA-Z_][a-zA-Z0-9_]*/
               | '`'@:?"[0-9a-zA-Z/#%._:-]+"'`';

    comparison_operator = '==' | '<' | '<=' | '>=' | '>' | '!=' ;

    text = '"' /[a-zA-Z0-9 ]*/ '"'
          | "'" /[a-zA-Z0-9 ]*/ "'" ;

    number = float | integer ;
    integer = [ '+' | '-' ] /[0-9]+/ ;
    float = [ '+' | '-' ] /[0-9]*/'.'/[0-9]+/
          | [ '+' | '-' ] /[0-9]+/'.'/[0-9]*/ ;
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

    def probabilistic_fact(self, ast):
        return Implication(
            ProbabilisticPredicate(ast[0], ast[2]),
            Constant(True),
        )

    def constant_predicate(self, ast):
        return ast[0](*ast[2])

    def rule(self, ast):
        head = ast[0]
        if isinstance(head, Expression) and head.functor == Symbol("ans"):
            return Query(ast[0], ast[2])
        else:
            return Implication(ast[0], ast[2])

    def probabilistic_rule(self, ast):
        head = ast[0]
        probability = ast[2]
        body = ast[4]
        return Implication(
            ProbabilisticPredicate(probability, head),
            body,
        )

    def constraint(self, ast):
        return RightImplication(ast[0], ast[2])

    def body(self, ast):
        return Conjunction(ast)

    def condition(self, ast):

        conditioned = ast[0]
        if isinstance(conditioned, list):
            conditioned = Conjunction(tuple(conditioned))

        condition = ast[2]
        if isinstance(condition, list):
            condition = Conjunction(tuple(condition))

        return Condition(conditioned, condition)

    def head_predicate(self, ast):
        if not isinstance(ast, Expression):
            if len(ast) == 4:
                arguments = []
                for arg in ast[2]:
                    arguments.append(arg)

                if PROB in arguments:
                    ix_prob = arguments.index(PROB)
                    arguments_not_prob = (
                        arguments[:ix_prob] +
                        arguments[ix_prob + 1:]
                    )
                    prob_arg = PROB(*arguments_not_prob)
                    arguments = (
                        arguments[:ix_prob] +
                        [prob_arg] +
                        arguments[ix_prob + 1:]
                    )

                ast = ast[0](*arguments)
            else:
                ast = ast[0]()
        return ast

    def query(self, ast):
        if len(ast) == 3:
            # Query head has arguments
            arguments = ast[1]
            return Symbol("ans")(*arguments)
        else:
            # Query head has no arguments
            return Symbol("ans")()

    def predicate(self, ast):
        if not isinstance(ast, Expression):
            ast = ast[0](*ast[2])
        return ast

    def negated_predicate(self, ast):
        return Negation(ast[1])

    def existential_predicate(self, ast):
        exp = ast[2]
        if isinstance(exp, list):
            exp = Conjunction(tuple(exp))

        for arg in ast[0]:
            exp = ExistentialPredicate(arg, exp)
        return exp

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
        if not isinstance(ast[0], Expression):
            f = Symbol(ast[0])
        else:
            f = ast[0]
        return FunctionApplication(f, args=ast[2])

    def identifier(self, ast):
        return Symbol(ast)

    def argument(self, ast):
        if ast == "...":
            return Symbol.fresh()
        else:
            return ast

    def text(self, ast):
        return Constant(ast[1])

    def integer(self, ast):
        return Constant(int("".join(ast)))

    def float(self, ast):
        return Constant(float("".join(ast)))

    def _default(self, ast):
        return ast


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR,
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
    )
