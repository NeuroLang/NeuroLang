from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu

from ...datalog import Implication
from ...datalog.aggregation import AggregationApplication
from ...expression_walker import (
    ExpressionWalker,
    PatternMatcher,
    ReplaceExpressionWalker,
    add_match
)
from ...expressions import (
    Constant,
    Expression,
    FunctionApplication,
    Lambda,
    Symbol
)
from ...logic import (
    Conjunction,
    ExistentialPredicate,
    Negation,
    UniversalPredicate
)
from ...logic.transformations import (
    CollapseConjunctions,
    RemoveTrivialOperations
)
from ...probabilistic.expressions import ProbabilisticPredicate
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

    probability = (float | int_ext_identifier );

    expression = montague | ['or'] @:rule | constraint | fact ;
    fact = constant_predicate ;
    rule = head implication body ;
    constraint = body right_implication head ;
    montague = 'montague' @:S ;
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

    S = NP VP ;
    NP = DET RCN
       | DET CN
       | '?'identifier
       | literal ;
    DET = 'some'
        | 'a'
        | 'every'
        | 'no' ;
    VP = AuxBe
       | TV NP
       | intransverb ;
    AuxBe = 'is' @:VPbe;
    VPbe = [ 'a' | 'an' ] @:CN;
    TV = transverb ;
    RCN = @:CN 'that' @:NP @:TV
        | @:CN '-' 'that' @:NP @:TV '-'
        | @:CN 'that' @:VP
        | @:CN '-' 'that' @:VP '-';
    CN = identifier ;
    intransverb = identifier ;
    transverb = identifier ;

    comparison = argument comparison_operator argument ;

    arguments = ','.{ argument }+ ;
    argument = arithmetic_operation
             | function_application
             | '...' ;

    int_ext_identifier = identifier | ext_identifier ;
    ext_identifier = '@'identifier;

    function_application = @:int_ext_identifier'(' [ @:arguments ] ')';

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

    def S(self, ast):
        exp = FunctionApplication(ast[0], (ast[1],))
        res = LambdaSolver().walk(exp)
        res = RemoveTrivialOperations().walk(res)
        res = CollapseConjunctions().walk(res)
        return res

    def NP(self, ast):
        P = Symbol.fresh()
        if isinstance(ast, Constant):
            res = Lambda((P,), FunctionApplication(P, (ast,)))
        elif ast[0] == "?":
            res = Lambda((P,), FunctionApplication(P, (ast[1],)))
        else:
            res = FunctionApplication(ast[0], (ast[1],))
        return res

    def DET(self, ast):
        P = Symbol.fresh()
        Q = Symbol.fresh()
        x = Symbol.fresh()

        if ast in ("some", "a"):
            res = Lambda((P,), Lambda((Q,), ExistentialPredicate(x, Conjunction((P(x), Q(x))))))
        elif ast == "every":
            res = Lambda((P,), Lambda((Q,), UniversalPredicate(x, Implication(Q(x), P(x)))))
        elif ast == "no":
            #res = Lambda((P,), Lambda((Q,), UniversalPredicate(x, Implication(Negation(Q(x)), P(x)))))
            res = Lambda((P,), Lambda((Q,), Negation(ExistentialPredicate(x, Conjunction((Q(x), P(x)))))))
        return res

    def VP(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        if isinstance(ast, Expression):
            return Lambda(
                (x,),  FunctionApplication(ast, (x,))
            )
        else:
            TV, NP = ast
            return Lambda(
                (x,), NP(Lambda((y,), TV(y, x)))
            )

    def TV(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        return Lambda(
            (y, x),
            FunctionApplication(ast, (x, y))
        )

    def RCN(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        if len(ast) == 2:
            CN, VP = ast
            return Lambda(
                (x,),
                Conjunction((CN(x), VP(x)))
            )
        elif len(ast) == 3:
            CN, NP, TV = ast
            return Lambda(
                (x,),
                Conjunction((CN(x), NP(Lambda((y,), TV(y, x)))))
            )
        else:
            raise ValueError()

    def CN(self, ast):
        x = Symbol.fresh()
        return Lambda(
            (x,), FunctionApplication(ast, (x,))
        )


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR,
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
    )


class LambdaSolver(ExpressionWalker):
    @add_match(FunctionApplication(Lambda, ...))
    def solve_lambda(self, expression):
        functor = expression.functor
        args = self.walk(expression.args)
        lambda_args = functor.args
        lambda_fun = self.walk(functor.function_expression)
        replacements = {arg: value for arg, value in zip(lambda_args, args)}
        res = ReplaceExpressionWalker(replacements).walk(lambda_fun)
        return self.walk(res)
