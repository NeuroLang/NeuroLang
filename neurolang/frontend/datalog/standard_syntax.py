from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu
from lark import Lark
from lark import Transformer

from neurolang.logic import ExistentialPredicate

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...datalog.constraints_representation import RightImplication
from ...expressions import (
    Command,
    Constant,
    Expression,
    FunctionApplication,
    Lambda,
    Query,
    Statement,
    Symbol
)
from ...probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticFact
)

GRAMMAR_lark = u"""
start: expressions
expressions : (expression)+

?expression : rule
            | constraint
            | fact
            | probabilistic_rule
            | probabilistic_fact
            | statement
            | statement_function
            | command

probabilistic_rule : head "::" ( arithmetic_operation | int_ext_identifier ) IMPLICATION (condition | body)

rule : (head | query) IMPLICATION (condition | body)
IMPLICATION : ":-" | "\N{LEFTWARDS ARROW}"

condition : composite_predicate "//" composite_predicate
?composite_predicate : "(" conjunction ")"
                        | predicate

constraint : body RIGHT_IMPLICATION head
RIGHT_IMPLICATION : "-:" | "\N{RIGHTWARDS ARROW}"

existential_predicate : exists "(" existential_body ")"
?existential_body : arguments SUCH_THAT predicate ( CONJUNCTION_SYMBOL predicate )*
SUCH_THAT : "st" | ";"

?head : head_predicate
// Previous version :
// head_predicate : identifier "(" [ arguments ] ")"
// (See comment of identifier rule)
head_predicate : identifier L_PAR_HEAD [ arguments ] R_PAR_HEAD
R_PAR_HEAD : ")"
L_PAR_HEAD : "("

?body : conjunction
// tatsu version : ( conjunction_symbol ).{ predicate }
conjunction : predicate (CONJUNCTION_SYMBOL predicate)*
CONJUNCTION_SYMBOL : "," | "&" | "\N{LOGICAL AND}"

negated_predicate : ("~" | "\u00AC" ) predicate

// Previous version :
// predicate : int_ext_identifier "(" [ arguments ] ")"
//          | negated_predicate
//          | existential_predicate
//          | comparison
//          | logical_constant
//          | "(" predicate ")"
// (See comments of int_ext_identifier rule (Collision message 2))
predicate : int_ext_identifier L_PAR_PRED [ arguments ] R_PAR_PRED
          | negated_predicate
          | existential_predicate
          | comparison
          | logical_constant
          | L_PAR_PRED predicate R_PAR_PRED
R_PAR_PRED : ")"
L_PAR_PRED : "("

comparison : argument COMPARISON_OPERATOR argument
COMPARISON_OPERATOR : "==" | "<" | "<=" | ">=" | ">" | "!="

query : "ans(" [ arguments ] ")"

// Previous version :
//statement : identifier ":=" ( lambda_expression | arithmetic_operation | int_ext_identifier )
// (See comments of lambda_expression and int_ext_identifier (Collision message 1) rules)
statement : identifier ":=" arithmetic_operation
statement_function : identifier "(" [ arguments ] ")" ":=" argument

probabilistic_fact : ( arithmetic_operation | int_ext_identifier ) "::" constant_predicate

command : "." cmd_identifier "(" [ cmd_args ] ")"
cmd_args : pos_args [ "," keyword_args ] | keyword_args

keyword_args : keyword_item ( "," keyword_item )*
keyword_item : identifier "=" pos_item

pos_args : pos_item ("," pos_item)*
pos_item : arithmetic_operation | python_string

python_string : PYTHON_STRING
PYTHON_STRING : DOUBLE_QUOTE NO_DBL_QUOTE_STR DOUBLE_QUOTE
              | SINGLE_QUOTE NO_DBL_QUOTE_STR SINGLE_QUOTE
NO_DBL_QUOTE_STR : /[^"]*/

// Previous version :
//function_application : "(" lambda_expression ")" "(" [ arguments ] ")" -> lambda_application
//                     | int_ext_identifier "(" [ arguments ] ")"        -> id_application
// (See comment of int_ext_identifier rule (Collision message 2))
// Collision message :
// collision occurred in state: {
//	<argument : function_application * >
//	<exponent : function_application * >
// Solution :
// - in argument rule : remove 'function_application'
function_application : L_PAR_FUNC lambda_expression R_PAR_FUNC L_PAR_FUNC [ arguments ] R_PAR_FUNC -> lambda_application
                     | int_ext_identifier L_PAR_FUNC [ arguments ] R_PAR_FUNC        -> id_application
R_PAR_FUNC : ")"
L_PAR_FUNC : "("

signed_int_ext_identifier : int_ext_identifier     -> signed_int_ext_identifier
                          | "-" int_ext_identifier -> minus_signed_id

// Collision message 1 :
// collision occurred in state: {
//	<function_application : int_ext_identifier * LPAR arguments RPAR>
//	<signed_int_ext_identifier : int_ext_identifier * >
//	<statement : identifier __ANON_0 int_ext_identifier * >
//	<function_application : int_ext_identifier * LPAR RPAR>
// Solution :
// - in statement rule : remove 'int_ext_identifier'
//
// Collision message 2 :
// collision occurred in state: {
//	<predicate : int_ext_identifier LPAR arguments RPAR * >
//	<function_application : int_ext_identifier LPAR arguments RPAR * >
// and
// collision occurred in state: {
//	<predicate : int_ext_identifier LPAR RPAR * >
//	<function_application : int_ext_identifier LPAR RPAR * >
// Solution :
// - Create terminals 'R_PAR_PRED : ")"' and 'L_PAR_PRED : "("' and then use them in predicate rule
// - create terminals 'R_PAR_FUNC : ")"' and 'L_PAR_FUNC : "("' and then use them in function_application rule
?int_ext_identifier : identifier
                    | ext_identifier
                    | lambda_expression

// Collision message :
// collision occurred in state: {
//	<statement : identifier __ANON_0 lambda_expression * >
//	<int_ext_identifier : lambda_expression * >
// Solution :
// - in statement rule : remove 'lambda_expression'
lambda_expression : "lambda" arguments ":" argument

arguments : argument ("," argument)*
// Previous version :
// argument : arithmetic_operation | function_application | DOTS
// (See comment of function_application rule)
argument : arithmetic_operation | DOTS
DOTS : "..."

arithmetic_operation : term                          -> sing_op
                     | arithmetic_operation "+" term -> plus_op
                     | arithmetic_operation "-" term -> minus_op
term : factor          -> sing_term
     | term "*" factor -> mul_term
     | term "/" factor -> div_term
factor : exponent             -> sing_factor
       | factor "**" exponent -> pow_factor
?exponent : literal | function_application | signed_int_ext_identifier | "(" argument ")"

fact : constant_predicate
// Previous version :
// constant_predicate : identifier "(" literal ("," literal)* ")"
//                    | identifier "(" ")"
// (See comment of ext_identifier rule)
constant_predicate : identifier "(" (literal | ext_identifier) ("," (literal | ext_identifier))* ")"
                   | identifier "(" ")"

// Previous version :
// ?literal : number | text | ext_identifier
// (See comment of ext_identifier rule)
?literal : number | text

// Collision message :
// collision occurred in state: {
//        <literal : ext_identifier * >
//        <int_ext_identifier : ext_identifier * >
// Solution :
// - in literal rule : remove 'ext_identifier'
// - in constant_predicate rule : replace 'literal' by 'literal | ext_identifier'
ext_identifier : "@" identifier

// Bug arisen during the process of adaptation of the grammar :
// 1/ Input : '''
// A(x)-:B(x)'''
// 2/ Error message :
// UnexpectedToken: Unexpected token Token('RIGHT_IMPLICATION', '-:') at line 2, column 5.
// Expected one of: 
//	* ":="
// Previous tokens: [Token('R_PAR_FUNC', ')')]
// 3/ Cause : the parser recognise an identifier instead of 
// 4/ Solution :
// - Separate identifier from int_ext_identifier
// - replace int_ext_identifier by (int_ext_identifier | identifier) in all rules
// 
// Collision message :
//collision occurred in state: {
//	<head_predicate : identifier L_PAR_FUNC R_PAR_FUNC * >
//	<constant_predicate : identifier L_PAR_FUNC R_PAR_FUNC * >
//	<statement_function : identifier L_PAR_FUNC R_PAR_FUNC * __ANON_0 argument>
// Solution :
// - Create terminals 'R_PAR_HEAD : ")"' and 'L_PAR_HEAD : "("' and then use them in head_predicate rule
identifier : cmd_identifier | identifier_regexp
identifier_regexp : IDENTIFIER_REGEXP
IDENTIFIER_REGEXP : "`" /[0-9a-zA-Z\/#%\._:-]+/ "`"

cmd_identifier : CMD_IDENTIFIER
CMD_IDENTIFIER : /\\b(?!\\bexists\\b)(?!\\b\\u2203\\b)(?!\\bEXISTS\\b)(?!\\bst\\b)(?!\\bans\\b)[a-zA-Z_][a-zA-Z0-9_]*\\b/

// Tatsu version :
//reserved_words : exists | ST | ANS
//ST : "st"
//ANS : "ans"

exists : EXISTS
EXISTS : "exists" | "\u2203" | "EXISTS"

text : TEXT
TEXT : DOUBLE_QUOTE /[a-zA-Z0-9 ]*/ DOUBLE_QUOTE
     | SINGLE_QUOTE /[a-zA-Z0-9 ]*/ SINGLE_QUOTE
DOUBLE_QUOTE : "\\""
SINGLE_QUOTE : "'"

?number : integer | float
?integer : INT     -> pos_int
         | "-" INT -> neg_int
?float : FLOAT     -> pos_float
       | "-" FLOAT -> neg_float

logical_constant : FALSE | TRUE
TRUE             : "True" | "\u22A4"
FALSE            : "False" | "\u22A5"

WHITESPACE : /[\t ]+/
//NEWLINE : /[(\\u000C)? (\\r)? \\n]+/

%import common.INT
%import common.FLOAT
%import common.NEWLINE

%ignore WHITESPACE
%ignore NEWLINE
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


COMPILED_GRAMMAR_lark = Lark(GRAMMAR_lark, debug=True)


class ExternalSymbol(Symbol):
    def __repr__(self):
        return "@S{{{}: {}}}".format(self.name, self.__type_repr__)


class DatalogTransformer(Transformer):

    def __init__(self, locals=None, globals=None):
        super().__init__()

        if locals is None:
            locals = {}
        if globals is None:
            globals = {}

        self.locals = locals
        self.globals = globals

    def start(self, ast):
        return ast[0]

    def expressions(self, ast):
        if isinstance(ast[0], Expression):
            return Union(ast)
        else:
            ast = ast[0]
        return Union(ast)

    #     return 0

    def probabilistic_rule(self, ast):
        head = ast[0]
        probability = ast[1]
        body = ast[3]
        return Implication(
            ProbabilisticFact(probability, head),
            body,
        )

    def rule(self, ast):
        head = ast[0]
        if isinstance(head, Expression) and head.functor == Symbol("ans"):
            return Query(ast[0], ast[2])
        else:
            return Implication(ast[0], ast[2])

    def condition(self, ast):
        conditioned = ast[0]
        condition = ast[1]
        return Condition(conditioned, condition)

    def constraint(self, ast):
        return RightImplication(ast[0], ast[2])

    def conjunction(self, ast):
        conj = []
        for c in ast:
            if isinstance(c, Expression):
                conj.append(c)
        return Conjunction(conj)

    def existential_predicate(self, ast):
        ast1 = ast[1].children
        exp = []
        for i in ast1:
            if isinstance(i, FunctionApplication):
                exp.append(i)

        if len(exp) == 1:
            exp = exp[0]
        else:
            exp = Conjunction(tuple(exp))

        for arg in ast1[0]:
            exp = ExistentialPredicate(arg, exp)

        return exp

    def negated_predicate(self, ast):
        return Negation(ast[0])

    def comparison(self, ast):
        operator = Constant(OPERATOR[ast[1].value])
        return operator(ast[0], ast[2])

    def predicate(self, ast):
        # print("")
        # print("___predicate()___")
        # print("ast :", ast)
        if not isinstance(ast, Expression):
            # print("not isinstance(ast, Expression)")
            if isinstance(ast, list):
                # print("isinstance(ast, list)")
                if len(ast) == 1:
                    # print("len(ast) == 1")
                    ast = ast[0]
                else:
                    # print("len(ast) != 1")
                    ast = ast[0](*ast[1])
            # else:
                # print("not isinstance(ast, list)")
        # print("isinstance(ast, Expression)")
        # print("res = ast :", ast)
        return ast

    def head_predicate(self, ast):
        if ast[1] != None:
            arguments = list(ast[1])

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
        ast = ast[0]
        if ast != None:
            if isinstance(ast, tuple):
                arguments = ast
            else:
                arguments = tuple(ast)

            return Symbol("ans")(*arguments)
        else:
            return Symbol("ans")()

    def statement(self, ast):
        return Statement(ast[0], ast[1])

    def statement_function(self, ast):
        return Statement(
            ast[0],
            Lambda(ast[1], ast[2])
        )

    def probabilistic_fact(self, ast):
        return Implication(
            ProbabilisticFact(ast[0], ast[1]),
            Constant(True),
        )

    def command(self, ast):
        if ast[1] == None:
            cmd = Command(ast[0], (), ())
        else:
            # only args, only kwargs or both
            name = ast[0]
            args, kwargs = ast[1]
            cmd = Command(name, args, kwargs)
        return cmd

    def cmd_args(self, ast):

        if len(ast) == 1:
            # No args
            args = ()
            kwargs = ast[0]
        else:
            if (ast[1] == None):
                # only args. ex : A,"http://myweb/file.csv"
                args = ast[0]
                kwargs = ()
            else:
                # args and kwargs
                args, kwargs = ast

        return args, kwargs

    def keyword_args(self, ast):
        return tuple(ast)

    def keyword_item(self, ast):
        key = ast[0]
        return (key, ast[1])

    def pos_args(self, ast):
        return tuple(ast)

    def pos_item(self, ast):
        ast = ast[0]
        if not isinstance(ast, Expression):
            return Constant((ast.children[0]).value.replace('"', ''))
        return ast

    def lambda_application(self, ast):
        print("")
        print("___lambda_application()___")
        print("ast :", ast)
        if not isinstance(ast[0], Expression):
            print("not isinstance(ast[0], Expression)")
            f = Symbol(ast[0])
        else:
            print("isinstance(ast[0], Expression)")
            f = ast[0]
        print("res = FunctionApplication(f, args=ast[1]) :")
        print(FunctionApplication(f, args=ast[1]))
        return FunctionApplication(f, args=ast[1])

    def id_application(self, ast):
        if not isinstance(ast[0], Expression):
            f = Symbol(ast[0])
        else:
            f = ast[0]

        return FunctionApplication(f, args=ast[1])

    def minus_signed_id(self, ast):
        return Constant(mul)(Constant(-1), ast[0])

    def signed_int_ext_identifier(self, ast):
        return ast[0]

    def lambda_expression(self, ast):
        return Lambda(ast[0], ast[1])

    def arguments(self, ast):
        return tuple(ast)

    def argument(self, ast):
        ast = ast[0]
        if isinstance(ast, Expression):
            return ast
        else:
            return Symbol.fresh()

    def minus_op(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]

            else:
                return ast[0]

        op_str = "-"
        op = Constant(OPERATOR[op_str])
        return op(*ast)

    def plus_op(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return ast[0]

        op_str = "+"
        op = Constant(OPERATOR[op_str])
        return op(*ast)

    def sing_op(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]

            else:
                return ast[0]

        op = Constant(OPERATOR[ast[1]])
        return 0

    def term(self, ast):
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            return ast[0]

    def div_term(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0

        op_str = "/"
        op = Constant(OPERATOR[op_str])

        return op(ast[0], ast[1])

    def mul_term(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0

        op_str = "*"
        op = Constant(OPERATOR[op_str])

        return op(ast[0], ast[1])

    def sing_term(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0
        else:
            return 0

    def factor(self, ast):
        if isinstance(ast[0], Expression):
            return ast[0]
        elif len(ast) == 1:
            return 0
        else:
            return 0

    def pow_factor(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0
        else:
            return Constant(pow)(ast[0], ast[1])

    def sing_factor(self, ast):
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0
        else:
            return 0

    def fact(self, ast):
        return Fact(ast[0])

    def constant_predicate(self, ast):
        predicate_name = ast[0]
        del ast[0]
        return predicate_name(*ast)

    def ext_identifier(self, ast):
        ast = ast[0]
        return ExternalSymbol[ast.type](ast.name)

    def identifier(self, ast):
        ast = ast[0]
        if not isinstance(ast, Symbol):
            return Symbol(ast)

        return ast

    def identifier_regexp(self, ast):
        return (ast[0]).value.replace('`', '')

    def cmd_identifier(self, ast):
        return Symbol((ast[0]).value)

    def text(self, ast):
        return Constant((ast[0].replace("'", "")).replace('"', ''))

    def pos_int(self, ast):
        return Constant(eval(ast[0]))

    def neg_int(self, ast):
        return Constant(0 - eval(ast[0]))

    def pos_float(self, ast):
        return Constant(eval(ast[0]))

    def neg_float(self, ast):
        return Constant(0 - eval(ast[0]))

    def _default(self, ast):
        return ast


def parser(code, locals=None, globals=None, interactive=None):
    if interactive:
        return None
    else:
        jp = COMPILED_GRAMMAR_lark.parse(code.strip())
        return DatalogTransformer().transform(jp)
