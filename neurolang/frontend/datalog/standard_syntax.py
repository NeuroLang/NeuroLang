from __future__ import annotations

from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

from lark import Lark, Transformer, UnexpectedCharacters
from lark.exceptions import UnexpectedToken
from lark.parsers.lalr_interactive_parser import InteractiveParser

from dataclasses import dataclass

from ...logic import ExistentialPredicate
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
from ...exceptions import UnexpectedTokenError

GRAMMAR = u"""
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

probabilistic_rule : head PROBA_OP arithmetic_operation IMPLICATION (condition | body)

rule : (head | query) IMPLICATION (condition | body)
IMPLICATION : ":-" | "\N{LEFTWARDS ARROW}"

query : "ans" "(" [ arguments ] ")"

condition : composite_predicate CONDITION_OP composite_predicate
CONDITION_OP : "//"
?composite_predicate : "(" conjunction ")"
                        | predicate

constraint : body RIGHT_IMPLICATION head
RIGHT_IMPLICATION : "-:" | "\N{RIGHTWARDS ARROW}"

existential_predicate : exists "(" existential_body ")"
?existential_body : arguments SUCH_THAT predicate ( "," predicate )*
SUCH_THAT : "st" | ";"

?head : head_predicate
head_predicate : identifier "(" [ arguments ] ")"

?body : conjunction
//conjunction : predicate ("," predicate)*
conjunction : predicate (("," | CONJUNCTION_SYMBOL) predicate)*
CONJUNCTION_SYMBOL : "&" | "\N{LOGICAL AND}"

negated_predicate : ("~" | NEG_UNICODE ) predicate
NEG_UNICODE : "\u00AC"
predicate : id_application
          | negated_predicate
          | existential_predicate
          | "(" comparison ")"
          | logical_constant

id_application : int_ext_identifier "(" [ arguments ] ")"

comparison : argument COMPARISON_OPERATOR argument
COMPARISON_OPERATOR : "==" | "<" | "<=" | ">=" | ">" | "!="

statement : identifier STATEMENT_OP arithmetic_operation
statement_function : identifier "(" [ arguments ] ")" STATEMENT_OP argument
STATEMENT_OP : ":="

probabilistic_fact : ( arithmetic_operation | int_ext_identifier ) PROBA_OP constant_predicate
PROBA_OP : "::"

command : "." cmd_identifier "(" [ cmd_args ] ")"
cmd_args : cmd_arg ("," cmd_arg)*
?cmd_arg : pos_item
         | keyword_item

keyword_item : identifier "=" pos_item
pos_item : arithmetic_operation | python_string

python_string : PYTHON_STRING
PYTHON_STRING : DOUBLE_QUOTE NO_DBL_QUOTE_STR DOUBLE_QUOTE
              | SINGLE_QUOTE NO_DBL_QUOTE_STR SINGLE_QUOTE
NO_DBL_QUOTE_STR : /[^"]*/

?function_application : "(" lambda_expression ")" "(" [ arguments ] ")" -> lambda_application
                     | id_application

signed_int_ext_identifier : int_ext_identifier     -> signed_int_ext_identifier
                          | "-" int_ext_identifier -> minus_signed_id
?int_ext_identifier : identifier
                    | ext_identifier
                    | lambda_expression

lambda_expression : "lambda" arguments ":" argument

arguments : argument ("," argument)*
argument : arithmetic_operation | DOTS
DOTS : "..."

arithmetic_operation : term                          -> sing_op
                     | arithmetic_operation "+" term -> plus_op
                     | arithmetic_operation "-" term -> minus_op
term : factor          -> sing_term
     | term "*" factor -> mul_term
     | term "/" factor -> div_term
factor : exponent             -> sing_factor
       | factor POW exponent -> pow_factor
POW : "**"
?exponent : literal | function_application | signed_int_ext_identifier | "(" argument ")"

fact : constant_predicate
constant_predicate : identifier "(" (literal | ext_identifier) ("," (literal | ext_identifier))* ")"
                   | identifier "(" ")"
//COMMA : ","
?literal : number | text

ext_identifier : "@" identifier
identifier : cmd_identifier | identifier_regexp
identifier_regexp : IDENTIFIER_REGEXP
IDENTIFIER_REGEXP : "`" /[0-9a-zA-Z\/#%\._:-]+/ "`"
cmd_identifier : CMD_IDENTIFIER
CMD_IDENTIFIER : /\\b(?!\\bexists\\b)(?!\\b\\u2203\\b)(?!\\bEXISTS\\b)(?!\\bst\\b)(?!\\bans\\b)[a-zA-Z_][a-zA-Z0-9_]*\\b/

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


COMPILED_GRAMMAR = Lark(GRAMMAR, parser='lalr', debug=True)


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
        print("")
        print("___start()___")
        print("ast :", ast)
        return ast[0]

    def expressions(self, ast):
        print("")
        print("___expressions()___")
        print("ast :", ast)
        if isinstance(ast[0], Expression):
            return Union(ast)
        else:
            ast = ast[0]
        return Union(ast)

    def probabilistic_rule(self, ast):
        print("")
        print("___probabilistic_rule()___")
        print("ast :", ast)
        head = ast[0]
        probability = ast[2]
        body = ast[4]
        return Implication(
            ProbabilisticFact(probability, head),
            body,
        )

    def rule(self, ast):
        print("")
        print("___rule()___")
        print("ast :", ast)
        head = ast[0]
        if isinstance(head, Expression) and head.functor == Symbol("ans"):
            return Query(ast[0], ast[2])
        else:
            return Implication(ast[0], ast[2])

    def condition(self, ast):
        print("")
        print("___condition()___")
        print("ast :", ast)
        conditioned = ast[0]
        condition = ast[2]
        return Condition(conditioned, condition)

    def constraint(self, ast):
        print("")
        print("___constraint()___")
        print("ast :", ast)
        return RightImplication(ast[0], ast[2])

    def conjunction(self, ast):
        print("")
        print("___conjunction()___")
        print("ast :", ast)
        conj = []
        for c in ast:
            if isinstance(c, Expression):
                conj.append(c)
        return Conjunction(conj)

    def existential_predicate(self, ast):
        print("")
        print("___existential_predicate()___")
        print("ast :", ast)
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
        print("")
        print("___negated_predicate()___")
        print("ast :", ast)
        return Negation(ast[0])

    def comparison(self, ast):
        print("")
        print("___comparison()___")
        print("ast :", ast)
        operator = Constant(OPERATOR[ast[1].value])
        return operator(ast[0], ast[2])

    def predicate(self, ast):
        print("")
        print("___predicate()___")
        print("ast :", ast)
        if not isinstance(ast, Expression):
            if isinstance(ast, list):
                if len(ast) == 1:
                    ast = ast[0]
                else:
                    ast = ast[0](*ast[1])
        return ast

    def head_predicate(self, ast):
        print("")
        print("___head_predicate()___")
        print("ast :", ast)
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
        print("")
        print("___query()___")
        print("ast :", ast)
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
        print("")
        print("___statement()___")
        print("ast :", ast)
        return Statement(ast[0], ast[2])

    def statement_function(self, ast):
        print("")
        print("___statement_function()___")
        print("ast :", ast)
        return Statement(
            ast[0],
            Lambda(ast[1], ast[3])
        )

    def probabilistic_fact(self, ast):
        print("")
        print("___probabilistic_fact()___")
        print("ast :", ast)
        return Implication(
            ProbabilisticFact(ast[0], ast[2]),
            Constant(True),
        )

    def command(self, ast):
        print("")
        print("___command()___")
        print("ast :", ast)
        if ast[1] == None:
            cmd = Command(ast[0], (), ())
        else:
            # only args, only kwargs or both
            name = ast[0]
            args, kwargs = ast[1]
            cmd = Command(name, args, kwargs)
        return cmd

    def cmd_args(self, ast):
        print("")
        print("___cmd_args()___")
        print("ast :", ast)
        args = ()
        kwargs = ()
        for a in ast :
            if isinstance(a, tuple):
                kwargs = kwargs + (a,)
            else:
                args = args + (a,)
        return args, kwargs

    def keyword_args(self, ast):
        print("")
        print("___keyword_args()___")
        print("ast :", ast)
        return tuple(ast)

    def keyword_item(self, ast):
        print("")
        print("___keyword_item()___")
        print("ast :", ast)
        key = ast[0]
        return (key, ast[1])

    def pos_args(self, ast):
        print("")
        print("___pos_args()___")
        print("ast :", ast)
        return tuple(ast)

    def pos_item(self, ast):
        print("")
        print("___pos_item()___")
        print("ast :", ast)
        ast = ast[0]
        if not isinstance(ast, Expression):
            return Constant((ast.children[0]).value.replace('"', ''))
        return ast

    def lambda_application(self, ast):
        print("")
        print("___lambda_application()___")
        print("ast :", ast)
        if not isinstance(ast[0], Expression):
            f = Symbol(ast[0])
        else:
            f = ast[0]

        return FunctionApplication(f, args=ast[1])

    def id_application(self, ast):
        print("")
        print("___id_application()___")
        print("ast :", ast)
        if not isinstance(ast[0], Expression):
            f = Symbol(ast[0])
        else:
            f = ast[0]

        return FunctionApplication(f, args=ast[1])

    def minus_signed_id(self, ast):
        print("")
        print("___minus_signed_id()___")
        print("ast :", ast)
        return Constant(mul)(Constant(-1), ast[0])

    def signed_int_ext_identifier(self, ast):
        print("")
        print("___signed_int_ext_identifier()___")
        print("ast :", ast)
        return ast[0]

    def lambda_expression(self, ast):
        print("")
        print("___lambda_expression()___")
        print("ast :", ast)
        return Lambda(ast[0], ast[1])

    def arguments(self, ast):
        print("")
        print("___arguments()___")
        print("ast :", ast)
        return tuple(ast)

    def argument(self, ast):
        print("")
        print("___argument()___")
        print("ast :", ast)
        ast = ast[0]
        if isinstance(ast, Expression):
            return ast
        else:
            return Symbol.fresh()

    def minus_op(self, ast):
        print("")
        print("___minus_op()___")
        print("ast :", ast)
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]

            else:
                return ast[0]

        op_str = "-"
        op = Constant(OPERATOR[op_str])
        return op(*ast)

    def plus_op(self, ast):
        print("")
        print("___plus_op()___")
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return ast[0]

        op_str = "+"
        op = Constant(OPERATOR[op_str])
        return op(*ast)

    def sing_op(self, ast):
        print("")
        print("___sing_op()___")
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]

            else:
                return ast[0]
        return 0

    def term(self, ast):
        print("")
        print("___term()___")
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            return ast[0]

    def div_term(self, ast):
        print("")
        print("___div_term()___")
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0

        op_str = "/"
        op = Constant(OPERATOR[op_str])

        return op(ast[0], ast[1])

    def mul_term(self, ast):
        print("")
        print("___mul_term()___")
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0

        op_str = "*"
        op = Constant(OPERATOR[op_str])

        return op(ast[0], ast[1])

    def sing_term(self, ast):
        print("")
        print("___sing_term()___")
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0
        else:
            return 0

    def factor(self, ast):
        print("")
        print("___factor()___")
        print("ast :", ast)
        print("len(ast) :", len(ast))
        for i in ast:
            print("* ", i)
        if isinstance(ast[0], Expression):
            return ast[0]
        elif len(ast) == 1:
            return 0
        else:
            return 0

    def pow_factor(self, ast):
        print("")
        print("___pow_factor()___")
        print("ast :", ast)
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0
        else:
            return Constant(pow)(ast[0], ast[2])

    def sing_factor(self, ast):
        print("")
        print("___sing_factor()___")
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                return ast[0]
            else:
                return 0
        else:
            return 0

    def fact(self, ast):
        print("")
        print("___fact()___")
        return Fact(ast[0])

    def constant_predicate(self, ast):
        print("")
        print("___constant_predicate()___")
        print("ast :", ast)
        predicate_name = ast[0]
        del ast[0]
        return predicate_name(*ast)

    def ext_identifier(self, ast):
        print("")
        print("___ext_identifier()___")
        ast = ast[0]
        return ExternalSymbol[ast.type](ast.name)

    def identifier(self, ast):
        print("")
        print("___identifier()___")
        print("ast :", ast)
        ast = ast[0]
        if not isinstance(ast, Symbol):
            return Symbol(ast)

        return ast

    def identifier_regexp(self, ast):
        print("")
        print("___identifier_regexp()___")
        return (ast[0]).value.replace('`', '')

    def cmd_identifier(self, ast):
        print("")
        print("___cmd_identifier()___")
        print("ast :", ast)
        return Symbol((ast[0]).value)

    def text(self, ast):
        print("")
        print("___text()___")
        return Constant((ast[0].replace("'", "")).replace('"', ''))

    def pos_int(self, ast):
        print("")
        print("___pos_int()___")
        print("ast :", ast)
        return Constant(int((ast[0]).value))

    def neg_int(self, ast):
        print("")
        print("___neg_int()___")
        return Constant(0 - int((ast[0]).value))

    def pos_float(self, ast):
        print("")
        print("___pos_float()___")
        return Constant(float((ast[0]).value))

    def neg_float(self, ast):
        print("")
        print("___neg_float()___")
        return Constant(0 - float((ast[0]).value))

    def _default(self, ast):
        print("")
        print("______")
        return ast

@dataclass
class CompleteResult:
    pos: int
    prefix: str
    token_options: set[str]

    def to_dictionary(self):
        return {"pos" : self.pos, "prefix" : self.prefix, "token_options" : self.token_options}

class LarkCompleter:
    def __init__(self, lark: Lark, start: str = None):
        if not lark.options.parser == "lalr":
            raise TypeError("The given Lark instance must be using the LALR(1) parser")
        self.parser = lark
        self.start = start

    def complete(self, text: str) -> CompleteResult:
        interactive = self.parser.parse_interactive(text, self.start)
        lex_stream = interactive.lexer_thread.lex(interactive.parser_state)
        while True:
            try:
                token = next(lex_stream)
            except StopIteration:
                break
            except UnexpectedCharacters as e:
                return self.compute_options_unexpected_char(interactive, text, e)
            interactive.feed_token(token)
        return self.compute_options_no_error(interactive, text)

    def compute_options_unexpected_char(self, interactive: InteractiveParser, text: str,
                                        e: UnexpectedCharacters) -> CompleteResult:
        prefix = text[e.pos_in_stream:]

        return CompleteResult(e.pos_in_stream, prefix, interactive.accepts())

    def compute_options_no_error(self, interactive: InteractiveParser, text: str) -> CompleteResult:
        return CompleteResult(len(text), "", interactive.accepts())

def parser(code, locals=None, globals=None, interactive=False):

    try :
        if (interactive) :
            completer = LarkCompleter(COMPILED_GRAMMAR)
            res = completer.complete(code.strip())
            # print(res)
            return res.token_options
            # return 0
        else :
            jp = COMPILED_GRAMMAR.parse(code.strip())
            return DatalogTransformer().transform(jp)
    except UnexpectedToken as e :
        raise UnexpectedTokenError from e

