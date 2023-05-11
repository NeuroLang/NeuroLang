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

GRAMMAR_tatsu = u"""
    @@grammar::Datalog
    @@parseinfo :: True
    @@whitespace :: /[\t ]+/
    @@eol_comments :: /#([^\n]*?)$/

    start = expressions $ ;

    expressions = ( newline ).{ expression };


    expression = rule
               | constraint
               | fact
               | probabilistic_rule
               | probabilistic_fact
               | statement
               | statement_function
               | command ;
    fact = constant_predicate ;
    probabilistic_fact = ( arithmetic_operation | int_ext_identifier ) '::' constant_predicate ;
    rule = (head | query) implication (condition | body) ;
    probabilistic_rule = head '::' ( arithmetic_operation | int_ext_identifier ) implication (condition | body) ;
    constraint = body right_implication head ;
    statement = identifier ':=' ( lambda_expression | arithmetic_operation | int_ext_identifier ) ;
    statement_function = identifier'(' [ arguments ] ')' ':=' argument ;
    command = '.' @:cmd_identifier '(' [ @:cmd_args ] ')';
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
    existential_body = arguments such_that ( conjunction_symbol ).{ predicate }+ ;
    existential_predicate = \
        exists '(' @:existential_body ')' ;

    comparison = argument comparison_operator argument ;

    arguments = ','.{ argument }+ ;
    argument = arithmetic_operation
             | function_application
             | '...' ;

    signed_int_ext_identifier = [ '-' ] int_ext_identifier ;
    int_ext_identifier = identifier | ext_identifier | lambda_expression;
    ext_identifier = '@'identifier;

    lambda_expression = 'lambda' arguments ':' argument;

    function_application = '(' @:lambda_expression ')' '(' [ @:arguments ] ')'
                         | @:int_ext_identifier '(' [ @:arguments ] ')' ;

    arithmetic_operation = term [ ('+' | '-') term ] ;

    term = factor [ ( '*' | '/' ) factor ] ;

    factor =  exponent [ '**' exponential ];

    exponential = exponent ;

    exponent = literal
             | function_application
             | signed_int_ext_identifier
             | '(' @:argument ')' ;

    literal = number
            | text
            | ext_identifier ;

    identifier = !reserved_words /[a-zA-Z_][a-zA-Z0-9_]*/
               | '`'@:?"[0-9a-zA-Z/#%._:-]+"'`';

    cmd_identifier = !reserved_words /[a-zA-Z_][a-zA-Z0-9_]*/ ;
    cmd_args = @:pos_args [ ',' @:keyword_args ]
             | @:keyword_args ;
    pos_args = pos_item { ',' pos_item }* ;
    pos_item = ( arithmetic_operation | python_string ) !'=' ;
    keyword_args = keyword_item { ',' keyword_item }* ;
    keyword_item = identifier '=' pos_item ;
    python_string = '"' @:/[^"]*/ '"'
                  | "'" @:/[^']*/ "'" ;

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

GRAMMAR = u"""
start: expressions
expressions : (_expression)+

_expression : (comparison_operator | python_string | reserved_words | cmd_identifier | term)+

fact : constant_predicate
probabilistic_fact : (arithmetic_operation | int_ext_identifier) "::" constant_predicate
rule : (head | query) implication (condition | body)
probabilistic_rule : head "::" ( arithmetic_operation | int_ext_identifier ) implication (condition | body)
constraint : body right_implication head 
statement : identifier ":=" ( lambda_expression | arithmetic_operation | int_ext_identifier )
statement_function : identifier"(" [ arguments ] ")" ":=" argument
command : "." cmd_identifier "(" [ cmd_args ] ")"
head : head_predicate
body : conjunction
condition :	composite_predicate "//" composite_predicate
conjunction : predicate [conjunction_symbol predicate]+ ;
composite_predicate "(" conjunction ")"
	| predicate

such_that : "st"
	| ";"

conjunction_symbol : ","
	| "&"
	| "\N{LOGICAL AND}"

implication : ":-"
 | "\N{LEFTWARDS ARROW}"

right_implication : "-:" 
| "\N{RIGHTWARDS ARROW}"

head_predicate : identifier"(" [ arguments ] ")"

query : "ans(" [ arguments ] ")"

predicate : int_ext_identifier"(" [ arguments ] ")"
             | negated_predicate
             | existential_predicate
             | comparison
             | logical_constant
             | "(" predicate ")"

constant_predicate : identifier "(" literal ["," literal]+

negated_predicate : ("~" | "\u00AC"" ) predicate

existential_body : arguments such_that predicate [conjunction_symbol predicate ]+

existential_predicate : \
       exists "(" existential_body ")"

comparison : argument comparison_operator argument

arguments : argument ["," argument]+

argument : arithmetic_operation
            | function_application
            | "..."

signed_int_ext_identifier : [ "-" ] int_ext_identifier

int_ext_identifier :  ext_identifier 
| lambda_expression

lambda_expression : "lambda" arguments ":" argument

function_application : "(" lambda_expression ")" "(" [ arguments ] ")"
                        | int_ext_identifier "(" [ arguments ] ")"

arithmetic_operation = term [ ("+" | "-") term ]

term : factor [ ( "*" | "/" ) factor ]

factor :  exponent [ "**" exponential ]

exponential : exponent

exponent : literal

literal : number
        | text
        | ext_identifier

ext_identifier : "@" identifier

identifier : reserved_words~0 /[a-zA-Z_][a-zA-Z0-9_]*/
           | "`" /[0-9a-zA-Z\/#%\._:-]+/ "`"

cmd_identifier : reserved_words~0 /[a-zA-Z_][a-zA-Z0-9_]*/

reserved_words : exists
                   | ST
                   | ANS
ST : "st"
ANS : "ans"

exists : "exists" | "\u2203" | "EXISTS"

python_string : PYTHON_STRING
PYTHON_STRING : DOUBLE_QUOTE NO_DBL_QUOTE_STR DOUBLE_QUOTE
              | SINGLE_QUOTE NO_DBL_QUOTE_STR SINGLE_QUOTE
NO_DBL_QUOTE_STR : /[^"]*/

comparison_operator : COMPARISON_OPERATOR
COMPARISON_OPERATOR : "==" | "<" | "<=" | ">=" | ">" | "!="

text : TEXT
TEXT : DOUBLE_QUOTE ALPHANUM_STR DOUBLE_QUOTE
     | SINGLE_QUOTE ALPHANUM_STR SINGLE_QUOTE
ALPHANUM_STR : /[a-zA-Z0-9 ]*/
DOUBLE_QUOTE : "\\""
SINGLE_QUOTE : "'"

number : integer | float
integer : SIGNED_INT
float : SIGNED_FLOAT

logical_constant : FALSE | TRUE
TRUE             : "True" | "\u22A4"
FALSE            : "False" | "\u22A5"

WHITESPACE : /[\t ]+/
%ignore WHITESPACE

NEWLINE : /[\\u000C)? (\\r)? \\n]+/
%ignore NEWLINE

%import common.SIGNED_INT
%import common.SIGNED_FLOAT
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


COMPILED_GRAMMAR_tatsu = tatsu.compile(GRAMMAR_tatsu)
COMPILED_GRAMMAR = Lark(GRAMMAR, parser='lalr')

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
            ProbabilisticFact(ast[0], ast[2]),
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
            ProbabilisticFact(probability, head),
            body,
        )

    def constraint(self, ast):
        return RightImplication(ast[0], ast[2])

    def statement(self, ast):
        return Statement(ast[0], ast[2])

    def statement_function(self, ast):
        return Statement(
            ast[0],
            Lambda(ast[2], ast[-1])
        )

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
        if len(exp) == 1:
            exp = exp[0]
        else:
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
        #print()
        #print("___ext_identifier___")
        #print("ast :", ast)
        ast = ast[1]
        #print("ast[1] :", ast)
        #print("(ast[1]).type :", ast.type)
        #print("(ast[1]).name :", ast.name)
        #print("ExternalSymbol[(ast[1]).type]((ast[1]).name) :", ExternalSymbol[ast.type](ast.name))
        return ExternalSymbol[ast.type](ast.name)

    def lambda_expression(self, ast):
        return Lambda(ast[1], ast[3])

    def arithmetic_operation(self, ast):
        if isinstance(ast, Expression):
            return ast

        if len(ast) == 1:
            return ast[0]

        op = Constant(OPERATOR[ast[1]])

        return op(*ast[0::2])

    def term(self, ast):
        print()
        print("___term___")
        print("ast :", ast)
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            print("len(ast) == 1")
            print("ast[0] :", ast[0])
            return ast[0]

        print("len(ast) != 1")
        print("ast[0] :", ast[0])
        print("ast[1] :", ast[1])
        print("ast[2] :", ast[2])
        print("OPERATOR[ast[1]] :", OPERATOR[ast[1]])
        op = Constant(OPERATOR[ast[1]])
        print("op = Constant(OPERATOR[ast[1]]) :", Constant(OPERATOR[ast[1]]))
        print("op(ast[0], ast[2]) :",op(ast[0], ast[2]) )

        return op(ast[0], ast[2])

    def factor(self, ast):
        #print()
        #print("___factor___")
        #print("ast :", ast)
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            #print("* len(ast) == 1")
            #print("ast[0] :", ast[0])
            return ast[0]
        else:
            #print("* len(ast) != 1")
            #print("ast[0] :", ast[0])
            #print("ast[1] :", ast[1])
            #print("ast[2] :", ast[2])
            #print("Constant(pow)(ast[0], ast[2]) :", Constant(pow)(ast[0], ast[2]))
            return Constant(pow)(ast[0], ast[2])

    def function_application(self, ast):
        if not isinstance(ast[0], Expression):
            f = Symbol(ast[0])
        else:
            f = ast[0]
        return FunctionApplication(f, args=ast[1])

    def signed_int_ext_identifier(self, ast):
        if isinstance(ast, Expression):
            return ast
        else:
            return Constant(mul)(Constant(-1), ast[1])

    def identifier(self, ast):
        #print()
        #print("___identifier___")
        #print("ast :", ast)
        #print("type(ast) :", type(ast))
        #print("Symbol(ast) :", Symbol(ast))
        return Symbol(ast)

    def argument(self, ast):
        if ast == "...":
            return Symbol.fresh()
        else:
            return ast

    def text(self, ast):
        #print()
        #print("___text___")
        #print("ast :", ast)
        #print("ast[1] :", ast[1])
        #print("Constant(ast[1]) :", Constant(ast[1]))
        return Constant(ast[1])

    def integer(self, ast):
        return Constant(int("".join(ast)))

    def float(self, ast):
        #print()
        #print("___float___")
        #print("ast :", ast)
        #print("\"\".join(ast) :", "".join(ast))
        #print("float(\"\".join(ast)) :", float("".join(ast)))
        #print("Constant(float(\"\".join(ast))) :", Constant(float("".join(ast))))
        return Constant(float("".join(ast)))

    def cmd_identifier(self, ast):
        #print()
        #print("___cmd_identifier___")
        #print("ast :", ast)
        #print("type(ast) :", type(ast))
        #print("Symbol(ast) :", Symbol(ast))
        return Symbol(ast)

    def pos_args(self, ast):
        args = [ast[0]]
        for arg in ast[1]:
            args.append(arg[1])
        return tuple(args)

    def keyword_item(self, ast):
        key = ast[0]
        return (key, ast[2])

    def pos_item(self, ast):
        if not isinstance(ast, Expression):
            return Constant(ast)
        return ast

    def keyword_args(self, ast):
        kwargs = [ast[0]]
        for kwarg in ast[1]:
            kwargs.append(kwarg[1])
        return tuple(kwargs)

    def cmd_args(self, ast):
        if isinstance(ast, list) and len(ast) == 2:
            args, kwargs = ast
        elif isinstance(ast[0], tuple):
            args = ()
            kwargs = ast
        else:
            args = ast
            kwargs = ()
        return args, kwargs

    def command(self, ast):
        if not isinstance(ast, list):
            cmd = Command(ast, (), ())
        else:
            name = ast[0]
            args, kwargs = ast[1]
            cmd = Command(name, args, kwargs)
        return cmd

    def _default(self, ast):
        return ast


class DatalogTransformer(Transformer):

    def __init__(self, locals=None, globals=None):
        super().__init__()

        if locals is None:
            locals = {}
        if globals is None:
            globals = {}

        self.locals = locals
        self.globals = globals

    def ext_identifier(self, ast):
        #print()
        #print("___ext_identifier___")
        #print("ast :", ast)
        name = ast[0]
        #print("name = ast[0] :", name)
        type = ""
        #print("type = \"\"")
        return ExternalSymbol[type](name)

    def identifier(self, ast):
        return Symbol(str(ast[0]))

    def cmd_identifier(self, ast):
        return Symbol(str(ast[0]))

    def text(self, ast):
        return Constant((ast[0].replace("'", "")).replace('"', ''))

    def integer(self, ast):
        return Constant(int(ast[0]))

    def float(self, ast):
        return Constant(float(ast[0]))

    def _default(self, ast):
        return ast


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR_tatsu,
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
    )

def parser_lark(code, locals=None, globals=None):
    lark_tree = COMPILED_GRAMMAR.parse(
        code.strip()
    )
    return DatalogTransformer().transform(lark_tree)