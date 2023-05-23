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
    existential_predicate = exists '(' @:existential_body ')' ;

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

GRAMMAR_lark = u"""
start: expressions
expressions : (expression)+

?expression : fact | probabilistic_fact | statement | statement_function | command | query
//?expression : head

existential_predicate : exists "(" existential_body ")"
existential_body : arguments such_that predicate ( conjunction_symbol predicate )*
such_that : "st" | ";"

probabilistic_rule : head "::" ( arithmetic_operation | int_ext_identifier ) implication (condition | body)

rule : (head | query) implication (condition | body)

implication : ":-" | "\N{LEFTWARDS ARROW}"

condition : composite_predicate "//" composite_predicate
composite_predicate : "(" conjunction ")"
                        | predicate

negated_predicate : ("~" | "\u00AC" ) predicate

constraint : body right_implication head
right_implication : "-:" | "\N{RIGHTWARDS ARROW}"


body : conjunction
conjunction : (predicate | function_application_identifier) ( conjunction_symbol (predicate | function_application_identifier) )*
            | "()"

conjunction_symbol : "," | "&" | "\N{LOGICAL AND}"
predicate : function_application_identifier
              | negated_predicate
              | comparison
              | logical_constant
              | "(" predicate ")"

function_application_identifier : int_ext_identifier "(" [ arguments ] ")"

comparison : argument COMPARISON_OPERATOR argument
COMPARISON_OPERATOR : "==" | "<" | "<=" | ">=" | ">" | "!="

?head : head_predicate
head_predicate : identifier "(" [ arguments ] ")"

query : "ans(" [ arguments ] ")"

statement : identifier ":=" ( lambda_expression | arithmetic_operation | int_ext_identifier )
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

function_application : "(" lambda_expression ")" "(" [ arguments ] ")" -> lambda_application
                     | int_ext_identifier "(" [ arguments ] ")"        -> id_application

signed_int_ext_identifier : int_ext_identifier     -> signed_int_ext_identifier
                          | "-" int_ext_identifier -> minus_signed_id
?int_ext_identifier : identifier
                    | ext_identifier
                    | lambda_expression

lambda_expression : "lambda" arguments ":" argument

arguments : argument ("," argument)*
argument : arithmetic_operation | function_application | DOTS
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
constant_predicate : identifier "(" literal ("," literal)* ")"
                   | identifier "(" ")"

?literal : number | text | ext_identifier

ext_identifier : "@" identifier
identifier : cmd_identifier | "`" /[0-9a-zA-Z\/#%\._:-]+/ "`"
cmd_identifier : /\\b(?!\\bexists\\b)(?!\\b\\u2203\\b)(?!\\bEXISTS\\b)(?!\\bst\\b)(?!\\bans\\b)[a-zA-Z_][a-zA-Z0-9_]*\\b/

reserved_words : exists | ST | ANS
ST : "st"
ANS : "ans"

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


COMPILED_GRAMMAR_tatsu = tatsu.compile(GRAMMAR_tatsu)
COMPILED_GRAMMAR_lark = Lark(GRAMMAR_lark, debug=True)


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
        # print()
        # print("___start___")
        # print("ast :", ast)
        return ast

    def expressions(self, ast):
        # print()
        # print("___expressions___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        if isinstance(ast, Expression):
            # print("isinstance(ast, Expression)")
            ast = (ast,)
            # print("ast = (ast,) :", ast)
        # else:
        #     print("not isinstance(ast, Expression)")
        # print("res = Union(ast) :", Union(ast))
        # print("type(res) = type(Union(ast)) :", type(Union(ast)))
        return Union(ast)

    def fact(self, ast):
        # print()
        # print("___fact___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("Fact(ast) :", Fact(ast))
        return Fact(ast)

    def probabilistic_fact(self, ast):
        # print()
        # print("___probabilistic_fact___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("ast[2] :", ast[2])
        # print("Constant(True) :", Constant(True))
        # print("res = Implication(ProbabilisticFact(ast[0], ast[2]), Constant(True), ) :", Implication(ProbabilisticFact(ast[0], ast[2]), Constant(True), ))
        return Implication(
            ProbabilisticFact(ast[0], ast[2]),
            Constant(True),
        )

    def constant_predicate(self, ast):
        # print()
        # print("___constant_predicate___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("type(ast[0]) :", type(ast[0]))
        # print("ast[2] :", ast[2])
        # print("*ast[2] :", *ast[2])
        # print("ast[0](*ast[2]) :", ast[0](*ast[2]))
        return ast[0](*ast[2])

    def rule(self, ast):
        #print()
        #print("___rule___")
        #print("ast :", ast)
        head = ast[0]
        #print("head = ast[0] :", head)
        #print("ast[2] :", ast[2])
        if isinstance(head, Expression) and head.functor == Symbol("ans"):
            #print("isinstance(head, Expression) and head.functor == Symbol(\"ans\")")
            #print("res = Query(ast[0], ast[2]) :", Query(ast[0], ast[2]))
            return Query(ast[0], ast[2])
        else:
            #print("not isinstance(head, Expression) and head.functor == Symbol(\"ans\")")
            #print("res = Implication(ast[0], ast[2]) :", Implication(ast[0], ast[2]))
            return Implication(ast[0], ast[2])

    def probabilistic_rule(self, ast):
        #print()
        #print("___probabilistic_rule___")
        #print("ast :", ast)
        #print()
        head = ast[0]
        #print("head = ast[0] :", head)
        probability = ast[2]
        #print("probability = ast[2] :", probability)
        body = ast[4]
        #print("body = ast[4] :", body)
        #print("ProbabilisticFact(probability, head) :", ProbabilisticFact(probability, head))
        #print("res = Implication(ProbabilisticFact(probability, head), body,) :", Implication(
        #    ProbabilisticFact(probability, head),
        #    body,
        #))
        return Implication(
            ProbabilisticFact(probability, head),
            body,
        )

    def constraint(self, ast):
        # print()
        # print("___constraint___")
        # print("ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        # print("ast[2] :", ast[2])
        # print("res = RightImplication(ast[0], ast[2]) :", RightImplication(ast[0], ast[2]))
        return RightImplication(ast[0], ast[2])

    def statement(self, ast):
        # print()
        # print("___statement___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("ast[2] :", ast[2])
        # print("res = Statement(ast[0], ast[2]) :", Statement(ast[0], ast[2]))
        return Statement(ast[0], ast[2])

    def statement_function(self, ast):
        # print()
        # print("___statement_function___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("ast[2] :", ast[2])
        # print("ast[-1] :", ast[-1])
        # print("Lambda(ast[2], ast[-1]) :", Lambda(ast[2], ast[-1]))
        # print("res = Statement(ast[0], Lambda(ast[2], ast[-1])) :",
        #       Statement(ast[0], Lambda(ast[2], ast[-1])))
        return Statement(
            ast[0],
            Lambda(ast[2], ast[-1])
        )

    def body(self, ast):
        # print()
        # print()
        # print("___body___")
        # print("ast :", ast)
        # print("res = Conjunction(ast) :", Conjunction(ast))
        return Conjunction(ast)

    def condition(self, ast):
        print()
        print("___condition___")
        print("ast :", ast)
        conditioned = ast[0]
        print("conditioned = ast[0] :", conditioned)
        if isinstance(conditioned, list):
            print("isinstance(conditioned, list)")
            conditioned = Conjunction(tuple(conditioned))
            print("conditioned = Conjunction(tuple(conditioned)) :", conditioned)
        else:
           print(" not isinstance(conditioned, list) :")

        condition = ast[2]
        print("condition = ast[2] :", condition)
        if isinstance(condition, list):
            print("isinstance(condition, list)")
            condition = Conjunction(tuple(condition))
            print("condition = Conjunction(tuple(condition)) :", condition)

        print("res = Condition(conditioned, condition) :", Condition(conditioned, condition))
        return Condition(conditioned, condition)

    def head_predicate(self, ast):
        # print()
        # print("___head_predicate___")
        # print("ast :", ast)
        if not isinstance(ast, Expression):
            # print("not isinstance(ast, Expression)")
            if len(ast) == 4:
                # print("len(ast) == 4")
                arguments = []
                for arg in ast[2]:
                    arguments.append(arg)
                # print("arguments after append :", arguments)

                if PROB in arguments:
                    # print("PROB in arguments")
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
                # print("len(ast) != 4")
                ast = ast[0]()
        # else:
        #     print("is isinstance(ast, Expression)")
        return ast

    def query(self, ast):
        # print()
        # print("___query___")
        # print("ast :", ast)
        # print("len(ast) :", len(ast))
        if len(ast) == 3:
            # Query head has arguments
            # print("len(ast) == 3")
            arguments = ast[1]
            # print("arguments = ast[1] :", ast[1])
            # print("res = Symbol(\"ans\")(*arguments) :", Symbol("ans")(*arguments))
            return Symbol("ans")(*arguments)
        else:
            # Query head has no arguments
            # print("len(ast) != 3")
            # print("res = Symbol(\"ans\")() :", Symbol("ans")())
            return Symbol("ans")()

    def predicate(self, ast):
        # print()
        # print("___predicate___")
        # print("ast :", ast)
        if not isinstance(ast, Expression):
            # print("not isinstance(ast, Expression)")
            # print("ast[0] :", ast[0])
            # print("ast[2] :", ast[2])
            # print("*ast[2] :", *ast[2])
            ast = ast[0](*ast[2])
            # print("res = ast = ast[0](*ast[2]) :", ast)
        # else:
            # print("isinstance(ast, Expression)")
            # print("nothing to do")
            # print("res = ast :", ast)
        return ast

    def negated_predicate(self, ast):
        # print()
        # print("___negated_predicate___")
        # print("ast :", ast)
        # print("ast[1] : ", ast[1])
        # print("res = Negation(ast[1] :", Negation(ast[1]))
        return Negation(ast[1])

    def existential_predicate(self, ast):
        print()
        print("___existential_predicate___")
        print("ast :", ast)
        exp = ast[2]
        print("exp = ast[2] :", exp)
        if len(exp) == 1:
            print("len(exp) == 1")
            exp = exp[0]
            print("exp = exp[0] :", exp)
        else:
            print("len(exp) != 1")
            print("exp avant :", exp)
            print("tuple(exp) :", tuple(exp))
            exp = Conjunction(tuple(exp))
            print("exp = Conjunction(tuple(exp)) :", exp)

        print("ast[0] :", ast[0])
        print("for arg in ast[0]")
        for arg in ast[0]:
            print("arg :", arg)
            print("type(arg) :", type(arg))
            print("exp 1 :", exp)
            print("type(exp) :", type(exp))
            exp = ExistentialPredicate(arg, exp)
            print("exp = ExistentialPredicate(arg, exp) :", exp)
        print("res = exp : ", exp)
        return exp
        # return 0

    def comparison(self, ast):
        # print()
        # print("___comparison___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        # print("ast[2] :", ast[2])
        # print("OPERATOR[ast[1]] :", OPERATOR[ast[1]])
        operator = Constant(OPERATOR[ast[1]])
        # print("operator = Constant(OPERATOR[ast[1]]) :", operator)
        # print("res = operator(ast[0], ast[2]) :", operator(ast[0], ast[2]))
        return operator(ast[0], ast[2])

    def arguments(self, ast):
        # print()
        # print("___arguments___")
        # print("ast :", ast)
        if isinstance(ast, Expression):
            # print("is isinstance(ast, Expression)")
            # print("res = (ast,) :", (ast,))
            return (ast,)
        # print("not isinstance(ast, Expression)")
        # print("res = tuple(ast) : ", tuple(ast))
        return tuple(ast)

    def ext_identifier(self, ast):
        # print()
        # print("___ext_identifier___")
        # print("ast :", ast)
        ast = ast[1]
        # print("ast[1] :", ast)
        # print("(ast[1]).type :", (ast).type)
        # print("(ast[1]).name :", (ast).name)
        # print("ExternalSymbol[ast[1].type](ast[1].name) :", ExternalSymbol[ast.type](ast.name))
        return ExternalSymbol[ast.type](ast.name)

    def lambda_expression(self, ast):
        # print()
        # print("___lambda_expression___")
        # print("ast :", ast)
        # print("ast[1] :", ast[1])
        # print("ast[3] :", ast[3])
        # print("Lambda(ast[1], ast[3]) :", Lambda(ast[1], ast[3]))
        return Lambda(ast[1], ast[3])

    def arithmetic_operation(self, ast):
        # print()
        # print("___arithmetic_operation___")
        # print("ast :", ast)
        if isinstance(ast, Expression):
            # print("is isinstance(ast, Expression)")
            # print("res = ast :", ast)
            return ast

        if len(ast) == 1:
            # print("not isinstance(ast, Expression) - len(ast) ==1")
            # print("res = ast[0] :", ast[0])
            return ast[0]

        # print("not isinstance(ast, Expression) - len(ast) !=1")
        op = Constant(OPERATOR[ast[1]])
        # print("ast[1] :", ast[1])
        # print("OPERATOR[ast[1]] :", OPERATOR[ast[1]])
        # print("op = Constant(OPERATOR[ast[1]]) :", op)
        # print("*ast :", *ast)
        # print("*ast[0::2] :", *ast[0::2])
        print("op(*ast[0::2]) :", op(*ast[0::2]))

        return op(*ast[0::2])

    def term(self, ast):
        # print()
        # print("___term___")
        # print("ast :", ast)
        if isinstance(ast, Expression):
            # print("isinstance(ast, Expression)")
            # print("res = ast :", ast)
            return ast
        elif len(ast) == 1:
            # print("not isinstance(ast, Expression) AND len(ast) == 1")
            # print("res = ast[0] :", ast[0])
            return ast[0]

        # print("not isinstance(ast, Expression) AND len(ast) != 1")
        # print("ast[1] :", ast[1])
        # print("type(ast[1]) :", type(ast[1]))
        # print("len(ast[1]) :", len(ast[1]))
        # print("OPERATOR[ast[1]] :", OPERATOR[ast[1]])
        op = Constant(OPERATOR[ast[1]])
        # print("Constant(OPERATOR[ast[1]]) :", op)

        # print("ast[2] :", ast[2])
        # print("res = op(ast[0], ast[2]) :", op(ast[0], ast[2]))
        return op(ast[0], ast[2])

    def factor(self, ast):
        # print()
        # print("___factor___")
        if isinstance(ast, Expression):
            # print("isinstance(ast, Expression)")
            # print("res = ast :", ast)
            return ast
        elif len(ast) == 1:
            # print("not isinstance(ast, Expression) AND len(ast) == 1")
            # print("ast :", ast)
            # print("res = ast[0] :", ast[0])
            return ast[0]
        else:
            # print("not isinstance(ast, Expression) AND len(ast) != 1")
            # print("ast :", ast)
            # print("ast[0] :", ast[0])
            # print("res = ast[2] :", ast[2])
            # print("Constant(pow)(ast[0], ast[2]) :", Constant(pow)(ast[0], ast[2]))
            return Constant(pow)(ast[0], ast[2])

    def function_application(self, ast):
        # print()
        # print("____function_application___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        if not isinstance(ast[0], Expression):
        #     print("not isinstance(ast[0], Expression)")
            f = Symbol(ast[0])
        #     print("Symbol(ast[0]) :", f)
        #     print("ast[1] :", ast[1])
        #     print("res = FunctionApplication(Symbol(ast[0]), args=ast[1]) :", FunctionApplication(f, args=ast[1]))
        else:
            # print("is isinstance(ast[0], Expression)")
            f = ast[0]
            # print("ast[1] :", ast[1])
            # print("res = FunctionApplication(ast[0], args=ast[1]) :", FunctionApplication(f, args=ast[1]))

        return FunctionApplication(f, args=ast[1])

    def signed_int_ext_identifier(self, ast):
        # print()
        # print("___signed_int_ext_identifier___")
        # print("ast :", ast)
        if isinstance(ast, Expression):
            # print("is isinstance(ast, Expression)")
            # print("res = ast :", ast)
            return ast
        else:
            # print("not isinstance(ast, Expression)")
            # print("ast[1] :", ast[1])
            # print("Constant(-1) :", Constant(-1))
            # print("res = Constant(mul)(Constant(-1), ast[1]) :", Constant(mul)(Constant(-1), ast[1]))
            return Constant(mul)(Constant(-1), ast[1])

    def identifier(self, ast):
        # print()
        # print("___identifier___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("Symbol(ast) :", Symbol(ast))
        return Symbol(ast)

    def argument(self, ast):
        # print()
        # print("___argument___")
        # print("ast :", ast)
        if ast == "...":
            return Symbol.fresh()
        else:
            return ast

    def text(self, ast):
        # print()
        # print("___text___")
        # print("ast :", ast)
        # print("ast[1] :", ast[1])
        # print("Constant(ast[1]) :", Constant(ast[1]))
        return Constant(ast[1])

    def integer(self, ast):
        # print()
        # print("___integer___")
        # print("ast :", ast)
        # print("\"\".join(ast) :", "".join(ast))
        # print("int(\"\".join(ast)) :", int("".join(ast)))
        # print("Constant(int(\"\".join(ast))) :", Constant(int("".join(ast))))
        return Constant(int("".join(ast)))

    def float(self, ast):
        # print()
        # print("___float___")
        # print("ast :", ast)
        # print("\"\".join(ast) :", "".join(ast))
        # print("float(\"\".join(ast)) :", float("".join(ast)))
        # print("Constant(float(\"\".join(ast))) :", Constant(float("".join(ast))))
        return Constant(float("".join(ast)))

    def cmd_identifier(self, ast):
        # print()
        # print("___cmd_identifier___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("res = Symbol(ast) :", Symbol(ast))
        return Symbol(ast)

    def pos_args(self, ast):
        # print()
        # print("___pos_args___")
        # print("ast :", ast)
        args = [ast[0]]
        # print("args = [ast[0]] :", args)
        for arg in ast[1]:
            args.append(arg[1])
        # print("args after append :", args)
        # print("res = tuple(args)", tuple(args))
        return tuple(args)

    def keyword_item(self, ast):
        # print()
        # print("___keyword_item___")
        # print("ast :", ast)
        key = ast[0]
        # print("key = ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        # print("ast[2] :", ast[2])
        # print("res = (key, ast[2]) :", (key, ast[2]))
        return (key, ast[2])

    def pos_item(self, ast):
        # print()
        # print("___pos_item___")
        # print("ast :", ast)
        if not isinstance(ast, Expression):
            # print("not isinstance(ast, Expression)")
            # print("res = Constant(ast) :", Constant(ast))
            return Constant(ast)
        # print("is isinstance(ast, Expression)")
        # print("res = ast :", ast)
        return ast

    def keyword_args(self, ast):
        # print()
        # print("___keyword_args___")
        # print("ast :", ast)
        kwargs = [ast[0]]
        # print("kwargs = [ast[0]] :", kwargs)
        for kwarg in ast[1]:
            kwargs.append(kwarg[1])
        # print("kwargs after append :", kwargs)
        # print("tuple(kwargs) :", tuple(kwargs))
        return tuple(kwargs)

    def cmd_args(self, ast):
        # print()
        # print("___cmd_args___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        if isinstance(ast, list) and len(ast) == 2:
            # print("is isinstance(ast, list) and len(ast) == 2")
            args, kwargs = ast
            # print("args from \"args, kwargs = ast\" :", args)
            # print("kwargs from \"args, kwargs = ast\" :", kwargs)
        elif isinstance(ast[0], tuple):
            # print(" is isinstance(ast[0], tuple)")
            args = ()
            kwargs = ast
            # print("kwargs = ast :", kwargs)
        else:
            # print("else")
            args = ast
            kwargs = ()
            # print("args = ast :", args)

        return args, kwargs

    def command(self, ast):
        # print()
        # print("___command___")
        # print("ast :", ast)
        if not isinstance(ast, list):
            # print("not isinstance(ast, list)")
            cmd = Command(ast, (), ())
            # print("res = Command(ast, (), ()) :", cmd)
        else:
            # print("is isinstance(ast, list)")
            name = ast[0]
            args, kwargs = ast[1]
            # print("name = ast[0] :", name)
            # print("args, kwargs = ast[1]", ast[1])
            cmd = Command(name, args, kwargs)
            # print("res = Command(name, args, kwargs) :", cmd)
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

    def start(self, ast):
        print()
        print("___start___")
        print("ast :", ast)
        print("res = ast[0] :", ast[0])
        return ast[0]

    def expressions(self, ast):
        # print()
        # print("___expressions___")
        # print("ast :", ast)
        #     type_ast = type(ast)
        #     print("type(ast) :", type_ast)
        # print("ast[0] :", ast[0])
        if isinstance(ast[0], Expression):
            # print("isinstance(ast, Expression)")
            # print("res = Union(ast) :", Union(ast))
            return Union(ast)
        else:
            # print("not isinstance(ast, Expression)")
            ast = ast[0]
            type_ast = type(ast)
            # print("type(ast) :", type_ast)
        # print("res = Union(ast) :", Union(ast))
        return Union(ast)

    #     return 0

    def head_predicate(self, ast):
        print()
        print("___head_predicate___")
        print("ast :", ast)
        print("len(ast) :", len(ast))
        if ast[1] != None:
            print("ast[1] != None")
            arguments = list(ast[1])
            print("arguments = list(ast[1]) :", arguments)

            if PROB in arguments:
                print("PROB in arguments")
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
            print("ast[1] == None")
            ast = ast[0]()
        print("res =", ast)
        return ast

    def query(self, ast):
        print()
        print("___query___")
        print("ast :", ast)
        print("len(ast) :", len(ast))
        ast = ast[0]
        if ast != None:
            # Query head has arguments
            print("ast[0] != None")
            arguments = ast[0]
            print("arguments = ast[1] :", ast[0])
            print("res = Symbol(\"ans\")(*arguments) :", Symbol("ans")(*arguments))
            return Symbol("ans")(*arguments)
        else:
            # Query head has no arguments
            print("ast[0] == None")
            print("res = Symbol(\"ans\")() :", Symbol("ans")())
            return Symbol("ans")()

    def statement(self, ast):
        print()
        print("___statement___")
        print("ast :", ast)
        print("len(ast) :", len(ast))
        print("ast[0] :", ast[0])
        print("ast[1] :", ast[1])
        print("res = Statement(ast[0], ast[1]) :", Statement(ast[0], ast[1]))
        return Statement(ast[0], ast[1])

    def statement_function(self, ast):
        print()
        print("___statement_function___")
        print("ast :", ast)
        print("len(ast) :", len(ast))
        print("ast[0] :", ast[0])
        print("ast[1] :", ast[1])
        print("ast[2] :", ast[2])
        print("Lambda(ast[1], ast[2]) :", Lambda(ast[1], ast[2]))
        print("res = Statement(ast[0], Lambda(ast[1], ast[2])) :",
              Statement(ast[0], Lambda(ast[1], ast[2])))
        return Statement(
            ast[0],
            Lambda(ast[1], ast[2])
        )
        # return 0

    def probabilistic_fact(self, ast):
        # print()
        # print("___probabilistic_fact___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        # print("Constant(True) :", Constant(True))
        # print("res = Implication(ProbabilisticFact(ast[0], ast[1]), Constant(True), ) :",
        #       Implication(ProbabilisticFact(ast[0], ast[1]), Constant(True), ))
        return Implication(
            ProbabilisticFact(ast[0], ast[1]),
            Constant(True),
        )

    def command(self, ast):
        # print()
        # print("___command___")
        # print("ast :", ast)
        # print("len(ast) :", len(ast))
        # print("ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        if ast[1] == None:
            # No arguments. ex : .load_csv()
            # print("ast[1] == None")
            cmd = Command(ast[0], (), ())
            # print("res = Command(ast[0], (), ()) :", cmd)
        else:
            # only args, only kwargs or both
            # print("is isinstance(ast, list)")
            name = ast[0]
            args, kwargs = ast[1]
            # print("name = ast[0] :", name)
            # print("args :", args)
            # print("kwargs :", kwargs)
            cmd = Command(name, args, kwargs)
            # print("res = Command(name, args, kwargs) :", cmd)
        return cmd

    def cmd_args(self, ast):
        # print()
        # print("___cmd_args___")
        # print("ast :", ast)
        # print("len(ast) :", len(ast))
        # print("ast[0] :", ast[0])

        if len(ast) == 1:
            # only kwargs. ex : sep=",", header=None
            # print("len(ast) == 1")
            args = ()
            # print("args = () :", args)
            kwargs = ast[0]
            # print("kwargs = ast[0] :", kwargs)
        else:
            # len(ast) == 2
            # print("ast[1] :", ast[1])
            # print("type(ast[1]) :", type(ast[1]))
            if (ast[1] == None):
                # only args. ex : A,"http://myweb/file.csv"
                # print("(len(ast) == 2) and (ast[1] == None)")
                args = ast[0]
                # print("args = ast[0] :", args)
                kwargs = ()
                print("kwargs = () :", kwargs)
            else:
                # args and kwargs
                # print("(len(ast) == 2) and (ast[1] != None)")
                args, kwargs = ast
                # print("args from \"args, kwargs = ast\" :", args)
                # print("kwargs from \"args, kwargs = ast\" :", kwargs)

        # print("res = args, kwargs")
        return args, kwargs

    def keyword_args(self, ast):
        # print()
        # print("___keyword_args___")
        # print("ast :", ast)
        # print("res = tuple(ast) :", tuple(ast))
        return tuple(ast)

    def keyword_item(self, ast):
        # print()
        # print("___keyword_item___")
        # print("ast :", ast)
        key = ast[0]
        # print("key = ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        print("(key, ast[1]) :", (key, ast[1]))
        return (key, ast[1])

    def pos_args(self, ast):
        # print()
        # print("___pos_args___")
        # print("ast :", ast)
        # print("res = tuple(ast) :", tuple(ast))
        return tuple(ast)

    def pos_item(self, ast):
        # print()
        # print("___pos_item___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        ast = ast[0]
        if not isinstance(ast, Expression):
            # print("not isinstance(ast, Expression)")
            # print("ast.children[0] : ", ast.children[0])
            # print("(ast.children[0]).value : ", (ast.children[0]).value)
            # print("(ast.children[0]).value.replace('\", '') : ", (ast.children[0]).value.replace('"', ''))
            # print("res = Constant((ast.children[0]).value.replace('\"', '')) :", Constant((ast.children[0]).value.replace('"', '')))
            return Constant((ast.children[0]).value.replace('"', ''))
        # print("is isinstance(ast, Expression)")
        # print("res = ast :", ast)
        return ast

    def lambda_application(self, ast):
        # print()
        # print("____lambda_application___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        if not isinstance(ast[0], Expression):
            # print("not isinstance(ast[0], Expression)")
            f = Symbol(ast[0])
            # print("Symbol(ast[0]) :", f)
            # print("ast[1] :", ast[1])
            # print("res = FunctionApplication(Symbol(ast[0]), args=ast[1]) :", FunctionApplication(f, args=ast[1]))
        else:
            # print("is isinstance(ast[0], Expression)")
            f = ast[0]
            # print("ast[1] :", ast[1])
            # print("res = FunctionApplication(ast[0], args=ast[1]) :", FunctionApplication(f, args=ast[1]))

        return FunctionApplication(f, args=ast[1])

    def id_application(self, ast):
        # print()
        # print("____id_application___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        if not isinstance(ast[0], Expression):
            # print("not isinstance(ast[0], Expression)")
            f = Symbol(ast[0])
            # print("Symbol(ast[0]) :", f)
            # print("ast[1] :", ast[1])
            # print("res = FunctionApplication(Symbol(ast[0]), args=ast[1]) :", FunctionApplication(f, args=ast[1]))
        else:
            # print("is isinstance(ast[0], Expression)")
            f = ast[0]
            # print("ast[1] :", ast[1])
            # print("res = FunctionApplication(ast[0], args=ast[1]) :", FunctionApplication(f, args=ast[1]))

        return FunctionApplication(f, args=ast[1])

    def minus_signed_id(self, ast):
        # print()
        # print("___minus_signed_id___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("ast[0] :", ast[0])
        # print("Constant(-1) :", Constant(-1))
        # print("res = Constant(mul)(Constant(-1), ast[0]) :", Constant(mul)(Constant(-1), ast[0]))
        return Constant(mul)(Constant(-1), ast[0])

    def signed_int_ext_identifier(self, ast):
        # print()
        # print("___signed_int_ext_identifier___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("res = ast[0] :", ast[0])
        return ast[0]

    def lambda_expression(self, ast):
        # print()
        # print("___lambda_expression___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("ast[1] :", ast[1])
        # print("res = Lambda(ast[0], ast[1]) :", Lambda(ast[0], ast[1]))
        return Lambda(ast[0], ast[1])

    def arguments(self, ast):
        print()
        print("___arguments___")
        print("ast :", ast)
        if len(ast) == 1:
            print("is isinstance(ast, Expression)")
            print("res = (ast,) :", (ast,))
            return (ast)
        print("not isinstance(ast, Expression)")
        print("res = tuple(ast) : ", tuple(ast))
        return tuple(ast)

    def argument(self, ast):
        print()
        print("___argument___")
        print("ast :", ast)
        ast = ast[0]
        print("ast = ast[0] :", ast)
        if isinstance(ast, Expression):
            print("is isinstance(ast, Expression)")
            return ast
        else:
            print("not isinstance(ast, Expression)")
            return Symbol.fresh()

    def minus_op(self, ast):
        # print()
        # print("___minus_op___")
        # print("ast :", ast)

        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("is isinstance(ast, Expression)")
                # print("ast[0] :", ast[0])
                return ast[0]

            else:
                # print("not isinstance(ast, Expression) - len(ast) ==1")
                # print("ast[0] :", ast[0])
                return ast[0]

        # print("not isinstance(ast, Expression) - len(ast) !=1")
        op_str = "-"
        # print("op_str :", op_str)
        op = Constant(OPERATOR[op_str])
        # print("OPERATOR[op_str] :", OPERATOR[op_str])
        # print("op = Constant(OPERATOR[op_str]) :", op)
        # print("*ast :", *ast)
        # print("op(*ast) :", op(*ast))
        return op(*ast)

    def plus_op(self, ast):
        # print()
        # print("___plus_op___")
        # print("ast :", ast)

        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("is isinstance(ast, Expression)")
                # print("ast[0] :", ast[0])
                return ast[0]

            else:
                # print("not isinstance(ast, Expression) - len(ast) ==1")
                # print("ast[0] :", ast[0])
                return ast[0]

        # print("not isinstance(ast, Expression) - len(ast) !=1")
        op_str = "+"
        # print("op_str :", op_str)
        op = Constant(OPERATOR[op_str])
        # print("OPERATOR[op_str] :", OPERATOR[op_str])
        # print("op = Constant(OPERATOR[op_str]) :", op)
        # print("*ast :", *ast)
        # print("op(*ast) :", op(*ast))
        return op(*ast)

    def sing_op(self, ast):
        # print()
        # print("___sing_op___")
        # print("ast :", ast)
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("is isinstance(ast, Expression)")
                # print("ast[0] :", ast[0])
                return ast[0]

            else:
                # print("not isinstance(ast, Expression) - len(ast) ==1")
                # print("ast[0] :", ast[0])
                return ast[0]

        print("not isinstance(ast, Expression) - len(ast) !=1")
        op = Constant(OPERATOR[ast[1]])
        # print("ast[1] :", ast[1])
        # print("OPERATOR[ast[1]] :", OPERATOR[ast[1]])
        # print("op = Constant(OPERATOR[ast[1]]) :", op)
        # print("*ast :", *ast)
        # print("*ast[0::2] :", *ast[0::2])
        # print("op(*ast[0::2]) :", op(*ast[0::2]))

        return 0

    def term(self, ast):
        # print()
        # print("___term___")
        # print("ast :", ast)
        if isinstance(ast, Expression):
            return ast
        elif len(ast) == 1:
            return ast[0]

    def div_term(self, ast):
        # print()
        # print("___div_term___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("isinstance(ast[0], Expression)")
                # print("res = ast[0] :", ast[0])
                return ast[0]
            else:
                # print("not isinstance(ast[0], Expression) AND len(ast) == 1")
                return 0

        # print("not isinstance(ast[0], Expression) AND len(ast) != 1")
        op_str = "/"
        # print("m :", m)
        # print("type(m) :", type(m))
        # print("OPERATOR[m] :", OPERATOR[m])
        op = Constant(OPERATOR[op_str])
        # print("Constant(OPERATOR[m]) :", Constant(OPERATOR[m]))

        # print("ast[1] :", ast[1])
        # print("res = op(ast[0], ast[1]) :", op(ast[0], ast[1]))
        return op(ast[0], ast[1])

    def mul_term(self, ast):
        # print()
        # print("___mul_term___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("isinstance(ast[0], Expression)")
                # print("res = ast[0] :", ast[0])
                return ast[0]
            else:
                # print("not isinstance(ast[0], Expression) AND len(ast) == 1")
                return 0

        # print("not isinstance(ast[0], Expression) AND len(ast) != 1")
        op_str = "*"
        # print("m :", m)
        # print("type(m) :", type(m))
        # print("OPERATOR[m] :", OPERATOR[m])
        op = Constant(OPERATOR[op_str])
        # print("Constant(OPERATOR[m]) :", Constant(OPERATOR[m]))

        # print("ast[1] :", ast[1])
        # print("res = op(ast[0], ast[1]) :", op(ast[0], ast[1]))
        return op(ast[0], ast[1])

    def sing_term(self, ast):
        # print()
        # print("___sing_term___")
        # print("ast :", ast)
        # print("len(ast) :", len(ast))
        # print("ast[0] :", ast[0])
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("isinstance(ast[0], Expression)")
                # print("res = ast[0] :", ast[0])
                return ast[0]
            else:
                # print("not isinstance(ast[0], Expression) AND len(ast) == 1")
                return 0
        else:
            # print("not isinstance(ast[0], Expression) AND len(ast) != 1")
            return 0

    def factor(self, ast):
        # print()
        # print("___factor___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("type(ast[0]) :", type(ast[0]))
        if isinstance(ast[0], Expression):
            # print("isinstance(ast, Expression)")
            return ast[0]
        elif len(ast) == 1:
            # print("not isinstance(ast, Expression) AND len(ast) == 1")
            # return ast[0]
            return 0
        else:
            # print("not isinstance(ast, Expression) AND len(ast) != 1")
            # return Constant(pow)(ast[0], ast[2])
            return 0

    def pow_factor(self, ast):
        # print()
        # print("___pow_factor___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("ast[0] :", ast[0])
        # print("type(ast[0]) :", type(ast[0]))
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("isinstance(ast, Expression)")
                # print("res = ast[0] :", ast[0])
                return ast[0]
            else:
                # print("not isinstance(ast, Expression) AND len(ast) == 1")
                # return ast[0]
                return 0
        else:
            # print("not isinstance(ast, Expression) AND len(ast) != 1")
            # print("ast[1] :", ast[1])
            # print("type(ast[1]) :", type(ast[1]))
            # print("res = Constant(pow)(ast[0], ast[1]) :",Constant(pow)(ast[0], ast[1]))
            return Constant(pow)(ast[0], ast[1])

    def sing_factor(self, ast):
        # print()
        # print("___sing_factor___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("ast[0] :", ast[0])
        # print("type(ast[0]) :", type(ast[0]))
        if len(ast) == 1:
            if isinstance(ast[0], Expression):
                # print("isinstance(ast, Expression)")
                # print("res = ast[0] :", ast[0])
                return ast[0]
            else:
                # print("not isinstance(ast, Expression) AND len(ast) == 1")
                return ast[0]
                return 0
        else:
            # print("not isinstance(ast, Expression) AND len(ast) != 1")
            return Constant(pow)(ast[0], ast[2])
            return 0

    def fact(self, ast):
        print()
        print("___fact___")
        print("ast :", ast)
        print("ast[0] :", ast[0])
        print("type(ast[0]) :", type(ast[0]))
        print("Fact(ast[0]) :", Fact(ast[0]))
        return Fact(ast[0])

    def constant_predicate(self, ast):
        print()
        print("___constant_predicate___")
        print("ast :", ast)
        print("len(ast) :", len(ast))
        predicate_name = ast[0]
        print("predicate_name = ast[0] :", predicate_name)
        print("type(ast[0]) :", type(predicate_name))
        del ast[0]
        print("ast :", ast)
        print("len(ast) :", len(ast))
        print("*ast :", *ast)
        print("res = predicate_name(*ast) :", predicate_name(*ast))
        return predicate_name(*ast)

    def ext_identifier(self, ast):
        # print()
        # print("___ext_identifier___")
        # print("ast :", ast)
        ast = ast[0]
        # print("ast[0] :", ast)
        # print("type(ast[0]) :", type(ast))
        # print("(ast[0]).type :", ast.type)
        # print("(ast[0]).name :", ast.name)
        # print("ExternalSymbol[ast.type](ast.name) :", ExternalSymbol[ast.type](ast.name))
        return ExternalSymbol[ast.type](ast.name)

    def identifier(self, ast):
        print()
        print("___identifier___")
        print("ast :", ast)
        ast = ast[0]
        print("ast[0] :", ast)
        print("type(ast[0]) :", type(ast))

        if not isinstance(ast, Symbol):
            print("not isinstance(ast, Symbol)")
            print("(ast[0]).value :", ast.value)
            print("type((ast[0]).value) :", type(ast.value))
            print("res = Symbol((ast[0]).value) :", Symbol(ast.value))
            return Symbol(ast.value)
        else:
            print("is isinstance(ast, Symbol)")

        return ast

    def cmd_identifier(self, ast):
        # print()
        # print("___cmd_identifier___")
        # print("ast :", ast)
        # print("type(ast) :", type(ast))
        # print("len(ast) :", len(ast))
        # print("ast[0] :", ast[0])
        # print("type(ast[0]) :", type(ast[0]))
        # print("(ast[0]).value :", (ast[0]).value)
        # print("type((ast[0]).value) :", type((ast[0]).value))
        # print("res = Symbol((ast[0]).value) :", Symbol((ast[0]).value))
        return Symbol((ast[0]).value)

    def text(self, ast):
        # print()
        # print("___text___")
        # print("ast :", ast)
        # print("ast[0] :", ast[0])
        # print("(ast[0].replace("'", "")).replace('"', '') :", (ast[0].replace("'", "")).replace('"', ''))
        # print("res = Constant((ast[0].replace("'", "")).replace('"', '')) :", Constant((ast[0].replace("'", "")).replace('"', '')))
        return Constant((ast[0].replace("'", "")).replace('"', ''))

    def pos_int(self, ast):
        # print()
        # print("___pos_int___")
        # print("ast :", ast)
        # print("eval(ast[0]) :", eval(ast[0]))
        # print("type(eval(ast[0])) :", type(eval(ast[0])))
        # print("Constant(eval(ast[0])) :", Constant(eval(ast[0])))
        return Constant(eval(ast[0]))

    def neg_int(self, ast):
        # print()
        # print("___pos_int___")
        # print("ast :", ast)
        # print("eval(ast[0]) :", eval(ast[0]))
        # print("type(eval(ast[0])) :", type(eval(ast[0])))
        # print("Constant(eval(ast[0])) :", Constant(eval(ast[0])))
        return Constant(0 - eval(ast[0]))

    def pos_float(self, ast):
        # print()
        # print("__ pos_float __ :")
        # print("ast :", ast)
        # print("eval(ast[0]) :", eval(ast[0]))
        # print("type(eval(ast[0])) :", type(eval(ast[0])))
        # print("Constant(eval(ast[0])) :", Constant(eval(ast[0])))
        return Constant(eval(ast[0]))

    def neg_float(self, ast):
        # print()
        # print("__ neg_float __ :")
        # print("ast :", ast)
        # print("eval(ast[0]) :", eval(ast[0]))
        # print("type(eval(ast[0])) :", type(eval(ast[0])))
        # print("Constant(eval(ast[0])) :", Constant(eval(ast[0])))
        return Constant(0 - eval(ast[0]))

    def _default(self, ast):
        return ast


def parser(code, locals=None, globals=None):
    return tatsu.parse(
        COMPILED_GRAMMAR_tatsu,
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
    )

def parser_lark(code, locals=None, globals=None):
    jp = COMPILED_GRAMMAR_lark.parse(code.strip())
    return DatalogTransformer().transform(jp)