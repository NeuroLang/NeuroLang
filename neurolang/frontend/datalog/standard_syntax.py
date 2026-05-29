import json
import os

from lark import Lark, Transformer
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, LarkError
from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

from ...datalog import Conjunction, Fact, Implication, Negation, Union
from ...datalog.constraints_representation import RightImplication
from ...datalog.expressions import AggregationApplication
from ...exceptions import UnexpectedTokenError, UnexpectedCharactersError, NeuroLangException
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
from ...logic import ExistentialPredicate
from ...probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticFact,
)
from ...utils.interactive_parsing import LarkCompleter

# Aggregation function names recognised in rule head arguments.
AGGREGATION_FUNCS = frozenset({"count", "sum", "max", "min", "mean", "avg", "std"})


GRAMMAR = u"""
start: expressions
expressions : (expression)+

?expression : rule
            | constraint
            | fact
            | probabilistic_rule
            | probabilistic_fact
            | prob_head_rule
            | statement
            | statement_function
            | command

probabilistic_rule : head PROBA_OP arithmetic_operation IMPLICATION (condition | body)

rule : (head | query) IMPLICATION (condition | body)
IMPLICATION : ":-" | "\N{LEFTWARDS ARROW}"

// PROB[pred] :- body  —  declare a probabilistic rule
prob_head_rule : "PROB" "[" predicate "]" IMPLICATION (condition | body)
               | "PROB" "[" predicate CONDITION_OP body "]" IMPLICATION (condition | body)

query : "ans" "(" [ arguments ] ")"

condition : composite_predicate CONDITION_OP composite_predicate
CONDITION_OP : "//"
?composite_predicate : "(" conjunction ")"
                        | predicate

constraint : body RIGHT_IMPLICATION head
RIGHT_IMPLICATION : "-:" | "\N{RIGHTWARDS ARROW}"

existential_predicate : exists "(" existential_body ")"
?existential_body : arguments (";" | SUCH_THAT_WORD) predicate ( "," predicate )*
SUCH_THAT_WORD : "st"

?head : head_predicate
head_predicate : identifier "(" [ arguments ] ")"

?body : conjunction
//conjunction : predicate ("," predicate)*
conjunction : predicate (("," | "&" | AND_SYMBOL) predicate)*
AND_SYMBOL : "\N{LOGICAL AND}"

negated_predicate : ("~" | NEG_UNICODE ) predicate
NEG_UNICODE : "\u00AC"
predicate : id_application
          | negated_predicate
          | existential_predicate
          | "(" comparison ")"
          | logical_constant
          | prob_body_predicate
          | marg_body_predicate
          | succ_body_predicate

// PROB[pred] = var  or  PROB[pred | cond] = var  —  probability query inside a rule body
prob_body_predicate : "PROB" "[" predicate "]" "=" argument
                    | "PROB" "[" predicate CONDITION_OP body "]" "=" argument

// MARG[pred] = var  —  marginal probability query inside a rule body
// MARG[pred1, pred2] = var  —  conjunction inside MARG (same semantics as PROB)
marg_body_predicate : "MARG" "[" conjunction "]" "=" argument
                    | "MARG" "[" conjunction CONDITION_OP body "]" "=" argument

// SUCC[...] = var  —  success probability query (skipped in execution)
succ_body_predicate : "SUCC" "[" SUCC_INNER "]" "=" argument
SUCC_INNER : /[^]]+/

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
              | SINGLE_QUOTE NO_SING_QUOTE_STR SINGLE_QUOTE
NO_DBL_QUOTE_STR : /[^"]*/
NO_SING_QUOTE_STR : /[^']*/

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

?literal : number | text

ext_identifier : "@" identifier
identifier : cmd_identifier | identifier_regexp
identifier_regexp : IDENTIFIER_REGEXP
IDENTIFIER_REGEXP : "`" /[0-9a-zA-Z\\/#%._:-]+/ "`"
cmd_identifier : CMD_IDENTIFIER
CMD_IDENTIFIER : /\\b(?!\\bexists\\b)(?!\\b\\u2203\\b)(?!\\bEXISTS\\b)(?!\\bst\\b)(?!\\bans\\b)[a-zA-Z_][a-zA-Z0-9_]*\\b/

exists : EXISTS_WORD | EXISTS_SYMBOL
EXISTS_WORD : "exists" | "EXISTS"
EXISTS_SYMBOL : "\u2203"

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
        return ast[0]

    def expressions(self, ast):
        if isinstance(ast[0], Expression):
            all_formulas = []
            for item in ast:
                if isinstance(item, Union):
                    all_formulas.extend(item.formulas)
                elif isinstance(item, Expression):
                    all_formulas.append(item)
            return Union(tuple(all_formulas))
        else:
            ast = ast[0]
        return Union(ast)

    def probabilistic_rule(self, ast):
        head = ast[0]
        probability = ast[2]
        body = ast[4]
        return Implication(
            ProbabilisticFact(probability, head),
            body,
        )

    def prob_head_rule(self, ast):
        # PROB[pred] :- body        → [pred, IMPLICATION_token, body]
        # PROB[pred | cond] :- body → [pred, COND_OP_token, cond, IMPLICATION_token, body]
        if len(ast) >= 4:
            # Conditional form: PROB[pred | cond] :- body
            pred = ast[0]
            cond_body = ast[2]
            body = ast[4] if len(ast) > 4 else ast[3]
            return Implication(pred, Condition(pred, cond_body))
        else:
            # Simple form: PROB[pred] :- body
            pred = ast[0]
            body = ast[2]
            return Implication(pred, body)

    @staticmethod
    def _extract_special_body_atoms(conjunction):
        """Separate __PROB__ marker atoms from regular atoms in a conjunction.

        Returns (regular_formulas, prob_specs) where each prob_spec is
        a tuple (inner_predicate, cond_body, result_var) or
        (inner_predicate, result_var).
        """
        formulas = list(conjunction.formulas)
        regular = []
        prob_specs = []
        for f in formulas:
            if (isinstance(f, FunctionApplication)
                    and isinstance(f.functor, Symbol)
                    and f.functor.name == "__PROB__"):
                prob_specs.append(f.args)
            else:
                regular.append(f)
        return regular, prob_specs

    def rule(self, ast):
        head = ast[0]
        body_or_cond = ast[2]

        if isinstance(body_or_cond, Conjunction):
            regular, prob_specs = self._extract_special_body_atoms(
                body_or_cond
            )
            if prob_specs:
                new_body = Conjunction(regular) if regular else Constant(True)
                return self._build_prob_rule(head, prob_specs, new_body)

        if isinstance(head, Expression) and head.functor == Symbol("ans"):
            return Query(ast[0], body_or_cond)
        else:
            return Implication(ast[0], body_or_cond)

    def _build_prob_rule(self, head, prob_specs, body):
        """Build rules from __PROB__ markers in the body.

        Each PROB[pred(x)]=p in the body desugars into a fresh predicate rule:
            fresh(x, PROB(x)) :- pred(x)
        and the main query uses the fresh predicate to extract the probability:
            ans(x, p) :- fresh(x, p)
        """
        head_args = list(head.args) if head.args else []
        body_formulas = (
            list(body.formulas) if isinstance(body, Conjunction) else []
        )
        fresh_rules = []
        query_body_atoms = []

        for spec in prob_specs:
            if len(spec) == 3:
                pred, cond_body, result_var = spec
            else:
                pred, result_var = spec
                cond_body = None

            fresh_pred = Symbol.fresh()

            if isinstance(pred, FunctionApplication):
                prob_vars = pred.args
                fresh_body_atoms = [pred.functor(*prob_vars)]
                if cond_body is not None:
                    fresh_body_atoms.append(cond_body)
                fresh_body = Conjunction(tuple(fresh_body_atoms))
                fresh_head = fresh_pred(
                    *prob_vars, FunctionApplication(PROB, prob_vars)
                )
                fresh_rules.append(Implication(fresh_head, fresh_body))

                query_body_atoms.append(
                    fresh_pred(*prob_vars, result_var)
                )

            elif isinstance(pred, Conjunction):
                fresh_body_atoms = list(pred.formulas)
                if cond_body is not None:
                    fresh_body_atoms.append(cond_body)
                fresh_body = Conjunction(tuple(fresh_body_atoms))
                fresh_head = fresh_pred(FunctionApplication(PROB, (pred,)))
                fresh_rules.append(Implication(fresh_head, fresh_body))

                query_body_atoms.append(
                    fresh_pred(result_var)
                )

            # result var stays in head as the probability column
            if result_var not in head_args:
                head_args.append(result_var)

        if isinstance(head, Query) or (isinstance(head, Expression)
                and hasattr(head, 'functor')
                and head.functor == Symbol("ans")):
            new_head = head.functor(*head_args)
        elif hasattr(head, 'functor'):
            new_head = head.functor(*head_args)
        else:
            new_head = head

        all_body_atoms = body_formulas + query_body_atoms
        if all_body_atoms:
            new_body = Conjunction(tuple(all_body_atoms))
        else:
            new_body = Constant(True)

        main_fml = Query(new_head, new_body)
        return Union(tuple(fresh_rules + [main_fml]))

    def prob_body_predicate(self, ast):
        # PROB[pred] = var          → ast = [predicate, argument] (2 elements)
        # PROB[pred // body] = var  → ast = [predicate, //_token, body, argument] (4 elements)
        if len(ast) == 4:
            pred = ast[0]
            cond_body = ast[2]   # ast[1] is the CONDITION_OP token
            result_var = ast[3]
        else:
            pred = ast[0]
            cond_body = None
            result_var = ast[1]
        return FunctionApplication(
            Symbol("__PROB__"),
            (pred, cond_body, result_var)
        )

    def marg_body_predicate(self, ast):
        # MARG[pred] = var          → ast = [conjunction, argument]
        # MARG[pred // body] = var  → ast = [conjunction, //_token, body, argument]
        if len(ast) == 4:
            pred = ast[0]
            cond_body = ast[2]
            result_var = ast[3]
            return FunctionApplication(
                Symbol("__PROB__"),
                (pred, cond_body, result_var)
            )
        pred = ast[0]
        result_var = ast[1]
        return FunctionApplication(
            Symbol("__PROB__"),
            (pred, result_var)
        )

    def succ_body_predicate(self, ast):
        # SUCC[...] = var → skipped (no-op in execution)
        # Return a __SUCC__ marker that _extract_special_body_atoms discards
        return FunctionApplication(
            Symbol("__SUCC__"),
            (ast[1],)
        )

    def cond_prob_predicate(self, ast):
        left = ast[0]
        right = ast[2]
        return Condition(left, right)

    def condition(self, ast):
        conditioned = ast[0]
        condition = ast[2]
        return Condition(conditioned, condition)

    def constraint(self, ast):
        return RightImplication(ast[0], ast[2])

    def conjunction(self, ast):
        conj = []
        for c in ast:
            if isinstance(c, Expression):
                if (isinstance(c, FunctionApplication)
                        and isinstance(c.functor, Symbol)
                        and c.functor.name == "__SUCC__"):
                    # SUCC markers are no-ops — discard at conjunction level
                    continue
                conj.append(c)
        if len(conj) == 0:
            return Constant(True)
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
        if not isinstance(ast, Expression):
            if isinstance(ast, list):
                if len(ast) == 1:
                    ast = ast[0]
                else:
                    ast = ast[0](*ast[1])
        return ast

    @staticmethod
    def _wrap_aggregation_args(arguments):
        """Replace FunctionApplication nodes whose functor is an
        aggregation function name with AggregationApplication."""
        result = []
        for arg in arguments:
            if (isinstance(arg, FunctionApplication)
                    and isinstance(arg.functor, Symbol)
                    and arg.functor.name in AGGREGATION_FUNCS):
                result.append(
                    AggregationApplication(arg.functor, arg.args)
                )
            elif (isinstance(arg, FunctionApplication)
                    and isinstance(arg.functor, FunctionApplication)
                    and isinstance(arg.functor.functor, Symbol)
                    and arg.functor.functor.name in AGGREGATION_FUNCS):
                result.append(
                    AggregationApplication(arg.functor, arg.args)
                )
            else:
                result.append(arg)
        return result

    def head_predicate(self, ast):
        if ast[1] is not None:
            arguments = list(ast[1])
            arguments = self._wrap_aggregation_args(arguments)

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
        arguments_raw = ast[0]
        if arguments_raw is not None:
            if isinstance(arguments_raw, tuple):
                arguments = list(arguments_raw)
            else:
                arguments = list(arguments_raw) if arguments_raw else []

            arguments = self._wrap_aggregation_args(arguments)
            return Symbol("ans")(*arguments)
        else:
            return Symbol("ans")()

    def statement(self, ast):
        return Statement(ast[0], ast[2])

    def statement_function(self, ast):
        return Statement(
            ast[0],
            Lambda(ast[1], ast[3])
        )

    def probabilistic_fact(self, ast):
        return Implication(
            ProbabilisticFact(ast[0], ast[2]),
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
        args = ()
        kwargs = ()
        for a in ast:
            if isinstance(a, tuple):
                kwargs = kwargs + (a,)
            else:
                args = args + (a,)
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
        if not isinstance(ast[0], Expression):
            f = Symbol(ast[0])
        else:
            f = ast[0]

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
            if isinstance(ast, Symbol) and ast.name == "_":
                return Symbol.fresh()
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
            return Constant(pow)(ast[0], ast[2])

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
        return Constant(int((ast[0]).value))

    def neg_int(self, ast):
        return Constant(0 - int((ast[0]).value))

    def pos_float(self, ast):
        return Constant(float((ast[0]).value))

    def neg_float(self, ast):
        return Constant(0 - float((ast[0]).value))

    def _default(self, ast):
        return ast


def _preprocess(code):
    """Strip % comments and trailing Prolog-style dots before Lark parsing.

    LLMs sometimes emit Prolog-style trailing dots (``.``) and ``%``
    line comments.  Lark would choke on both.
    """
    lines = code.split('\n')
    processed = []
    for line in lines:
        # Strip % comments (not inside strings)
        in_single = False
        in_double = False
        for i, c in enumerate(line):
            if c == "'" and not in_double:
                in_single = not in_single
            elif c == '"' and not in_single:
                in_double = not in_double
            elif c == '%' and not in_single and not in_double:
                line = line[:i]
                break
        # Strip trailing Prolog-style dot
        stripped = line.rstrip()
        if stripped.endswith('.') and stripped != '.':
            line = stripped[:-1]
        else:
            line = stripped
        processed.append(line)
    return '\n'.join(processed)


def parser(code, locals=None, globals=None, interactive=False):

    try:
        if (interactive):
            completer = LarkCompleter(COMPILED_GRAMMAR)
            res = completer.complete(code.strip())
            return res.token_options
        else:
            code = _preprocess(code)
            jp = COMPILED_GRAMMAR.parse(code.strip())
            return DatalogTransformer().transform(jp)
    except UnexpectedToken as e:
        raise UnexpectedTokenError(str(e), line=e.line - 1, column=e.column - 1) from e
    except UnexpectedCharacters as e:
        raise UnexpectedCharactersError(str(e), line=e.line - 1, column=e.column - 1) from e
    except LarkError as e:
        raise NeuroLangException from e


def parse_rules():
    curdir = os.path.dirname(os.path.realpath(__file__))
    rules_file = os.path.join(
        curdir,
        "rules.json"
    )

    # Opening JSON file
    with open(rules_file, 'r') as f:
        # json.load() returns JSON object as a dictionary
        rules_dico = json.load(f)
    return rules_dico
