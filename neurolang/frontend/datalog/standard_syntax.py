import json
import os
from typing import Tuple

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
    Symbol,
)
from ...logic import ExistentialPredicate
from ...logic.expression_processing import extract_logic_free_variables
from ...probabilistic.expressions import (
    PROB,
    Condition,
    ProbabilisticChoice,
    ProbabilisticFact,
)
from ...type_system import Unknown
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
            | probabilistic_choice_rule
            | probabilistic_choice_fact
            | prob_head_rule
            | marg_head_rule
            | succ_head_rule
            | statement
            | statement_function
            | command

agg_assign : "AGGREGATE" "[" agg_group_vars "]" "(" conjunction "@" agg_fn_call ")" "=" argument -> agg_assign_vars
           | "AGGREGATE" "[" "(" ")" "]" "(" conjunction "@" agg_fn_call ")" "=" argument -> agg_assign_empty
           | "AGGREGATE" "[" "]" "(" conjunction "@" agg_fn_call ")" "=" argument -> agg_assign_empty

agg_group_vars : cmd_identifier ("," cmd_identifier)*

agg_fn_call : cmd_identifier "(" argument ")"

probabilistic_rule : head PROBA_OP arithmetic_operation IMPLICATION (condition | body)

rule : (head | query) IMPLICATION (condition | body | agg_assign)
IMPLICATION : ":-" | "\N{LEFTWARDS ARROW}"

// PROB[pred] :- body  —  declare a probabilistic rule
prob_head_rule : "PROB" "[" predicate "]" IMPLICATION (condition | body)
               | "PROB" "[" predicate PROB_SEP body "]" IMPLICATION (condition | body)

query : "ans" "(" [ arguments ] ")"

condition : composite_predicate PROB_SEP composite_predicate
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

// PROB[pred] = var  or  PROB[pred // cond] = var  —  probability query inside a rule body
prob_body_predicate : "PROB" "[" conjunction "]" "=" argument
                    | "PROB" "[" conjunction PROB_SEP body "]" "=" argument

// MARG[pred] = var  —  marginal probability query inside a rule body
// MARG[pred1, pred2] = var  —  conjunction inside MARG (same semantics as PROB)
marg_body_predicate : "MARG" "[" conjunction "]" "=" argument
                    | "MARG" "[" conjunction PROB_SEP body "]" "=" argument

// SUCC[...] = var  —  success probability query (skipped in execution)
succ_body_predicate : "SUCC" "[" SUCC_INNER "]" "=" argument
SUCC_INNER : /[a-zA-Z_*+.,()&@\/ \t-]+/

// Separator inside PROB/MARG for the conditional form: accepts both // and |
PROB_SEP : "//" | "|"

// MARG head rule  —  declare a marginal probability rule
marg_head_rule : "MARG" "[" conjunction "]" IMPLICATION (condition | body)
               | "MARG" "[" conjunction PROB_SEP body "]" IMPLICATION (condition | body)

// SUCC head rule  —  success probability rule (skipped in execution)
succ_head_rule : "SUCC" "[" SUCC_INNER "]" IMPLICATION (condition | body)

id_application : int_ext_identifier "(" [ arguments ] ")"

comparison : argument COMPARISON_OPERATOR argument
COMPARISON_OPERATOR : "==" | "<" | "<=" | ">=" | ">" | "!="

statement : identifier STATEMENT_OP arithmetic_operation
statement_function : identifier "(" [ arguments ] ")" STATEMENT_OP argument
STATEMENT_OP : ":="

probabilistic_fact : ( arithmetic_operation | int_ext_identifier ) PROBA_OP constant_predicate
PROBA_OP : "::"

probabilistic_choice_fact : ( arithmetic_operation | int_ext_identifier ) CHOICE_OP constant_predicate
probabilistic_choice_rule : head CHOICE_OP arithmetic_operation IMPLICATION (condition | body)
CHOICE_OP : "^" | ":~:"

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

?literal : number | text | tuple_literal

tuple_literal : "(" argument ("," argument)+ ")"

ext_identifier : "@" identifier
identifier : cmd_identifier | identifier_regexp
identifier_regexp : IDENTIFIER_REGEXP
IDENTIFIER_REGEXP : "`" /[0-9a-zA-Z\\/#%._:-]+/ "`"
cmd_identifier : CMD_IDENTIFIER
CMD_IDENTIFIER : /\\b(?!\\bexists\\b)(?!\\b\\u2203\\b)(?!\\bEXISTS\\b)(?!\\bst\\b)(?!\\bans\\b)(?!\\bAGGREGATE\\b)[a-zA-Z_][a-zA-Z0-9_]*\\b/

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
        # PROB[pred] :- body              → [pred, IMPLICATION_token, body]
        # PROB[pred // cond] :- body      → [pred, SEP_token, cond, IMPLICATION_token, body]
        # PROB[pred | cond] :- body       → same ast shape
        if len(ast) >= 4:
            # Conditional form: PROB[pred // cond] :- body
            pred = ast[0]
            cond_body = ast[2]
            body = ast[4] if len(ast) > 4 else ast[3]
            prob_vars = tuple(
                v for v in extract_logic_free_variables(pred)
            )
            prob_head = self._add_prob_arg_to_predicate(pred, prob_vars)
            return Implication(prob_head, Condition(pred, cond_body))
        else:
            # Simple form: PROB[pred] :- body
            pred = ast[0]
            body = ast[2]
            prob_vars = tuple(
                v for v in extract_logic_free_variables(pred)
            )
            prob_head = self._add_prob_arg_to_predicate(pred, prob_vars)
            return Implication(prob_head, body)

    @staticmethod
    def _add_prob_arg_to_predicate(pred, prob_vars):
        """Add PROB(prob_vars) as an extra argument to a predicate."""
        if isinstance(pred, FunctionApplication) and hasattr(pred, 'args'):
            return pred.functor(*pred.args, FunctionApplication(PROB, prob_vars))
        return pred

    def marg_head_rule(self, ast):
        # MARG[pred] :- body         → [conjunction, IMPLICATION_token, body]
        # MARG[pred // cond] :- body → [conjunction, SEP_token, cond, IMPLICATION_token, body]
        if len(ast) >= 4:
            conjunction = ast[0]
            cond_body = ast[2]
            body = ast[4] if len(ast) > 4 else ast[3]
            prob_vars = tuple(
                v for v in extract_logic_free_variables(conjunction)
            )
            prob_head = self._add_prob_arg_to_conjunction(
                conjunction, prob_vars, unwrap=True
            )
            return Implication(prob_head, Condition(conjunction, cond_body))
        else:
            conjunction = ast[0]
            body = ast[2]
            prob_vars = tuple(
                v for v in extract_logic_free_variables(conjunction)
            )
            prob_head = self._add_prob_arg_to_conjunction(
                conjunction, prob_vars, unwrap=False
            )
            return Implication(prob_head, body)

    @staticmethod
    def _add_prob_arg_to_conjunction(conjunction, prob_vars, unwrap=True):
        """Add PROB(prob_vars) as an extra argument to each atom in a conjunction.

        When *unwrap* is True and the conjunction has a single atom, return
        the augmented atom directly (not wrapped in Conjunction). Otherwise
        return Conjunction with PROB appended to each atom.
        """
        prob_arg = FunctionApplication(PROB, prob_vars)
        if isinstance(conjunction, Conjunction):
            augmented = tuple(
                a.functor(*a.args, prob_arg)
                if isinstance(a, FunctionApplication) and hasattr(a, 'args')
                else a
                for a in conjunction.formulas
            )
            if unwrap and len(augmented) == 1:
                return augmented[0]
            return Conjunction(augmented)
        elif isinstance(conjunction, FunctionApplication) and hasattr(conjunction, 'args'):
            return conjunction.functor(*conjunction.args, prob_arg)
        return conjunction

    def succ_head_rule(self, ast):
        # SUCC[...] :- body → skipped (no-op)
        return Constant(True)

    def agg_assign_vars(self, ast):
        # "AGGREGATE" "[" agg_group_vars "]" "(" conjunction "@" agg_fn_call ")" "=" argument
        # Children: [group_vars, conjunction, agg_fn_call, argument]
        group_vars, conjunction, agg_fn_call, result_var = ast
        return (group_vars, conjunction, agg_fn_call, result_var)

    def agg_assign_empty(self, ast):
        # "AGGREGATE" "[" ("(" ")")? "]" "(" conjunction "@" agg_fn_call ")" "=" argument
        # Children: [conjunction, agg_fn_call, argument] (no group_vars)
        conjunction, agg_fn_call, result_var = ast
        return ((), conjunction, agg_fn_call, result_var)

    def agg_group_vars(self, ast):
        # CMD_IDENTIFIER ("," CMD_IDENTIFIER)*
        # Children are the Symbol tokens from each CMD_IDENTIFIER
        # Return them as a tuple
        return tuple(ast)

    def agg_fn_call(self, ast):
        # CMD_IDENTIFIER "(" argument ")"
        # Children: [cmd_identifier_symbol, argument_expression]
        fn_sym = ast[0]     # Symbol, e.g. Symbol("count")
        arg_expr = ast[1]   # argument expression, e.g. Symbol("s")
        return fn_sym(arg_expr)

    @staticmethod
    def _extract_special_body_atoms(conjunction):
        """Separate __PROB__ and __MARG__ marker atoms from regular atoms.

        Returns (regular_formulas, prob_specs) where each prob_spec is
        a tuple of (marker_name, args).
        """
        formulas = list(conjunction.formulas)
        regular = []
        prob_specs = []
        for f in formulas:
            if (isinstance(f, FunctionApplication)
                    and isinstance(f.functor, Symbol)
                    and f.functor.name in ("__PROB__", "__MARG__")):
                prob_specs.append((f.functor.name, f.args))
            else:
                regular.append(f)
        return regular, prob_specs

    def rule(self, ast):
        head = ast[0]
        body_or_cond = ast[2]

        if isinstance(body_or_cond, tuple):
            group_vars, conjunction, agg_fn_call, result_var = body_or_cond
            return self._build_aggregate_rule(head, group_vars, conjunction, agg_fn_call, result_var)

        if isinstance(body_or_cond, Conjunction):
            regular, prob_specs = self._extract_special_body_atoms(
                body_or_cond
            )
            if prob_specs:
                new_body = Conjunction(regular) if regular else Constant(True)
                result = self._build_prob_rule(head, prob_specs, new_body)
                # PROB body desugaring now produces an Implication for the
                # main formula.  Wrap as Query when the head is ans so that
                # execute_datalog_program can dispatch it as a query.
                if isinstance(head, Expression) and head.functor == Symbol("ans"):
                    formulas = list(result.formulas)
                    main = formulas[-1]
                    if isinstance(main, Implication):
                        formulas[-1] = Query(
                            main.consequent, main.antecedent
                        )
                        return Union(tuple(formulas))
                return result

        if isinstance(head, Expression) and head.functor == Symbol("ans"):
            return Query(ast[0], body_or_cond)
        else:
            return Implication(ast[0], body_or_cond)

    @staticmethod
    def _classify_prob_predicate(pred):
        """Extract (prob_vars, subject, is_marg) from a PROB/MARG spec predicate.

        Returns a tuple ``(prob_vars, subject, is_marg)`` or *None* for
        unsupported predicate types (which are silently skipped).
        """
        if isinstance(pred, FunctionApplication):
            return pred.args, pred, False
        elif isinstance(pred, Conjunction):
            return (), pred, True
        elif isinstance(pred, Negation):
            inner = pred.formula
            if isinstance(inner, FunctionApplication):
                return inner.args, pred, False
            return None
        elif isinstance(pred, ExistentialPredicate):
            body = getattr(pred, 'body', None)
            if isinstance(body, FunctionApplication):
                prob_vars = tuple(
                    v for v in body.args if v != pred.head
                )
                return prob_vars, pred, False
            return None
        return None

    @staticmethod
    def _filter_prob_vars_outside(prob_vars, outside_connect):
        """Keep only prob_vars that connect outside the probabilistic expression."""
        return tuple(v for v in prob_vars if v in outside_connect)

    @staticmethod
    def _prob_vars_from_conjunction(subject, outside_connect):
        """Extract prob_vars from a Conjunction, filtered to outside-connecting vars."""
        return tuple(
            v for v in extract_logic_free_variables(subject)
            if v in outside_connect
        )

    @staticmethod
    def _prob_vars_for_condition(subject, cond_body, outside_connect):
        """Extract prob_vars from both sides of a ``//`` Condition, outside-connect filtered."""
        all_vars = extract_logic_free_variables(subject)
        all_vars |= extract_logic_free_variables(cond_body)
        return tuple(v for v in all_vars if v in outside_connect)

    def _build_prob_rule(self, head, prob_specs, body):
        """Build rules from PROB/MARG markers in the body.

        Each PROB[pred(x)]=p in the body desugars into a fresh predicate rule::

            fresh(x, PROB(x)) :- pred(x)

        and the main query uses the fresh predicate to extract the probability::

            ans(x, p) :- fresh(x, p)
        """
        head_args = list(head.args) if head.args else []
        body_formulas = (
            list(body.formulas) if isinstance(body, Conjunction) else []
        )
        fresh_rules = []
        query_body_atoms = []

        # Compute outside-connecting variables: variables appearing in the head
        # (excluding the probability result variable) or in non-prob body atoms.
        # Only these should be projected out of PROB/MARG as grouping variables.
        # Variables only inside the prob formula are marginalized by the solver.
        outside_var_set: set[Expression] = set()
        for arg in head_args:
            if isinstance(arg, Symbol):
                outside_var_set.add(arg)
        for f in body_formulas:
            if isinstance(f, FunctionApplication):
                for arg in f.args:
                    if isinstance(arg, Symbol):
                        outside_var_set.add(arg)

        for spec in prob_specs:
            # --- unpack spec ---
            if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
                marker_name, args = spec
                if len(args) == 3:
                    pred, cond_body, result_var = args
                else:
                    pred, result_var = args
                    cond_body = None
            else:
                marker_name = "__PROB__"
                if len(spec) == 3:
                    pred, cond_body, result_var = spec
                else:
                    pred, result_var = spec
                    cond_body = None

            # Unwrap single-formula conjunctions
            inner = pred
            if isinstance(pred, Conjunction) and len(pred.formulas) == 1:
                inner = pred.formulas[0]
            classified = self._classify_prob_predicate(inner)
            if classified is None:
                continue

            _, subject, _ = classified
            is_marg = marker_name == "__MARG__"
            fresh_pred_sym = Symbol.fresh()
            outside_connect = {
                v for v in outside_var_set if v != result_var
            }

            # --- determine prob_vars and body ---
            if cond_body is not None:
                prob_vars = self._prob_vars_for_condition(
                    subject, cond_body, outside_connect
                )
                fresh_body = Condition(subject, cond_body)
            elif is_marg and isinstance(subject, Conjunction) and len(subject.formulas) > 1:
                prob_vars = self._prob_vars_from_conjunction(subject, outside_connect)
                fresh_body = subject
            elif is_marg:
                prob_vars = self._filter_prob_vars_outside(
                    classified[0], outside_connect
                )
                formulas = (
                    subject.formulas
                    if isinstance(subject, Conjunction)
                    else (subject,)
                )
                fresh_body = Conjunction(formulas)
                if not prob_vars or not body_formulas:
                    # No outside-connecting vars or no regular body atoms
                    # alongside this MARG — PROB wraps the full conjunction.
                    fresh_head = fresh_pred_sym(
                        FunctionApplication(
                            PROB, (Conjunction(formulas),)
                        )
                    )
                    query_body_atoms.append(fresh_pred_sym(result_var))
                    fresh_rules.append(
                        Implication(fresh_head, fresh_body)
                    )
                    if result_var not in head_args:
                        head_args.append(result_var)
                    continue
            elif isinstance(subject, Conjunction) and len(subject.formulas) > 1:
                prob_vars = self._prob_vars_from_conjunction(subject, outside_connect)
                fresh_body = subject
            else:
                prob_vars = self._filter_prob_vars_outside(
                    classified[0], outside_connect
                )
                fresh_body = Conjunction((subject,))

            # --- common head / atom (4 of 5 branches) ---
            fresh_head = fresh_pred_sym(
                *prob_vars, FunctionApplication(PROB, prob_vars)
            )
            query_body_atoms.append(fresh_pred_sym(*prob_vars, result_var))
            fresh_rules.append(Implication(fresh_head, fresh_body))

            if result_var not in head_args:
                head_args.append(result_var)

        if isinstance(head, Query) or (
            isinstance(head, Expression)
            and hasattr(head, 'functor')
            and head.functor == Symbol("ans")
        ):
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

        main_fml = Implication(new_head, new_body)
        return Union(tuple(fresh_rules + [main_fml]))

    def _build_aggregate_rule(self, head, group_vars, conjunction, agg_fn_call, result_var):
        fresh_pred_sym = Symbol.fresh()

        fresh_rules = []
        if isinstance(conjunction, Conjunction):
            regular, prob_specs = self._extract_special_body_atoms(conjunction)
            # Separate MARG specs (which need desugaring) from PROB specs
            # (which stay as markers in the conjunction body).
            marg_specs = []
            prob_marker_atoms = []
            for spec in prob_specs:
                marker_name = spec[0] if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str) else "__PROB__"
                if marker_name == "__MARG__":
                    marg_specs.append(spec)
                else:
                    # Reconstruct the __PROB__ marker atom to keep it in the body
                    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
                        prob_marker_atoms.append(
                            FunctionApplication(Symbol(marker_name), spec[1])
                        )

            if marg_specs:
                # MARG specs need desugaring. When aggregating over MARG,
                # the fresh predicate carries the group vars so that the
                # aggregate can group over them.
                fresh_rules_marg = []
                for spec in marg_specs:
                    spec_result_var = self._extract_prob_result_var(spec)
                    if spec_result_var is None:
                        continue
                    spec_args = spec[1] if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str) else spec[1:]
                    pred = spec_args[0] if len(spec_args) < 3 else spec_args[0]

                    inner = pred
                    if isinstance(pred, Conjunction) and len(pred.formulas) == 1:
                        inner = pred.formulas[0]

                    fresh_sym = Symbol.fresh()
                    prob_vars = tuple(
                        v for v in extract_logic_free_variables(pred)
                        if v in set(group_vars)
                    )
                    prob_arg = FunctionApplication(PROB, prob_vars) if prob_vars else FunctionApplication(PROB, (Conjunction((pred,) if not isinstance(pred, Conjunction) else pred.formulas),))

                    if isinstance(inner, FunctionApplication) and prob_vars:
                        fresh_head = fresh_sym(*prob_vars, prob_arg)
                        fresh_body = inner
                    elif isinstance(inner, FunctionApplication):
                        fresh_head = fresh_sym(prob_arg)
                        fresh_body = Conjunction((inner,))
                    else:
                        fresh_head = fresh_sym(prob_arg)
                        fresh_body = Conjunction((inner,)) if not isinstance(inner, Conjunction) else inner

                    fresh_rules_marg.append(Implication(fresh_head, fresh_body))
                    marg_fresh_sym = fresh_sym
                    marg_result_var = spec_result_var

                head_args = list(head.args) if hasattr(head, 'args') and head.args else []

                aggregation = AggregationApplication(agg_fn_call.functor, agg_fn_call.args)
                new_head_args = []
                for arg in head_args:
                    if (isinstance(arg, Symbol)
                            and result_var is not None
                            and isinstance(result_var, Symbol)
                            and arg.name == result_var.name):
                        new_head_args.append(aggregation)
                    else:
                        new_head_args.append(arg)
                new_head = head.functor(*new_head_args)

                body_atoms = []
                if prob_vars:
                    body_atoms.append(marg_fresh_sym(*prob_vars, marg_result_var))
                else:
                    body_atoms.append(marg_fresh_sym(marg_result_var))

                new_body = Conjunction(tuple(body_atoms))

                main_rule = Implication(new_head, new_body)
                return Union(tuple(fresh_rules_marg + [main_rule]))

            elif prob_marker_atoms:
                conjunction = Conjunction(tuple(regular + prob_marker_atoms)) if (regular + prob_marker_atoms) else conjunction

        head_args = list(head.args) if hasattr(head, 'args') and head.args else []

        if group_vars:
            aggregation = AggregationApplication(agg_fn_call.functor, agg_fn_call.args)
            new_args = []
            for arg in head_args:
                if (isinstance(arg, Symbol)
                        and result_var is not None
                        and isinstance(result_var, Symbol)
                        and arg.name == result_var.name):
                    new_args.append(aggregation)
                else:
                    new_args.append(arg)
            new_head = head.functor(*new_args)
            return Implication(new_head, conjunction)
        else:
            fresh_head = fresh_pred_sym(agg_fn_call)
            fresh_rules.append(Implication(fresh_head, conjunction))

            main_args = []
            for arg in head_args:
                if (isinstance(arg, Symbol)
                        and result_var is not None
                        and isinstance(result_var, Symbol)
                        and arg.name == result_var.name):
                    continue
                main_args.append(arg)

            fresh_query_atom = fresh_pred_sym(result_var)
            main_body = Conjunction((fresh_query_atom,))
            main_head = head.functor(*main_args, result_var)
            main_rule = Implication(main_head, main_body)

            return Union(tuple(fresh_rules + [main_rule]))

    @staticmethod
    def _extract_prob_result_var(spec):
        """Extract the result variable from a PROB/MARG spec tuple."""
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
            _, spec_args = spec
            if len(spec_args) == 3:
                _, _, rv = spec_args
            else:
                _, rv = spec_args
            return rv
        elif isinstance(spec, tuple):
            if len(spec) == 3:
                _, _, rv = spec
            else:
                _, rv = spec
            return rv
        return None

    def prob_body_predicate(self, ast):
        # PROB[pred] = var              → ast = [predicate, argument] (2 elements)
        # PROB[pred // body] = var      → ast = [predicate, SEP_token, body, argument] (4)
        # PROB[pred | body] = var       → same 4-element shape
        if len(ast) == 4:
            pred = ast[0]
            cond_body = ast[2]   # ast[1] is the separator token
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
        # MARG[pred] = var              → ast = [conjunction, argument] (2 elements)
        # MARG[pred // body] = var      → ast = [conjunction, SEP_token, body, argument] (4)
        # MARG[pred | body] = var       → same 4-element shape
        if len(ast) == 4:
            pred = ast[0]
            cond_body = ast[2]
            result_var = ast[3]
            return FunctionApplication(
                Symbol("__MARG__"),
                (pred, cond_body, result_var)
            )
        pred = ast[0]
        result_var = ast[1]
        return FunctionApplication(
            Symbol("__MARG__"),
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

    def probabilistic_choice_fact(self, ast):
        return Implication(
            ProbabilisticChoice(ast[0], ast[2]),
            Constant(True),
        )

    def probabilistic_choice_rule(self, ast):
        head = ast[0]
        probability = ast[2]
        body = ast[4]
        return Implication(
            ProbabilisticChoice(probability, head),
            body,
        )

    def command(self, ast):
        if ast[1] is None:
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

    def tuple_literal(self, ast):
        # Filter out Lark terminal tokens (commas, parens) which are str/Token
        # instances. Actual values are already transformed to Constant/Symbol
        # by other grammar rules, so no string value slips through here.
        values = tuple(a for a in ast if not isinstance(a, str))
        # Explicitly mark each element's type as Unknown since at parse time
        # we cannot determine the concrete types of tuple elements.
        value_types = tuple(Unknown for _ in values)
        result = Constant[Tuple[value_types]](
            values, auto_infer_type=False, verify_type=False
        )
        for v in values:
            if isinstance(v, Symbol):
                result._symbols.add(v)
        return result

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
        raise UnexpectedTokenError(
            str(e), line=e.line, column=e.column
        ) from e
    except UnexpectedCharacters as e:
        raise UnexpectedCharactersError(
            str(e), line=e.line, column=e.column
        ) from e
    except LarkError as e:
        raise NeuroLangException(str(e)) from e


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
