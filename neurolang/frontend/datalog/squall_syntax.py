from operator import add, eq, ge, gt, le, lt, mul, ne, neg, pow, sub, truediv
from pyclbr import Function
from re import I
from typing import Any, Callable, NewType, Tuple, TypeVar

import tatsu

from neurolang.datalog.expressions import Fact

from ...datalog import Implication
from ...datalog.aggregation import AggregationApplication
from ...expression_walker import (
    ChainedWalker,
    ExpressionWalker,
    PatternMatcher,
    PatternWalker,
    ReplaceExpressionWalker,
    add_match,
    expression_iterator
)
from ...expressions import (
    Constant,
    Definition,
    Expression,
    FunctionApplication,
    Lambda,
    Symbol
)
from ...logic import (
    TRUE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    NaryLogicOperator,
    Negation,
    Quantifier,
    Union,
    UniversalPredicate
)
from ...logic.transformations import (
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    LogicExpressionWalker,
    FactorQuantifiersMixin,
    PushExistentialsDown,
    PushExistentialsDownMixin,
    RemoveTrivialOperationsMixin
)
from ...probabilistic.expressions import ProbabilisticPredicate
from ...type_system import Unknown, get_args, unify_types
from .standard_syntax import DatalogSemantics as DatalogClassicSemantics

S = TypeVar("Statement")
E = TypeVar("Entity")
P1 = Callable[[E], S]
P2 = Callable[[E, E], S]
S1 = Callable[[P1], S]
S2 = Callable[[P1], S1]


def M(type_):
    return Callable[[type_], type_]


def K(type_):
    return Callable[[Callable[[E], type_]], type_]


class Label(FunctionApplication):
    def __init__(self, variable, label):
        self.functor = Constant(None)
        if isinstance(label, tuple):
            self.args = (variable,) + label
        else:
            self.args = (variable, label)
        self.variable = variable
        self.label = label

    def __repr__(self):
        return "Label[{}:={}]".format(self.label, self.variable)


class Ref(Expression):
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return "Ref[{}]".format(self.label)


class Arg(Definition):
    def __init__(self, preposition, args):
        self.preposition = preposition
        self.args = args

    def __repr__(self):
        return "Arg[{}, {}]".format(self.preposition, self.args)


class From(FunctionApplication):
    def __init__(self, args):
        self.functor = "TO"
        self.args = args

    def __repr__(self):
        return "To[{}]".format(self.args)


class ForArg(Definition):
    def __init__(self, preposition, arg):
        self.preposition = preposition
        self.arg = arg

    def __repr__(self):
        return "ForArg[{}, {}]".format(self.preposition, self.arg)


class Expr(Definition):
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return "Expr({})".format(self.arg)


class The(Quantifier):
    def __init__(self, head, arg1, arg2):
        self.head = head
        self.arg1 = arg1
        self.arg2 = arg2

    def __repr__(self):
        return "The[{}; {}, {}]".format(self.head, self.arg1, self.arg2)


TO = Constant("to")
FROM = Constant("from")

EQ = Constant(eq)
NE = Constant(ne)
ADD = Constant(add)
DIV = Constant(truediv)
MUL = Constant(mul)
NEG = Constant(neg)
POW = Constant(pow)
SUB = Constant(sub)


GRAMMAR = u"""
    @@grammar::Datalog
    @@parseinfo :: True
    @@eol_comments :: /#([^\n]*?)$/
    @@keyword :: a all an and are belong belongs every for from
    @@keyword :: has hasnt have had in is not of or relate relates that
    @@keyword :: the there to such was were where which who whom
    @@keyword :: ADD
    @@left_recursion :: True

    start = squall $ ;

    expressions = ( newline ).{ probabilistic_expression | expression };

    probabilistic_expression = probability:probability '::' \
                               expression:expression
                             | 'with' 'probability'\
                               probability:probability [','] \
                               expression:expression ;

    probability = (float | int_ext_identifier );

    expression = squall | ['or'] @:rule | constraint | fact ;
    fact = constant_predicate ;
    rule = head implication body ;
    constraint = body right_implication head ;
    squall = 'squall' @:s ;
    head = head_predicate ;
    body = ( conjunction ).{ predicate } ;

    conjunction = '&' | '\N{LOGICAL AND}' | 'and';
    disjunction = '|' | '\N{LOGICAL OR}' | 'or' ;
    implication = ':-' | '\N{LEFTWARDS ARROW}' | 'if' ;
    not = 'not' | '\N{Not Sign}' ;
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

    negated_predicate = ( '\u00AC' | 'not' ) predicate ;

    s = s_or ;
    s_or = (disjunction).{@+:s_and} ;
    s_and = (conjunction).{@+:s_not};
    s_not = @:[ not ] @+:s_
          | @+:[ not ] '[' @+:s_ ']' ;

    s_ = s1
       | s2
       | s3
       | s4
       ;

    s1 = @:np [','] @:vp ;
    s2 = 'for' @:np ',' @:s ;
    s3 = 'there' ( 'is' | 'are' | 'was' | 'were' ) @:np ;
    s4 = pp s ;

    np = np2 'of' np
       | det ng1
       | expr_np
       ;

    term_ = label
         | literal
         ;

    vp = vp1
       | vpdo
       | vppp
       ;

    vp1 = &HAVE Aux vphave
        | &BE Aux vpbe ;

    vpdo = transitive:Verb2 op:op
         | intransitive:Verb1 cp:[ cp ]
         ;

    vphave = vphave1
           | vphave2
           ;

    vphave1 = noun2 op ;
    vphave2 = @+:np2 [ @+:rel ];

    vpbe = vpbe1
         | vpbe2
         ;

    vpbe1 = 'there' ;
    vpbe2 = rel ;

    vppp = pp vp ;

    Aux = ( HAVE | BE ) 'not'
        | ( HAVE | BE )
        ;

    HAVE = 'has' | 'have' | 'had' ;
    BE = 'is' | 'are' | 'was' | 'were' ;

    op = op1
       | op2
       ;

    op1 = np [ cp ] ;
    op2 = pp op ;

    Verb1 = belongs
          | intransitive ;
    Verb2 = relates
          |transitive ;

    belongs = 'belong' | 'belongs' ;
    relates = 'relate' | 'relates' ;

    intransitive = @:identifier ;
    transitive = '~'@:identifier ;

    cp = cp1 ;
    cp1 = pp [ cp ] ;

    det = det1
        | ('every' | 'all')
        | 'the'
        ;

    det1 = ('a' | 'an' | 'some' )
         | 'no'
         ;

    np2 = det ng2 ;

    ng1 = noun1 [ app ] [ rel ] ;
    ng2 = noun2:noun2 app:[ app ] ;

    noun1 = intransitive ;
    noun2 = transitive ;
    app = 'in' /\s+[0-9]+D/
        | label ;
    label = tuple_label
          | '?'@+:identifier
          ;

    tuple_label = '(' ';'.{'?'@+:identifier}+ ')';

    rel = ( conjunction ).{ rel_ }+ ;

    rel_ = rel2
         | rel1
         | rel4
         | rel7
         ;

    rel1 = ('that' | 'which' | 'where' | 'who' ) vp ;
    rel2 = ('that' | 'which' | 'where' | 'whom' ) np Verb2 [ cp ] ;
    rel3 = np2 'of which' vp ;
    rel4 = 'whose' ng2 vp ;
    rel5 = adj1 [ cp ] ;
    rel6 = adj2 op ;
    rel7 = 'such' 'that' @:s
         | '--'~ 'such' 'that' @:s '--' ;

    adj1 = intransitive ;
    adj2 = transitive ;

    pp = pp1
       ;

    pp1 = Prep np ;
    Prep = ( "to" | "from" ) ;

    expr_np = expr_np_add ;

    sum_op = '+'
           | '-'
           ;

    expr_np_add = expr_np_add ( '+' | '-' ) ~ expr_np_mul
                | expr_np_mul
                ;

    expr_np_mul = expr_np_mul ( '*' | '/' ) ~ expr_np_factor
                | expr_np_factor
                ;

    expr_np_factor = expr_np_exponent [ '**' expr_np_exponent ] ;
    expr_np_exponent = '(' ~ @:expr_np_add ')'
                     | expr_np_np
                     ;
    expr_np_np = term_
               | expr_np_function
               | np
               ;

    expr_np_function = @:identifier '(' ~ (',').{ @:expr_np_add } ')' ;

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

    identifier = identifier1
               | '`'@:?"[0-9a-zA-Z/#%._:-]+"'`';

    @name
    identifier1 = /[a-zA-Z_][a-zA-Z0-9_]*/ ;

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

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

    def s_or(self, ast):
        if len(ast) == 1:
            return ast[0]
        else:
            return Disjunction(tuple(ast))

    def s_and(self, ast):
        if len(ast) == 1:
            return ast[0]
        else:
            return Conjunction(tuple(ast))

    def s_not(self, ast):
        if len(ast) > 1:
            return Negation(ast[1])
        else:
            return ast[0]

    def s(self, ast):
        return squall_to_fol(ast)

    def s1(self, ast):
        return FunctionApplication[S](ast[0], (ast[1],))

    def s2(self, ast):
        np, s = ast
        x = Symbol[E].fresh()
        return np(Lambda[P1]((x,), s))

    def s3(self, ast):
        np = ast
        x = Symbol[E].fresh()
        return np(Lambda[P1]((x,), TRUE))

    def s4(self, ast):
        pp, s = ast
        return pp(s)

    def np(self, ast):
        if isinstance(ast, Expression):
            res = ast
        elif len(ast) == 2:
            det, ng1 = ast
            d = Symbol[P1].fresh()
            res = Lambda[S1](
                (d,),
                det(ng1)(d)
            )
        elif len(ast) == 3 and ast[1] == "of":
            np2, _, np = ast
            d = Symbol[P1].fresh()
            x = Symbol[E].fresh()
            res = Lambda[S1](
                (d,),
                np(Lambda[Callable[[E], S1]]((x,), np2(x)(d)))
            )
        return res

    def np2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        d = Symbol[P1].fresh()
        det, ng2 = ast
        res = Lambda[Callable[[E], S1]](
            (x,),
            Lambda(
                (d,),
                det(
                    Lambda[P2]((y,), ng2(x)(y))
                )(d)
            )
        )
        return res

    def term_(self, ast):
        d = Symbol[P1].fresh()
        return Lambda[S1]((d,), d(ast))

    def vpdo(self, ast):
        x = Symbol[E].fresh()
        if 'intransitive' in ast:
            res = ast['intransitive'].cast(P1)(x)
            if ast['cp']:
                res = ast['cp'](res)
            res = Lambda[P1]((x,), res)
        if 'transitive' in ast:
            y = Symbol[E].fresh()
            res = Lambda[P1](
                (x,),
                ast['op'](
                    Lambda[P1]((y,), ast['transitive'].cast(P2)(x, y))
                )
            )
        return res

    def vppp(self, ast):
        pp, vp = ast
        x = Symbol[E].fresh()

        return Lambda[P1]((x,), pp(vp(x)))

    def op1(self, ast):
        return ast

    def op2(self, ast):
        pp, op = ast
        d = Symbol[S].fresh()
        return Lambda[S1]((d,), pp(op(d)))

    def ng1(self, ast):
        x = Symbol[E].fresh()
        if isinstance(ast, Expression):
            ast = [ast]
        return Lambda[P1]((x,), Conjunction[S](tuple(
            FunctionApplication(a.cast(P1), (x,))
            for a in ast
        )))

    def ng2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()

        noun2 = ast['noun2'].cast(P2)
        app = ast['app']
        adj1 = ast['adj1']

        conjunction = (noun2(x, y),)
        if app:
            conjunction += (app(y),)
        if adj1:
            conjunction += (adj1(y),)

        return Lambda[P2]((x,), Lambda[P1]((y,), Conjunction[S](conjunction)))

    def det(self, ast):
        d1 = Symbol[P1].fresh()
        d2 = Symbol[P1].fresh()
        x = Symbol[E].fresh()

        if isinstance(ast, Expression):
            det1 = ast
            res = Lambda[S2](
                (d2,), 
                Lambda[S1](
                    (d1,),
                    det1(Lambda[P1](
                        (x,),
                        Conjunction[S]((d1(x), d2(x)))
                    ))
                )
            )
        elif ast in ("every", "all"):
            res = Lambda[S2](
                (d2,),
                Lambda[S1](
                    (d1,),
                    UniversalPredicate[S]((x,), Implication[S](d1(x), d2(x)))
                )
            )
        elif ast == "the":
            res = Lambda[S2](
                (d2,),
                Lambda[S1](
                    (d1,),
                    The[S]((x,), d1(x), d2(x))
                )
            )
        return res

    def det1(self, ast):
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        if ast in ("a", "an", "some"):
            res = Lambda[S1]((d,), ExistentialPredicate[S]((x,), d(x)))
        elif ast == "no":
            res = Lambda[S1]((d,), Negation[S](ExistentialPredicate[S]((x,), d(x))))
        return res

    def vp(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        if isinstance(ast, Expression):
            return Lambda[P1](
                (x,),  FunctionApplication[S](ast, (x,))
            )
        else:
            tv, np = ast
            return Lambda[P1](
                (x,), np(Lambda[P1]((y,), tv(y, x)))
            )

    def Aux(self, ast, *args, **kwargs):
        s = Symbol.fresh()
        if isinstance(ast, str):
            res = Lambda[M(S)]((s,), s)
        elif ast[1] in ('not', 'n\'t'):
            res = Lambda[M(S)]((s,), Negation[S](s))
        return res

    def vp1(self, ast):
        x = Symbol[E].fresh()
        aux, vp = ast
        return Lambda[P1]((x,), aux(vp(x)))

    def vphave1(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        noun2, op = ast
        return Lambda[P1](
            (x,),
            op(Lambda[P1]((y,), noun2.cast(P2)(x, y)))
        )

    def vphave2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        np2 = ast[0]
        rel = Lambda((y,), y)
        if ast[1:]:
            rel = ast[1]
        res = np2(x)(rel)
        return Lambda[P1]((x,), res)

    def vpbe1(self, ast):
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), TRUE)

    def vpbe2(self, ast):
        rel = ast
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), rel(x))

    def vpbe3(self, ast):
        npc = ast
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), npc(x))

    def belongs(self, ast):
        x = Symbol[E].fresh()
        c = Symbol.fresh()
        return Lambda[P1](
            (x,),
            ForArg(TO, Lambda((c,), c(x)))
        )

    def relates(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        p = Symbol.fresh()

        return Lambda(
            (p,),
            Lambda[P1](
                (x,),
                ForArg(TO, Lambda((y,), p(x, y)))
            )
        )

    def app(self, ast):
        if isinstance(ast, tuple) and ast[0] == 'in':
            ast = tuple(Symbol.fresh() for _ in range(int(ast[1][:-1])))
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), Label(x, ast))

    def label(self, ast):
        if len(ast) == 1:
            label = ast[0].cast(E)
        else:
            label = tuple(arg.cast(E) for arg in ast)
        return label

    def rel(self, ast):
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), Conjunction[S](tuple(a(x) for a in ast)))

    def rel1(self, ast):
        x = Symbol[E].fresh()
        _, vp = ast
        res = Lambda[P1]((x,), vp(x))
        return res

    def rel2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        _, np, verb2 = ast[:3]
        res = Lambda[P1](
            (y,),
            np(
                Lambda[P1]((x,), verb2.cast(P2)(x, y))
            )
        )
        return res

    def rel3(self, ast):
        np2, _, vp = ast
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), np2(x)(vp))

    def rel4(self, ast):
        _, ng2, vp = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()

        res = Lambda[P1](
            (x,),
            ExistentialPredicate[S](
                (y,),
                Lambda[P1](
                    (y,),
                    Conjunction[S]((ng2(x)(y), vp(y)))
                )(y)
            )
        )

        return res

    def rel5(self, ast):
        z = Symbol.fresh()
        cp = Lambda((z,), z)
        if isinstance(ast, list):
            adj1 = ast[0]
            if ast[1:]:
                cp = ast[1]
        elif isinstance(ast, Expression):
            adj1 = ast
        x = Symbol.fresh()
        res = cp(adj1(x))
        res = Lambda((x,), res)
        return res

    def rel6(self, ast):
        adj2, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        return Lambda[P1]((x,), op(Lambda((y,), adj2(x, y))))

    def rel7(self, ast):
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), ast)

    def rel8(self, ast):
        x = Symbol[E].fresh()
        if len(ast) == 2 and ast[0] == 'not':
            res = Negation[S](ast(x))
        if len(ast) == 3:
            if ast[1] == "or":
                res = Disjunction[S]((ast[0](x), ast[1]))
            elif ast[1] == "and":
                res = Conjunction[S]((ast[0](x), ast[1](x)))
        return Lambda[P1]((x,), res)

    @staticmethod
    def apply(k, d, alpha):
        k_ = Symbol.fresh()
        d_ = Symbol.fresh()
        y = Symbol.fresh()

        alpha_args = get_args(alpha)
        z_ = tuple(Symbol[arg].fresh() for arg in alpha_args[:-1])
        y = Symbol[E].fresh()
        res = The[S](
            (y,),
            EQ(d_, y),
            k_(y, *z_)
        )

        lambda_args = (k_, d_) + z_
        for arg in lambda_args[::-1]:
            res = Lambda((arg,), res)

        res = res(k)(d)

        return res

    def expr_np(self, ast):
        expr = ast

        d_ = Symbol.fresh()
        v = Symbol.fresh()
        res = expr(
            Lambda(
                (v,),
                Lambda(
                    (d_,),
                    d_(v)
                )
            )
        )

        return res

    def expr_np_add(self, ast):
        if not isinstance(ast, Expression):
            print(LS.walk(ast[0]))
        res = self.expression_infix_operation(ast, {'+': ADD, '-': SUB}, S1)
        res = LS.walk(res)
        print(res)
        return res

    def expr_np_mul(self, ast):
        return self.expression_infix_operation(ast, {'*': MUL, '/': DIV}, S1)

    def expr_np_factor(self, ast):
        return self.expression_infix_operation(ast, {'**': POW}, S1)

    def expr_np_function(self, ast):
        # arg_type = [Unknown] * (len(ast) - 1)
        # identifier = ast[0].cast(Callable[arg_type, Unknown])
        # arguments = tuple(ast[1:])
        print(LS.walk(ast[1]))
        ast = [ast[1], '+', ast[2]]
        res = self.expression_infix_operation(ast, {'+': ADD}, S1)
        res = LS.walk(res)
        print(res)
        return res
        # return self.apply_expression(ADD, (ast[1], ast[2]), S1)

    def expression_infix_operation(self, ast, ops, alpha):
        if isinstance(ast, Expression):
            return ast
        expr1, op, expr2 = ast
        fun = ops[op]
        return self.apply_expression(fun, (expr1, expr2), S1)

    def apply_expression(self, fun, args, alpha):
        k = Symbol[K(alpha)].fresh()

        xs = tuple(Symbol[arg.type].fresh() for arg in args)
        res = self.apply(k, fun(*xs), alpha)

        for x, arg in zip(xs[::-1], args[::-1]):
            res = arg(Lambda((x,), res))

        res = Lambda((k,), res)

        return res

    def expr_np_np(self, ast):
        k = Symbol.fresh()
        d = Symbol.fresh()
        x = Symbol.fresh()
        np = ast

        res = Lambda(
            (k,),
            Lambda(
                (d,),
                np(
                    Lambda((x,), k(x)(d))
                )
            )
        )

        return res

    def pp1(self, ast):
        prep, np = ast
        s = Symbol.fresh()
        z = Symbol.fresh()
        res = Lambda((s,), np(Lambda((z,), Arg(prep, ((z, s))))))
        return res

    def Prep(self, ast):
        if ast == "to":
            return TO
        elif ast == "from":
            return FROM


def parser(code, locals=None, globals=None, **kwargs):
    return COMPILED_GRAMMAR.parse(
        code.strip(),
        semantics=DatalogSemantics(locals=locals, globals=globals),
        **kwargs
    )


class LambdaSolverMixin(PatternMatcher):
    @add_match(Lambda, lambda exp: len(exp.args) > 1)
    def explode_lambda(self, expression):
        res = expression.function_expression
        for arg in expression.args:
            res = Lambda((arg,), res)
        return res

    @add_match(FunctionApplication(Lambda, ...))
    def solve_lambda(self, expression):
        functor = self.walk(expression.functor)
        args = self.walk(expression.args)
        lambda_args = functor.args
        lambda_fun = self.walk(functor.function_expression)
        replacements = {arg: value for arg, value in zip(lambda_args, args)}
        res = LambdaReplacementsWalker(replacements).walk(lambda_fun)
        if len(lambda_args) < len(args):
            res = FunctionApplication(res, args[len(lambda_args):])
        elif len(lambda_args) > len(args):
            res = Lambda(lambda_args[len(args):], res)
        return self.walk(res)


class LambdaReplacementsWalker(ExpressionWalker):
    def __init__(self, replacements):
        self.replacements = replacements

    @add_match(Lambda)
    def process_lambda(self, expression):
        new_replacements = {
            k: v for k, v in self.replacements.items()
            if k not in expression.args
        }
        if new_replacements:
            new_function_expression = (
                LambdaReplacementsWalker(new_replacements)
                .walk(expression.function_expression)
            )
            if new_function_expression != expression.function_expression:
                expression = Lambda(
                    expression.args,
                    new_function_expression
                )
        return expression

    @add_match(Symbol)
    def process_symbol(self, expression):
        return self.replacements.get(expression, expression)


class LogicPreprocessing(FactorQuantifiersMixin, LogicExpressionWalker):
    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifier_tuples(self, expression):
        head = expression.head
        res = expression.body
        for h in sorted(head, key=repr):
            res = expression.apply(h, res)
        return self.walk(res)


def _label_in_quantifier_body(expression):
    return any(
        isinstance(l, Label) and l.variable == expression.head
        for _, l in expression_iterator(expression.body)
    )


class ExplodeTupleArguments(LogicExpressionWalker):
    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifier_tuples(self, expression):
        head = expression.head
        res = expression.body
        for h in sorted(head, key=repr):
            res = expression.apply(h, res)
        return self.walk(res)

    @add_match(
        FunctionApplication,
        lambda exp: any(isinstance(arg, tuple) for arg in exp.args)
    )
    def explode_function_applications(self, expression):
        new_args = tuple()
        for arg in expression.args:
            if not isinstance(arg, tuple):
                arg = (arg,)
            new_args += arg
            type_args = get_args(expression.functor.type)
            new_type = Callable[[arg.type for arg in new_args], type_args[-1]]
        return self.walk(FunctionApplication(
            expression.functor.cast(new_type), new_args
        ))


class SolveLabels(LogicExpressionWalker):
    @add_match(
        Quantifier,
        _label_in_quantifier_body
    )
    def solve_label(self, expression):
        labels = [
            l for _, l in expression_iterator(expression.body)
            if isinstance(l, Label)
            and l.variable == expression.head
        ]
        if labels:
            expression = ReplaceExpressionWalker(
                {labels[0]: TRUE, labels[0].variable: labels[0].label}
            ).walk(expression)
        return self.walk(expression.apply(expression.head, self.walk(expression.body)))


class SimplifyNestedImplicationsMixin(PatternMatcher):
    @add_match(Implication(Implication, ...))
    def implication_implication_other(self, expression):
        consequent = expression.consequent.consequent
        antecedent = Conjunction((expression.consequent.antecedent, expression.antecedent))
        return Implication(consequent, antecedent)

    @add_match(Implication(UniversalPredicate, ...))
    def push_universal_up_implication(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent
        new_head = consequent.head
        new_consequent = consequent.body
        if new_head in antecedent._symbols:
            new_head = new_head.fresh()
            new_consequent = ReplaceExpressionWalker(
                {consequent.head[0]: new_head}
            ).walk(consequent.body)
        return UniversalPredicate(
            (new_head,), Implication(new_consequent, antecedent)
        )


class PrepositionSolverMixin(PatternMatcher):
    @add_match(
        Arg(..., (..., ForArg)),
        lambda exp: exp.preposition == exp.args[1].preposition
    )
    def arg_forarg(self, expression):
        return expression.args[1].arg(expression.args[0])

    @add_match(Arg(..., (..., Lambda)))
    def arg(self, expression):
        lambda_exp = expression.args[1]
        new_lambda_exp = Lambda(
            lambda_exp.args,
            Arg(expression.preposition, (expression.args[0], lambda_exp.function_expression))
        )

        return new_lambda_exp


class SquallIntermediateSolver(PatternWalker):
    @add_match(The)
    def replace_the_existential(self, expression):
        head, d1, d2 = expression.unapply()
        return ExistentialPredicate[S](head, Conjunction[S]((d2, d1)))


class SquallSolver(
    LambdaSolverMixin,
    PrepositionSolverMixin,
    SquallIntermediateSolver,
    ExpressionWalker
):
    pass


class LambdaSolver(LambdaSolverMixin, ExpressionWalker):
    pass


class MergeQuantifiersMixin(PatternWalker):
    @add_match(UniversalPredicate(..., UniversalPredicate))
    def merge_universals(self, expression):
        new_head = tuple(sorted(expression.head + expression.body.head))
        new_body = expression.body.body
        return UniversalPredicate(new_head, new_body)


class SplitNestedImplicationsMixin(PatternWalker):
    @add_match(
        Implication(..., Conjunction),
        lambda exp: any(
            isinstance(formula, UniversalPredicate) and
            isinstance(formula.body, Implication)
            for formula in exp.antecedent.formulas
        )
    )
    def nested_implications_to_union(self, expression):
        code = tuple()
        new_formulas = tuple()
        for formula in expression.antecedent.formulas:
            if (
                isinstance(formula, UniversalPredicate) and
                isinstance(formula.body, Implication)
            ):
                new_formulas += (formula.body.consequent,)
                code += (formula,)
        new_rule = Implication(expression.consequent, Conjunction(new_formulas))

        code += (new_rule,)

        res = Union(code) 
        return res


class SplitQuantifiersMixin(PatternWalker):
    @add_match(UniversalPredicate, lambda exp: isinstance(exp.head, tuple) and len(exp.head) > 1)
    def split_universals(self, expression):
        exp = expression.body
        for head in expression.head[::-1]:
            exp = UniversalPredicate((head,), exp)
        return exp


class SimplifiyEqualitiesMixin(PatternWalker):
    @add_match(
        Conjunction,
        lambda exp: any(
            isinstance(formula, FunctionApplication) and
            formula.functor == EQ and
            any(
                isinstance(formula_, FunctionApplication) and
                formula_.functor == EQ and
                formula_.args[1] in formula_._symbols
                for formula_ in exp.formulas
                if formula_ != formula
            )
            for formula in exp.formulas
        )
    )
    def simplify_equalities(self, expression):
        equalities = []
        non_equalities = []
        for formula in expression.formulas:
            if (
                isinstance(formula, FunctionApplication) and
                formula.functor == EQ
            ):
                equalities.append(formula)
            else:
                non_equalities.append(formula)
        equality_replacements = {
            formula.args[1]: formula.args[0]
            for formula in equalities
        }
        rew = ReplaceExpressionWalker(equality_replacements)
        changed = True
        while changed:
            changed = False
            new_equalities = []
            for equality in equalities:
                left, right = equality.args
                left = rew.walk(left)
                if left != equality.args[0]:
                    equality = EQ(left, right)
                    changed = True
                new_equalities.append(equality)
            equalities = new_equalities

        return Conjunction(tuple(non_equalities + equalities))

    @add_match(Negation(FunctionApplication(EQ, ...)))
    def negation_eq_to_ne(self, expression):
        return FunctionApplication(
            NE, expression.formula.args
        )


class LogicSimplifier(
    SimplifyNestedImplicationsMixin,
    RemoveTrivialOperationsMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    FactorQuantifiersMixin,
    SimplifiyEqualitiesMixin,
    LogicExpressionWalker
):
    @add_match(Quantifier, lambda exp: isinstance(exp.head, tuple))
    def explode_quantifiers(self, expression):
        res = expression.body
        for h in expression.head:
            res = expression.apply(h, res)

        return res



NONE = Constant(None)


class EliminateSpuriousEqualities(
    PushExistentialsDownMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    RemoveTrivialOperationsMixin,
    LogicExpressionWalker
):
    @add_match(ExistentialPredicate(..., EQ(..., ...)))
    def eliminate_trivial_existential(self, _):
        return NONE

    @add_match(
        NaryLogicOperator,
        lambda exp: any(e == NONE for e in exp.formulas)
    )
    def eliminate_element(self, expression):
        return expression.apply(
            tuple(e for e in expression.formulas if e != NONE)
        )


def squall_to_fol(expression):
    cw = ChainedWalker(
        SquallSolver(),
        ExplodeTupleArguments(),
        LogicPreprocessing(),
        SolveLabels(),
        ExplodeTupleArguments(),       
        LogicSimplifier(),
        EliminateSpuriousEqualities()
    )

    return cw.walk(expression)


LS = LambdaSolver()
