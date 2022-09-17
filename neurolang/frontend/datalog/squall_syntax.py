from operator import add, eq, ge, gt, le, lt, mul, ne, pow, sub, truediv

import tatsu

from ...datalog import Implication
from ...datalog.aggregation import AggregationApplication
from ...expression_walker import (
    ChainedWalker,
    ExpressionWalker,
    PatternMatcher,
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
    Negation,
    Quantifier,
    UniversalPredicate
)
from ...logic.transformations import (
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    LogicExpressionWalker,
    RemoveTrivialOperations,
    RemoveTrivialOperationsMixin
)
from ...probabilistic.expressions import ProbabilisticPredicate
from .standard_syntax import DatalogSemantics as DatalogClassicSemantics


class Label(FunctionApplication):
    def __init__(self, variable, label):
        self.functor = Constant(None)
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


TO = Constant("to")
FROM = Constant("from")

EQ = Constant(eq)


GRAMMAR = u"""
    @@grammar::Datalog
    @@parseinfo :: True
    @@whitespace :: /[\t ]+/
    @@eol_comments :: /#([^\n]*?)$/
    @@keyword :: a all an and are belong belongs every for from
    @@keyword :: has hasnt have had is not of or relate relates that
    @@keyword :: the there to such was were where which who whom
    @@left_recursion :: False

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
       | term_
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
    app = label ;
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
        return FunctionApplication(ast[0], (ast[1],))

    def s2(self, ast):
        np, s = ast
        x = Symbol.fresh()
        return np(Lambda((x,), s))

    def s3(self, ast):
        np = ast
        x = Symbol.fresh()
        return np(Lambda((x,), TRUE))

    def s4(self, ast):
        pp, s = ast
        return pp(s)

    def np(self, ast):
        if isinstance(ast, Expression):
            res = ast
        elif len(ast) == 2:
            det, ng1 = ast
            d = Symbol.fresh()
            res = Lambda(
                (d,),
                det(ng1)(d)
            )
        elif len(ast) == 3 and ast[1] == "of":
            np2, _, np = ast
            d = Symbol.fresh()
            x = Symbol.fresh()
            res = Lambda(
                (d,),
                np(Lambda((x,), np2(x)(d)))
            )
        return res

    def np2(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        d = Symbol.fresh()
        det, ng2 = ast
        res = Lambda(
            (x,),
            Lambda(
                (d,),
                det(
                    Lambda((y,), ng2(x)(y))
                )(d)
            )
        )
        return res

    def term_(self, ast):
        d = Symbol.fresh()
        return Lambda((d,), d(ast))

    def vpdo(self, ast):
        x = Symbol.fresh()
        if 'intransitive' in ast:
            res = ast['intransitive'](x)
            if ast['cp']:
                res = ast['cp'](res)
            res = Lambda((x,), res)
        if 'transitive' in ast:
            y = Symbol.fresh()
            res = Lambda(
                (x,),
                ast['op'](
                    Lambda((y,), ast['transitive'](x, y))
                )
            )
        return res

    def vppp(self, ast):
        pp, vp = ast
        x = Symbol.fresh()

        return Lambda((x,), pp(vp(x)))

    def op1(self, ast):
        return ast

    def op2(self, ast):
        pp, op = ast
        d = Symbol.fresh()
        return Lambda((d,), pp(op(d)))

    def ng1(self, ast):
        x = Symbol.fresh()
        if isinstance(ast, Expression):
            ast = [ast]
        return Lambda((x,), Conjunction(tuple(
            FunctionApplication(a, (x,))
            for a in ast
        )))

    def ng2(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()

        noun2 = ast['noun2']
        app = ast['app']
        adj1 = ast['adj1']

        conjunction = (noun2(x, y),)
        if app:
            conjunction += (app(y),)
        if adj1:
            conjunction += (adj1(y),)

        return Lambda((x,), Lambda((y,), Conjunction(conjunction)))

    def det(self, ast):
        d1 = Symbol.fresh()
        d2 = Symbol.fresh()
        x = Symbol.fresh()

        if isinstance(ast, Expression):
            det1 = ast
            res = Lambda(
                (d1, d2),
                det1(Lambda(
                    (x,),
                    Conjunction((d1(x), d2(x)))
                ))
            )
        elif ast in ("every", "all"):
            res = Lambda(
                (d1, d2),
                UniversalPredicate(x, Implication(d1(x), d2(x)))
            )
        return res

    def det1(self, ast):
        d = Symbol.fresh()
        x = Symbol.fresh()
        if ast in ("a", "an", "some"):
            res = Lambda((d,), ExistentialPredicate(x, d(x)))
        elif ast == "no":
            res = Lambda((d,), Negation(ExistentialPredicate(x, d(x))))
        return res

    def vp(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        if isinstance(ast, Expression):
            return Lambda(
                (x,),  FunctionApplication(ast, (x,))
            )
        else:
            tv, np = ast
            return Lambda(
                (x,), np(Lambda((y,), tv(y, x)))
            )

    def Aux(self, ast, *args, **kwargs):
        s = Symbol.fresh()
        if isinstance(ast, str):
            res = Lambda((s,), s)
        elif ast[1] in ('not', 'n\'t'):
            res = Lambda((s,), Negation(s))
        return res

    def vp1(self, ast):
        x = Symbol.fresh()
        aux, vp = ast
        return Lambda((x,), aux(vp(x)))

    def vphave1(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        noun2, op = ast
        return Lambda(
            (x,),
            op(Lambda((y,), noun2(x, y)))
        )

    def vphave2(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        np2 = ast[0]
        rel = Lambda((y,), y)
        if ast[1:]:
            rel = ast[1]
        res = np2(x)(rel)
        return Lambda((x,), res)

    def vpbe1(self, ast):
        x = Symbol.fresh()
        return Lambda((x,), TRUE)

    def vpbe2(self, ast):
        rel = ast
        x = Symbol.fresh()
        return Lambda((x,), rel(x))

    def vpbe3(self, ast):
        npc = ast
        x = Symbol.fresh()
        return Lambda((x,), npc(x))

    def belongs(self, ast):
        x = Symbol.fresh()
        c = Symbol.fresh()
        return Lambda(
            (x,),
            ForArg(TO, Lambda((c,), c(x)))
        )

    def relates(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        p = Symbol.fresh()

        return Lambda(
            (p,),
            Lambda(
                (x,),
                ForArg(TO, Lambda((y,), p(x, y)))
            )
        )

    def app(self, ast):
        x = Symbol.fresh()
        return Lambda((x,), Label(x, ast))

    def label(self, ast):
        if len(ast) == 1:
            label = ast[0]
        else:
            label = tuple(ast)
        return label

    def rel(self, ast):
        x = Symbol.fresh()
        return Lambda((x,), Conjunction(tuple(a(x) for a in ast)))

    def rel1(self, ast):
        x = Symbol.fresh()
        _, vp = ast
        res = Lambda((x,), vp(x))
        return res

    def rel2(self, ast):
        x = Symbol.fresh()
        y = Symbol.fresh()
        _, np, verb2 = ast[:3]
        res = Lambda(
            (y,),
            np(
                Lambda((x,), verb2(x, y))
            )
        )
        return res

    def rel3(self, ast):
        np2, _, vp = ast
        x = Symbol.fresh()
        return Lambda((x,), np2(x)(vp))

    def rel4(self, ast):
        _, ng2, vp = ast
        x = Symbol.fresh()
        y = Symbol.fresh()

        res = Lambda(
            (x,),
            ExistentialPredicate(
                y,
                Lambda(
                    (y,),
                    Conjunction((ng2(x)(y), vp(y)))
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
        x = Symbol.fresh()
        y = Symbol.fresh()
        return Lambda((x,), op(Lambda((y,), adj2(x, y))))

    def rel7(self, ast):
        x = Symbol.fresh()
        return Lambda((x,), ast)

    def rel8(self, ast):
        x = Symbol.fresh()
        if len(ast) == 2 and ast[0] == 'not':
            res = Negation(ast(x))
        if len(ast) == 3:
            if ast[1] == "or":
                res = Disjunction((ast[0](x), ast[1]))
            elif ast[1] == "and":
                res = Conjunction((ast[0](x), ast[1](x)))
        return Lambda((x,), res)

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
            CN, vp = ast
            return Lambda(
                (x,),
                Conjunction((CN(x), vp(x)))
            )
        elif len(ast) == 3:
            CN, np, TV = ast
            return Lambda(
                (x,),
                Conjunction((CN(x), np(Lambda((y,), TV(y, x)))))
            )
        else:
            raise ValueError()

    def CN(self, ast):
        x = Symbol.fresh()
        return Lambda(
            (x,), FunctionApplication(ast, (x,))
        )

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


class SolveLabels(LogicExpressionWalker):
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
        return self.walk(FunctionApplication(expression.functor, new_args))

    @add_match(
        Quantifier,
        lambda expression: any(
            isinstance(l, Label)
            for _, l in expression_iterator(expression.body)
        )
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
                {consequent.head: new_head}
            ).walk(consequent.body)
        return UniversalPredicate(
            new_head, Implication(new_consequent, antecedent)
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


class SquallSolver(LambdaSolverMixin, PrepositionSolverMixin, ExpressionWalker):
    pass


class LambdaSolver(LambdaSolverMixin, ExpressionWalker):
    pass


class LogicSimplifier(
    SimplifyNestedImplicationsMixin,
    RemoveTrivialOperationsMixin,
    CollapseConjunctionsMixin,
    CollapseDisjunctionsMixin,
    LogicExpressionWalker
):
    pass


def squall_to_fol(expression):
    cw = ChainedWalker(
        SquallSolver(),
        SolveLabels(),
        LogicSimplifier()
    )

    return cw.walk(expression)
