import ast
from operator import add, eq, ge, gt, le, lt, mul, ne, neg, pow, sub, truediv
from typing import Callable, TypeVar

import lark

from ...expression_walker import ExpressionWalker, add_match
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
    Implication,
    Negation,
    UniversalPredicate
)
from ...logic.transformations import RemoveTrivialOperations, FactorQuantifiersMixin
from ...type_system import (
    Unknown,
    get_args,
    get_parameters,
    is_leq_informative,
    is_parameterized
)
from .squall_syntax import (
    FROM,
    TO,
    Arg,
    ForArg,
    Label,
    LambdaSolver,
    SquallSolver,
    The,
    squall_to_fol
)

S = TypeVar("Statement")
E = TypeVar("Entity")
alpha = TypeVar("alpha")
P1 = Callable[[E], S]
P2 = Callable[[E, E], S]
S1 = Callable[[P1], S]
S2 = Callable[[P1], S1]
K_alpha = Callable[[Callable[[E], alpha]], alpha]


def M(type_):
    return Callable[[type_], type_]


def K(type_):
    return Callable[[Callable[[E], type_]], type_]


RTO = RemoveTrivialOperations()
LS = LambdaSolver()
SS = SquallSolver()

EQ = Constant(eq)
GT = Constant(gt)
GE = Constant(ge)
LT = Constant(lt)
LE = Constant(le)
NE = Constant(ne)
ADD = Constant(add)
DIV = Constant(truediv)
MUL = Constant(mul)
NEG = Constant(neg)
POW = Constant(pow)
SUB = Constant(sub)

OfType = Symbol[Callable[[E], E]]("rdf:type")


GRAMMAR = r"""
?start: ["squall"] squall

squall : s

?s : bool{s_b}
?s_b : np [ "," ] vp  -> s_np_vp
    | "for" np "," s  -> s_for
    | pp s            -> s_pp  

?np : expr{np_b}    -> expr_np
?np_b : det ng1     -> np_quantified
      | np2 "of" np -> np_np2
//      | term        -> np_term

det : det1              -> det_some
    | ("every" | "all") -> det_every
    | "the"             -> det_the

det1 : "some" -> det1_some
     | /an{0,1}/   -> det1_some
     | "no"   -> det1_no

np2 : det ng2

ng1 : noun1 [ app ] [ rel ]
ng2 : noun2 [ app ]

app : "in" number"D" -> app_dimension
    | label          -> app_label

?rel : bool{rel_b}
rel_b : ("that" | "which" | "where" | "who" ) vp               -> rel_vp
      | ("that" | "which" | "where" | "whom" ) np verb2 [ cp ] -> rel_vp2
      | np2 "of" "which" vp                                    -> rel_np2
      | "whose" ng2 vp                                         -> rel_ng2
      | "such" "that" s                                        -> rel_s
      | comparison "than" op                                   -> rel_comp
      | "--" rel "--"


!comparison : "greater" [ "equal" ]
            | "lower"   [ "equal" ]
            | [ "not" ] "equal"


term : label
     | literal

?vp : bool{vp_b}
?vp_b : vpdo
      | aux{be} vpbe      -> vp_aux
      | aux{have} vphave  -> vp_aux
      | aux{do} vpdo      -> vp_aux
      | pp vp             -> vp_pp

vpdo : verb1 [ cp ]     -> vpdo_v1
     | verb2 [ DOPREP ] op  -> vpdo_v2

DOPREP : "with"

vpbe : "there"       -> vpbe_there
     | rel           -> vpbe_rel
     | npc{a_an_the} -> vpbe_npc
     | npc_p{in}     -> vpbe_npc

a_an_the : "a"
         | "an"
         | "the"

in : "in"

vphave : noun2 op -> vphave_noun2
       | np2 [ rel ] -> vphave_np2

aux{verb} : verb                         -> aux_id
          | (verb "\s+not" | verb"n't")  -> aux_not

npc{det_} : term     -> npc_term
          | det_ WS ng1 -> npc_det

npc_p{prep_} : prep_ WS ng1 -> npc_det

?be : ( "is" | "are" | "was" | "were" )
?have: ( "has" | "had" | "have" )
?do: ( "does" | "do" | "did" )

// ?adj1 : intransitive
?adj2 : transitive

?noun1 : intransitive
?noun2 : transitive

?verb1 : belong
       | intransitive
?verb2 : relate
       | transitive

belong : /belong[s]{0,1}/
relate : /relate[s]{0,1}/

?intransitive : identifier
?transitive : "~" identifier

op : np [ cp ] -> op_np
   | pp op     -> op_pp

pp : prep np -> pp_np

!prep : "to"
      | "from"

cp : pp [ cp ] -> cp_pp

label : "?" identifier
      | "(" "?" identifier (";" "?" identifier )* ")"

identifier : NAME
           | "`"string"`"

?literal : "'"string"'"
         | number

string : STRING
number : SIGNED_INT
       | SIGNED_FLOAT


?bool{x} : bool_disjunction{x}
bool_disjunction{x} : bool_conjunction{x}
                    | bool_disjunction{x} _disjunction bool_conjunction{x}
bool_conjunction{x} : bool_atom{x}
                    | bool_conjunction{x} _conjunction bool_atom{x}
bool_atom{x} : _negation bool_atom{x} -> bool_negation
         | "(" bool{x} ")"
         | x

_conjunction : "&" | "\N{LOGICAL AND}" | "and"
_disjunction : "|" | "\N{LOGICAL OR}" | "or"
_implication : ":-" | "\N{LEFTWARDS ARROW}" | "if"
_negation : "not" | "\N{Not Sign}"

?expr{x} : expr_sum{x}
expr_sum{x} : ( expr_sum{x} SUM )? expr_mul{x}
expr_mul{x} : ( expr_mul{x} MUL )? expr_pow{x}
expr_pow{x} : expr_exponent{x} (pow expr_exponential{x})?
?expr_exponent{x} : expr_atom{x}
?expr_exponential{x} : expr_atom{x}
expr_atom{x} : "(" expr{x} ")"
             | identifier"(" (expr{x} ("," expr{x})*)? ")" -> expr_atom_fun
             | term                                        -> expr_atom_term
             | x

SUM : "+" 
    | "-"

MUL : "*" 
    | "/"

?pow : "**"

NAME : /[a-zA-Z_]\w*/
STRING : /[^']+/

%import common._STRING_ESC_INNER
%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.WS
%ignore WS
"""


class Apply_(Definition):
    """Apply Operator defined in the SQUALL paper
    by Ferré, section 4.4.11.

    Parameters
    ----------
    Definition : _type_
        _description_
    """
    def __init__(self, k, d):
        self.k = k
        self.d = d

    def __repr__(self):
        return (
            "\N{GREEK SMALL LETTER LAMDA}.z*"
            f"APPLY[{self.k}; {self.d}]"
        )


Apply = Apply_[Callable[[Callable[[Callable[[E], alpha]], P1]], alpha]]


class Expr(Definition):
    """Expr Operator defined in the SQUALL paper
    by Ferré, section 4.4.11.

    Parameters
    ----------
    Definition : _type_
        _description_
    """
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Expr[{self.expr}]"


class ChangeApplies(ExpressionWalker):
    """Change operation Apply for the
    apply operation in the SQUALL paper,
    section 4.4.11

    Parameters
    ----------
    alpha: type parameter of the expression

    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.types_for_z = get_args(alpha)[:-1]

    @add_match(Apply)
    def apply_apply(self, expression):
        k = expression.k
        d = expression.d

        return ChangeApplies.apply(k, d, self.alpha)

    @staticmethod
    def apply(k, d, alpha):
        k_ = Symbol[K(alpha)].fresh()
        d_ = Symbol[P1].fresh()
        y = Symbol[E].fresh()

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


class ChangeAlphaTypes(ExpressionWalker):
    def __init__(self, alpha):
        self.alpha = alpha

    @add_match(
        Expression,
        lambda exp: (
            is_parameterized(exp.type) and
            alpha in get_parameters(exp.type)
        )
    )
    def change_alpha_type(self, expression):
        type_params = expression.type.__parameters__

        new_params = tuple()
        for t in type_params:
            if t == alpha:
                new_params += (self.alpha,)
            else:
                new_params += (t,)

        new_expression = expression.cast(expression.type[new_params])

        return new_expression


class CastExpr(ExpressionWalker):
    def __init__(self, cast_operation):
        self.cast_operation = cast_operation

    @add_match(Expr)
    def expr(self, expression):
        return self.cast_operation(expression.expr)


class SquallTransformer(lark.Transformer):
    def squall(self, ast):
        return squall_to_fol(ast[0])

    def s_np_vp(self, ast):
        np, vp = ast
        res = np(vp)
        return res

    def s_for(self, ast):
        x = Symbol[E].fresh()
        np, s = ast
        return np(Lambda((x,), s))

    def s_pp(self, ast):
        pp, s = ast
        return pp(s)

    def np_term(self, ast):
        d = Symbol[P1].fresh()
        return Lambda[S1]((d,), d(ast[0]))

    def np_quantified(self, ast):
        det, ng1 = ast
        d = Symbol[P1].fresh()
        res = Lambda[S1](
            (d,),
            det(ng1)(d)
        )
        return res

    def np_np2(self, ast):
        np2, np = ast
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()

        res = Lambda((d,), np(Lambda((x,), np2(x)(d))))
        return res
 
    def vp_aux(self, ast):
        aux, vp = ast
        x = Symbol[E].fresh()
        res = Lambda((x,), aux(vp(x)))
        return res

    def vp_pp(self, ast):
        pp, vp = ast
        x = Symbol[E].fresh()
        res = Lambda((x,), pp(vp(x)))

        return res

    def vpdo_v1(self, ast):
        x = Symbol[E].fresh()
        verb1, cp = ast
        res = verb1.cast(P1)(x)
        if cp:
            res = cp(res)
        res = Lambda[P1]((x,), res)
        return res

    def vpdo_v2(self, ast):
        verb2, _, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        res = Lambda(
            (x,),
            op(
                Lambda((y,), verb2.cast(P2)(x, y))
            )
        )
        return res

    def aux_id(self, ast):
        s = Symbol[S].fresh()
        return Lambda((s,), s)

    def aux_not(self, ast):
        s = Symbol[S].fresh()
        return Lambda((s,), Negation(s))

    def vpbe_there(self, ast):
        x = Symbol[E].fresh()
        return Lambda((x,), TRUE)

    def vpbe_rel(self, ast):
        rel = ast[0]
        x = Symbol[E].fresh()
        return Lambda((x,), rel(x))

    def vpbe_npc(self, ast):
        npc = ast[0]
        x = Symbol[E].fresh()

        return Lambda((x,), npc(x))

    def vphave_noun(self, ast):
        noun2, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()

        res = Lambda(
            (x,),
            op(Lambda((y,), noun2(x, y)))
        )

        return res

    def vphave_np2(self, ast):
        np2, rel = ast
        if not rel:
            y = Symbol[E].fresh()
            rel = Lambda((y,), TRUE)
        x = Symbol[E].fresh()

        res = Lambda((x,), np2(x)(rel))
        return res

    def npc_term(self, ast):
        term = ast[0]
        x = Symbol[E].fresh()

        return Lambda((x,), EQ(x, term))

    def npc_det(self, ast):
        ng1 = ast[-1]
        x = Symbol[E].fresh()

        return Lambda((x,), ng1(x))

    def op_np(self, ast):
        np = ast[0]

        d = Symbol[P1].fresh()
        y = Symbol[E].fresh()

        res = Lambda(
            (d,),
            np(
                Lambda(
                    (y,),
                    d(y)
                )
            )
        )

        return res

    def op_pp(self, ast):
        pp, op = ast
        x = Symbol[E].fresh()
        res = Lambda((x,), pp(op(x)))
        return res

    def cp_pp(self, ast):
        pp, cp = ast
        s = Symbol[S].fresh()
        if cp:
            res = cp(s)
        else:
            res = s
        res = Lambda((s,), pp(res))
        return res

    def prep(self, ast):
        if ast[0].lower() == "from":
            res = FROM
        elif ast[0].lower() == "to":
            res = TO
        return res

    def pp_np(self, ast):
        prep, np = ast
        s = Symbol[S].fresh()
        z = Symbol[E].fresh()
        res = Lambda((s,), np(Lambda((z,), Arg(prep, ((z, s))))))
        return res

    def ng1(self, ast):
        x = Symbol[E].fresh()
        noun1, app, rel = ast
        noun1 = noun1.cast(P1)
        args = (noun1, app, rel)
        return Lambda[P1]((x,), Conjunction[S](tuple(
            FunctionApplication(a, (x,))
            for a in args if a is not None
        )))

    def ng2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()

        noun2, app = ast
        noun2 = noun2.cast(P2)

        conjunction = (noun2(x, y),)
        if app:
            conjunction += (app(y),)

        return Lambda((x,), Lambda((y,), Conjunction[S](conjunction)))

    def det1_some(self, ast):
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        res = Lambda(
            (d,),
            ExistentialPredicate[S]((x,), d(x))
        )
        return res

    def det1_no(self, ast):
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()

        res = Lambda[S1](
            (d,),
            Negation[S](
                ExistentialPredicate[S]((x,), d(x)))
        )
        return res

    def np2(self, ast):
        det, ng2 = ast
        x = Symbol[E].fresh()
        d = Symbol[P1].fresh()
        y = Symbol[E].fresh()

        res = Lambda(
            (x,),
            Lambda(
                (d,),
                det(Lambda((y,), ng2(x)(y)))(d)
            )
        )
        return res  

    def det_some(self, ast):
        det1 = ast[0]
        d1 = Symbol[P1].fresh()
        d2 = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        res = Lambda(
            (d2,),
            Lambda(
                (d1,),
                det1(Lambda((x,), Conjunction[S]((d1(x), d2(x)))))
            )
        )
        return res

    def det_every(self, ast):
        d1 = Symbol[P1].fresh()
        d2 = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        res = Lambda[S2](
            (d2,),
            Lambda[S1](
                (d1,),
                UniversalPredicate[S]((x,), Implication[S](d1(x), d2(x)))
            )
        )
        return res

    def det_the(self, ast):
        d1 = Symbol[P1].fresh()
        d2 = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        res = Lambda[S2](
            (d2,),
            Lambda[S1](
                (d1,),
                The[S]((x,), d1(x), d2(x))
            )
        )
        return res

    def term(self, ast):
        return ast[0]

    def app_dimension(self, ast):
        dimensions = ast[0].value
        ast = tuple(Symbol[E].fresh() for _ in range(dimensions))
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), Label(x, ast))

    def app_label(self, ast):
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), Label(x, ast[0]))

    def rel_vp(self, ast):
        x = Symbol[E].fresh()
        return Lambda((x,), ast[0](x))

    def rel_vp2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        np, verb2, cp = ast
        res = verb2.cast(P2)(x, y)
        if cp:
            res = cp(res)
        res = np(Lambda((x,), res))
        res = Lambda((y,), res)
        return res

    def rel_np2(self, ast):
        x = Symbol[E].fresh()
        np2, vp = ast
        res = Lambda(
            (x,),
            np2(x)(vp)
        )
        return res

    def rel_ng2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()

        ng2, vp = ast

        res = Lambda(
            (x,),
            ExistentialPredicate(
                y,
                Lambda(
                    (y,),
                    Conjunction((
                        ng2(x)(y),
                        vp(y)
                    ))
                )
            )
        )
        return res

    def rel_s(self, ast):
        s = ast[0]
        x = Symbol[E].fresh()
        return Lambda((x,), s)

    def rel_comp(self, ast):
        comp, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        return Lambda(
            (x,),
            op(Lambda((y,), comp(x, y)))
        )

    def comparison(self, ast):
        comp = ' '.join(a for a in ast if a)
        comp_dict = {
            "greater": GT,
            "greater equal": GE,
            "lower": LT,
            "lower equal": LE,
            "equal": EQ,
            "not equal": NE
        }

        return comp_dict[comp]

    def label(self, ast):
        ast = tuple(ast)
        if len(ast) == 1:
            ast = ast[0]
        return ast

    def identifier(self, ast):
        return Symbol[E](ast[0])

    def number(self, ast):
        return Constant(ast[0])

    def string(self, ast):
        return Constant[str](ast[0])

    def bool_disjunction(self, ast):
        return self._boolean_application_by_type(
            ast, Disjunction, True
        )

    def bool_conjunction(self, ast):
        return self._boolean_application_by_type(
            ast, Conjunction, True
        )

    def bool_negation(self, ast):
        res = self._boolean_application_by_type(
            ast, Negation, False
        )
        res = res.apply(*res.unapply())
        return res

    def bool_atom(self, ast):
        return ast[0]

    @staticmethod
    def expr_2_np(np):
        k = Symbol.fresh()
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        return Lambda((k,), Lambda((d,), np(Lambda((x,), (k(x)(d))))))

    def expr_np(self, ast):
        v = Symbol[E].fresh()
        d = Symbol[P1].fresh()

        expr = ast[0]
        expr = CastExpr(self.expr_2_np).walk(expr)
        expr = ChangeApplies(S1).walk(expr)
        expr = ChangeAlphaTypes(S1).walk(expr)
        res = expr(Lambda((v,), Lambda((d,), d(v))))

        return res

    def expr_sum(self, ast):
        if len(ast) == 1:
            return ast[0]
        else:
            if ast[1][0] == "+":
                op = ADD
            else:
                op = SUB
            ast = tuple((ast[0], ast[-1]))
        return self.apply_expression(op, ast)

    def expr_mul(self, ast):
        if len(ast) == 1:
            return ast[0]
        else:
            if ast[1][0] == "*":
                op = MUL
            else:
                op = DIV
            ast = tuple((ast[0], ast[-1]))
        return self.apply_expression(op, ast)

    def expr_pow(self, ast):
        ast = tuple(ast)
        if len(ast) == 1:
            return ast[0]
        return self.apply_expression(POW, ast)

    def expr_atom_term(self, ast):
        term = ast[0]
        k = Symbol[Callable[[E], alpha]].fresh()
        return Lambda((k,), k(term))

    def expr_atom_fun(self, ast):
        functor = ast[0].cast(Unknown)
        params = tuple(ast[1:])
        return self.apply_expression(functor, params)

    def expr_atom(self, ast):
        return Expr(ast[0])

    @staticmethod
    def apply_expression(fun, args, alpha=alpha):
        k = Symbol[Callable[[E], alpha]].fresh()

        xs = tuple(Symbol[arg.type].fresh() for arg in args)
        res = Apply(k, fun(*xs))

        for x, arg in zip(xs[::-1], args[::-1]):
            res = arg(Lambda((x,), res))

        res = Lambda((k,), res)

        return res

    @staticmethod
    def _boolean_application_by_type(ast, op, nary):
        if nary and len(ast) == 1:
            return ast[0]

        type_ = ast[0].type
        if (
            not isinstance(type_, TypeVar) and
            is_leq_informative(type_, Callable)
        ):
            type_args = get_args(type_)
            args = tuple(
                Symbol[t].fresh() for t in type_args[:-1]
            )
            formulas = tuple(
                a(*args) for a in ast
            )
            if op is Negation:
                formulas = formulas[0]
            return Lambda(args, op[type_](formulas))
        else:
            if op is Negation:
                return Negation[type_](ast[0])
            else:
                return op[type_](tuple(ast))

    def belong(self, ast):
        x = Symbol.fresh()
        c = Symbol[Callable[[alpha], Callable[[alpha], alpha]]].fresh()
        return Lambda[P1](
            (x,),
            ForArg(TO, Lambda((c,), OfType(x, c)))
        )

    def relate(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        p = Symbol.fresh()

        return Lambda(
            (p,),
            Lambda(
                (x,),
                ForArg(TO, Lambda((y,), p(x, y)))
            )
        )

    NAME = str
    SIGNED_INT = int
    SIGNED_FLOAT = float
    STRING = str


COMPILED_GRAMMAR = lark.Lark(GRAMMAR)


def parser(code, locals=None, globals=None, return_tree=False, process=True, **kwargs):
    tree = COMPILED_GRAMMAR.parse(code)

    if process:
        intermediate_representation = SquallTransformer().transform(tree)
    else:
        intermediate_representation = None

    if return_tree:
        return intermediate_representation, tree
    else:
        return intermediate_representation
