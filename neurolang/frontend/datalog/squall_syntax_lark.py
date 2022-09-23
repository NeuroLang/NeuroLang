

import lark
from typing import Callable, TypeVar

from ...expressions import Constant, FunctionApplication, Lambda, Symbol
from ...logic import (
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Negation,
    UniversalPredicate
)
from ...logic.transformations import RemoveTrivialOperations
from ...type_system import get_args, is_leq_informative
from .squall_syntax import (
    Label,
    LambdaSolver,
    SquallSolver,
    The,
    squall_to_fol
)


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


RTO = RemoveTrivialOperations()
LS = LambdaSolver()
SS = SquallSolver()


GRAMMAR=r"""
?start: ["squall"] squall

squall : s

?s : s_b
?s_b : np [ "," ] vp  -> s_np_vp
    | "for" np "," s -> s_for

?np : np_b
?np_b : det ng1 -> np_quantified
      | np2 "of" np
      | term -> np_term

det : det1              -> det_some
    | ("every" | "all") -> det_every
    | "the"             -> det_the

det1 : ("a" | "an" | "some" ) -> det1_some
     | "no"                   -> det1_no

np2 : det ng2

ng1 : noun1 [ app ] [ rel ]
ng2 : [ adj1 ] noun2 [ app ]

app : "in" number"D" -> app_dimension
    | label          -> app_label

rel : ("that" | "which" | "where" | "who" ) vp               -> rel_vp
    | ("that" | "which" | "where" | "whom" ) np verb2 [ cp ] -> rel_vp2
    | np2 "of" "which" vp                                    -> rel_np2
    | "whose" ng2 vp                                         -> rel_ng2
    | "such" "that" s                                        -> rel_s


term : label
     | literal

?vp : vpdo

vpdo :  verb1 [ cp ] -> vpdo_v1
     |  verb2 op    -> vpdo_v2

?adj1 : intransitive
?adj2 : transitive

?noun1 : intransitive
?noun2 : transitive

?verb1 : intransitive
?verb2 : transitive

?intransitive : identifier
?transitive : "~" identifier

op : np [ cp ]

cp : "NULL"

label : "?" identifier
      | "(" "?" identifier (";" "?" identifier )* ")"

identifier : NAME

?literal : "'"string"'"
         | number

string : STRING
number : SIGNED_NUMBER


_bool{x} : bool_disjunction{x}
bool_disjunction{x} : ( bool_disjunction{x} _disjunction ) * bool_conjunction{x}
bool_conjunction{x} : ( bool_conjunction{x} _conjunction ) * bool_atom{x}
bool_atom{x} : _negation bool_atom{x} -> bool_negation
         | "(" _bool{x} ")"
         | x

_conjunction : "&" | "\N{LOGICAL AND}" | "and"
_disjunction : "|" | "\N{LOGICAL OR}" | "or"
_implication : ":-" | "\N{LEFTWARDS ARROW}" | "if"
_negation : "not" | "\N{Not Sign}"

NAME : /[a-zA-Z_]\w*/
STRING : /[^']+/

%import common._STRING_ESC_INNER
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""


class SquallTransformer(lark.Transformer):
    def squall(self, ast):
        return squall_to_fol(ast[0])

    def s_np_vp(self, ast):
        np, vp = ast
        res = np(vp)
        print(res.type)
        return res

    def s_for(self, ast):
        x = Symbol[E].fresh()
        np, s = ast
        return np(Lambda((x,), s))

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

    def vpdo_v1(self, ast):
        x = Symbol[E].fresh()
        verb1, cp = ast
        res = verb1.cast(P1)(x)
        if cp:
            res = cp(res)
        res = Lambda[P1]((x,), res)
        return res

    def vpdo_v2(self, ast):
        verb2, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        res = Lambda(
            (x,),
            op(
                Lambda((y,), verb2.cast(P2)(x, y))
            )
        )
        return res

    def op(self, ast):
        np = ast[0]

        d = Symbol.fresh()
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

    def ng1(self, ast):
        x = Symbol[E].fresh()
        return Lambda[P1]((x,), Conjunction[S](tuple(
            FunctionApplication(a.cast(P1), (x,))
            for a in ast if a is not None
        )))

    def ng2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()

        adj1, noun2, app = ast
        noun2 = noun2.cast(P2)

        conjunction = (noun2(x, y),)
        if app:
            conjunction += (app(y),)
        if adj1:
            conjunction += (adj1(y),)

        return Lambda[P2]((x,), Lambda[P1]((y,), Conjunction[S](conjunction)))

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
        return Lambda((x,), ast[0])

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
            ast, Disjunction
        )

    def bool_conjunction(self, ast):
        return self._boolean_application_by_type(
            ast, Conjunction
        )

    def bool_negation(self, ast):
        res = self._boolean_application_by_type(
            ast, Negation
        )
        res = res.apply(*res.unapply())
        return res

    def bool_atom(self, ast):
        return ast[0]

    @staticmethod
    def _boolean_application_by_type(ast, op):
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

    NAME = str
    SIGNED_NUMBER = int
    STRING = str


COMPILED_GRAMMAR = lark.Lark(GRAMMAR)


def parser(code, locals=None, globals=None, return_tree=False, **kwargs):
    tree = COMPILED_GRAMMAR.parse(code)
    intermediate_representation = SquallTransformer().transform(tree)

    if return_tree:
        return intermediate_representation, tree
    else:
        return intermediate_representation
