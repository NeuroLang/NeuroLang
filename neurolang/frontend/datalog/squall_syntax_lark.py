import re
from operator import (
    add,
    eq,
    ge,
    gt,
    itemgetter,
    le,
    lt,
    mul,
    ne,
    neg,
    pow,
    sub,
    truediv
)
from typing import Callable, List, TypeVar
from warnings import warn

import lark
from nltk.corpus import wordnet
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from ...exceptions import (
    NeuroLangException,
    NeuroLangFailedParseException,
    NeuroLangFrontendException
)
from ...expression_walker import ExpressionWalker, add_match
from ...expressions import (
    Command,
    Constant,
    Definition,
    Expression,
    FunctionApplication,
    Lambda,
    Symbol,
    expressions_behave_as_objects
)
from ...logic import (
    TRUE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication,
    Negation,
    Union,
    UniversalPredicate
)
from ...logic.transformations import RemoveTrivialOperations
from ...probabilistic.expressions import PROB, Condition
from ...type_system import (
    Unknown,
    get_args,
    get_parameters,
    is_leq_informative,
    is_parameterized
)
from .squall import (
    P1,
    P2,
    PN,
    S1,
    S2,
    Aggregation,
    CollapseUnions,
    E,
    ExpandListArgument,
    K,
    Label,
    LambdaSolver,
    ProbabilisticChoiceSymbol,
    ProbabilisticFactSymbol,
    Query,
    S,
    SquallSolver,
    The,
    TheToExistential,
    TheToUniversal,
    squall_to_fol
)

alpha = TypeVar("alpha")

LEMMATIZER = WordNetLemmatizer()
STEMMER = EnglishStemmer()


def lemmatize(word, pos):
    try:
        lemmatized = LEMMATIZER.lemmatize(word, pos)
        if (
            lemmatized == word and
            word not in wordnet.all_lemma_names()
        ):
            lemmatized = STEMMER.stem(word)
            if lemmatized == word and " " not in word:
                warn(f"Word {word} couldn't be lemmatized")
    except Exception:
        lemmatized = lemmatize(word, pos)
    return lemmatized


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


KEYWORDS = [
    'a',
    'all',
    'an',
    'and',
    'are',
    'as',
    'by',
    'choice',
    'conditioned',
    'define',
    'defines',
    'do',
    'does',
    'did',
    'every',
    'equal',
    'from',
    'for',
    'greater',
    'given',
    'obtain',
    'has',
    'hasn\'t',
    'have',
    'had',
    'if',
    'in',
    'is',
    'isn\'t',
    'lower',
    'no',
    'not',
    'of',
    'or',
    'per',
    'probability',
    'probably',
    'some',
    'such',
    'than',
    'that',
    'the',
    'then',
    'there',
    'to',
    'was',
    'were',
    'when',
    'where',
    'which',
    'with',
    'who',
    'whom',
    'whose',
]


GRAMMAR = (
    r"""
?start: squall

squall : ( rule )* rule

_SEPARATOR : "."

rule  : rule1 _SEPARATOR?
      | rulen _SEPARATOR?
      | query _SEPARATOR?
      | COMMENT
      | command _SEPARATOR?

command : "#" identifier "(" arguments ")"

arguments: (argument ("," argument)* )*
argument : term                 -> arg
         | identifier "=" term  -> kwd

query : _OBTAIN ops

_rule_start : _DEFINE _AS

?rule1 : _rule_start rule1_body
rule1_body : [ PROBABLY ] verb1 prep? rule_body1
           |  PROBABLY verb1 prep? rule_body1_cond
           |  verb1 _WITH _PROBABILITY op rule_body1 -> rule_op_fact
           |  _CHOICE verb1 _WITH _PROBABILITY op rule_body1 -> rule_op_choice

rule_body1 : det ng1
rule_body1_cond : det ng1 _CONDITIONED _TO s -> rule_body1_cond_prior
                | s _CONDITIONED _TO det ng1 -> rule_body1_cond_posterior

?rulen : _rule_start rulen_body
?rulen_body : PROBABLY? verbn [ prep ] rule_body1 ";" ops -> rule_opn
            | PROBABLY verbn condition -> rule_opnc_per

condition : ops _CONDITIONED prep ops -> condition_oo
          | s _CONDITIONED prep ops   -> condition_so
          | ops _CONDITIONED prep s   -> condition_os
          | s _CONDITIONED s          -> condition_ss

PROBABLY : _PROBABLY

?s : bool{s_b}
?s_b : np [ "," ]  vp          -> s_np_vp
      | _FOR np ","  s  -> s_for
      | _THERE be np   -> s_be

_CONSEQUENCE : /define[s]{0,1}/

?np : expr{np_b}    -> expr_np
?np_b : det ng1     -> np_quantified
      | np2 _OF np -> np_np2

det : det1  -> det_some
    | EVERY -> det_every
    | THE   -> det_the

EVERY : _EVERY | _ALL
THE : _THE

det1 : _SOME -> det1_some
     | _A   -> det1_some
     | _AN  -> det1_some
     | _NO   -> det1_no

np2 : det ng2

ng1 : noun1 [ app ] [ rel ]                                                   -> ng1_noun
    | noun1 [ app ] [ _COMMA ops _COMMA ] ( _OF | _FROM ) npc{_THE}  [ dims ] -> ng1_agg_npc
    // | noun1 [ app ] ng1 [ dims ]           -> ng1_agg_ng1
ng2 : noun2 [ app ]



app : _IN number"D" -> app_dimension
    | label          -> app_label

dims : dim                    -> dims_base
     | dim _CONJUNCTION dims  -> dims_rec

dim : _PER ng2      -> dim_ng2
    | _PER npc{THE} -> dim_npc

?rel : bool{rel_b}

rel_b : (_THAT | _WHICH | _WHERE | _WHO ) vp                -> rel_vp
      | (_THAT | _WHICH | _WHERE | _WHOM ) np verbn [ ops ] -> rel_vpn
      | np2 _OF _WHICH vp                                   -> rel_np2
      | _WHOSE ng2 vp                                       -> rel_ng2
      | _SUCH _THAT s                                       -> rel_s
      | comparison (_THAN | _TO) op                         -> rel_comp
      | adj1 [ cp ]                                         -> rel_adj1
//      | adjn opn                                            -> rel_adjn


!comparison : _GREATER [ _EQUAL ]
            | _LOWER   [ _EQUAL ]
            | [ _NOT ] _EQUAL


term : label
     | literal

?vp : vp_b
?vp_b : vpdo
      | aux{be} vpbe      -> vp_aux
      | aux{have} vphave  -> vp_aux
      | aux{do} vpdo      -> vp_aux

vpdo : verb1 [ cp ] -> vpdo_v1
     | verbn opn    -> vpdo_vn

vpbe : "there"       -> vpbe_there
     | rel           -> vpbe_rel
     | npc{a_an_the} -> vpbe_npc
     | npc_p{in}     -> vpbe_npc

a_an_the : _A
         | _AN
         | _THE

in : _IN

vphave : noun2 op -> vphave_noun2
       | np2 [ rel ] -> vphave_np2

aux{verb} : verb not?
not : _NOT

npc{det_} : term        -> npc_term
          | det_ ng1 -> npc_det

npc_p{prep_} : prep_ ng1 -> npc_det

?be : ( _IS | _ARE | _WAS | _WERE )
?have: ( _HAS | _HAD | _HAVE )
?do: ( _DOES | _DO | _DID )

adj1 : intransitive
adjn : transitive_multiple

noun1 : intransitive
noun2 : transitive

verb1 : intransitive
verbn : transitive_multiple

intransitive : upper_identifier
transitive : identifier
transitive_multiple : identifier

op_np : np

op : prep? op_np

?opn : ops

ops : [prep] ( op_np prep )* op_np

_COMMA : ","

!prep : _BY
      | _FOR
      | _FROM
      | _OF
      | _THEN
      | _TO
      | _WHEN
      | _WHERE
      | _WITH

?cp : opn

label : _LABEL_MARKER label_identifier
      | ANONYMOUS_LABEL
      | "(" _LABEL_MARKER label_identifier (";" _LABEL_MARKER label_identifier )* ")"

_LABEL_MARKER : "?"
              | "@"

ANONYMOUS_LABEL : "_"

upper_identifier : UPPER_NAME
                 | UPPER_NAME_QUOTED

identifier : LOWER_NAME
           | LOWER_NAME_QUOTED

label_identifier : CNAME

?literal : "'"string"'"
         | number
         | "^"NAME      -> external_literal

string : STRING
number : SIGNED_INT
       | SIGNED_FLOAT

?bool{x} : bool_disjunction{x}
bool_disjunction{x} : bool_conjunction{x}
                    | bool_disjunction{x} _DISJUNCTION bool_conjunction{x}
bool_conjunction{x} : bool_atom{x}
                    | bool_conjunction{x} _CONJUNCTION bool_atom{x}
bool_atom{x} : _NEGATION bool_atom{x} -> bool_negation
             | "--" bool{x} "--"
             | "(" bool{x} ")"
             | "[" bool{x} "]"
             | "," bool{x} ","
             | x

?parenthesized{x} : x
                  | "(" x ")"
                  | "," x ","
                  | "--" x "--"

_CONJUNCTION : "&" | "," | "\N{LOGICAL AND}" | _AND
_DISJUNCTION : "|" | "\N{LOGICAL OR}" | _OR
_IMPLICATION : ":-" | "\N{LEFTWARDS ARROW}" | _IF
_NEGATION : _NOT | "\N{Not Sign}"

?expr{x} : expr_sum{x}
expr_sum{x} : ( expr_sum{x} SUM )? expr_mul{x}
expr_mul{x} : ( expr_mul{x} MUL )? expr_pow{x}
expr_pow{x} : expr_exponent{x} (pow expr_exponential{x})?
?expr_exponent{x} : expr_atom{x}
?expr_exponential{x} : expr_atom{x}
expr_atom{x} : "(" expr{x} (";" expr{x})+ ")"              -> expr_atom_tuple
             | "(" expr{x} ")"                             -> expr_atom_par
             | identifier"(" (expr{x} ("," expr{x})*)? ")" -> expr_atom_fun
             | term                                        -> expr_atom_term
             | x

SUM : "+"
    | "-"

MUL : "*"
    | "/"

?pow : "**"

__KEYWORD_RULES__

LOWER_NAME : /(?!(\b(_KEYWORD)\b))//[a-z_]\w*/
UPPER_NAME : /(?!(\b(_KEYWORD)\b))//[A-Z]\w*/
NAME : /(?!(\b(_KEYWORD)\b))//[a-zA-Z_]\w*/
UPPER_NAME_QUOTED : /`[A-Z][^`]*`/
LOWER_NAME_QUOTED : /`[a-z][^`]*`/
STRING : /[^']+/
COMMENT : /\%\%[^\n]*/ NEWLINE

%import common._STRING_ESC_INNER
%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.WS
%import unicode.WS_INLINE
%import common.NEWLINE
%import common.CNAME
%ignore WS
%ignore WS_INLINE
%ignore NEWLINE  // comment out to avoid crash
%ignore COMMENT

    """
    .replace(
        '__KEYWORD_RULES__',
        '\n'.join(
            '_{} : "{}"'.
            format(kw.upper().replace("'", "_"), kw.lower())
            for kw in KEYWORDS
        ) + '\n'
    )
    .replace("_KEYWORD", "|".join(KEYWORDS))
)


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
            k_(y, *z_),
            EQ(d_, y)
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
    def __init__(self, type_predicate_symbols=None, locals=None, globals=None):
        super().__init__()

        if type_predicate_symbols is None:
            type_predicate_symbols = {}
        if locals is None:
            locals = {}
        if globals is None:
            globals = {}

        self.type_predicate_symbols = type_predicate_symbols
        self.locals = locals
        self.globals = globals

    def squall(self, ast):
        return CollapseUnions().walk(Union(tuple(ast)))

    def rule(self, ast):
        try:
            return squall_to_fol(ast[0], self.type_predicate_symbols)
        except NeuroLangFrontendException as e:
            raise e from None
        except NeuroLangException as e:
            raise NeuroLangFrontendException(str(e))

    def query(self, ast):
        ops = ast[0]
        ops = TheToUniversal[ops.type](ops)
        d = Query[Callable[[List[E]], S]]()
        x = Symbol[E].fresh()
        lx = Symbol[List[E]].fresh()
        res = ExpandListArgument[S](ops(lx)(Lambda((x,), d(x))), lx)
        return res

    def command(self, ast):
        args = tuple()
        kws = tuple()
        for a in ast[1]:
            if isinstance(a, Label):
                kws += (a.unapply(),)
            else:
                args += (a,)
        res = Command(ast[0], args, kws)
        return res

    def arguments(self, ast):
        return tuple(ast)

    def arg(self, ast):
        return ast[0]

    def kwd(self, ast):
        return Label(ast[0], ast[1])

    def rule1_body(self, ast):
        probably, verb1, op = ast
        x = Symbol[E].fresh()
        if probably:
            prob = PROB.cast(Callable[[E], float])
            verb = verb1(x, FunctionApplication[float](prob, (x,)))
        else:
            verb = verb1(x)

        op = TheToUniversal[op.type](op)
        return op(Lambda((x,), verb))

    def rule_op_fact(self, ast):
        verb1, probability, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        pps = (
            ProbabilisticFactSymbol
            .cast(Callable[[E, S], S])
        )
        verb = Lambda(
            (x,),
            TheToUniversal[probability.type](
                probability(Lambda((y,), pps(y, verb1(x))))
            )
        )
        op = TheToUniversal[op.type](op)
        return op(verb)

    def rule_op_choice(self, ast):
        verb1, probability, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        pps = (
            ProbabilisticChoiceSymbol
            .cast(Callable[[E, S], S])
        )
        verb = Lambda(
            (x,),
            TheToUniversal[probability.type](
                probability(Lambda((y,), pps(y, verb1(x))))
            )
        )
        res = op(verb)
        return TheToUniversal[res.type](res)

    def rule_opn(self, ast):
        verbn, _, rule_body1, ops = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        ly = Symbol[List[E]].fresh()
        ops = TheToUniversal[ops.type](ops)
        verb_obj = ExpandListArgument[P1](
            Lambda((x,), ops(ly)(Lambda((y,), verbn(x, y)))),
            ly
        )
        res = rule_body1(verb_obj)
        return TheToUniversal[res.type](res)

    def rule_opnc_per(self, ast):
        _, verbn, condition = ast
        x = Symbol[E].fresh()
        lx = Symbol[List[E]].fresh()
        prob = PROB.cast(Callable[[List[E]], float])
        verb_obj = verbn(x, prob(x))
        verb_obj = ExpandListArgument[P1](
            condition(lx)(Lambda((x,), verb_obj)),
            lx
        )
        res = verb_obj
        return TheToUniversal[res.type](res)

    def rule_body1(self, ast):
        res = self.np_quantified(ast[-2:])
        return TheToUniversal[res.type](res)

    def rule_body1_cond_prior(self, ast):
        det, ng1, s = ast[-3:]
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        body = Condition[S](ng1(x), s)
        res = Lambda[S1](
            (d,),
            det(Lambda((x,), body))(d)
        )
        return res

    def rule_body1_cond_posterior(self, ast):
        s, det, ng1 = ast
        d = Symbol[P1].fresh()
        x = Symbol[E].fresh()
        res = Lambda[S1](
            (d,),
            det(Lambda((x,), Condition[S](s, ng1(x))))(d)
        )
        return res

    def s_np_vp(self, ast):
        np, vp = ast
        res = np(vp)
        return res

    def s_for(self, ast):
        x = Symbol[E].fresh()
        np, s = ast
        return np(Lambda((x,), s))

    def s_be(self, ast):
        np = ast[-1]
        x = Symbol[E].fresh()
        return np(Lambda((x,), TRUE))

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

    def vpdo_v1(self, ast):
        x = Symbol[E].fresh()
        verb1, cp = ast
        if cp:
            verb1 = verb1.cast(PN)
            y = Symbol[E].fresh()
            ly = Symbol[List[E]].fresh()
            res = ExpandListArgument[P1](
                Lambda((x,), cp(ly)(Lambda((y,), verb1(x, y)))),
                ly
            )(x)
        else:
            res = verb1.cast(P1)(x)
        res = Lambda[P1]((x,), res)
        return res

    def vpdo_vn(self, ast):
        verbn, ops = ast
        x = Symbol[E].fresh()
        y = Symbol[List[E]].fresh()
        ly = Symbol[List[E]].fresh()
        res = ExpandListArgument[P1](
            Lambda((x,), ops(ly)(Lambda((y,), verbn(x, y)))),
            ly
        )
        return res

    def aux(self, ast):
        s = Symbol[S].fresh()
        if len(ast) > 1:
            return Lambda((s,), Negation(s))
        else:
            return Lambda((s,), s)

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

    def vphave_noun2(self, ast):
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
        np = ast[-1]

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

    def op(self, ast):
        return ast[-1]

    def ops(self, ast):
        ops = ast[1::2]
        lz = Symbol[List[E]].fresh()
        d = Symbol[Callable[[List[E]], S]].fresh()

        body = d(lz)
        with expressions_behave_as_objects():
            for i, op in enumerate(ops):
                z = Symbol[E].fresh()
                body = op(Lambda(
                    (z,),
                    Conjunction[S]((Label[S](z, lz[Constant[int](i)]), body)))
                )

        return Lambda((lz,), Lambda((d,), body))

    def prep(self, ast):
        res = Constant[str](ast[0].lower())
        return res

    def ng1_noun(self, ast):
        x = Symbol[E].fresh()
        noun1, app, rel = ast
        prep = None

        if prep is not None:
            noun1 = noun1.cast[P2]
            y = Symbol[E].fresh()
            z = Symbol[E].fresh()
            noun1 = Lambda((y,), prep(Lambda((z,), noun1(y, z))))

        args = (noun1, app, rel)
        return Lambda[P1]((x,), Conjunction[S](tuple(
            FunctionApplication(a, (x,))
            for a in args if a is not None
        )))

    def ng1_agg_npc(self, ast):
        noun1, app, ops, npc, dims = ast
        if ops:
            x = Symbol[E].fresh()
            w = Symbol[E].fresh()
            lw = Symbol[List[E]].fresh()
            noun1 = noun1.cast(Callable[[E, List[E]], bool])
            noun1 = ExpandListArgument[P1](
                Lambda((x,), ops(lw)(Lambda((w,), noun1(x, w)))),
                lw
            )
        aggreg = self.adj_aggreg([noun1])
        v = Symbol[S].fresh()
        y = Symbol[E].fresh()
        lz = Symbol[List[E]].fresh()

        inner = (npc(y),)
        formulas = tuple()
        if dims:
            inner += (dims(y)(lz),)

        if app:
            formulas += (app(v),)

        inner = Conjunction[S](inner)

        formulas += (aggreg(Lambda(
            (lz,),
            Lambda(
                (y,),
                inner
            )
        ))(v),)

        formulas = Conjunction[S](formulas)

        res = Lambda((v,), formulas)
        return res

    def ng1_agg_ng1(self, ast):
        return self.ng1_agg_npc(ast)

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

    def adj1(self, ast):
        name = lemmatize(ast[0].name.lower(), 'a')
        return ast[0].apply(name)

    def adjn(self, ast):
        name = lemmatize(ast[0].name.lower(), 'a')
        return ast[0].apply(name)

    def noun1(self, ast):
        name = lemmatize(ast[0].name.lower(), 'n')
        return ast[0].apply(name)

    def noun2(self, ast):
        name = lemmatize(ast[0].name.lower(), 'n')
        return ast[0].apply(name)

    def verb1(self, ast):
        name = lemmatize(ast[0].name.lower(), 'v')
        return ast[0].apply(name)

    def verb2(self, ast):
        name = lemmatize(ast[0].name.lower(), 'v')
        return ast[0].apply(name)

    def verbn(self, ast):
        name = lemmatize(ast[0].name.lower(), 'v')
        return ast[0].apply(name)

    def intransitive(self, ast):
        return ast[0].cast(P1)

    def transitive(self, ast):
        return ast[0].cast(P2)

    def transitive_multiple(self, ast):
        return ast[0].cast(PN)

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

    def dims_base(self, ast):
        dim = ast[0]
        y = Symbol[E].fresh()
        lz = Symbol[List[E]].fresh()
        with expressions_behave_as_objects():
            res = Lambda(
                (y,),
                Lambda(
                    (lz,),
                    dim(y)(lz[Constant(0)])
                )
            )
        return res

    def dims_rec(self, ast):
        dim, dims = ast
        y = Symbol[E].fresh()
        lz = Symbol[List[E]].fresh()

        with expressions_behave_as_objects():
            res = Lambda(
                (y,),
                Lambda(
                    (lz,),
                    Conjunction[S]((
                        dim(y, lz[Constant(0)]),
                        dims(y, lz[Constant(slice(1, None))])
                    ))
                )
            )
        return res

    def dim_ng2(self, ast):
        ng2 = ast[0]
        y = Symbol[E].fresh()
        z = Symbol[E].fresh()
        res = Lambda((z,), Lambda((y,), ng2(y)(z)))
        return res

    def dim_npc(self, ast):
        npc = ast[0]
        y = Symbol[E].fresh()
        z = Symbol[E].fresh()
        res = Lambda((y,), Lambda((z,), npc(z)))
        return res

    def rel_vp(self, ast):
        x = Symbol[E].fresh()
        res = Lambda((x,), ast[0](x))
        return TheToExistential[res.type](res)

    def rel_vp2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        np, verb2 = ast
        res = verb2.cast(P2)(x, y)
        res = np(Lambda((x,), res))
        res = Lambda((y,), res)
        return TheToExistential[res.type](res)

    def rel_vpn(self, ast):
        np, verbn, ops = ast
        if ops is None:
            return self.rel_vp2(ast[:2])
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        z = Symbol[E].fresh()
        lz = Symbol[List[E]].fresh()
        res = ExpandListArgument[P1](
            Lambda((x,), ops(lz)(Lambda((z,), verbn(x, y, z)))),
            lz
        )
        res = Lambda((y,), np(res))
        return TheToExistential[res.type](res)

    def rel_np2(self, ast):
        x = Symbol[E].fresh()
        np2, vp = ast
        res = Lambda(
            (x,),
            np2(x)(vp)
        )
        return TheToExistential[res.type](res)

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
                )(y)
            )
        )
        return TheToExistential[res.type](res)

    def rel_s(self, ast):
        s = ast[0]
        x = Symbol[E].fresh()
        res = Lambda((x,), s)
        return TheToExistential[res.type](res)

    def rel_comp(self, ast):
        comp, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        res = Lambda(
            (x,),
            op(Lambda((y,), comp(x, y)))
        )
        return TheToExistential[res.type](res)

    def rel_adj1(self, ast):
        adj1, cp = ast
        x = Symbol[E].fresh()
        if cp:
            adj1 = adj1.cast(PN)
            y = Symbol[E].fresh()
            ly = Symbol[List[E]].fresh()
            res = ExpandListArgument[P1](
                Lambda((x,), cp(ly)(Lambda((y,), adj1(x, y)))),
                ly
            )
        else:
            res = Lambda((x,), adj1(x))
        return TheToExistential[res.type](res)

    def rel_adjn(self, ast):
        adjn, ops = ast
        x = Symbol[E].fresh()
        adjn = adjn.cast(Callable[[E, List[E]], S])
        y = Symbol[E].fresh()
        ly = Symbol[List[E]].fresh()
        res = ExpandListArgument[P1](
            Lambda((x,), ops(ly)(Lambda((y,), adjn(x, y)))),
            ly
        )
        return res

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

    def upper_identifier(self, ast):
        ast = ''.join(ast)
        return Symbol[E](ast)

    def identifier(self, ast):
        return Symbol[E](ast[0])

    def label_identifier(self, ast):
        return Symbol[E](ast[0])

    def number(self, ast):
        return Constant(ast[0])

    def string(self, ast):
        return Constant[str](ast[0])

    def condition_oo(self, ast):
        ops1, _, ops2 = ast
        lx = Symbol[List[E]].fresh()
        d = Symbol[P1].fresh()
        res = Lambda(
            (lx,),
            Lambda((d,), Condition[S](ops1(lx)(d), ops2(lx)(d)))
        )

        return res

    def condition_so(self, ast):
        s, _, ops = ast
        lx = Symbol[List[E]].fresh()
        d = Symbol[P1].fresh()
        res = Lambda((lx,), Lambda((d,), Condition[S](s, ops(lx)(d))))

        return res

    def condition_os(self, ast):
        ops, _, s = ast
        lx = Symbol[List[E]].fresh()
        d = Symbol[P1].fresh()
        res = Lambda((lx,), Lambda((d,), Condition[S](ops(lx)(d), s)))

        return res

    def condition_ss(self, ast):
        s, _, s = ast
        lx = Symbol[List[E]].fresh()
        d = Symbol[P1].fresh()
        res = Lambda((lx,), Lambda((d,), Condition[S](s, s)))

        return res

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
        if len(ast) == 1:
            return ast[0]
        ast = tuple((ast[0], ast[-1]))
        return self.apply_expression(POW, ast)

    def expr_atom_par(self, ast):
        return ast[0]

    def expr_atom_tuple(self, ast):
        params = tuple(ast)
        return self.apply_expression(Constant(tuple), params)

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

    def external_literal(self, ast):
        name = ast[0]
        if name in self.locals:
            return Constant(self.locals[name])
        elif name in self.globals:
            return Constant(self.globals[name])
        else:
            raise NeuroLangFrontendException(
                f"Variable {name} not found in environment"
            )

    def adj_aggreg(self, ast):
        functor = ast[0].cast(Callable[[Callable[[List[E]], P1]], P1])
        d = Symbol[List[E]].fresh()
        x = Symbol[P1].fresh()
        res = Lambda(
            (d,),
            Lambda(
                (x,),
                Aggregation[P1](functor, d, x)
            )
        )
        return res

    CNAME = str
    NAME = str
    LOWER_NAME = str
    LOWER_NAME_QUOTED = itemgetter(slice(1, -1))
    UPPER_NAME = str
    UPPER_NAME_QUOTED = itemgetter(slice(1, -1))
    SIGNED_INT = int
    SIGNED_FLOAT = float
    STRING = str


COMPILED_GRAMMAR = lark.Lark(GRAMMAR, parser="lalr")


# Errors to check:
#   * If there is an existential variable at the beginning of a rule, then the variable was badly quantified
def parser(
    code,
    type_predicate_symbols=None, locals=None, complete=False,
    globals=None, return_tree=False, process=True, **kwargs
):
    global COMPILED_GRAMMAR
    if (
        "parser" in kwargs and
        kwargs["parser"] != COMPILED_GRAMMAR.options.parser
    ):
        COMPILED_GRAMMAR = lark.Lark(GRAMMAR, parser=kwargs["parser"])
    if not complete:
        try:
            return parse_to_neurolang_ir(
                code,
                type_predicate_symbols=type_predicate_symbols, locals=locals,
                globals=globals, return_tree=return_tree, process=process
            )
        except lark.exceptions.VisitError as ex:
            raise ex.orig_exc
        except Exception as ex:
            raise ex from None
    else:
        completions = extract_completions(code)
        return list(sorted(completions))


def _callback(collection):
    def callback_(token):
        collection.add(token.value)
        return token
    return callback_


def extract_completions(code: str):
    transitive = set()
    intransitive = set()
    labels = set()
    lexer_sets = {
        'UPPER_NAME': transitive, 'UPPER_NAME_QUOTED': transitive,
        'LOWER_NAME': intransitive, 'LOWER_NAME_QUOTED': intransitive,
        'CNAME': labels
    }
    lexer_callbacks = {k: _callback(v) for k, v in lexer_sets.items()}

    completions = []
    try:
        parser = lark.Lark(
            COMPILED_GRAMMAR.grammar,
            parser=COMPILED_GRAMMAR.options.parser,
            lexer_callbacks=lexer_callbacks
        )
        parser.parse(code)
        parser.parse("")
    except lark.exceptions.UnexpectedInput as ex:
        if ex.token.type != '$END':
            return []
        expected = ex.interactive_parser.accepts()
        completions += terminals_to_options(
            lexer_sets, expected, ex.interactive_parser
        )

    return list(sorted(set(completions)))


def terminals_to_options(lexer_sets, expected, interactive_parser=None) -> List[str]:
    completions = []
    for candidate in expected:
        token = COMPILED_GRAMMAR.get_terminal(candidate)
        if isinstance(token.pattern, lark.lexer.PatternStr):
            completions.append(token.pattern.value)
        elif token.name in lexer_sets:
            completions += list(lexer_sets[token.name])
        elif isinstance(token.pattern, lark.lexer.PatternRE):
            match = re.match(r"^\(\?\:([^\|]+)(?:\|([^\|]+))+\)$", token.pattern.value)
            if match:
                completions += list(match.groups())
    if (
        interactive_parser and
        len(interactive_parser.parser_state.value_stack) > 1 and
        [
            t.type for t in interactive_parser.parser_state.value_stack[-2:]
            if isinstance(t, lark.common.Token)
        ] in [["_DEFINE", "_AS"], ["PROBABLY"]]
    ):
        completions += list(lexer_sets[token.name])
        completions.append("NEW IDENTIFIER")

    return completions


def parse_to_neurolang_ir(
    code: str,
    type_predicate_symbols=None,
    locals=None, globals=None,
    return_tree=False, process=True, **kwargs
) -> Expression:
    try:
        tree = COMPILED_GRAMMAR.parse(code)
    except lark.exceptions.UnexpectedEOF as ex:
        err = ex.get_context(code, span=80)
        expected = set(ex.expected)
        expected_formatted = '\n\t* ' + '\n\t* '.join(
            "%s : %s" % (t, COMPILED_GRAMMAR.get_terminal(t).pattern.to_regexp())
            for t in sorted(expected)
        )
        raise NeuroLangFailedParseException(
            "\n" + err + expected_formatted, line=ex.line - 1, column=ex.column
        ) from None
    except lark.exceptions.UnexpectedInput as ex:
        err = ex.get_context(code, span=80)
        if hasattr(ex, "interactive_parser"):
            expected = terminals_to_options({}, ex.accepts)
            expected_formatted = '\n\t* ' + '\n\t* '.join(sorted(expected))
        else:
            expected = ex.allowed
            expected_formatted = '\n\t* ' + '\n\t* '.join(
                "%s : %s" % (t, COMPILED_GRAMMAR.get_terminal(t).pattern.to_regexp())
                for t in sorted(expected)
            )
        raise NeuroLangFailedParseException(
            "\n" + err + expected_formatted, line=ex.line - 1, column=ex.column
        ) from None
    except lark.exceptions.LarkError as ex:
        raise ex from None
    except NeuroLangException as ex:
        raise ex from None

    try:
        if process:
            intermediate_representation = SquallTransformer(
                type_predicate_symbols=type_predicate_symbols,
                locals=locals,
                globals=globals
            ).transform(tree)
        else:
            intermediate_representation = None

        if return_tree:
            return intermediate_representation, tree
        else:
            return intermediate_representation
    except NeuroLangException as ex:
        raise ex from None
