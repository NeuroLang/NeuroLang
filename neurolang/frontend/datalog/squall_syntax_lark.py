import re
from operator import add, eq, ge, gt, le, lt, mul, ne, neg, pow, sub, truediv
from typing import Callable, List, TypeVar
from warnings import warn

import lark
from nltk.corpus import wordnet
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from ...exceptions import NeuroLangException, NeuroLangFailedParseException, NeuroLangFrontendException
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
    FROM,
    P1,
    P2,
    PN,
    S1,
    S2,
    TO,
    Aggregation,
    Arg,
    CollapseUnions,
    Cons,
    E,
    ExpandListArgument,
    ForArg,
    K,
    Label,
    LambdaSolver,
    ProbabilisticPredicateSymbol,
    Query,
    S,
    SquallSolver,
    The,
    squall_to_fol
)

# nltk.download('omw-1.4')


alpha = TypeVar("alpha")
K_alpha = Callable[[Callable[[E], alpha]], alpha]

LEMMATIZER = WordNetLemmatizer()
STEMMER = EnglishStemmer()


def lemmatize(word, pos):
    lemmatized = LEMMATIZER.lemmatize(word, pos)
    if (
        lemmatized == word and
        word not in wordnet.all_lemma_names()
    ):
        lemmatized = STEMMER.stem(word)
        if lemmatized == word and " " not in word:
            warn(f"Word {word} couldn't be lemmatized")
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

OfType = Symbol[Callable[[E], E]]("rdf:type")


KEYWORDS = [
    'a',
    'all',
    'an',
    'and',
    'are',
    'as',
    'by',
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

squall : ( rule _LINE_BREAK )* rule _LINE_BREAK

_LINE_BREAK : ( WS* "." WS* NEWLINE* WS* )+

?rule  : rule_op
       | rulen
       | query
       | COMMENT
       | command

command : "#" identifier "(" (term ("," term)* )* ")" "."

query : _OBTAIN ops

rule_op : "define" "as" [ PROBABLY ] verb1 rule_body1 "."?
        | "define" "as" PROBABLY verb1 rule_body1_cond "."?
        | "define" "as" verb1 _WITH _PROBABILITY np rule_body1 "."? -> rule_op_prob

?rulen : "define" "as" verbn rule_body1 _BREAK? ops "."? -> rule_opnn
       | "define" "as" PROBABLY verbn rule_body1 _CONDITIONED? _BREAK? ops "."? -> rule_opnn_per

rule_body1 : prep? _GIVEN? det ng1
rule_body1_cond : det ng1 _CONDITIONED _TO s -> rule_body1_cond_prior
                | s _CONDITIONED _TO det ng1 -> rule_body1_cond_posterior

rule_body2_cond : det ng1 _CONDITIONED _TO det ng1

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

det1 : SOME -> det1_some
     | AN   -> det1_some
     | NO   -> det1_no

SOME : _SOME
AN : _A
   | _AN
NO : _NO

np2 : det ng2

ng1 : noun1 [ app ] [ rel ] -> ng1_noun
    | noun1 [ app ] ( _OF | _FROM ) npc{_THE}  [ dims ]   -> ng1_agg_npc
    // | noun1 [ app ] ng1 [ dims ]           -> ng1_agg_ng1
ng2 : noun2 [ app ]



app : _IN number"D" -> app_dimension
    | label          -> app_label

dims : dim                    -> dims_base
     | dim _CONJUNCTION dims  -> dims_rec

dim : _PER ng2      -> dim_ng2
    | _PER npc{THE} -> dim_npc

?rel : bool{rel_b}
     | _DASH bool{rel_b} _DASH

_DASH : "--"

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

?vp : bool{vp_b}
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

aux{verb} : verb       -> aux_id
          | verb _NOT  -> aux_not

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

op : prep? np             -> op_np
   | _DASH prep? np _DASH -> op_np

?opn : ops
     | _DASH ops _DASH

ops : op                              -> ops_base
    | ops _BREAK? prep op             -> ops_rec
    | ops _BREAK? _DASH prep op _DASH -> ops_rec

pp : prep np -> pp_np

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
         | "(" bool{x} ")"
         | x

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
expr_atom{x} : "(" expr{x} ")"                             -> expr_atom_par
             | "(" expr{x} (";" expr{x})+ ")"              -> expr_atom_tuple
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
%import unicode.NEWLINE
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
.replace("_BREAK", '(";"|",")')
)


with open(os.path.join(os.path.dirname(__file__), "neurolang_natural.lark"), 'w') as f:
    f.write(GRAMMAR)

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
        return CollapseUnions().walk(
            Union(tuple(
                squall_to_fol(node, self.type_predicate_symbols)
                for node in ast)
            )
        )

    def rule_simple(self, ast):
        np, vpcons = ast
        res = np(vpcons)
        return res

    def rule_rec(self, ast):
        x = Symbol[E].fresh()
        np, rule = ast
        return np(Lambda((x,), rule))

    def query(self, ast):
        ops = ast[0]
        d = Query[Callable[[List[E]], S]]()
        x = Symbol[E].fresh()
        lx = Symbol[List[E]].fresh()
        res = ExpandListArgument[S](ops(lx)(Lambda((x,), d(x))), lx)
        return res

    def command(self, ast):
        res = Command(ast[0], tuple(ast[1:]), ())
        return res

    def rule_op(self, ast):
        probably, verb1, op = ast
        x = Symbol[E].fresh()
        if probably:
            prob = PROB.cast(Callable[[E], float])
            verb = verb1(x, FunctionApplication[float](prob, (x,)))
        else:
            verb = verb1(x)
        return op(Lambda((x,), verb))

    def rule_op_prob(self, ast):
        verb1, probability, op = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        pps = (
            ProbabilisticPredicateSymbol
            .cast(Callable[[E, S], S])
        )
        verb = Lambda((x,), probability(Lambda((y,), pps(y, verb1(x)))))
        return op(verb)

    def rule_op_cond1(self, ast):
        probably, verb1, op = ast
        x = Symbol[E].fresh()
        if probably:
            prob = PROB.cast(Callable[[E], float])
            verb = verb1(x, FunctionApplication[float](prob, (x,)))
        else:
            verb = verb1(x)
        return op(Lambda((x,), verb))

    def rule_op2_cond(self, ast):
        verb2, np2 = ast[-2:]
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        prob = PROB.cast(Callable[[E, E], float])

        verb = verb2(x, y, FunctionApplication[float](prob, (x, y)))
        res = np2(verb)
        return res

    def rule_op2(self, ast):
        probably, verb2, np, prep, op = ast
        s = Symbol[S].fresh()
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        z = Symbol[E].fresh()

        if probably:
            prob = PROB.cast(Callable[[E, E], float])
            verb = verb2(x, y, FunctionApplication[float](prob, (x, y)))
        else:
            verb = verb2(x, y)

        pp = Lambda((s,), op(Lambda((z,), Arg(prep, (z, s)))))
        vp = Lambda((x,), ForArg(prep, Lambda((y,), verb)))
        vp_pp = Lambda((x,), pp(vp(x)))
        res = np(vp_pp)
        return res

    def rule_opn(self, ast):
        probably, verb2, np = ast[:3]
        n_op = len(ast[3:]) // 2
        args = tuple(Symbol[E].fresh() for _ in range(n_op + 1))
        if probably:
            args += (FunctionApplication[float](Symbol[Callable[[E], float]]("PROB"), args),)

        vp_pp = verb2(*args)
        arg_ant = args[0]
        for prep, op, arg in zip(ast[3::2], ast[4::2], args[1:]):
            s = Symbol[S].fresh()
            z = Symbol[E].fresh()
            x = Symbol[E].fresh()
            pp = Lambda((s,), op(Lambda((z,), Arg(prep, (z, s)))))
            vp = Lambda((arg_ant,), ForArg(prep, Lambda((arg,), vp_pp)))
            vp_pp = Lambda((x,), pp(vp(x)))
            arg_ant = arg

        res = np(vp_pp)
        return res

    def rule_op2_b(self, ast):
        probably, verb2, prep, op, np = ast
        s = Symbol[S].fresh()
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        z = Symbol[E].fresh()

        if probably:
            verb = verb2(x, y, FunctionApplication[float](Symbol[Callable[[E], float]]("PROB"), (x, y)))
        else:
            verb = verb2(x, y)

        pp = Lambda((s,), op(Lambda((z,), Arg(prep, (z, s)))))
        vp = Lambda((x,), ForArg(prep, Lambda((y,), verb)))
        vp_pp = Lambda((x,), pp(vp(x)))
        res = np(vp_pp)
        return res

    def rule_opnn(self, ast):
        verbn, rule_body1, ops = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        ly = Symbol[List[E]].fresh()
        verb_obj = ExpandListArgument[P1](
            Lambda((x,), ops(ly)(Lambda((y,), verbn(x, y)))),
            ly
        )
        res = rule_body1(verb_obj)
        return res

    def rule_opnn_per(self, ast):
        _, verbn, rule_body1, ops = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        ly = Symbol[List[E]].fresh()
        prob = PROB.cast(Callable[[E, List[E]], float])
        verb_obj = ExpandListArgument[P1](
            Lambda((x,), ops(ly)(Lambda((y,), verbn(x, y, prob(x, y))))),
            ly
        )
        res = rule_body1(verb_obj)
        return res

    def rule_body1(self, ast):
        return self.np_quantified(ast[-2:])

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

    def rule_body2_cond(self, ast):
        det_1, ng1_1, det_2, ng1_2 = ast
        d = Symbol[P2].fresh()
        x = Symbol[E].fresh() 
        y = Symbol[E].fresh()       

        np = self.np_quantified((det_1, ng1_1))
        op = self.np_quantified((det_2, ng1_2))

        res = Lambda((d,), np(Lambda((x,), op(Lambda((y,), d(x, y))))))
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

    def np_every_1(self, ast):
        det, ng1 = ast
        d = Symbol[P1].fresh()
        res = Lambda[S1](
            (d,),
            det(ng1)(d)
        )
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

    def vpdo_vn(self, ast):
        verbn, ops = ast
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        ly = Symbol[List[E]].fresh()
        res = ExpandListArgument[P1](
            Lambda((x,), ops(ly)(Lambda((y,), verbn(x, y)))),
            ly
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

    def op_pp(self, ast):
        pp, op = ast
        x = Symbol[E].fresh()
        res = Lambda((x,), pp(op(x)))
        return res

    def ops_base(self, ast):
        op = ast[0]
        lz = Symbol[List[E]].fresh()
        d = Symbol[P1].fresh()
        with expressions_behave_as_objects():
            lz_head = lz[Constant(0)]
            lz_tail = lz[Constant(slice(1, None))]
            z = Symbol[E].fresh()
            pred = Lambda(
                (z,),
                Conjunction[S]((
                    Label[S](z, lz_head),
                    d(Cons(lz_head, lz_tail))
                ))
            )
            res = Lambda((lz,), Lambda((d,), op(pred)))
        return res

    def ops_rec(self, ast):
        ops, _, op = ast
        lz = Symbol[List[E]].fresh()
        d = Symbol.fresh()
        zz = Symbol.fresh()

        with expressions_behave_as_objects():
            lz_head = lz[Constant(0)]
            lz_tail = lz[Constant(slice(1, None))]
            z = Symbol[E].fresh()
            label = Lambda(
                (z,),
                Conjunction((
                    Label[S](z, lz_head),
                    ops(lz_tail)(Lambda((zz,), d(Cons(lz_head, zz))))
                ))
            )
            res = Lambda((lz,), Lambda((d,), op(label)))
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
        else:
            res = Constant[str](ast[0].lower())
        return res

    def pp_np(self, ast):
        prep, np = ast
        s = Symbol[S].fresh()
        z = Symbol[E].fresh()
        res = Lambda((s,), np(Lambda((z,), Arg(prep, ((z, s))))))
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
        noun1, app, npc, dims = ast
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

    def adj2(self, ast):
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
        return Lambda((x,), ast[0](x))

    def rel_vp2(self, ast):
        x = Symbol[E].fresh()
        y = Symbol[E].fresh()
        np, verb2 = ast
        res = verb2.cast(P2)(x, y)
        res = np(Lambda((x,), res))
        res = Lambda((y,), res)
        return res

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
                )(y)
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
        return res

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

    def noun_aggreg(self, ast):
        return self.adj_aggreg(ast)

    def adj_aggreg(self, ast):
        functor = ast[0].apply(ast[0].name.lower())
        functor = functor.cast(Callable[[Callable[[List[E]], P1]], P1])
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
    LOWER_NAME_QUOTED = lambda self, s: s[1:-1]
    UPPER_NAME = str
    UPPER_NAME_QUOTED = lambda self, s: s[1:-1]
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
    if "parser" in kwargs and kwargs["parser"] != COMPILED_GRAMMAR.options.parser:
        COMPILED_GRAMMAR = lark.Lark(GRAMMAR, parser=kwargs["parser"])
    if not complete:
        try:
            return parse_to_neurolang_ir(
                code,
                type_predicate_symbols=type_predicate_symbols, locals=locals,
                globals=globals, return_tree=return_tree, process=process
            )
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
        completions += terminals_to_options(lexer_sets, expected)

    return list(sorted(set(completions)))


def terminals_to_options(lexer_sets, expected) -> List[str]:
    completions = []
    for candidate in expected:
        token = COMPILED_GRAMMAR.get_terminal(candidate)
        if isinstance(token.pattern, lark.lexer.PatternStr):
            completions.append(token.pattern.value)
        elif token.name in lexer_sets:
            completions += list(lexer_sets[token.name])
        elif isinstance(token.pattern, lark.lexer.PatternRE):
            match = re.match(r"^\(\?\:(\w+)(?:\|(\w+))+\)$", token.pattern.value)
            if match:
                completions += list(match.groups())
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
    except lark.exceptions.UnexpectedException as ex:
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
