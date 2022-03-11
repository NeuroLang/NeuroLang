import operator

import pandas as pd
import pytest
from pytest import raises


from ... import expression_walker, expressions
from ...frontend.datalog.sugar import TranslateProbabilisticQueryMixin
from ...frontend.probabilistic_frontend import NeurolangPDL
from ...logic import Negation
from .. import DatalogProgram, Fact, Implication, Conjunction
from ..aggregation import (
    AGG_COUNT,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from ..chase import Chase
from ..exceptions import (
    BoundAggregationApplicationError,
    NegationInMagicSetsRewriteError,
)
from ..magic_sets import (
    AdornedSymbol,
    LeftToRightSIPS,
    ReplaceAdornedSymbolWalker,
    magic_rewrite,
)
from ..negation import DatalogProgramNegationMixin

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
Cnj_ = Conjunction
F_ = Fact
Eb_ = expressions.ExpressionBlock


class Datalog(
    TranslateProbabilisticQueryMixin,
    TranslateToLogicWithAggregation,
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    DatalogProgramNegationMixin,
    DatalogProgram,
    expression_walker.ExpressionBasicEvaluator,
):
    pass


x = S_("x")
y = S_("y")
z = S_("z")
x1 = S_("x1")
y1 = S_("y1")
rsg = S_("rsg")
up = S_("up")
down = S_("down")
flat = S_("flat")
anc = S_("anc")
par = S_("par")
anc2 = S_("anc2")
q = S_("q")
a = C_("a")
b = C_("b")
c = C_("c")
d = C_("d")
eq = C_(operator.eq)

aq = AdornedSymbol(q, "f", 0)
aanc2 = AdornedSymbol(anc2, "bf", 0)
aanc = AdornedSymbol(anc, "bf", 0)


def test_adorned_symbol():
    ax = AdornedSymbol(x, 'b', 1)
    ax_ = AdornedSymbol(x, 'b', 1)
    ax__ = AdornedSymbol(a, 'b', 1)

    assert ax.expression == x
    assert ax.adornment == 'b'
    assert ax.number == 1
    assert ax == ax_
    assert ax != ax__

    ax = AdornedSymbol(a, 'b', 1)
    ax_ = AdornedSymbol(a, 'b', 1)
    ax__ = AdornedSymbol(a, 'b', 2)

    assert ax.expression == a
    assert ax.adornment == 'b'
    assert ax.number == 1
    assert ax == ax_
    assert ax != ax__

    with raises(NotImplementedError):
        AdornedSymbol(a, 'b', 1).name


def test_unadorned_expression_walker():
    adorned_code = tuple(
        [
            Imp_(aq(x), aanc2(a, x)),
            Imp_(aanc(x, y), par(x, y)),
            Imp_(aanc(x, y), aanc(x, z) & par(z, y)),
            Imp_(aanc2(x, y), aanc(x, y) & Negation(aanc(x, y))),
        ]
    )
    unadorned_code = ReplaceAdornedSymbolWalker().walk(adorned_code)

    xanc2 = S_("anc2^bf_0")
    xanc = S_("anc^bf_0")
    xq = S_("q^f_0")
    expected_code = tuple(
        [
            Imp_(xq(x), xanc2(a, x)),
            Imp_(xanc(x, y), par(x, y)),
            Imp_(xanc(x, y), xanc(x, z) & par(z, y)),
            Imp_(xanc2(x, y), xanc(x, y) & Negation(xanc(x, y))),
        ]
    )
    assert unadorned_code == expected_code


def test_l2r_sips():
    edb = Eb_(
        [
            F_(par(a, b)),
        ]
    )
    dl = Datalog()
    dl.walk(edb)
    sips = LeftToRightSIPS(dl)

    # Default case
    arcs, _ = sips.creates_arcs(Imp_(q(x), anc(a, x)), aq(x))
    assert arcs == {aanc(a, x): (aq(x),)}

    # par is in EDB and should not be an arc
    arcs, _ = sips.creates_arcs(Imp_(anc(x, y), par(x, y)), aanc(x, y))
    assert arcs == {}

    # par is in EDB and should not be an arc
    arcs, _ = sips.creates_arcs(
        Imp_(anc(x, y), Cnj_((anc(x, z), par(z, y)))), aanc(x, y)
    )
    assert arcs == {aanc(x, z): (aanc(x, y),)}

    # bindings should propagate from predicate to predicate
    arcs, _ = sips.creates_arcs(
        Imp_(anc(x, y), Cnj_((anc(y, x), par(z, y), up(y, z), down(z, x)))),
        aanc(x, y),
    )
    ancfb = AdornedSymbol(anc, "fb", 0)(y, x)
    upbf = AdornedSymbol(up, "bf", 0)(y, z)
    downbb = AdornedSymbol(down, "fb", 0)(z, x)
    assert arcs == {
        ancfb: (aanc(x, y),),
        upbf: (aanc(x, y), ancfb),
        downbb: (aanc(x, y), ancfb, upbf),
    }

    # negative predicates should not be added to the tails of arcs
    arcs, _ = sips.creates_arcs(
        Imp_(anc2(x, y), Cnj_((Negation(anc(x, y)), anc2(y, x)))), aanc2(x, y)
    )
    anc2fb = AdornedSymbol(anc2, "fb", 0)
    assert arcs == {
        Negation(aanc(x, y)): (aanc2(x, y),),
        anc2fb(y, x): (aanc2(x, y),),
    }


def test_resolution_works():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_2():
    """
    Reverse Same Generation from Abiteboul et al. 1995.
    Foundations of databases. p.312

    l0:  l  m   n   o   p
    l1:  e  f   g   h   i   j   k
    l2:  a  b   c   d
    """
    edb = Eb_(
        [
            F_(up(C_("a"), C_("e"))),
            F_(up(C_("a"), C_("f"))),
            F_(up(C_("f"), C_("m"))),
            F_(up(C_("g"), C_("n"))),
            F_(up(C_("h"), C_("n"))),
            F_(up(C_("i"), C_("o"))),
            F_(up(C_("j"), C_("o"))),
            F_(flat(C_("g"), C_("f"))),
            F_(flat(C_("m"), C_("n"))),
            F_(flat(C_("m"), C_("o"))),
            F_(flat(C_("p"), C_("m"))),
            F_(down(C_("l"), C_("f"))),
            F_(down(C_("m"), C_("f"))),
            F_(down(C_("g"), C_("b"))),
            F_(down(C_("h"), C_("c"))),
            F_(down(C_("i"), C_("d"))),
            F_(down(C_("p"), C_("k"))),
        ]
    )

    code = Eb_(
        [
            Imp_(q(y), rsg(C_("a"), y)),
            Imp_(rsg(x, y), flat(x, y)),
            Imp_(rsg(x, y), up(x, x1) & rsg(y1, x1) & down(y1, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(C_("b"),), (C_("c"),), (C_("d"),)}


def test_resolution_works_query_constant():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x, a), anc(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e, a)) for e in (b, c, d)}


def test_resolution_works_query_free():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x), anc(a, x)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_builtin():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, y), anc(x, z) & eq(z, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}


def test_resolution_works_aggregation():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, AGG_COUNT(y)), anc(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {(3,)}


def test_bound_aggregation_raises_error():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(x, C_(3))),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, AGG_COUNT(y)), anc(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    with raises(BoundAggregationApplicationError):
        magic_rewrite(q(x), dl)


def test_resolution_works_negation():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(up(x, y), par(x, y)),
            Imp_(anc2(x, y), anc(x, y) & ~up(x, y)),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    goal, mr = magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = Chase(dl).build_chase_solution()
    assert solution[goal].value == {C_((e,)) for e in (c, d)}


def test_complex_negations_raise_errors():
    edb = Eb_(
        [
            F_(par(a, b)),
            F_(par(b, c)),
            F_(par(c, d)),
        ]
    )

    code = Eb_(
        [
            Imp_(q(x), anc2(a, x)),
            Imp_(anc(x, y), par(x, y)),
            Imp_(anc(x, y), anc(x, z) & par(z, y)),
            Imp_(anc2(x, y), anc(x, y) & ~(par(y, z) & anc(x, z))),
        ]
    )

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)

    with raises(NegationInMagicSetsRewriteError):
        magic_rewrite(q(x), dl)


@pytest.fixture
def nl():
    nl = NeurolangPDL()
    studies = (
        10191322,
        10227106,
        10349031,
        10402199,
        10571235,
        10407201,
        10197540,
        10022496,
        10022494,
        9819274,
        11430815,
        11275483,
        10913505,
        10808134,
        10751455,
    )
    nl.add_uniform_probabilistic_choice_over_set(
        list(map(lambda x: (x,), studies)), name="SelectedStudy"
    )
    peaks_data = (
        (-48, -74, 34, 10349031),
        (-33, 28, 7, 10191322),
        (21, 52, 20, 10349031),
        (-42, 28, 9, 10191322),
        (42, -24, 12, 10022496),
        (-38, 15, 26, 10191322),
        (-36, 26, 9, 10191322),
        (-53, -26, -21, 10349031),
        (-42, 12, 17, 10191322),
        (0, -73, 6, 10913505),
        (10, 41, 1, 10349031),
        (-40, 22, 32, 10191322),
        (-52, -28, 52, 9819274),
        (-48, -30, 12, 10022496),
        (-44, 27, 18, 10191322),
        (-48, -24, 12, 10022496),
        (-30, -31, -2, 10197540),
        (-35, 16, 39, 10191322),
        (18, -76, -16, 10022494),
        (-48, 14, 36, 10227106),
        (-48, 32, -16, 10808134),
        (-48, 26, 27, 10402199),
        (38, -84, 0, 10751455),
        (-40, 16, 19, 10191322),
        (23, -53, -19, 10913505),
        (48, -20, 57, 10197540),
        (32, -72, -20, 10022494),
        (-35, 24, 27, 10402199),
        (-42, -26, 12, 10022496),
        (-33, 9, 29, 10191322),
        (2, 4, -2, 10349031),
        (22, 44, 32, 10349031),
        (-60, -36, 16, 10571235),
        (-53, -61, -29, 10913505),
        (-31, -18, 46, 10913505),
        (-40, 16, 19, 10191322),
        (14, -13, -10, 10197540),
        (-52, -28, 12, 10407201),
        (32, 61, -16, 10913505),
        (40, -16, 12, 10022496),
        (-58, -41, -20, 10913505),
        (26, -93, 2, 10227106),
        (-52, -16, 8, 10407201),
        (59, -19, -12, 10197540),
        (-29, -31, -2, 10197540),
        (-56, -43, 19, 10197540),
        (55, -3, -1, 10913505),
        (-49, 28, 4, 10227106),
        (-42, -26, 8, 10022496),
        (5, -39, 22, 10913505),
    )
    peaks = pd.DataFrame(peaks_data, columns=("x", "y", "z", "tfidf"))

    terms_data = (
        (10751455, "emotion", 0.052538),
        (10808134, "emotion", 0.244368),
        (10913505, "emotion", 0.059463),
        (11275483, "emotion", 0.123356),
        (11430815, "emotion", 0.065204),
        (9819274, "auditory", 0.059705),
        (10751455, "auditory", 0.052538),
        (10022494, "auditory", 0.10052),
        (10022496, "auditory", 0.28146),
        (10197540, "auditory", 0.38657),
        (10407201, "auditory", 0.16817),
        (11430815, "auditory", 0.065204),
        (10191322, "language", 0.28298),
        (10751455, "language", 0.052538),
        (10227106, "language", 0.05432),
        (10349031, "language", 0.05329),
        (10402199, "language", 0.18801),
        (10571235, "language", 0.08257),
        (11430815, "language", 0.065204),
    )
    terms = pd.DataFrame(terms_data, columns=("study", "term", "tfidf"))
    nl.add_tuple_set(peaks, name="PeakReported")
    nl.add_tuple_set(terms, name="TermInStudyTFIDF")
    nl.add_tuple_set(terms[["term"]].drop_duplicates(), name="Term")
    return nl


def test_neurosynth_query(nl):
    query = """
    TermInStudy(term, study) :: (1 / (1 + exp(-300 * (tfidf - 0.001)))) :- TermInStudyTFIDF(study, term, tfidf)
    TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
    Activation(x, y, z) :- SelectedStudy(s) & PeakReported(x, y, z, s)
    ActivationGivenTerm(x, y, z, t, PROB) :- Activation(x, y, z) // TermAssociation(t)
    QueryActivation(x, y, z, p) :- ActivationGivenTerm(x, y, z, "emotion", p)
    ans(x, y, z, p) :- QueryActivation(x, y, z, p)
    """
    res = nl.execute_datalog_program(query)
    expected = (
        (-58, -41, -20, 0.2),
        (-53, -61, -29, 0.2),
        (-48, 32, -16, 0.2),
        (-31, -18, 46, 0.2),
        (0, -73, 6, 0.2),
        (5, -39, 22, 0.2),
        (23, -53, -19, 0.2),
        (32, 61, -16, 0.2),
        (38, -84, 0, 0.2),
        (55, -3, -1, 0.2),
    )
    assert res is not None
    assert len(res) == len(expected)
    p = res.as_pandas_dataframe()
    assert set(tuple(x) for x in p[["x", "y", "z"]].values) == set(
        (x, y, z) for x, y, z, _ in expected
    )
