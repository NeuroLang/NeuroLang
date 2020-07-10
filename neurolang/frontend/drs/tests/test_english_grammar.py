from ....expressions import Symbol, Constant as _C
from ....expression_walker import PatternWalker, add_match
from ..chart_parser import ChartParser
from ..english_grammar import (
    EnglishGrammar,
    EnglishBaseLexicon,
    S,
    NP,
    PN,
    VP,
    V,
    DET,
    N,
    num,
    gen,
    PRO,
    case,
    VAR,
)
import pytest


_eg = EnglishGrammar
_cp = ChartParser(_eg, EnglishBaseLexicon())


def test_indefinite_noun_phrase():
    t = _cp.parse("every book has an ending")[0]
    assert t == S(num.singular)(
        NP(num.singular, gen.thing, case.nom)(
            DET(num.singular)(_C("every")),
            N(num.singular, gen.thing)(_C("book")),
        ),
        VP(num.singular)(
            V(num.singular)(_C("has")),
            NP(num.singular, gen.thing, case.notnom)(
                DET(num.singular)(_C("an")),
                N(num.singular, gen.thing)(_C("ending")),
            ),
        ),
    )


@pytest.mark.skip(reason="Not priority now")
def test_nested_variable_unification():
    t = _cp.parse("Jones and a man like the book")[0]
    g = t.args[0].functor.args[1]
    e = S(num.plural)(
        NP(num.plural, g, case.nom)(
            NP(num.singular, gen.male, case.nom)(
                PN(num.singular, gen.male)(_C("Jones"))
            ),
            _C("and"),
            NP(num.singular, gen.male, case.nom)(
                DET(num.singular)(_C("a")),
                N(num.singular, gen.male)(_C("man")),
            ),
        ),
        VP(num.plural)(
            V(num.plural)(_C("like")),
            NP(num.singular, gen.thing, case.notnom)(
                DET(num.singular)(_C("the")),
                N(num.singular, gen.thing)(_C("book")),
            ),
        ),
    )
    assert t == e


def test_pronouns():
    t = _cp.parse("she owns it")[0]
    e = S(num.singular)(
        NP(num.singular, gen.female, case.nom)(
            PRO(num.singular, gen.female, case.nom)(_C("she"))
        ),
        VP(num.singular)(
            V(num.singular)(_C("owns")),
            NP(num.singular, gen.thing, case.notnom)(
                PRO(num.singular, gen.thing, case.notnom)(_C("it")),
            ),
        ),
    )
    assert t == e


def test_pronouns_2():
    assert not _cp.recognize("it owns he")


def test_apposition():
    t = _cp.parse("the man X owns a book Y")[0]
    e = S(num.singular)(
        NP(num.singular, gen.male, case.nom)(
            NP(num.singular, gen.male, case.nom)(
                DET(num.singular)(_C("the")),
                N(num.singular, gen.male)(_C("man")),
            ),
            VAR()(_C("X")),
        ),
        VP(num.singular)(
            V(num.singular)(_C("owns")),
            NP(num.singular, gen.thing, case.notnom)(
                NP(num.singular, gen.thing, case.notnom)(
                    DET(num.singular)(_C("a")),
                    N(num.singular, gen.thing)(_C("book")),
                ),
                VAR()(_C("Y")),
            ),
        ),
    )
    assert t == e


def test_variable_reference():
    t = _cp.parse("X owns Y")[0]
    fresh1 = t.args[0].functor.args[1]
    fresh2 = t.args[1].args[1].functor.args[0]
    fresh3 = t.args[1].args[1].functor.args[1]
    e = S(num.singular)(
        NP(num.singular, fresh1, case.nom)(VAR()(_C("X"))),
        VP(num.singular)(
            V(num.singular)(_C("owns")),
            NP(fresh2, fresh3, case.notnom)(VAR()(_C("Y"))),
        ),
    )
    assert t == e


def test_implication_if():
    t = _cp.parse("X intersects Y if Y intersects X")[0]
    fresh = (
        t.functor.args[0],
        t.args[0].args[0].functor.args[1],
        t.args[0].args[1].args[1].functor.args[0],
        t.args[0].args[1].args[1].functor.args[1],
        t.args[2].args[0].functor.args[1],
        t.args[2].args[1].args[1].functor.args[0],
        t.args[2].args[1].args[1].functor.args[1],
    )
    e = S(fresh[0])(
        S(num.singular)(
            NP(num.singular, fresh[1], case.nom)(VAR()(_C("X"))),
            VP(num.singular)(
                V(num.singular)(_C("intersects")),
                NP(fresh[2], fresh[3], case.notnom)(VAR()(_C("Y"))),
            ),
        ),
        _C("if"),
        S(num.singular)(
            NP(num.singular, fresh[4], case.nom)(VAR()(_C("Y"))),
            VP(num.singular)(
                V(num.singular)(_C("intersects")),
                NP(fresh[5], fresh[6], case.notnom)(VAR()(_C("X"))),
            ),
        ),
    )
    assert t == e


def test_implication_if_then():
    t = _cp.parse("if Y intersects X then X intersects Y")[0]
    fresh = (
        t.functor.args[0],
        t.args[1].args[0].functor.args[1],
        t.args[1].args[1].args[1].functor.args[0],
        t.args[1].args[1].args[1].functor.args[1],
        t.args[3].args[0].functor.args[1],
        t.args[3].args[1].args[1].functor.args[0],
        t.args[3].args[1].args[1].functor.args[1],
    )
    e = S(fresh[0])(
        _C("if"),
        S(num.singular)(
            NP(num.singular, fresh[1], case.nom)(VAR()(_C("Y"))),
            VP(num.singular)(
                V(num.singular)(_C("intersects")),
                NP(fresh[2], fresh[3], case.notnom)(VAR()(_C("X"))),
            ),
        ),
        _C("then"),
        S(num.singular)(
            NP(num.singular, fresh[4], case.nom)(VAR()(_C("X"))),
            VP(num.singular)(
                V(num.singular)(_C("intersects")),
                NP(fresh[5], fresh[6], case.notnom)(VAR()(_C("Y"))),
            ),
        ),
    )
    assert t == e


def test_sentence_conjunction():
    _cp.parse("Jones owns Ulysses and Smith likes Ulysses")
    _cp.parse("Jones owns Ulysses, Smith owns Ulysses, and Smith owns Ulysses")
    _cp.parse(
        "Jones and Smith own Ulysses, "
        + "Jones likes Ulysses and Odyssey, "
        + "and Smith likes every book"
    )
    _cp.parse(
        """
        if a region X provides a function F and X intersects
        a region Y then Y affects F
        """
    )
    _cp.parse(
        """
        if a region provides a function and that region intersects
        a region Y then Y affects that function
        """
    )
    _cp.parse(
        """
        if a region provides a function, that region intersects
        a region Y, and Ulysses references Y then if Jones owns
        Ulysses then the region Y affects that function
        """
    )
