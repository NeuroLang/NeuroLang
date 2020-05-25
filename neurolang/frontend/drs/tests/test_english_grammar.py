from ....expressions import Symbol, Constant as _C
from ....expression_walker import PatternWalker, add_match
from ..chart_parser import ChartParser
from ..english_grammar import EnglishGrammar, EnglishBaseLexicon, S, NP, PN, VP, V, DET, N, num, gen, PRO, case


_eg = EnglishGrammar(EnglishBaseLexicon())
_cp = ChartParser(_eg)


def test_indefinite_noun_phrase():
    t = _cp.parse("every book has an ending")[0]
    assert t == S(num.singular)(
        NP(num.singular, gen.thing, case.nom)(
            DET(num.singular)(_C("every")),
            N(num.singular, gen.thing)(_C("book"))
        ),
        VP(num.singular)(
            V(num.singular)(_C("has")),
            NP(num.singular, gen.thing, case.notnom)(
                DET(num.singular)(_C("an")),
                N(num.singular, gen.thing)(_C("ending"))
            )
        )
    )


def test_indefinite_noun_phrase_2():
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
                N(num.singular, gen.male)(_C("man"))
            ),
        ),
        VP(num.plural)(
            V(num.plural)(_C("like")),
            NP(num.singular, gen.thing, case.notnom)(
                DET(num.singular)(_C("the")),
                N(num.singular, gen.thing)(_C("book"))
            )
        )
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
            )
        )
    )
    assert t == e


def test_pronouns_2():
    assert not _cp.recognize("it owns he")
