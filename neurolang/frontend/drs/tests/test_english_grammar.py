from ....expressions import Symbol, Constant as _C
from ....expression_walker import PatternWalker, add_match
from ..chart_parser import ChartParser
from ..english_grammar import EnglishGrammar, BaseLexicon, S, NP, PN, VP, V, DET, N, num, gen


_eg = EnglishGrammar(BaseLexicon())
_cp = ChartParser(_eg)


def test_indefinite_verb():
    t = _cp.parse("every book has an ending")[0]
    assert t == S(num.singular)(
        NP(num.singular, gen.thing)(
            DET(num.singular)(_C("every")),
            N(num.singular, gen.thing)(_C("book"))
        ),
        VP(num.singular)(
            V(num.singular)(_C("has")),
            NP(num.singular, gen.thing)(
                DET(num.singular)(_C("an")),
                N(num.singular, gen.thing)(_C("ending"))
            )
        )
    )


def test_indefinite_verb_2():
    t = _cp.parse("Jones and a man like the book")[0]
    g = t.args[0].functor.args[1]
    e = S(num.plural)(
        NP(num.plural, g)(
            NP(num.singular, gen.male)(
                PN(num.singular, gen.male)(_C("Jones"))
            ),
            _C("and"),
            NP(num.singular, gen.male)(
                DET(num.singular)(_C("a")),
                N(num.singular, gen.male)(_C("man"))
            ),
        ),
        VP(num.plural)(
            V(num.plural)(_C("like")),
            NP(num.singular, gen.thing)(
                DET(num.singular)(_C("the")),
                N(num.singular, gen.thing)(_C("book"))
            )
        )
    )
    assert t == e
