from ....expressions import Symbol, Constant
from ..drs_builder import DRSBuilder
from ..chart_parser import ChartParser
from ..english_grammar import EnglishGrammar, BaseLexicon


_eg = EnglishGrammar(BaseLexicon())
_cp = ChartParser(_eg)


def test_simple_expression():
    b = DRSBuilder(_eg)
    t = _cp.parse("Jones owns Ulysses")[0]
    drs = b.walk(t)

    assert len(drs.referents) == 2
    x = drs.referents[0]
    y = drs.referents[1]
    assert len(drs.expressions) == 3

    assert drs.expressions[0] == Symbol("owns")(x, y)
    assert drs.expressions[1] == Symbol("Ulysses")(y)
    assert drs.expressions[2] == Symbol("Jones")(x)


def test_indefinite_noun_phrase():
    b = DRSBuilder(_eg)
    t = _cp.parse("Jones owns a book")[0]
    drs = b.walk(t)

    assert len(drs.referents) == 2
    x = drs.referents[0]
    y = drs.referents[1]
    assert len(drs.expressions) == 3

    assert drs.expressions[0] == Symbol("owns")(x, y)
    assert drs.expressions[1] == Symbol("book")(y)
    assert drs.expressions[2] == Symbol("Jones")(x)
