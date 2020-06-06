from ....expressions import Symbol, Constant
from ....logic import (
    Implication,
    Conjunction,
    ExistentialPredicate,
    UniversalPredicate,
)
from ..drs_builder import DRSBuilder, DRS2FOL
from ..chart_parser import ChartParser
from ..english_grammar import EnglishGrammar, EnglishBaseLexicon
import pytest


_eg = EnglishGrammar(EnglishBaseLexicon())
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


def test_var_noun_phrases():
    b = DRSBuilder(_eg)
    t = _cp.parse("X intersects Y")[0]
    drs = b.walk(t)

    assert len(drs.referents) == 2
    assert len(drs.expressions) == 1
    assert drs.expressions[0] == Symbol("intersects")(Symbol("X"), Symbol("Y"))


def test_apposition_variable_introduction():
    b = DRSBuilder(_eg)
    t = _cp.parse("a region X intersects a region Y")[0]
    drs = b.walk(t)
    x = Symbol("X")
    y = Symbol("Y")
    assert len(drs.referents) == 2
    assert x == drs.referents[0]
    assert y == drs.referents[1]
    assert len(drs.expressions) == 3

    assert drs.expressions[0] == Symbol("intersects")(x, y)
    assert drs.expressions[1] == Symbol("region")(y)
    assert drs.expressions[2] == Symbol("region")(x)


def test_conditional():
    b = DRSBuilder(_eg)
    t = _cp.parse("if a region Y intersects a region X then X intersects Y")[0]
    drs = b.walk(t)
    x = Symbol("X")
    y = Symbol("Y")
    assert len(drs.expressions) == 1
    assert isinstance(drs.expressions[0], Implication)
    ant = drs.expressions[0].antecedent
    con = drs.expressions[0].consequent
    assert len(ant.referents) == 2
    assert set(ant.referents) == {x, y}
    assert set(ant.expressions) == {
        Symbol("intersects")(y, x),
        Symbol("region")(x),
        Symbol("region")(y),
    }

    assert len(con.referents) == 2
    assert set(con.referents) == {x, y}
    assert set(con.expressions) == {
        Symbol("intersects")(x, y),
    }


def test_translation_1():
    b = DRSBuilder(_eg)
    t = _cp.parse("X intersects Y")[0]
    drs = b.walk(t)
    exp = DRS2FOL().walk(drs)
    x = Symbol("X")
    y = Symbol("Y")

    assert exp == ExistentialPredicate(
        y, ExistentialPredicate(x, Symbol("intersects")(x, y))
    )


def test_translation_2():
    b = DRSBuilder(_eg)
    t = _cp.parse("if a region Y intersects a region X then X intersects Y")[0]
    drs = b.walk(t)
    exp = DRS2FOL().walk(drs)
    x = Symbol("X")
    y = Symbol("Y")

    assert exp == UniversalPredicate(
        x,
        UniversalPredicate(
            y,
            Implication(
                Symbol("intersects")(x, y),
                Conjunction(
                    (
                        Symbol("intersects")(y, x),
                        Symbol("region")(x),
                        Symbol("region")(y),
                    )
                ),
            ),
        ),
    )
