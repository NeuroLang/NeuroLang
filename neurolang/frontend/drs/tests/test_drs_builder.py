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


_eg = EnglishGrammar
_cp = ChartParser(_eg, EnglishBaseLexicon())


def test_simple_expression():
    b = DRSBuilder(_eg)
    t = _cp.parse("Jones owns Ulysses")[0]
    drs = b.walk(t)

    assert len(drs.referents) == 0
    assert len(drs.expressions) == 1

    assert drs.expressions[0] == Symbol("owns")(
        Constant("Jones"), Constant("Ulysses")
    )


def test_indefinite_noun_phrase():
    b = DRSBuilder(_eg)
    t = _cp.parse("Jones owns a book")[0]
    drs = b.walk(t)

    assert len(drs.referents) == 1
    x = drs.referents[0]
    assert len(drs.expressions) == 2

    assert drs.expressions[0] == Symbol("owns")(Constant("Jones"), x)
    assert drs.expressions[1] == Symbol("book")(x)


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

    assert Symbol("intersects")(y, x) in set(ant.expressions)
    assert Symbol("region")(x) in set(ant.expressions)
    assert Symbol("region")(y) in set(ant.expressions)

    assert len(con.referents) == 0
    assert Symbol("intersects")(x, y) in set(con.expressions)


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


def test_same_implication():
    # Check that the next translate to the same:
    # - Jones likes a book X if X references Odyssey.
    # - if a book X references Odyssey then Jones likes X.
    pytest.skip("Incomplete")


def test_quoted_predicate():
    b = DRSBuilder(_eg)
    t = _cp.parse("if `intersects(Y, X)` then X intersects Y")[0]
    drs = b.walk(t)

    x = Symbol("X")
    y = Symbol("Y")

    assert len(drs.expressions) == 1
    assert isinstance(drs.expressions[0], Implication)
    ant = drs.expressions[0].antecedent
    con = drs.expressions[0].consequent
    assert len(ant.referents) == 2
    assert set(ant.referents) == {x, y}
    assert Symbol("intersects")(y, x) in set(ant.expressions)
    assert len(con.referents) == 0
    assert Symbol("intersects")(x, y) in set(con.expressions)


def test_conjunction_1():
    b = DRSBuilder(_eg)
    t = _cp.parse("X owns Y and Y references Z")[0]
    drs = b.walk(t)
    exp = DRS2FOL().walk(drs)
    x = Symbol("X")
    y = Symbol("Y")
    z = Symbol("Z")

    assert exp == ExistentialPredicate(
        z,
        ExistentialPredicate(
            y,
            ExistentialPredicate(
                x,
                Conjunction(
                    (Symbol("owns")(x, y), Symbol("references")(y, z),)
                ),
            ),
        ),
    )


def test_conjunction_2():
    b = DRSBuilder(_eg)
    t = _cp.parse(
        "a man X owns a book Y, Y references a book Z, and X likes Z"
    )[0]
    drs = b.walk(t)
    exp = DRS2FOL().walk(drs)
    x = Symbol("X")
    y = Symbol("Y")
    z = Symbol("Z")

    assert exp == ExistentialPredicate(
        z,
        ExistentialPredicate(
            y,
            ExistentialPredicate(
                x,
                Conjunction(
                    (
                        Symbol("owns")(x, y),
                        Symbol("book")(y),
                        Symbol("man")(x),
                        Symbol("references")(y, z),
                        Symbol("book")(z),
                        Symbol("likes")(x, z),
                    )
                ),
            ),
        ),
    )


def test_quoted_string_literal():
    b = DRSBuilder(_eg)
    t = _cp.parse('"Ulysses" references "Odyssey"')[0]
    drs = b.walk(t)
    o = Constant("Odyssey")
    u = Constant("Ulysses")

    assert len(drs.referents) == 0
    assert len(drs.expressions) == 1
    assert Symbol("references")(u, o) in set(drs.expressions)
