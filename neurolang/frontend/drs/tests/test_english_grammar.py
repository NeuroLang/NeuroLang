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
    assert _cp.recognize("every book has an ending")


@pytest.mark.skip(reason="Not priority now")
def test_nested_variable_unification():
    assert _cp.recognize("Jones and a man like the book")


def test_pronouns():
    assert _cp.recognize("she owns it")


def test_pronouns_2():
    assert not _cp.recognize("it owns he")


def test_apposition():
    assert _cp.recognize("the man X owns a book Y")


def test_variable_reference():
    assert _cp.recognize("X owns Y")


def test_implication_if():
    assert _cp.recognize("X intersects Y if Y intersects X")


def test_implication_if_then():
    assert _cp.recognize("if Y intersects X then X intersects Y")


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


def test_ands_in_implications():
    _cp.parse(
        """
        if Y contains Z and X contains Z
        then X intersects Y and Y intersects X
        """
    )


def test_disallow_implication_in_ands():
    assert not _cp.recognize(
        """
        if Y intersects Z then Z intersects Y and
        if X contains Y then Y intersects X
        """
    )


def test_verb_negation():
    _cp.parse(
        """
        Smith does not own Ulysses
        """
    )


def test_sentence_negation():
    _cp.parse(
        """
        is not the case that Smith owns Ulysses
        """
    )


def test_plural_verb_negation():
    _cp.parse(
        """
        Jones and Smith do not like Ulysses
        """
    )
