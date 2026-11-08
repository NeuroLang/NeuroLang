"""The ``%%`` documentation blocks of the predicates of a program."""

import pytest

from nl_core.datalog import Fact, Implication
from nl_expr.exceptions import ParserError
from nl_expr.expressions import Command, Query, Statement
from nl_core.datalog.pretty_printer import DatalogPrettyPrinter
from ..standard_syntax import DOCUMENT_COMMAND, parser


def docs(program):
    """Return ``(name, description, arguments)`` of each doc command."""
    return [
        (f.args[0].name, f.args[1].value, f.args[2].value)
        for f in parser(program).formulas
        if isinstance(f, Command) and f.name == DOCUMENT_COMMAND
    ]


def test_signature_and_description():
    program = (
        "%% uncinate(x: streamline, h: hemisphere)\n"
        "%% The uncinate fasciculus: streamlines joining the\n"
        "%% orbitofrontal cortex to the anterior temporal lobe.\n"
        "uncinate(x, h) :- q(x, h)\n"
    )
    formulas = parser(program).formulas
    assert len(formulas) == 2
    assert isinstance(formulas[0], Command)
    assert isinstance(formulas[1], Implication)
    assert docs(program) == [(
        "uncinate",
        "The uncinate fasciculus: streamlines joining the\n"
        "orbitofrontal cortex to the anterior temporal lobe.",
        (("x", "streamline"), ("h", "hemisphere")),
    )]


def test_description_without_signature():
    assert docs("%% The numbers.\n%% Small ones.\np(x) :- q(x)") == [
        ("p", "The numbers.\nSmall ones.", ())
    ]


def test_signature_without_sorts_or_description():
    assert docs("%% p(x, y: number)\np(x, y) :- q(x, y)") == [
        ("p", "", (("x", ""), ("y", "number")))
    ]


def test_first_line_naming_another_predicate_is_description():
    assert docs("%% q(x) holds the values.\np(x) :- q(x)") == [
        ("p", "q(x) holds the values.", ())
    ]


def test_signature_of_the_wrong_arity_is_an_error():
    with pytest.raises(ParserError, match="gives 1 arguments") as e:
        parser("a(1)\n%% p(x)\n%% desc\np(x, y) :- q(x, y)")
    assert e.value.line == 2


def test_badly_written_argument_is_an_error():
    with pytest.raises(ParserError, match="'x y'"):
        parser("%% p(x y)\np(x) :- q(x)")


def test_plain_comments_are_not_documentation():
    program = "% p(x: number)\n% The numbers.\np(x) :- q(x)"
    assert docs(program) == []
    assert len(parser(program).formulas) == 1


def test_block_followed_by_a_blank_line_or_comment_is_a_comment():
    assert docs("%% A header.\n\np(x) :- q(x)") == []
    assert docs("%% A header.\n% note\np(x) :- q(x)") == []
    assert docs("p(1)\n%% Trailing.") == []


def test_doc_inside_a_rule_is_a_comment():
    program = "p(x) :-\n%% not a doc\n  q(x)\nr(1)"
    assert docs(program) == []
    assert len(parser(program).formulas) == 2


def test_doc_of_facts_and_probabilistic_facts():
    program = (
        "%% Facts of a.\n"
        "a(1)\n"
        "a(2)\n"
        "%% c(x: number)\n"
        "%% A probabilistic fact.\n"
        "0.5 :: c(1)\n"
        "%% A choice.\n"
        "0.5 ^ d(1)\n"
        "%% A constant.\n"
        "k := 3\n"
    )
    assert docs(program) == [
        ("a", "Facts of a.", ()),
        ("c", "A probabilistic fact.", (("x", "number"),)),
        ("d", "A choice.", ()),
        ("k", "A constant.", ()),
    ]
    formulas = parser(program).formulas
    assert isinstance(formulas[1], Fact)
    assert isinstance(formulas[-1], Statement)


def test_doc_of_a_group_of_rules_and_of_another_predicate():
    program = (
        "%% The p.\n"
        "p(x) :- q(x)\n"
        "p(x) :- r(x)\n"
        "%% The s.\n"
        "s(x) :- p(x)\n"
        "%% More on p.\n"
        "p(x) :- t(x)\n"
    )
    assert docs(program) == [
        ("p", "The p.", ()), ("s", "The s.", ()), ("p", "More on p.", ())
    ]


def test_doc_of_a_query_or_command_is_ignored():
    program = '%% query\nans(x) :- p(x)\n%% load\n.load_csv(r, "f.csv")'
    formulas = parser(program).formulas
    assert docs(program) == []
    assert isinstance(formulas[0], Query)


def test_percent_in_a_doc_and_trailing_dot():
    assert docs('%% 50% of the values.\np("a%b").') == [
        ("p", "50% of the values.", ())
    ]


def test_program_without_doc_unchanged():
    program = "p(x) :- q(x)\nq(1)"
    assert len(parser(program).formulas) == 2


def test_pretty_printer_writes_the_doc_block_back():
    program = (
        "%% p(x: number)\n"
        "%% The numbers.\n"
        "%% Small ones.\n"
        "p(x) :- q(x)\n"
    )
    command = parser(program).formulas[0]
    assert DatalogPrettyPrinter().walk(command) == (
        "%% p(x: number)\n%% The numbers.\n%% Small ones."
    )
    assert docs(DatalogPrettyPrinter().walk(command) + "\np(x) :- q(x)") == (
        docs(program)
    )
