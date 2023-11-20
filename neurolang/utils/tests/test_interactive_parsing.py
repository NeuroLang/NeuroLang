import pytest

from ...expressions import Symbol
from ...exceptions import UnexpectedTokenError
from ...frontend.datalog.standard_syntax import parser


def test_interactive_empty_input():
    res = parser('', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'-', '~', '¬'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_facts():
    res = parser('A', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': {'/', '+', '*', '-', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {':=', '::'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(', interactive=True)
    expected = {'Signs': {'...', '@', ')', '('}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A()', interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'¬', '-', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'←', '.', ':=', ':-', '::'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(3', interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': {'/', '+', '*', '-', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(3)', interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'¬', '-', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A("x"', interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': {'/', '+', '*', '-', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A("x")', interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'¬', '-', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A("x",', interactive=True)
    expected = {'Signs': {'@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>'}}
    assert res == expected

    res = parser("A('x', 3", interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser("A('x', 3,", interactive=True)
    expected = {'Signs': {'@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>'}}
    assert res == expected

    res = parser("A('x', 3)", interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'¬', '-', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    input = '''
    `http://uri#test-fact`("x")
    A("x", 3
    '''
    res = parser(input, interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected


def test_interactive_rules():
    res = parser('A(x', interactive=True)
    expected = {'Signs': {')', ',', '('}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '+', '/', '*', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x)', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'::', ':-', ':=', '←'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('A(x):-B', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(', interactive=True)
    expected = {'Signs': {')', '...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B()', interactive=True)
    expected = {'Signs': {',', '∧', '(', '∃', '&', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'//', '-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x', interactive=True)
    expected = {'Signs': {')', ',', '('}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '+', '/', '*', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(x,', interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x, y', interactive=True)
    expected = {'Signs': {')', ',', '('}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '+', '/', '*', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(x, y)', interactive=True)
    expected = {'Signs': {',', '∧', '(', '∃', '&', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'//', '-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x, y),', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('A(x):-B(x, y), C(3, z)', interactive=True)
    expected = {'Signs': {',', '∧', '(', '∃', '&', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-~', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('A(x):-~B(x)', interactive=True)
    expected = {'Signs': {',', '∧', '(', '∃', '&', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'//', '-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-~B(x),', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('A(x):-B(x, ...)', interactive=True)
    expected = {'Signs': {',', '∧', '(', '∃', '&', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'//', '-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x, y), C(3, z), (', interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x, y), C(3, z), (z', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(),
                'Operators': {'-', '==', '<=', '+', '!=', '>=', '>', '<', '/', '*', '**'}, 'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(x, y), C(3, z), (z ==', interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x + 5 *', interactive=True)
    expected = {'Signs': {'(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x / 2', interactive=True)
    expected = {'Signs': {')', ','}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '+', '/', '*', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(f(x', interactive=True)
    expected = {'Signs': {')', ',', '('}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '+', '/', '*', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(x + (-5),', interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x):-B(x + (-5), "a"', interactive=True)
    expected = {'Signs': {')', ','}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '+', '/', '*', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x):-B(x - 5 * 2, @', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_aggregation():
    res = parser('A(x, f(', interactive=True)
    expected = {'Signs': {'...', '@', ')', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('A(x, f(y', interactive=True)
    expected = {'Signs': {',', ')', '('}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '*', '/', '**', '+'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x, f(y)', interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '*', '/', '**', '+'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('A(x, f(y))', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'←', ':=', ':-', '::'}, 'Python string': set(), 'Strings': set()}
    assert res == expected


def test_interactive_uri():
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(
        name='http://sig.biostr.washington.edu/fma3.0#regional_part_of'
    )

    res = parser(f'`{str(label.name)}`', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': {'/', '+', '*', '-', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'::', ':='}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser(f'`{str(label.name)}`(x)', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'::', '←', ':=', ':-'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser(f'`{str(label.name)}`(x):-', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'False', 'True', '⊤', '⊥'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser(f'`{str(label.name)}`(x):-`{str(regional_part.name)}`', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser(f'`{str(label.name)}`(x):-`{str(regional_part.name)}`(x, y)', interactive=True)
    expected = {'Signs': {'(', '&', ',', '@', '∧', '∃'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'//', '¬', '~', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'ans', 'EXISTS', 'exists'}, 'Boleans': {'False', 'True', '⊤', '⊥'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected


def test_interactive_probabilistic_fact():
    res = parser('p::', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('p::A', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('p::A(', interactive=True)
    expected = {'Signs': {')', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>'}}
    assert res == expected

    res = parser('p::A(3', interactive=True)
    expected = {'Signs': {')', ','}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('p::A(3)', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '-', '¬'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('0.8::A("a b",', interactive=True)
    expected = {'Signs': {'@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>'}}
    assert res == expected

    res = parser('0.8::A("a b", 3', interactive=True)
    expected = {'Signs': {')', ','}, 'Numbers': set(), 'Text': set(), 'Operators': set(),
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('0.8::A("a b", 3)', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '-', '¬'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('B(x) :: exp(-d / 5.0)', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': {'-', '/', '*', '+', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {':-', '←'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('B(x) :: exp(-d / 5.0) :-', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': set(), 'Text': set(), 'Operators': {'~', '¬'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('B(x) :: exp(-d / 5.0) :- A(x, d)', interactive=True)
    expected = {'Signs': {'@', '&', '(', '∃', '∧', ','}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'//', '~', '-', '¬'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('B(x) :: exp(-d / 5.0) :- A(x, d) &', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': set(), 'Text': set(), 'Operators': {'~', '¬'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('B(x) :: exp(-d / 5.0) :- A(x, d) & (d < 0.8)', interactive=True)
    expected = {'Signs': {'@', '&', '(', '∃', '∧', ','}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '-', '¬'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_condition():
    res = parser('C(x) :- A(x) //', interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'~', '¬'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('C(x) :- A(x) // B(x)', interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '¬', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('C(x) :- A(x) // (A(x),', interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'~', '¬'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('C(x) :- A(x) // (A(x), B(x))', interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '¬', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('C(x) :- (A(x), B(x)) // B(x)', interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '¬', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected


def test_interactive_existential():
    res = parser('C(x) :- B(x), exists', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('C(x) :- B(x), exists(', interactive=True)
    expected = {'Signs': {'(', '...', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), exists(s1', interactive=True)
    expected = {'Signs': {'(', ',', ';'}, 'Numbers': set(), 'Text': set(), 'Operators': {'**', '+', '*', '/', '-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': {'st'}, 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('C(x) :- B(x), exists(s1;', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), exists(s1; A(s1))', interactive=True)
    expected = {'Signs': {',', '@', '&', '∧', '∃', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'¬', '~', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), ∃', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('C(x) :- B(x), ∃(', interactive=True)
    expected = {'Signs': {'(', '...', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), ∃(s1', interactive=True)
    expected = {'Signs': {'(', ',', ';'}, 'Numbers': set(), 'Text': set(), 'Operators': {'**', '+', '*', '/', '-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': {'st'}, 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('C(x) :- B(x), ∃(s1 st', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), ∃(s1 st A(s1))', interactive=True)
    expected = {'Signs': {',', '@', '&', '∧', '∃', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'¬', '~', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), exists(s1, s2; A(s1),', interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('C(x) :- B(x), exists(s1, s2; A(s1), A(s2))', interactive=True)
    expected = {'Signs': {',', '@', '&', '∧', '∃', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'¬', '~', '-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    with pytest.raises(UnexpectedTokenError):
        res = parser('C(x) :- B(x), exists(s1; )', interactive=True)


def test_interactive_query():
    res = parser('ans(', interactive=True)
    expected = {'Signs': {'...', ')', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'-'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('ans(x)', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {':-', '←'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('ans(x) :-', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': set(), 'Text': set(), 'Operators': {'~', '¬'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'False', 'True', '⊤'}, 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('ans(x) :- B(x, y), C(3, y)', interactive=True)
    expected = {'Signs': {'∧', '@', '∃', ',', '(', '&'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '-', '¬'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊥', 'False', 'True', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_prob_implicit():
    res = parser('B(x, PROB', interactive=True)
    expected = {'Signs': {'(', ')', ','}, 'Numbers': set(), 'Text': set(), 'Operators': {'+', '-', '*', '**', '/'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('B(x, PROB,', interactive=True)
    expected = {'Signs': {'(', '@', '...'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('B(x, PROB, y)', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'←', ':-', ':=', '::'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('B(x, PROB, y) :- C(x, y)', interactive=True)
    expected = {'Signs': {'@', '(', '∃', '&', ',', '∧'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'-', '~', '¬', '//'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'True', '⊥', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_prob_explicit():
    res = parser('B(x, PROB', interactive=True)
    expected = {'Signs': {',', '(', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': {'+', '*', '-', '/', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('B(x, PROB(', interactive=True)
    expected = {'Signs': {'(', '...', ')', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('B(x, PROB(x,', interactive=True)
    expected = {'Signs': {'(', '...', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('B(x, PROB(x, y)', interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': {'+', '*', '-', '/', '**'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('B(x, PROB(x, y),', interactive=True)
    expected = {'Signs': {'(', '...', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'commands': set(), 'functions': set(), 'base symbols': set(), 'query symbols': set()}
    assert res == expected

    res = parser('B(x, PROB(x, y), y)', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'::', ':=', ':-', '←'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('B(x, PROB(x, y), y) :- C(x, y)', interactive=True)
    expected = {'Signs': {'&', '∃', '@', ',', '∧', '('}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '-', '¬', '//'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_lambda_definition():
    res = parser('c', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': {'**', '+', '-', '/', '*'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {':=', '::'}, 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('c :=', interactive=True)
    expected = {'Signs': {'@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('c := lambda', interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('c := lambda x', interactive=True)
    expected = {'Signs': {',', '(', ':'}, 'Numbers': set(), 'Text': set(), 'Operators': {'**', '+', '-', '/', '*'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('c := lambda x:', interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('c := lambda x: x + 1', interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'**', '~', '+', '¬', '-', '/', '*'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'ans', 'exists'}, 'Boleans': {'⊤', 'False', '⊥', 'True'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_lambda_definition_statement():
    res = parser('c(x, y) := x +', interactive=True)
    expected = {'Signs': {'@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('c(x, y) := x + y', interactive=True)
    expected = {'Signs': {'∃', '@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Text': set(),
                'Operators': {'**', '+', '¬', '-', '~', '/', '*'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'False', '⊥', '⊤', 'True'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_lambda_application():
    res = parser('c := (lambda', interactive=True)
    expected = {'Signs': {'...', '@', '('}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('c := (lambda x', interactive=True)
    expected = {'Signs': {',', '(', ':'}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '*', '**', '/', '+'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('c := (lambda x:', interactive=True)
    expected = {'Signs': {'...', '@', '('}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('c := (lambda x: x', interactive=True)
    expected = {'Signs': {'(', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': {'-', '*', '**', '/', '+'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('c := (lambda x: x + 1)', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('c := (lambda x: x + 1)(2)', interactive=True)
    expected = {'Signs': {'∃', '@', '('}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'~', '-', '*', '**', '/', '¬', '+'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'True', 'False', '⊤', '⊥'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_command_syntax():
    res = parser('.', interactive=True)
    expected = {'Signs': set(), 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<command identifier>'}, 'commands': set()}
    assert res == expected

    res = parser('.load_csv', interactive=True)
    expected = {'Signs': {'('}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('.load_csv(', interactive=True)
    expected = {'Signs': {')', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>', '<quoted string>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('.load_csv()', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'True', 'False', '⊥', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('.load_csv(A,', interactive=True)
    expected = {'Signs': {'(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>', '<quoted string>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('.load_csv(A, "http://myweb/file.csv", B)', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(),
                'Operators': {'-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'True', 'False', '⊥', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('.load_csv(sep', interactive=True)
    expected = {'Signs': {',', ')', '(', '='}, 'Numbers': set(), 'Text': set(), 'Operators': {'*', '**', '/', '+', '-'},
                'Cmd_identifier': set(), 'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('.load_csv(sep=', interactive=True)
    expected = {'Signs': {'(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-'},
                'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>', '<quoted string>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected

    res = parser('.load_csv(sep=","', interactive=True)
    expected = {'Signs': {',', ')'}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': set(), 'Python string': set(), 'Strings': set()}
    assert res == expected

    res = parser('.load_csv(sep=",")', interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Text': set(), 'Operators': {'-', '¬', '~'}, 'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(), 'Reserved words': {'exists', 'EXISTS', 'ans'},
                'Boleans': {'True', 'False', '⊥', '⊤'}, 'Expression symbols': {'.'}, 'Python string': set(), 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}, 'functions': set(), 'base symbols': set(), 'query symbols': set(), 'commands': set()}
    assert res == expected


def test_interactive_constraint():
    res = parser("(x == y)", interactive=True)
    expected = {'Signs': {',', '&', '∧'}, 'Numbers': set(), 'Text': set(), 'Operators': set(), 'Cmd_identifier': set(),
                'Functions': set(), 'Identifier_regexp': set(), 'Reserved words': set(), 'Boleans': set(), 'Expression symbols': {'→', '-:'}, 'Python string': set(), 'Strings': set()}
    assert res == expected
