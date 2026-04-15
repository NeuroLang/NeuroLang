import pytest

from ..interactive_parsing import LarkCompleter
from ...expressions import Symbol
from ...exceptions import UnexpectedTokenError
from ...frontend.datalog.standard_syntax import COMPILED_GRAMMAR


def test_interactive_empty_input():
    completer = LarkCompleter(COMPILED_GRAMMAR)
    res = completer.complete('').token_options
    expected = {'Signs': {'values': {'@', '∃', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'False', '⊤', 'True', '⊥'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected


def test_interactive_facts():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('A').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'/', '+', '*', '**', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {':=', '::'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(').token_options
    expected = {'Signs': {'values': {'@', '(', '...', ')'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A()').token_options
    expected = {'Signs': {'values': {'(', '∃', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '-', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists', 'ans'}}, 'Boleans': {'values': {'⊥', 'True', 'False', '⊤'}}, 'Expression symbols': {'values': {'←', '::', '.', ':=', ':-'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(3').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'+', '**', '-', '/', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(3)').token_options
    expected = {'Signs': {'values': {'(', '∃', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'~', '¬', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'⊥', 'False', 'True', '⊤'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A("x"').token_options
    expected = {'Signs': {'values': {')', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'*', '+', '**', '-', '/'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A("x")').token_options
    expected = {'Signs': {'values': {'@', '∃', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'ans', 'EXISTS'}}, 'Boleans': {'values': {'False', '⊤', 'True', '⊥'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A("x",').token_options
    expected = {'Signs': {'values': {'@'}}, 'Numbers': {'values': {'<integer>', '<float>'}}, 'Text': {'values': set()},
                'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>'}}}
    assert res == expected

    res = completer.complete("A('x', 3").token_options
    expected = {'Signs': {'values': {')', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete("A('x', 3,").token_options
    expected = {'Signs': {'values': {'@'}}, 'Numbers': {'values': {'<integer>', '<float>'}}, 'Text': {'values': set()},
                'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>'}}}
    assert res == expected

    res = completer.complete("A('x', 3)").token_options
    expected = {'Signs': {'values': {'(', '∃', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'False', '⊥', '⊤', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('''
    `http://uri#test-fact`("x")
    A("x", 3
    ''').token_options
    expected = {'Signs': {'values': {')', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected


def test_interactive_rules():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('A(x').token_options
    expected = {'Signs': {'values': {')', ',', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'*', '/', '**', '+', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x)').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {':-', '::', '←', ':='}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-').token_options
    expected = {'Signs': {'values': {'(', '∃', '@'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists'}}, 'Boleans': {'values': {'True', 'False', '⊤', '⊥'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(').token_options
    expected = {'Signs': {'values': {')', '(', '@', '...'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B()').token_options
    expected = {'Signs': {'values': {'∧', '∃', '(', '&', ',', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'//', '~', '¬', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'EXISTS', 'exists'}}, 'Boleans': {'values': {'⊤', 'False', '⊥', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x').token_options
    expected = {'Signs': {'values': {')', '(', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'*', '-', '/', '+', '**'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x,').token_options
    expected = {'Signs': {'values': {'(', '...', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y').token_options
    expected = {'Signs': {'values': {'(', ',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '*', '/', '+', '**'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y)').token_options
    expected = {'Signs': {'values': {',', '(', '@', '∧', '∃', '&'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'//', '¬', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'True', '⊤', 'False', '⊥'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y),').token_options
    expected = {'Signs': {'values': {'(', '@', '∃'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊥', 'True', '⊤', 'False'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y), C(3, z)').token_options
    expected = {'Signs': {'values': {'@', ',', '∧', '&', '∃', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'~', '¬', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'⊤', 'False', '⊥', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-~').token_options
    expected = {'Signs': {'values': {'∃', '(', '@'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'True', 'False', '⊥', '⊤'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-~B(x)').token_options
    expected = {'Signs': {'values': {'&', ',', '@', '(', '∧', '∃'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '~', '-', '//'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'True', '⊤', '⊥', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-~B(x),').token_options
    expected = {'Signs': {'values': {'@', '(', '∃'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊤', 'False', '⊥', 'True'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, ...)').token_options
    expected = {'Signs': {'values': {',', '∧', '@', '(', '∃', '&'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'~', '//', '-', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊥', '⊤', 'True', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y), C(3, z), (').token_options
    expected = {'Signs': {'values': {'@', '(', '...'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y), C(3, z), (z').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'<=', '==', '>=', '**', '>', '+', '/', '<', '-', '*', '!='}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x, y), C(3, z), (z ==').token_options
    expected = {'Signs': {'values': {'...', '@', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<identifier_regexp>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x + 5 *').token_options
    expected = {'Signs': {'values': {'(', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x / 2').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'/', '**', '*', '-', '+'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(f(x').token_options
    expected = {'Signs': {'values': {',', '(', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '*', '/', '**', '+'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x + (-5),').token_options
    expected = {'Signs': {'values': {'...', '@', '('}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x + (-5), "a"').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'+', '/', '*', '-', '**'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x):-B(x - 5 * 2, @').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected


def test_interactive_aggregation():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('A(x, f(').token_options
    expected = {'Signs': {'values': {')', '@', '(', '...'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x, f(y').token_options
    expected = {'Signs': {'values': {')', ',', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'**', '-', '/', '+', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x, f(y)').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'/', '+', '-', '**', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('A(x, f(y))').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {'::', ':=', ':-', '←'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected


def test_interactive_uri():
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(
        name='http://sig.biostr.washington.edu/fma3.0#regional_part_of'
    )
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete(f'`{str(label.name)}`').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'+', '**', '*', '/', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {':=', '::'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete(f'`{str(label.name)}`(x)').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {':-', '←', '::', ':='}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete(f'`{str(label.name)}`(x):-').token_options
    expected = {'Signs': {'values': {'@', '(', '∃'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'False', '⊥', 'True', '⊤'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete(f'`{str(label.name)}`(x):-`{str(regional_part.name)}`').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete(f'`{str(label.name)}`(x):-`{str(regional_part.name)}`(x, y)').token_options
    expected = {'Signs': {'values': {',', '∃', '&', '(', '∧', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '//', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊤', 'True', 'False', '⊥'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected


def test_interactive_probabilistic_fact():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('p::').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('p::A').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('p::A(').token_options
    expected = {'Signs': {'values': {'@', ')'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>'}}}
    assert res == expected

    res = completer.complete('p::A(3').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('p::A(3)').token_options
    expected = {'Signs': {'values': {'@', '∃', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'EXISTS', 'exists'}}, 'Boleans': {'values': {'⊥', 'True', '⊤', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('0.8::A("a b",').token_options
    expected = {'Signs': {'values': {'@'}}, 'Numbers': {'values': {'<integer>', '<float>'}}, 'Text': {'values': set()},
                'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>'}}}
    assert res == expected

    res = completer.complete('0.8::A("a b", 3').token_options
    expected = {'Signs': {'values': {')', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('0.8::A("a b", 3)').token_options
    expected = {'Signs': {'values': {'∃', '@', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'False', '⊥', '⊤', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x) :: exp(-d / 5.0)').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'**', '-', '/', '+', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {':-', '←'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x) :: exp(-d / 5.0) :-').token_options
    expected = {'Signs': {'values': {'@', '∃', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists'}}, 'Boleans': {'values': {'⊥', 'False', '⊤', 'True'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x) :: exp(-d / 5.0) :- A(x, d)').token_options
    expected = {'Signs': {'values': {'∧', '&', ',', '∃', '(', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '~', '¬', '//'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'False', '⊥', '⊤', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete(
        'B(x) :: exp(-d / 5.0) :- A(x, d) &').token_options
    expected = {'Signs': {'values': {'∃', '(', '@'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists'}}, 'Boleans': {'values': {'⊥', '⊤', 'False', 'True'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete(
        'B(x) :: exp(-d / 5.0) :- A(x, d) & (d < 0.8)').token_options
    expected = {'Signs': {'values': {'∧', '&', ',', '(', '∃', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'EXISTS', 'exists'}}, 'Boleans': {'values': {'True', '⊤', '⊥', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected


def test_interactive_condition():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('C(x) :- A(x) //').token_options
    expected = {'Signs': {'values': {'@', '(', '∃'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊤', 'True', 'False', '⊥'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- A(x) // B(x)').token_options
    expected = {'Signs': {'values': {'@', '∃', '('}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '-', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'ans', 'EXISTS'}}, 'Boleans': {'values': {'False', '⊥', 'True', '⊤'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<cmd_identifier>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- A(x) // (A(x),').token_options
    expected = {'Signs': {'values': {'∃', '@', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists'}}, 'Boleans': {'values': {'⊥', 'False', 'True', '⊤'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- A(x) // (A(x), B(x))').token_options
    expected = {'Signs': {'values': {'(', '@', '∃'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'~', '-', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'ans', 'EXISTS'}}, 'Boleans': {'values': {'⊥', '⊤', 'False', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- (A(x), B(x)) // B(x)').token_options
    expected = {'Signs': {'values': {'(', '@', '∃'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'⊥', 'True', 'False', '⊤'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<cmd_identifier>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected


def test_interactive_existential():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('C(x) :- B(x), exists').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), exists(').token_options
    expected = {'Signs': {'values': {'(', '...', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), exists(s1').token_options
    expected = {'Signs': {'values': {',', ';', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'/', '+', '-', '**', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'st'}}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), exists(s1;').token_options
    expected = {'Signs': {'values': {'∃', '@', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists'}}, 'Boleans': {'values': {'⊤', 'False', '⊥', 'True'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), exists(s1; A(s1))').token_options
    expected = {'Signs': {'values': {',', '@', '&', '∃', '(', '∧'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists', 'ans'}}, 'Boleans': {'values': {'⊤', 'False', '⊥', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), ∃').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), ∃(').token_options
    expected = {'Signs': {'values': {'...', '(', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), ∃(s1').token_options
    expected = {'Signs': {'values': {';', ',', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '*', '**', '/', '+'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'st'}}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), ∃(s1 st').token_options
    expected = {'Signs': {'values': {'(', '∃', '@'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'~', '¬'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊥', 'True', 'False', '⊤'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('C(x) :- B(x), ∃(s1 st A(s1))').token_options
    expected = {'Signs': {'values': {',', '&', '@', '∧', '(', '∃'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'True', '⊤', '⊥', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete(
        'C(x) :- B(x), exists(s1, s2; A(s1),').token_options
    expected = {'Signs': {'values': {'∃', '(', '@'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'True', '⊥', 'False', '⊤'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete(
        'C(x) :- B(x), exists(s1, s2; A(s1), A(s2))').token_options
    expected = {'Signs': {'values': {'(', ',', '@', '&', '∧', '∃'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'True', '⊥', '⊤', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    with pytest.raises(UnexpectedTokenError):
        completer.complete('C(x) :- B(x), exists(s1; )').token_options


def test_interactive_query():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('ans(').token_options
    expected = {'Signs': {'values': {')', '(', '@', '...'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('ans(x)').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {'←', ':-'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('ans(x) :-').token_options
    expected = {'Signs': {'values': {'∃', '@', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'¬', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊥', 'True', '⊤', 'False'}}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('ans(x) :- B(x, y), C(3, y)').token_options
    expected = {'Signs': {'values': {'(', ',', '@', '∧', '∃', '&'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '-', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'ans', 'EXISTS'}}, 'Boleans': {'values': {'⊤', '⊥', 'False', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected


def test_interactive_prob_implicit():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('B(x, PROB').token_options
    expected = {'Signs': {'values': {',', ')', '('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'/', '-', '+', '*', '**'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB,').token_options
    expected = {'Signs': {'values': {'(', '@', '...'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB, y)').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {'←', ':-', ':=', '::'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB, y) :- C(x, y)').token_options
    expected = {'Signs': {'values': {',', '(', '∃', '@', '&', '∧'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '¬', '~', '//'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'ans', 'exists'}}, 'Boleans': {'values': {'⊤', 'True', '⊥', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected


def test_interactive_prob_explicit():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('B(x, PROB').token_options
    expected = {'Signs': {'values': {')', '(', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '*', '/', '+', '**'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB(').token_options
    expected = {'Signs': {'values': {'(', '...', ')', '@'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB(x,').token_options
    expected = {'Signs': {'values': {'...', '(', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB(x, y)').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '/', '+', '**', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB(x, y),').token_options
    expected = {'Signs': {'values': {'@', '(', '...'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB(x, y), y)').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {'←', ':=', ':-', '::'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('B(x, PROB(x, y), y) :- C(x, y)').token_options
    expected = {'Signs': {'values': {'&', '@', ',', '∧', '(', '∃'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '-', '//', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'ans', 'exists'}}, 'Boleans': {'values': {'⊥', '⊤', 'True', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected


def test_interactive_lambda_definition():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('c').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'**', '-', '+', '/', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {':=', '::'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('c :=').token_options
    expected = {'Signs': {'values': {'@', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('c := lambda').token_options
    expected = {'Signs': {'values': {'@', '(', '...'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<cmd_identifier>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('c := lambda x').token_options
    expected = {'Signs': {'values': {'(', ',', ':'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'/', '*', '+', '**', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('c := lambda x:').token_options
    expected = {'Signs': {'values': {'@', '...', '('}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('c := lambda x: x + 1').token_options
    expected = {'Signs': {'values': {'(', '@', '∃'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'**', '¬', '~', '*', '+', '/', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists', 'ans'}}, 'Boleans': {'values': {'⊤', 'True', 'False', '⊥'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<identifier_regexp>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected


def test_interactive_lambda_definition_statement():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('c(x, y) := x +').token_options
    expected = {'Signs': {'values': {'@', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<identifier_regexp>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('c(x, y) := x + y').token_options
    expected = {'Signs': {'values': {'@', '∃', '('}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-', '/', '¬', '*', '+', '**', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists', 'ans'}}, 'Boleans': {'values': {'⊤', '⊥', 'False', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected


def test_interactive_lambda_application():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('c := (lambda').token_options
    expected = {'Signs': {'values': {'(', '@', '...'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<text>', '<cmd_identifier>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('c := (lambda x').token_options
    expected = {'Signs': {'values': {'(', ':', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '+', '/', '**', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('c := (lambda x:').token_options
    expected = {'Signs': {'values': {'...', '(', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('c := (lambda x: x').token_options
    expected = {'Signs': {'values': {'(', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'*', '-', '+', '**', '/'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('c := (lambda x: x + 1)').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('c := (lambda x: x + 1)(2)').token_options
    expected = {'Signs': {'values': {'(', '@', '∃'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '+', '*', '/', '**', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'EXISTS', 'exists', 'ans'}}, 'Boleans': {'values': {'False', '⊥', '⊤', 'True'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>', '<text>', '<identifier_regexp>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected


def test_interactive_command_syntax():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('.').token_options
    expected = {'Signs': {'values': set()}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<cmd_identifier>'}}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv').token_options
    expected = {'Signs': {'values': {'('}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv(').token_options
    expected = {'Signs': {'values': {'(', '@', ')'}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<string>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv()').token_options
    expected = {'Signs': {'values': {'@', '(', '∃'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '~', '-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'ans', 'exists', 'EXISTS'}}, 'Boleans': {'values': {'⊤', '⊥', 'True', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv(A,').token_options
    expected = {'Signs': {'values': {'(', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>', '<string>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete(
        '.load_csv(A, "http://myweb/file.csv", B)').token_options
    expected = {'Signs': {'values': {'∃', '@', '('}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '-', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'EXISTS', 'ans'}}, 'Boleans': {'values': {'⊥', '⊤', 'True', 'False'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<identifier_regexp>', '<cmd_identifier>', '<text>'}}, 'commands': {'values': set()}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv(sep').token_options
    expected = {'Signs': {'values': {')', '(', '=', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': {'-', '+', '/', '**', '*'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv(sep=').token_options
    expected = {'Signs': {'values': {'@', '('}}, 'Numbers': {'values': {'<float>', '<integer>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'-'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<identifier_regexp>', '<cmd_identifier>', '<string>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv(sep=","').token_options
    expected = {'Signs': {'values': {',', ')'}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': set()}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected

    res = completer.complete('.load_csv(sep=",")').token_options
    expected = {'Signs': {'values': {'(', '∃', '@'}}, 'Numbers': {'values': {'<integer>', '<float>'}},
                'Text': {'values': set()}, 'Operators': {'values': {'¬', '-', '~'}}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': {'lambda'}}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': {'exists', 'ans', 'EXISTS'}}, 'Boleans': {'values': {'⊤', 'True', 'False', '⊥'}}, 'Expression symbols': {'values': {'.'}}, 'Python string': {'values': set()}, 'Strings': {'values': {'<text>', '<cmd_identifier>', '<identifier_regexp>'}}, 'functions': {'values': set()}, 'base symbols': {'values': set()}, 'query symbols': {'values': set()}, 'commands': {'values': set()}}
    assert res == expected


def test_interactive_constraint():
    completer = LarkCompleter(COMPILED_GRAMMAR)

    res = completer.complete('(x == y)').token_options
    print(res)
    expected = {'Signs': {'values': {'∧', '&', ','}}, 'Numbers': {'values': set()}, 'Text': {'values': set()},
                'Operators': {'values': set()}, 'Cmd_identifier': {'values': set()}, 'Functions': {'values': set()}, 'Identifier_regexp': {'values': set()}, 'Reserved words': {'values': set()}, 'Boleans': {'values': set()}, 'Expression symbols': {'values': {'→', '-:'}}, 'Python string': {'values': set()}, 'Strings': {'values': set()}}
    assert res == expected
