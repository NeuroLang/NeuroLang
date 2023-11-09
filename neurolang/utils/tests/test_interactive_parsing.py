import pytest

from ...expressions import Symbol
from ...exceptions import UnexpectedTokenError
from ...frontend.datalog.standard_syntax import parser

def test_interactive_empty_input():
    input = ""
    res = parser(input, interactive=True)
    expected = {
        'Signs'             : {'(', '@', '∃'},
        'Numbers'           : {'<float>', '<integer>'},
        'Operators'         : {'-', '~', '¬'},
        'Functions'         : {'lambda'},
        'Reserved words'    : {'ans', 'exists', 'EXISTS'},
        'Boleans'           : {'True', '⊤', 'False', '⊥'},
        'Expression symbols': {'.'},
        'Strings'           : {'<text>', '<command identifier>', '<identifier regular expression>'}
    }
    assert res == expected


def test_interactive_facts():
    input = 'A'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}, 'Operators': {'*', '-', '**', '/', '+'}, 'Expression symbols': {':=', '::'}}
    assert res == expected

    input = 'A('
    res = parser(input, interactive=True)
    expected = {
        'Signs'    : {'@', ')','(', '...'},
        'Numbers'  : {'<float>', '<integer>'},
        'Operators': {'-'},
        'Functions': {'lambda'},
        'Strings'  : {'<identifier regular expression>', '<command identifier>', '<text>'}
    }
    assert res == expected

    input = 'A()'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'¬', '-', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊤', 'False', 'True', '⊥'}, 'Expression symbols': {':=', ':-', '←', '.', '::'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected

    input = 'A(3'
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ','}, 'Operators': {'/', '**', '+', '*', '-'}}
    assert res == expected

    input = 'A(3)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = 'A("x"'
    res = parser(input, interactive=True)
    expected = {
        'Signs'    : {')', ','},
        'Operators': {'**', '*', '-', '/', '+'}
    }
    assert res == expected

    input = 'A("x")'
    res = parser(input, interactive=True)
    expected = {
        'Signs': {'∃', '@', '('},
        'Numbers': {'<integer>', '<float>'},
        'Operators': {'-', '¬', '~'},
        'Functions': {'lambda'},
        'Reserved words': {'exists', 'ans', 'EXISTS'},
        'Boleans': {'True', '⊤', 'False', '⊥'},
        'Expression symbols': {'.'},        # because the grammar ignores spaces and new lines, so the dot is accepted as the command of the next line
        'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = 'A("x",'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Strings': {'<text>'}}
    assert res == expected

    input = "A('x', 3"
    res = parser(input, interactive=True)
    expected = {
        'Signs': {')',','}
    }
    assert res == expected

    input = "A('x', 3,"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Strings': {'<text>'}}
    assert res == expected

    input = "A('x', 3)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'¬', '~', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'False', '⊤', 'True', '⊥'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = '''
    `http://uri#test-fact`("x")
    A("x", 3
    '''
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ','}}
    assert res == expected


def test_interactive_rules():
    input = 'A(x'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ',', ')'}, 'Operators': {'/', '-', '*', '**', '+'}}
    assert res == expected

    input = 'A(x)'
    res = parser(input, interactive=True)
    expected = {'Expression symbols': {':=', ':-', '←', '::'}}
    assert res == expected

    input = 'A(x):-'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '@', '('}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'False', 'True'}, 'Strings': {'<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = 'A(x):-B'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = 'A(x):-B('
    res = parser(input, interactive=True)
    expected = {'Signs': {'...', '(', ')', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = 'A(x):-B()'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '@', '(', ',', '∧', '&'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '//', '~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'False', 'True'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = 'A(x):-B(x'
    res = parser(input, interactive=True)
    expected = {'Signs': {')', '(', ','}, 'Operators': {'-', '**', '/', '*', '+'}}
    assert res == expected

    input = 'A(x):-B(x,'
    res = parser(input, interactive=True)
    expected = {'Signs': {'...', '@', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = 'A(x):-B(x, y'
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ',', '('}, 'Operators': {'-', '**', '*', '/', '+'}}
    assert res == expected

    input = 'A(x):-B(x, y)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '∧', '∃', '(', '&', ','}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'~', '-', '¬', '//'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'EXISTS', 'exists'}, 'Boleans': {'True', '⊥', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}}
    assert res == expected

    input = 'A(x):-B(x, y),'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '@', '('}, 'Operators': {'~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = 'A(x):-B(x, y), C(3, z)'
    res = parser(input, interactive=True)
    expected = {'Signs': {',', '∧', '(', '@', '&', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊥', 'True', 'False', '⊤'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = 'A(x):-~'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'⊥', '⊤', 'False', 'True'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = 'A(x):-~B(x)'
    res = parser(input, interactive=True)
    expected = {'Signs': {',', '@', '&', '∧', '∃', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '~', '//', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = 'A(x):-~B(x),'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Operators': {'~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'True', 'False', '⊥', '⊤'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = 'A(x):-B(x, ...)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'&', '∃', '@', ',', '(', '∧'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'//', '-', '~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}}
    assert res == expected

    input = 'A(x):-B(x, y), C(3, z), ('
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@', '...'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = 'A(x):-B(x, y), C(3, z), (z'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}, 'Operators': {'*', '>', '==', '<', '-', '+', '<=', '**', '/', '!=', '>='}}
    assert res == expected

    input = 'A(x):-B(x, y), C(3, z), (z =='
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '...', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = 'A(x):-B(x + 5 *'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = 'A(x):-B(x / 2'
    res = parser(input, interactive=True)
    expected = {'Signs': {',', ')'}, 'Operators': {'+', '*', '/', '**', '-'}}
    assert res == expected

    input = 'A(x):-B(f(x'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ',', ')'}, 'Operators': {'*', '/', '-', '**', '+'}}
    assert res == expected

    input = 'A(x):-B(x + (-5),'
    res = parser(input, interactive=True)
    expected = {'Signs': {'...', '@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}}
    assert res == expected

    input = 'A(x):-B(x + (-5), "a"'
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ','}, 'Operators': {'+', '**', '*', '-', '/'}}
    assert res == expected

    input = 'A(x):-B(x - 5 * 2, @'
    res = parser(input, interactive=True)
    expected = {'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected


def test_interactive_aggregation():
    input = 'A(x, f('
    res = parser(input, interactive=True)
    expected = {'Signs': {')', '...', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = 'A(x, f(y'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ')', ','}, 'Operators': {'*', '/', '+', '**', '-'}}
    assert res == expected

    input = 'A(x, f(y)'
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ','}, 'Operators': {'+', '*', '/', '**', '-'}}
    assert res == expected

    input = 'A(x, f(y))'
    res = parser(input, interactive=True)
    expected = {'Expression symbols': {'←', ':=', '::', ':-'}}
    assert res == expected


def test_interactive_uri():
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(
        name='http://sig.biostr.washington.edu/fma3.0#regional_part_of'
    )

    input = f'`{str(label.name)}`'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}, 'Operators': {'/', '*', '**', '-', '+'}, 'Expression symbols': {'::', ':='}}
    assert res == expected

    input = f'`{str(label.name)}`(x)'
    res = parser(input, interactive=True)
    expected = {'Expression symbols': {':-', '←', '::', ':='}}
    assert res == expected

    input = f'`{str(label.name)}`(x):-'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@', '∃'}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'⊤', 'False', 'True', '⊥'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = f'`{str(label.name)}`(x):-`{str(regional_part.name)}`'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = f'`{str(label.name)}`(x):-`{str(regional_part.name)}`(x, y)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '&', ',', '∃', '∧', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'~', '¬', '-', '//'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'EXISTS', 'exists'}, 'Boleans': {'False', '⊥', 'True', '⊤'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected


def test_interactive_probabilistic_fact():
    input = 'p::'
    res = parser(input, interactive=True)
    expected = {'Strings': {'<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = 'p::A'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = 'p::A('
    res = parser(input, interactive=True)
    expected = {'Signs': {')', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Strings': {'<text>'}}
    assert res == expected

    input = 'p::A(3'
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ','}}
    assert res == expected

    input = 'p::A(3)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'¬', '-', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = '0.8::A("a b",'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Strings': {'<text>'}}
    assert res == expected

    input = '0.8::A("a b", 3'
    res = parser(input, interactive=True)
    expected = {'Signs': {',', ')'}}
    assert res == expected

    input = '0.8::A("a b", 3)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'¬', '-', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'True', 'False', '⊤', '⊥'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0)"
    res = parser(input, interactive=True)
    expected = {'Operators': {'*', '**', '-', '+', '/'}, 'Expression symbols': {':-', '←'}}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :-"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Operators': {'~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Strings': {'<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :- A(x, d)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', ',', '(', '@', '&', '∧'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'¬', '-', '~', '//'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'EXISTS', 'exists'}, 'Boleans': {'True', '⊤', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :- A(x, d) &"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '@', '('}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists'}, 'Boleans': {'False', 'True', '⊥', '⊤'}, 'Strings': {'<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :- A(x, d) & (d < 0.8)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∧', '∃', '@', '(', ',', '&'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '-', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', 'False', '⊤'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}}
    assert res == expected


def test_interactive_condition():
    input = 'C(x) :- A(x) //'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '∃', '('}, 'Operators': {'~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = 'C(x) :- A(x) // B(x)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'EXISTS', 'exists'}, 'Boleans': {'False', '⊤', '⊥', 'True'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = 'C(x) :- A(x) // (A(x),'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'True', '⊤', '⊥', 'False'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = 'C(x) :- A(x) // (A(x), B(x))'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '@', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊥', 'False', 'True', '⊤'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}}
    assert res == expected

    input = 'C(x) :- (A(x), B(x)) // B(x)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'~', '¬', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊤', 'False', 'True', '⊥'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected


def test_interactive_existential():
    input = "C(x) :- B(x), exists"
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = "C(x) :- B(x), exists("
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '(', '...'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected

    input = "C(x) :- B(x), exists(s1"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ',', ';'}, 'Operators': {'**', '+', '-', '*', '/'}, 'Reserved words': {'st'}}
    assert res == expected

    input = "C(x) :- B(x), exists(s1;"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '∃', '('}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'False', 'True', '⊤'}, 'Strings': {'<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = "C(x) :- B(x), exists(s1; A(s1))"
    res = parser(input, interactive=True)
    expected = {'Signs': {',', '&', '∃', '@', '∧', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-', '¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'⊥', '⊤', 'False', 'True'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = "C(x) :- B(x), ∃"
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = "C(x) :- B(x), ∃("
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@', '...'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = "C(x) :- B(x), ∃(s1"
    res = parser(input, interactive=True)
    expected = {'Signs': {',', '(', ';'}, 'Operators': {'/', '+', '*', '**', '-'}, 'Reserved words': {'st'}}
    assert res == expected

    input = "C(x) :- B(x), ∃(s1 st"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '∃', '('}, 'Operators': {'~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', '⊤', 'True', 'False'}, 'Strings': {'<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = "C(x) :- B(x), ∃(s1 st A(s1))"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '&', '∃', '∧', '(', ','}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-', '¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊤', '⊥', 'True', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = "C(x) :- B(x), exists(s1, s2; A(s1),"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Operators': {'¬', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = "C(x) :- B(x), exists(s1, s2; A(s1), A(s2))"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '&', '∃', '∧', ',', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '¬', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'EXISTS', 'exists'}, 'Boleans': {'True', 'False', '⊤', '⊥'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<text>', '<command identifier>'}}
    assert res == expected

    with pytest.raises(UnexpectedTokenError):
        input = "C(x) :- B(x), exists(s1; )"
        res = parser(input, interactive=True)


def test_interactive_query():
    input = "ans("
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '...', '@', ')'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected

    input = "ans(x)"
    res = parser(input, interactive=True)
    expected = {'Expression symbols': {':-', '←'}}
    assert res == expected

    input = "ans(x) :-"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Operators': {'~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊤', 'False', '⊥', 'True'}, 'Strings': {'<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = "ans(x) :- B(x, y), C(3, y)"
    res = parser(input, interactive=True)
    expected = {'Signs': {',', '(', '&', '∃', '@', '∧'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '¬', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊥', 'True', '⊤', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected


def test_interactive_prob_implicit():
    input = "B(x, PROB"
    res = parser(input, interactive=True)
    expected = {'Signs': {',', '(', ')'}, 'Operators': {'**', '/', '+', '*', '-'}}
    assert res == expected

    input = "B(x, PROB,"
    res = parser(input, interactive=True)
    expected = {'Signs': {'...', '@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected

    input = "B(x, PROB, y)"
    res = parser(input, interactive=True)
    expected = {'Expression symbols': {'←', ':=', ':-', '::'}}
    assert res == expected

    input = "B(x, PROB, y) :- C(x, y)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '∧', ',', '(', '&', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'//', '~', '-', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'False', '⊥', 'True', '⊤'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<identifier regular expression>', '<command identifier>'}}
    assert res == expected


def test_interactive_prob_explicit():
    input = "B(x, PROB"
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ',', '('}, 'Operators': {'*', '/', '+', '-', '**'}}
    assert res == expected

    input = "B(x, PROB("
    res = parser(input, interactive=True)
    expected = {'Signs': {')', '@', '...', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = "B(x, PROB(x,"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '...', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = "B(x, PROB(x, y)"
    res = parser(input, interactive=True)
    expected = {'Signs': {')', ','}, 'Operators': {'*', '/', '+', '-', '**'}}
    assert res == expected

    input = "B(x, PROB(x, y),"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '...', '('}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = "B(x, PROB(x, y), y)"
    res = parser(input, interactive=True)
    expected = {'Expression symbols': {'::', '←', ':=', ':-'}}
    assert res == expected

    input = "B(x, PROB(x, y), y) :- C(x, y)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∧', '@', '&', '∃', '(', ','}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '-', '¬', '//'}, 'Functions': {'lambda'}, 'Reserved words': {'ans', 'exists', 'EXISTS'}, 'Boleans': {'⊥', 'False', 'True', '⊤'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected


def test_interactive_lambda_definition():
    input = "c"
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}, 'Operators': {'**', '-', '/', '+', '*'}, 'Expression symbols': {':=', '::'}}
    assert res == expected

    input = "c :="
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected

    input = "c := lambda"
    res = parser(input, interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected

    input = "c := lambda x"
    res = parser(input, interactive=True)
    expected = {'Signs': {':', '(', ','}, 'Operators': {'**', '-', '/', '+', '*'}}
    assert res == expected

    input = "c := lambda x:"
    res = parser(input, interactive=True)
    expected = {'Signs': {'...', '(', '@'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected

    input = "c := lambda x: x + 1"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '∃', '('}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'**', '¬', '-', '/', '+', '*', '~'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'ans', 'EXISTS'}, 'Boleans': {'⊤', 'False', 'True', '⊥'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<text>', '<identifier regular expression>'}}
    assert res == expected


def test_interactive_lambda_definition_statement():
    input = "c(x, y) := x +"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '('}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected

    input = "c(x, y) := x + y"
    res = parser(input, interactive=True)
    expected = {'Signs': {'@', '(', '∃'}, 'Numbers': {'<integer>', '<float>'}, 'Operators': {'*', '+', '/', '-', '**', '~', '¬'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊤', 'True', 'False', '⊥'}, 'Expression symbols': {'.'}, 'Strings': {'<command identifier>', '<identifier regular expression>', '<text>'}}
    assert res == expected


def test_interactive_lambda_application():
    input = "c := (lambda"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@', '...'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = "c := (lambda x"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ',', ':'}, 'Operators': {'/', '**', '-', '*', '+'}}
    assert res == expected

    input = "c := (lambda x:"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@', '...'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected

    input = "c := (lambda x: x"
    res = parser(input, interactive=True)
    expected = {'Signs': {')', '('}, 'Operators': {'/', '**', '-', '*', '+'}}
    assert res == expected

    input = "c := (lambda x: x + 1)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = "c := (lambda x: x + 1)(2)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '∃', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'/', '**', '¬', '-', '*', '~', '+'}, 'Functions': {'lambda'}, 'Reserved words': {'exists', 'EXISTS', 'ans'}, 'Boleans': {'⊤', 'True', '⊥', 'False'}, 'Expression symbols': {'.'}, 'Strings': {'<identifier regular expression>', '<command identifier>', '<text>'}}
    assert res == expected


def test_interactive_command_syntax():
    input = '.'
    res = parser(input, interactive=True)
    expected = {'Strings': {'<command identifier>'}}
    assert res == expected

    input = '.load_csv'
    res = parser(input, interactive=True)
    expected = {'Signs': {'('}}
    assert res == expected

    input = '.load_csv('
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ')', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<quoted string>', '<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = ".load_csv()"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '¬', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊤', 'False', '⊥', 'True'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = '.load_csv(A,'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<quoted string>', '<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = '.load_csv(A, "http://myweb/file.csv", B)'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '¬', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊤', 'False', '⊥', 'True'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = '.load_csv(sep'
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', ',', '=', ')'}, 'Operators': {'*', '+', '/', '**', '-'}}
    assert res == expected

    input = '.load_csv(sep='
    res = parser(input, interactive=True)
    expected = {'Signs': {'(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'-'}, 'Functions': {'lambda'}, 'Strings': {'<quoted string>', '<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected

    input = '.load_csv(sep=","'
    res = parser(input, interactive=True)
    expected = {'Signs': {',', ')'}}
    assert res == expected

    input = '.load_csv(sep=",")'
    res = parser(input, interactive=True)
    expected = {'Signs': {'∃', '(', '@'}, 'Numbers': {'<float>', '<integer>'}, 'Operators': {'~', '¬', '-'}, 'Functions': {'lambda'}, 'Reserved words': {'EXISTS', 'exists', 'ans'}, 'Boleans': {'⊤', 'False', '⊥', 'True'}, 'Expression symbols': {'.'}, 'Strings': {'<text>', '<command identifier>', '<identifier regular expression>'}}
    assert res == expected


def test_interactive_constraint():
    input = "(x == y)"
    res = parser(input, interactive=True)
    expected = {'Signs': {'∧', ',', '&'}, 'Expression symbols': {'-:', '→'}}
    assert res == expected

