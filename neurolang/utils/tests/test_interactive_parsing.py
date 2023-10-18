# from operator import add, eq, lt, mul, pow, sub, truediv
#
import pytest
#
#
# from ....logic import ExistentialPredicate
#
# from ....datalog import Conjunction, Fact, Implication, Negation, Union
from  ...expressions import Symbol
# from ....expressions import (
#     Command,
#     Constant,
#     FunctionApplication,
#     Lambda,
#     Query,
#     Statement,
#     Symbol
# )
# from ....exceptions import UnexpectedTokenError
from ...exceptions import UnexpectedTokenError
# from ....probabilistic.expressions import (
#     PROB,
#     Condition,
#     ProbabilisticFact
# )
from ...frontend.datalog.standard_syntax import parser
# from ..standard_syntax import ExternalSymbol


def test_interactive_facts():
    # print("")
    # print("__________________________________")
    # print("____ test_facts() ____")
    input = 'A('
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'DOTS', 'LPAR', 'AT', 'INT', 'TEXT', 'MINUS', 'RPAR', 'LAMBDA', 'FLOAT'}
    assert res == expected

    input = 'A("x"'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'POW', 'STAR', 'PLUS', 'MINUS', 'RPAR', 'COMMA', 'SLASH'}
    assert res == expected

    input = "A('x', 3"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'RPAR', 'COMMA'}
    assert res == expected

    input = "A('x', 3,"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'AT', 'INT', 'TEXT', 'MINUS', 'FLOAT'}
    assert res == expected

    input = '''
    `http://uri#test-fact`("x")
    A("x", 3
    '''
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'RPAR', 'COMMA'}
    assert res == expected


def test_interactive_rules():
    # print("")
    # print("__________________________________")
    # print("____ test_rules() ____")
    input = 'A(x):-'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'LPAR', 'CMD_IDENTIFIER', 'NEG_UNICODE', 'EXISTS', 'LAMBDA', 'TILDE', 'FALSE', 'TRUE', 'AT'}
    assert res == expected

    input = 'A(x):-B(x, y'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'PLUS', 'LPAR', 'POW', 'STAR', 'RPAR', 'COMMA', 'SLASH'}
    assert res == expected

    input = 'A(x):-B(x, y)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'EXISTS', 'FALSE', '$END', 'ANS', 'NEG_UNICODE', 'TRUE', 'FLOAT', 'AT', 'COMMA', 'INT', 'CONDITION_OP', 'TEXT', 'LPAR', 'CMD_IDENTIFIER', 'CONJUNCTION_SYMBOL', 'LAMBDA', 'TILDE', 'DOT'}
    assert res == expected

    input = 'A(x):-B(x, y),'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'LPAR', 'CMD_IDENTIFIER', 'NEG_UNICODE', 'EXISTS', 'LAMBDA', 'TILDE', 'FALSE', 'TRUE', 'AT'}
    assert res == expected

    input = 'A(x):-B(x, y), C(3, z)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'EXISTS', 'FALSE', '$END', 'ANS', 'NEG_UNICODE', 'TRUE', 'FLOAT', 'AT', 'COMMA', 'INT', 'TEXT', 'LPAR', 'CMD_IDENTIFIER', 'CONJUNCTION_SYMBOL', 'LAMBDA', 'TILDE', 'DOT'}
    assert res == expected

    input = 'A(x):-~B(x)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'EXISTS', 'FALSE', '$END', 'ANS', 'NEG_UNICODE', 'TRUE', 'FLOAT', 'AT', 'COMMA', 'INT', 'CONDITION_OP', 'TEXT', 'LPAR', 'CMD_IDENTIFIER', 'CONJUNCTION_SYMBOL', 'LAMBDA', 'TILDE', 'DOT'}
    assert res == expected

    input = 'A(x):-B(x, ...)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'EXISTS', 'FALSE', '$END', 'ANS', 'NEG_UNICODE', 'TRUE', 'FLOAT', 'AT', 'COMMA', 'INT', 'CONDITION_OP', 'TEXT', 'LPAR', 'CMD_IDENTIFIER', 'CONJUNCTION_SYMBOL', 'LAMBDA', 'TILDE', 'DOT'}
    assert res == expected

    input = 'A(x):-B(x, y), C(3, z), (z =='
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'LPAR', 'CMD_IDENTIFIER', 'LAMBDA', 'DOTS', 'AT', 'INT', 'TEXT', 'FLOAT'}
    assert res == expected

    input = 'A(x):-B(x + 5 *'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'LPAR', 'CMD_IDENTIFIER', 'LAMBDA', 'AT', 'INT', 'TEXT', 'FLOAT'}
    assert res == expected

    input = 'A(x):-B(x / 2'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'PLUS', 'POW', 'STAR', 'RPAR', 'COMMA', 'SLASH'}
    assert res == expected

    input = 'A(x):-B(f(x'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'PLUS', 'LPAR', 'POW', 'STAR', 'RPAR', 'COMMA', 'SLASH'}
    assert res == expected

    input = 'A(x):-B(x + (-5), "a'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'IDENTIFIER_REGEXP', 'LPAR', 'CMD_IDENTIFIER', 'LAMBDA', 'DOTS', 'AT', 'INT', 'TEXT', 'FLOAT'}
    assert res == expected

    input = 'A(x):-B(x - 5 * 2, @'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER'}
    assert res == expected


def test_interactive_aggregation():
    # print("")
    # print("__________________________________")
    # print("____ test_aggregation() ____")
    input = 'A(x, f(y'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'RPAR', 'PLUS', 'STAR', 'POW', 'COMMA', 'SLASH', 'LPAR', 'MINUS'}
    assert res == expected


def test_interactive_uri():
    # print("")
    # print("__________________________________")
    # print("____ test_uri() ____")
    from rdflib import RDFS

    label = Symbol(name=str(RDFS.label))
    regional_part = Symbol(
        name='http://sig.biostr.washington.edu/fma3.0#regional_part_of'
    )

    input = f'`{str(label.name)}`'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'PLUS', 'POW', 'SLASH', 'STAR', 'PROBA_OP', 'MINUS', 'STATEMENT_OP', 'LPAR'}
    assert res == expected

    input = f'`{str(label.name)}`(x)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IMPLICATION', 'PROBA_OP', 'STATEMENT_OP'}
    assert res == expected

    input = f'`{str(label.name)}`(x):-'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TILDE', 'NEG_UNICODE', 'AT', 'TRUE', 'FALSE', 'EXISTS', 'LAMBDA', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LPAR'}
    assert res == expected

    input = f'`{str(label.name)}`(x):-`'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TILDE', 'NEG_UNICODE', 'AT', 'TRUE', 'FALSE', 'EXISTS', 'LAMBDA', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LPAR'}
    assert res == expected

    input = f'`{str(label.name)}`(x):-`{str(regional_part.name)}'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TILDE', 'NEG_UNICODE', 'AT', 'TRUE', 'FALSE', 'EXISTS', 'LAMBDA', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LPAR'}
    assert res == expected

    input = f'`{str(label.name)}`(x):-`{str(regional_part.name)}`'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'LPAR'}
    assert res == expected

    input = f'`{str(label.name)}`(x):-`{str(regional_part.name)}`(x, y)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TILDE', 'AT', 'CMD_IDENTIFIER', 'CONJUNCTION_SYMBOL', 'IDENTIFIER_REGEXP', 'INT', 'COMMA', 'DOT', 'TRUE', 'CONDITION_OP', 'LAMBDA', 'ANS', 'LPAR', 'FLOAT', 'EXISTS', '$END', 'NEG_UNICODE', 'FALSE', 'MINUS', 'TEXT'}
    assert res == expected


def test_interactive_probabilistic_fact():
    # print("")
    # print("__________________________________")
    # print("____ test_probabilistic_fact() ____")
    input = 'p::'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER'}
    assert res == expected

    input = 'p::A(3)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'CMD_IDENTIFIER', 'FLOAT', 'TILDE', 'NEG_UNICODE', 'INT', '$END', 'LPAR', 'LAMBDA', 'IDENTIFIER_REGEXP', 'TRUE', 'EXISTS', 'DOT', 'AT', 'FALSE', 'TEXT', 'ANS'}
    assert res == expected

    input = '0.8::A("a b", 3)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'MINUS', 'CMD_IDENTIFIER', 'FLOAT', 'TILDE', 'NEG_UNICODE', 'INT', '$END', 'LPAR', 'LAMBDA', 'IDENTIFIER_REGEXP', 'TRUE', 'EXISTS', 'DOT', 'AT', 'FALSE', 'TEXT', 'ANS'}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'POW', 'MINUS', 'PLUS', 'IMPLICATION', 'SLASH', 'STAR'}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :-"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'CMD_IDENTIFIER', 'NEG_UNICODE', 'TILDE', 'LPAR', 'IDENTIFIER_REGEXP', 'LAMBDA', 'TRUE', 'EXISTS', 'AT', 'FALSE'}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :- A(x, d) &"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'CMD_IDENTIFIER', 'NEG_UNICODE', 'TILDE', 'LPAR', 'IDENTIFIER_REGEXP', 'LAMBDA', 'TRUE', 'EXISTS', 'AT', 'FALSE'}
    assert res == expected

    input = "B(x) :: exp(-d / 5.0) :- A(x, d) & (d < 0.8)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'FLOAT', 'TILDE', 'INT', 'COMMA', 'LAMBDA', 'AT', 'MINUS', 'CONJUNCTION_SYMBOL', 'NEG_UNICODE', '$END', 'LPAR', 'IDENTIFIER_REGEXP', 'DOT', 'CMD_IDENTIFIER', 'TRUE', 'EXISTS', 'FALSE', 'TEXT', 'ANS'}
    assert res == expected


def test_interactive_condition():
    # print("")
    # print("__________________________________")
    # print("____ test_condition() ____")
    input = 'C(x) :- A(x) //'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'NEG_UNICODE', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LAMBDA', 'FALSE', 'EXISTS', 'AT', 'TILDE', 'LPAR', 'TRUE'}
    assert res == expected

    input = 'C(x) :- A(x) // B(x)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'DOT', 'NEG_UNICODE', 'FLOAT', '$END', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LAMBDA', 'INT', 'FALSE', 'AT', 'MINUS', 'EXISTS', 'TILDE', 'LPAR', 'ANS', 'TEXT', 'TRUE'}
    assert res == expected

    input = 'C(x) :- (A(x), B(x)) // B(x)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'DOT', 'NEG_UNICODE', 'FLOAT', '$END', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LAMBDA', 'INT', 'FALSE', 'AT', 'MINUS', 'EXISTS', 'TILDE', 'LPAR', 'ANS', 'TEXT', 'TRUE'}
    assert res == expected

    input = 'C(x) :- A(x) // (A(x),'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'NEG_UNICODE', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LAMBDA', 'FALSE', 'EXISTS', 'AT', 'TILDE', 'LPAR', 'TRUE'}
    assert res == expected

    input = 'C(x) :- A(x) // (A(x), B(x))'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'DOT', 'NEG_UNICODE', 'FLOAT', '$END', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'LAMBDA', 'INT', 'FALSE', 'AT', 'MINUS', 'EXISTS', 'TILDE', 'LPAR', 'ANS', 'TEXT', 'TRUE'}
    assert res == expected


def test_interactive_existential():
    # print("")
    # print("__________________________________")
    # print("____ test_existential() ____")

    input = "C(x) :- B(x), exists"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'LPAR'}
    assert res == expected

    input = "C(x) :- B(x), exists(s1;"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'CMD_IDENTIFIER', 'TRUE', 'LPAR', 'AT', 'TILDE', 'LAMBDA', 'FALSE', 'IDENTIFIER_REGEXP', 'NEG_UNICODE'}
    assert res == expected

    input = "C(x) :- B(x), exists(s1; A(s1))"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'LPAR', 'FLOAT', 'INT', 'DOT', 'MINUS', 'CMD_IDENTIFIER', 'ANS', 'CONJUNCTION_SYMBOL', 'AT', 'FALSE', 'IDENTIFIER_REGEXP', 'NEG_UNICODE', '$END', 'TRUE', 'TILDE', 'TEXT', 'LAMBDA', 'COMMA'}
    assert res == expected

    input = "C(x) :- B(x), ∃(s1 st A(s1))"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'LPAR', 'FLOAT', 'INT', 'DOT', 'MINUS', 'CMD_IDENTIFIER', 'ANS', 'CONJUNCTION_SYMBOL', 'AT', 'FALSE', 'IDENTIFIER_REGEXP', 'NEG_UNICODE', '$END', 'TRUE', 'TILDE', 'TEXT', 'LAMBDA', 'COMMA'}
    assert res == expected

    input = "C(x) :- B(x), exists(s1, s2; A(s1),"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'CMD_IDENTIFIER', 'TRUE', 'LPAR', 'AT', 'TILDE', 'LAMBDA', 'FALSE', 'IDENTIFIER_REGEXP', 'NEG_UNICODE'}
    assert res == expected

    input = "C(x) :- B(x), exists(s1, s2; A(s1), A(s2))"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'LPAR', 'FLOAT', 'INT', 'DOT', 'MINUS', 'CMD_IDENTIFIER', 'ANS', 'CONJUNCTION_SYMBOL', 'AT', 'FALSE', 'IDENTIFIER_REGEXP', 'NEG_UNICODE', '$END', 'TRUE', 'TILDE', 'TEXT', 'LAMBDA', 'COMMA'}
    assert res == expected

    with pytest.raises(UnexpectedTokenError):
        input = "C(x) :- B(x), exists(s1; )"
        # print("***")
        # print(input)
        # print("res :")
        res = parser(input, interactive=True)


def test_interactive_query():
    # print("")
    # print("__________________________________")
    # print("____ test_query() ____")
    input = "ans("
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TEXT', 'IDENTIFIER_REGEXP', 'MINUS', 'LAMBDA', 'DOTS', 'LPAR', 'CMD_IDENTIFIER', 'INT', 'AT', 'FLOAT', 'RPAR'}
    assert res == expected

    input = "ans(x)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IMPLICATION'}
    assert res == expected

    input = "ans(x) :-"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TRUE', 'IDENTIFIER_REGEXP', 'FALSE', 'LAMBDA', 'TILDE', 'LPAR', 'CMD_IDENTIFIER', 'NEG_UNICODE', 'AT', 'EXISTS'}
    assert res == expected

    input = "ans(x) :- B(x, y), C(3, y)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'COMMA', 'ANS', 'EXISTS', 'TRUE', '$END', 'MINUS', 'TILDE', 'NEG_UNICODE', 'TEXT', 'DOT', 'CONJUNCTION_SYMBOL', 'FALSE', 'INT', 'CMD_IDENTIFIER', 'AT', 'LAMBDA', 'LPAR', 'FLOAT'}
    assert res == expected


def test_interactive_prob_implicit():
    # print("")
    # print("__________________________________")
    # print("____ test_prob_implicit() ____")
    input = "B(x, PROB"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'STAR', 'MINUS', 'SLASH', 'PLUS', 'LPAR', 'POW', 'COMMA', 'RPAR'}
    assert res == expected

    input = "B(x, PROB, y) :- C(x, y)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'FLOAT', 'FALSE', 'CONJUNCTION_SYMBOL', '$END', 'EXISTS', 'INT', 'CONDITION_OP', 'ANS', 'MINUS', 'LAMBDA', 'COMMA', 'TEXT', 'NEG_UNICODE', 'IDENTIFIER_REGEXP', 'LPAR', 'DOT', 'TILDE', 'CMD_IDENTIFIER', 'AT', 'TRUE'}
    assert res == expected


def test_interactive_prob_explicit():
    # print("")
    # print("__________________________________")
    # print("____ test_prob_explicit() ____")
    input = "B(x, PROB("
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'CMD_IDENTIFIER', 'INT', 'FLOAT', 'RPAR', 'TEXT', 'IDENTIFIER_REGEXP', 'DOTS', 'MINUS', 'LPAR', 'LAMBDA', 'AT'}
    assert res == expected

    input = "B(x, PROB(x,"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'CMD_IDENTIFIER', 'INT', 'FLOAT', 'TEXT', 'IDENTIFIER_REGEXP', 'DOTS', 'MINUS', 'LPAR', 'LAMBDA', 'AT'}
    assert res == expected

    input = "B(x, PROB(x, y), y) :- C(x, y)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'FALSE', 'EXISTS', 'INT', 'TILDE', 'COMMA', '$END', 'DOT', 'LPAR', 'LAMBDA', 'AT', 'CMD_IDENTIFIER', 'ANS', 'IDENTIFIER_REGEXP', 'CONJUNCTION_SYMBOL', 'NEG_UNICODE', 'MINUS', 'TRUE', 'CONDITION_OP', 'FLOAT', 'TEXT'}
    assert res == expected


def test_interactive_lambda_definition():
    # print("")
    # print("__________________________________")
    # print("____ test_lambda_definition() ____")

    input = "c"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'PROBA_OP', 'STAR', 'SLASH', 'POW', 'STATEMENT_OP', 'LPAR', 'PLUS', 'MINUS'}
    assert res == expected

    input = "c :="
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'FLOAT', 'INT', 'CMD_IDENTIFIER', 'AT', 'LPAR', 'MINUS', 'LAMBDA', 'TEXT'}
    assert res == expected

    input = "c := lambda"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'FLOAT', 'INT', 'CMD_IDENTIFIER', 'AT', 'LPAR', 'DOTS', 'MINUS', 'LAMBDA', 'TEXT'}
    assert res == expected

    input = "c := lambda x"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'STAR', 'COMMA', 'SLASH', 'POW', 'LPAR', 'PLUS', 'MINUS', 'COLON'}
    assert res == expected

    input = "c := lambda x:"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'IDENTIFIER_REGEXP', 'FLOAT', 'INT', 'CMD_IDENTIFIER', 'AT', 'LPAR', 'DOTS', 'MINUS', 'LAMBDA', 'TEXT'}
    assert res == expected

    input = "c := lambda x: x + 1"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'INT', 'SLASH', 'AT', 'MINUS', 'TILDE', 'CMD_IDENTIFIER', 'TRUE', 'IDENTIFIER_REGEXP', 'LAMBDA', 'TEXT', 'DOT', 'FALSE', 'NEG_UNICODE', 'PLUS', 'ANS', 'EXISTS', 'STAR', '$END', 'FLOAT', 'POW', 'LPAR'}
    assert res == expected


def test_interactive_lambda_definition_statement():
    # print("")
    # print("__________________________________")
    # print("____ test_lambda_definition_statement() ____")

    input = "c(x, y) :="
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'LPAR', 'INT', 'DOTS', 'MINUS', 'CMD_IDENTIFIER', 'TEXT', 'IDENTIFIER_REGEXP', 'AT', 'LAMBDA', 'FLOAT'}
    assert res == expected

    input = "c(x, y) := x + y"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TRUE', 'NEG_UNICODE', 'DOT', 'INT', 'PLUS', 'ANS', 'CMD_IDENTIFIER', 'IDENTIFIER_REGEXP', 'STAR', 'MINUS', 'POW', 'FALSE', 'LAMBDA', 'FLOAT', 'TEXT', 'SLASH', 'AT', 'EXISTS', 'LPAR', 'TILDE', '$END'}
    assert res == expected


def test_interactive_lambda_application():
    # print("")
    # print("__________________________________")
    # print("____ test_lambda_application() ____")

    input = "c := (lambda"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'INT', 'TEXT', 'DOTS', 'IDENTIFIER_REGEXP', 'MINUS', 'FLOAT', 'LAMBDA', 'LPAR', 'AT', 'CMD_IDENTIFIER'}
    assert res == expected

    input = "c := (lambda x:"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'INT', 'TEXT', 'DOTS', 'IDENTIFIER_REGEXP', 'MINUS', 'FLOAT', 'LAMBDA', 'LPAR', 'AT', 'CMD_IDENTIFIER'}
    assert res == expected

    input = "c := (lambda x: x + 1)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'LPAR'}
    assert res == expected

    input = "c := (lambda x: x + 1)(2)"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'INT', 'TRUE', 'TILDE', 'EXISTS', 'NEG_UNICODE', 'CMD_IDENTIFIER', '$END', 'ANS', 'STAR', 'IDENTIFIER_REGEXP', 'LPAR', 'POW', 'DOT', 'TEXT', 'FALSE', 'FLOAT', 'SLASH', 'MINUS', 'PLUS', 'LAMBDA', 'AT'}
    assert res == expected


def test_interactive_command_syntax():
    # print("")
    # print("__________________________________")
    # print("____ test_command_syntax() ____")
    input = '.load_csv'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'LPAR'}
    assert res == expected

    input = '.load_csv('
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TEXT', 'MINUS', 'RPAR', 'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER', 'AT', 'LAMBDA', 'FLOAT', 'LPAR', 'INT', 'PYTHON_STRING'}
    assert res == expected

    input = '.load_csv(A,'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TEXT', 'MINUS', 'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER', 'AT', 'LAMBDA', 'FLOAT', 'LPAR', 'INT', 'PYTHON_STRING'}
    assert res == expected

    input = '.load_csv(A, "http://myweb/file.csv", B)'
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'TEXT', 'MINUS', 'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER', 'ANS', 'AT', 'NEG_UNICODE', '$END', 'LAMBDA', 'FALSE', 'FLOAT', 'INT', 'LPAR', 'TILDE', 'DOT', 'TRUE'}
    assert res == expected

    input = ".load_csv()"
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'EXISTS', 'TEXT', 'MINUS', 'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER', 'ANS', 'AT', 'NEG_UNICODE', '$END', 'LAMBDA', 'FALSE', 'FLOAT', 'INT', 'LPAR', 'TILDE', 'DOT', 'TRUE'}
    assert res == expected

    input = '.load_csv(sep='
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'TEXT', 'MINUS', 'IDENTIFIER_REGEXP', 'CMD_IDENTIFIER', 'AT', 'LAMBDA', 'FLOAT', 'LPAR', 'INT', 'PYTHON_STRING'}
    assert res == expected


def test_interactive_empty_input():
    # print("")
    # print("__________________________________")
    # print("____ test_empty_input() ____")
    input = ""
    # print("***")
    # print(input)
    # print("res :")
    res = parser(input, interactive=True)
    expected = {'FLOAT', 'IDENTIFIER_REGEXP', 'MINUS', 'LAMBDA', 'EXISTS', 'NEG_UNICODE', 'DOT', 'INT', 'CMD_IDENTIFIER', 'FALSE', 'TEXT', 'LPAR', 'ANS', 'TRUE', 'AT', 'TILDE'}
    assert res == expected
