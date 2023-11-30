import pytest

from ..standard_syntax import parse_rules


def test_rules_properties_parsing():
    res = parse_rules()
    expected = {
        'expression': {
            'values': [
                'rule'
            ]
        },
        'rule': {
            'values': [
                '<identifier> ( <arguments> ) :- <condition>',
                '<identifier> ( <arguments> ) :- <body>',
                '<query> :- <condition>',
                '<query> :- <body>'
            ]
        },
        'head': {
            'values': [
                '<identifier> ( <arguments> )'
            ]
        }
    }
    assert res == expected
