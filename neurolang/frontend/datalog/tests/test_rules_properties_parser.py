import pytest

from ..standard_syntax import parse_rules


def test_rules_properties_parsing():
    res = parse_rules()
    expected = {
        "expression": {
            "values": [
                "<rule>"
            ]
        }
    }
    assert res == expected
