import pytest

from ...exceptions import NeuroLangException
from ...expressions import Symbol
from ..expressions import ProbabilisticPredicate

p = Symbol("p")
x = Symbol("x")


def test_probabilistc_predicate_not_predicate():
    with pytest.raises(NeuroLangException):
        ProbabilisticPredicate(p, x)
