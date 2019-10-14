import pytest

from ...exceptions import NeuroLangException
from ...expressions import Symbol, Constant, ExpressionBlock
from ...datalog.expressions import Implication
from ..probdatalog import ProbFact
from ..probdatalog_bn import (
    TranslatorGroundedProbDatalogToBN, pfact_cpd_functor
)
from ..distributions import TableDistribution

shop = Symbol('shop')
bought = Symbol('bought')
john = Constant[str]('john')
mary = Constant[str]('mary')
spaghetti = Constant[str]('spaghetti')
fish = Constant[str]('fish')

shopping_program_code = ExpressionBlock([
    ProbFact(Constant[float](0.2), shop(john)),
    ProbFact(Constant[float](0.6), shop(mary)),
    Implication(bought(spaghetti), shop(john)),
    Implication(bought(fish), shop(mary)),
])


def test_probdatalog_bn_translation():
    translator = TranslatorGroundedProbDatalogToBN()
    bn = translator.walk(shopping_program_code)
    expected_rv_names = {
        'shop(john)',
        'shop(mary)',
        'bought(spaghetti)',
        'bought(fish)',
        'c_1',
        'c_2',
    }
    assert bn.random_variables == expected_rv_names
    assert bn.edges == {
        'shop(john)': {'c_1'},
        'shop(mary)': {'c_2'},
        'bought(spaghetti)': {'shop(john)'},
        'bought(fish)': {'shop(mary)'},
    }
    assert bn.rv_to_cpd_functor['shop(john)']({
        'c_1': 1
    }) == TableDistribution({
        0: 0.0,
        1: 1.0,
    })
    assert bn.rv_to_cpd_functor['shop(john)']({
        'c_1': 0
    }) == TableDistribution({
        0: 1.0,
        1: 0.0,
    })
    assert bn.rv_to_cpd_functor['shop(mary)']({
        'c_2': 1
    }) == TableDistribution({
        0: 0.0,
        1: 1.0,
    })
    assert bn.rv_to_cpd_functor['c_1']({}) == TableDistribution({
        0: 0.8,
        1: 0.2
    })


def test_bn_translation_unique_rv_names():
    translator = TranslatorGroundedProbDatalogToBN()
    translator.walk(ExpressionBlock([]))
    translator._add_choice_variable(
        Symbol('c'), TableDistribution({
            0: 0,
            1: 1
        })
    )
    with pytest.raises(NeuroLangException):
        translator._add_choice_variable(
            Symbol('c'), TableDistribution({
                0: 0,
                1: 1
            })
        )


def test_forbid_parents_for_choice_vars():
    with pytest.raises(NeuroLangException):
        pfact_cpd_functor(ProbFact(Constant(0.2), shop(john)))({'x': 4})
