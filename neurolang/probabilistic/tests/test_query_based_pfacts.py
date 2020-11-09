import pytest

from ...expressions import Constant, Symbol
from ...logic import Conjunction, Implication, Union
from ..cplogic.program import CPLogicProgram
from ..query_resolution import (
    QueryBasedProbFactToDetRule,
    compute_probabilistic_solution,
)


class CPLogicWithQueryBasedPfactProgram(
    QueryBasedProbFactToDetRule,
    CPLogicProgram,
):
    pass


def test_query_based_pfact():
    pass


def test_prevent_combination_of_query_based_and_set_based():
    pass
