from .general import (
    ChaseGeneral,
    ChaseNaive,
    ChaseNode,
    ChaseNonRecursive,
    ChaseSemiNaive,
    ChaseStratified,
    NeuroLangNonLinearProgramException,
    NeuroLangProgramHasLoopsException,
)
from .mgu import ChaseMGUMixin
from .relational_algebra import (
    ChaseNamedRelationalAlgebraMixin,
    ChaseRelationalAlgebraPlusCeriMixin,
)

__all__ = [
    "ChaseGeneral",
    "ChaseNode",
    "ChaseNaive",
    "ChaseSemiNaive",
    "ChaseStratified",
    "NeuroLangNonLinearProgramException",
    "ChaseMGUMixin",
    "ChaseNonRecursive",
    "ChaseRelationalAlgebraPlusCeriMixin",
    "ChaseNamedRelationalAlgebraMixin",
    "NeuroLangProgramHasLoopsException",
]


class ChaseSN(ChaseSemiNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass

class ChaseNR(ChaseNonRecursive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass

class ChaseN(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass

class Chase(ChaseStratified, ChaseNamedRelationalAlgebraMixin):

    def __init__(self, datalog_program, rules=None, chase_classes=[
        ChaseNR, ChaseSN, ChaseN
    ]):
        super().__init__(datalog_program, chase_classes, rules=rules)