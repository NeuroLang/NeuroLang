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


class ChaseNR(
    ChaseNonRecursive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral
):
    pass


class ChaseN(ChaseNaive, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


DEFAULT_STRATIFIED_CHASE_CLASSES = (ChaseNR, ChaseSN, ChaseN)


class Chase(ChaseStratified, ChaseNamedRelationalAlgebraMixin):
    def __init__(
        self,
        datalog_program,
        rules=None,
        chase_classes=DEFAULT_STRATIFIED_CHASE_CLASSES,
    ):
        super().__init__(datalog_program, chase_classes, rules=rules)