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
from ..aggregation import ChaseAggregationMixin

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


class ChaseAggregationNR(ChaseAggregationMixin, ChaseNR):
    pass


class ChaseAggregationSN(ChaseAggregationMixin, ChaseSN):
    pass


class ChaseAggregationN(ChaseAggregationMixin, ChaseN):
    pass


DEFAULT_STRATIFIED_CHASE_CLASSES = (ChaseNR, ChaseSN, ChaseN)


class ChaseNamedStratified(ChaseStratified, ChaseNamedRelationalAlgebraMixin):
    def __init__(
        self,
        datalog_program,
        rules=None,
        chase_classes=DEFAULT_STRATIFIED_CHASE_CLASSES,
    ):
        super().__init__(datalog_program, chase_classes, rules=rules)


class Chase(ChaseNamedStratified):
    """
    Stratified Chase which will try to run (Aggregation + ChaseNonRecursive),
    (Aggregation + ChaseSemiNaive) and then (Aggregation + ChaseNaive)
    for each stratum.
    """

    def __init__(self, datalog_program, rules=None):
        chase_classes = (
            ChaseAggregationNR,
            ChaseAggregationSN,
            ChaseAggregationN,
        )
        super().__init__(
            datalog_program, rules=rules, chase_classes=chase_classes
        )
