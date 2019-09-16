from .general import (ChaseGeneral, ChaseGeneralToModularAdapter, ChaseNaive,
                      ChaseNode, ChaseSemiNaive, ChaseStepModular,
                      NeuroLangNonLinearProgramException)
from .mgu import ChaseMGUMixin
from .relational_algebra import (ChaseNamedRelationalAlgebraMixin,
                                 ChaseRelationalAlgebraMixin)

__all__ = [
    "ChaseGeneral", "ChaseNode", "ChaseNaive", "ChaseSemiNaive",
    "NeuroLangNonLinearProgramException", "ChaseMGUMixin",
    "ChaseRelationalAlgebraMixin", "ChaseNamedRelationalAlgebraMixin"
]


# class Chase(ChaseGeneral, ChaseSemiNaive, ChaseRelationalAlgebraMixin):
#    pass

class Chase(
    # ChaseStepModular, ChaseGeneralToModularAdapter,
    ChaseSemiNaive, ChaseRelationalAlgebraMixin,
    ChaseGeneral,
):
    pass
