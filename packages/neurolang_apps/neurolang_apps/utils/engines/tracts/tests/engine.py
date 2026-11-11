"""The engine ``tracts`` of the apps without its atlases, for the tests and
the tools: a ``NeurolangPDL`` whose solver has the commands of the engine,
with :func:`.frontend.init_engine`, as ``engines.yaml`` puts
them together."""

from neurolang.frontend.probabilistic_frontend import (
    NeurolangPDL,
    RegionFrontendCPLogicSolver,
)
from ..frontend import TractsCommandsMixin, init_engine


class TractsSolver(TractsCommandsMixin, RegionFrontendCPLogicSolver):
    """The commands of the engine before the ``CommandsMixin`` that refuses
    the commands it does not know."""


def tracts_engine(**kwargs) -> NeurolangPDL:
    """The engine, with the options of ``NeurolangPDL``."""
    engine = NeurolangPDL(solver_class=TractsSolver, **kwargs)
    init_engine(engine)
    return engine
