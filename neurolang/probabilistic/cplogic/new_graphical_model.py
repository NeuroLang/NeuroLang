import enum

from ...expression_pattern_matching import add_match
from ...expressions import Constant, FunctionApplication
from ...relational_algebra_provenance import ProvenanceAlgebraSet
from .noisy_or_probability_provenance import NoisyORProbabilityProvenanceSolver


class CPD(enum.Enum):
    SingleNaryChoice = enum.auto()
    MultiBinaryChoice = enum.auto()
    DeterministicSingleNaryChoiceParent = enum.auto()
    DeterministicMultiBinaryChoiceParent = enum.auto()
    AndMultiBinaryParents = enum.auto()


class CPDApplication(FunctionApplication):
    def __init__(self, cpd, value, parent_values):
        super().__init__(cpd, (value,) + parent_values)

    @property
    def cpd(self):
        return self.functor

    @property
    def value(self):
        return self.args[0]

    @property
    def parent_values(self):
        if len(self.args) > 1:
            return tuple(self.args[1:])
        return tuple()


class CPLogicGraphicalModelQuerySolver(NoisyORProbabilityProvenanceSolver):
    @add_match(
        CPDApplication(
            Constant[CPD](SingleNaryChoice),
            ProvenanceAlgebraSet,
        )
    )
    def single_nary_choice(self, cpd_app):
        pass

    @add_match(
        CPDApplication(
            Constant[CPD](CPD.DeterministicSingleNaryChoiceParent),
            ProvenanceAlgebraSet,
            (ProvenanceAlgebraSet,),
        )
    )
    def deterministic_single_nary_choice_parent(self, cpd_app):
        pass
