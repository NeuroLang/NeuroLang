import os
from collections import defaultdict
import typing
import itertools

import neurosynth as ns
from neurosynth.base import imageutils
from neurosynth import Dataset
from nilearn import plotting
import numpy as np
import pandas as pd

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.expression_walker import (
    ExpressionBasicEvaluator,
    ReplaceSymbolsByConstants,
)
from neurolang.datalog.expressions import Implication, Fact, Conjunction
from neurolang.datalog.instance import SetInstance
from neurolang.probabilistic.expressions import ProbabilisticPredicate
from neurolang.probabilistic.probdatalog import ProbDatalogProgram
from neurolang.probabilistic.probdatalog_gm import (
    full_observability_parameter_estimation,
    AlgebraSet,
    succ_query,
    ExtendedRelationalAlgebraSolver,
    DivideColumns,
    NaturalJoin,
)

estimations = pd.read_hdf(
    "examples/global_connectivity/estimations.h5", "estimations"
).fillna(0)

estimations = dict(
    zip(estimations.__parameter_name__, estimations.__parameter_estimate__)
)

selected_terms = np.array(["cognitive control", "default mode"])
selected_voxel_ids = np.arange(228453)

CoActivation = Symbol("CoActivation")
DoesCoActivate = Symbol("DoesCoActivate")
TermInStudy = Symbol("TermInStudy")
TermDoesAppearInStudy = Symbol("TermDoesAppearInStudy")
VoxelReported = Symbol("VoxelReported")
VoxelIsReported = Symbol("VoxelIsReported")
Voxel = Symbol("Voxel")
Term = Symbol("Term")
v = Symbol("v")
t = Symbol("t")

term_facts = [Fact(Term(Constant(term))) for term in selected_terms]
voxel_facts = [Fact(Voxel(Constant(vid))) for vid in selected_voxel_ids]
extensional_database = term_facts + voxel_facts
intensional_database = [
    Implication(
        DoesCoActivate(v, t),
        Conjunction([Voxel(v), Term(t), CoActivation(v, t)]),
    ),
    Implication(
        TermDoesAppearInStudy(t), Conjunction([Term(t), TermInStudy(t)])
    ),
    Implication(VoxelIsReported(v), Conjunction([Voxel(v), VoxelReported(v)])),
]
voxel_term_probfacts = [
    Implication(
        ProbabilisticPredicate(
            Constant[float](estimations.get(f"p_{voxel_id}_{term}", 0.0)),
            CoActivation(Constant(voxel_id), Constant(term)),
        ),
        Constant[bool](True),
    )
    for voxel_id, term in itertools.product(selected_voxel_ids, selected_terms)
]
term_probfacts = [
    Implication(
        ProbabilisticPredicate(
            Constant[float](estimations.get(f"p_{term}", 0.0)),
            TermInStudy(Constant(term)),
        ),
        Constant[bool](True),
    )
    for term in selected_terms
]
voxel_probfacts = [
    Implication(
        ProbabilisticPredicate(
            Constant[float](estimations.get(f"p_{voxel_id}", 0.0)),
            VoxelReported(Constant(voxel_id)),
        ),
        Constant[bool](True),
    )
    for voxel_id in selected_voxel_ids
]
probabilistic_database = (
    voxel_term_probfacts + term_probfacts + voxel_probfacts
)

program_code = ExpressionBlock(
    extensional_database + intensional_database + probabilistic_database
)

succ1 = succ_query(program_code, CoActivation(v, t))
succ2 = succ_query(program_code, VoxelReported(v))

succ1_prob_col = next(c for c in succ1.value.columns if c.startswith("fresh"))
succ2_prob_col = next(c for c in succ2.value.columns if c.startswith("fresh"))

result = ExtendedRelationalAlgebraSolver({}).walk(
    DivideColumns(
        NaturalJoin(succ1, succ2),
        Constant(succ1_prob_col),
        Constant(succ2_prob_col),
    )
)

result.value._container.to_hdf(
    "examples/global_connectivity/estimated_pFgA.h5", 'estimation'
)
