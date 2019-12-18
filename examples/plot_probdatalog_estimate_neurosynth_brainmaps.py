# coding: utf-8
r"""
ProbDatalog Neurosynth per-term forward inference brain maps reconstruction
===========================================================================

This example reconstructs the forward inference brain maps associated with each
term in the Neurosynth [1]_ database.

.. [1] Yarkoni et al., "Large-scale automated synthesis of human functional
       neuroimaging data"

"""

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
)


def get_dataset():
    if not os.path.isfile("database.txt"):
        ns.dataset.download(path=".", unpack=True)
    if not os.path.isfile("dataset.pkl"):
        dataset = Dataset("database.txt")
        dataset.add_features("features.txt")
        dataset.save("dataset.pkl")
    else:
        dataset = Dataset.load("dataset.pkl")
    return dataset


dataset = get_dataset()


def study_ids_to_study_indices(study_ids):
    return np.argwhere(
        np.isin(np.array(dataset.image_table.ids), np.array(study_ids))
    ).flatten()


selected_terms = np.array(["cognitive control", "default mode"])
selected_voxel_ids = np.arange(dataset.image_table.data.shape[0])

term_study_dfs = []
for term in selected_terms:
    study_indices = study_ids_to_study_indices(
        dataset.feature_table.get_ids(features=[term], threshold=0.05)
    )
    term_study_dfs.append(
        pd.DataFrame(
            {
                "term": np.repeat(term, study_indices.shape[0]),
                "study_id": study_indices,
            }
        )
    )
term_study_df = pd.concat(term_study_dfs)

activations_df = pd.DataFrame(
    np.argwhere(
        dataset.image_table.data[:, term_study_df.study_id.values] > 0
    ),
    columns=["voxel_id", "study_id"],
)
activations_df["study_id"] = term_study_df.study_id.values[
    activations_df.study_id.values
]
activations_df = activations_df.loc[
    activations_df.voxel_id.isin(selected_voxel_ids)
]

big_data_table = term_study_df.merge(activations_df, on="study_id")


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

n_interpretations = len(set(big_data_table.study_id))
interpretations = dict()
interpretations[CoActivation] = AlgebraSet(
    columns=["v", "t", "__interpretation_id__"],
    iterable=big_data_table.rename(
        columns={
            "study_id": "__interpretation_id__",
            "voxel_id": "v",
            "term": "t",
        }
    )[["v", "t", "__interpretation_id__"]],
)
interpretations[DoesCoActivate] = interpretations[CoActivation]
interpretations[TermInStudy] = AlgebraSet(
    columns=["t", "__interpretation_id__"],
    iterable=big_data_table[["term", "study_id"]]
    .drop_duplicates()
    .rename(columns={"term": "t", "study_id": "__interpretation_id__"}),
)
interpretations[TermDoesAppearInStudy] = interpretations[TermInStudy]
interpretations[VoxelReported] = AlgebraSet(
    columns=["v", "__interpretation_id__"],
    iterable=big_data_table[["voxel_id", "study_id"]]
    .drop_duplicates()
    .rename(columns={"voxel_id": "v", "study_id": "__interpretation_id__"}),
)
interpretations[VoxelIsReported] = interpretations[VoxelReported]

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
            Symbol(f"p_{voxel_id}_{term}"),
            CoActivation(Constant(voxel_id), Constant(term)),
        ),
        Constant[bool](True),
    )
    for voxel_id, term in itertools.product(selected_voxel_ids, selected_terms)
]
term_probfacts = [
    Implication(
        ProbabilisticPredicate(
            Symbol(f"p_{term}"), TermInStudy(Constant(term))
        ),
        Constant[bool](True),
    )
    for term in selected_terms
]
voxel_probfacts = [
    Implication(
        ProbabilisticPredicate(
            Symbol(f"p_{voxel_id}"), VoxelReported(Constant(voxel_id))
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

estimations = full_observability_parameter_estimation(
    program_code, interpretations, n_interpretations
)

# grounded = ReplaceSymbolsByConstants(
# {
# parameter: Constant(estimation)
# for parameter, estimation in estimations.items()
# }
# ).walk(program_code)

# # compare estimations with neurosynth's meta analysis
# results = dict()
# for term in selected_terms:
# ma = ns.meta.MetaAnalysis(
# dataset,
# list(per_term_study_ids[term]),
# list(set(selected_study_ids) - set(per_term_study_ids[term])),
# )
# assert set(ma.selected_ids) == set(per_term_study_ids[term])
# results[term] = ma.images["pAgF"]

# for term in selected_terms:
# for voxel_id in selected_voxel_ids:
# symbol = Symbol(f"p_{term}_{voxel_id}")
# actual = results[term][voxel_id]
# estimated = estimations[symbol] / estimations[Symbol(f"p_{term}")]
# print(symbol.name)
# print("actual = {}, estimated = {}".format(actual, estimated))
