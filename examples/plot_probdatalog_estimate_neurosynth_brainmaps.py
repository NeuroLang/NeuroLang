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

selected_terms = {"cognitive control", "default mode network"}
selected_voxel_ids = set(list(range(dataset.image_table.data.shape[0])))
per_term_study_ids = {
    t: set(dataset.feature_table.get_ids(features=[t], threshold=0.2))
    for t in selected_terms
}
selected_study_ids = list(set.union(*per_term_study_ids.values()))
selected_study_indices = np.argwhere(
    np.isin(np.array(dataset.image_table.ids), np.array(selected_study_ids))
).flatten()
per_term_study_indices = {
    term: np.argwhere(
        np.isin(
            np.array(selected_study_ids),
            np.array(list(per_term_study_ids[term])),
        )
    ).flatten()
    for term in selected_terms
}
terms_per_study_id = {
    study_id: {
        term for term in selected_terms if study_id in per_term_study_ids[term]
    }
    for study_id in selected_study_ids
}

image_data = dataset.image_table.data.todense()

Activation = Symbol("Activation")
DoesActivate = Symbol("DoesActivate")
TermInStudy = Symbol("TermInStudy")
DoesAppearInStudy = Symbol("DoesAppearInStudy")
Voxel = Symbol("Voxel")
Term = Symbol("Term")
v = Symbol("v")
t = Symbol("t")

term_tuples = frozenset({(Constant[str](term),) for term in selected_terms})
voxel_tuples = frozenset(
    {(Constant[int](voxel_id),) for voxel_id in selected_voxel_ids}
)


def study_id_to_idx(study_id):
    return np.argwhere(
        np.array(dataset.image_table.ids) == study_id
    ).flatten()[0]


def get_study_reported_voxel_ids(study_id):
    study_idx = study_id_to_idx(study_id)
    return (
        set(np.argwhere(image_data[:, study_idx] > 0).flatten().astype(int))
        & selected_voxel_ids
    )


def build_interpretation(study_id):
    terms = terms_per_study_id[study_id]
    voxel_ids = get_study_reported_voxel_ids(study_id)
    voxel_term_tuples = set.union(
        *(
            [set()]
            + [
                {
                    (Constant[int](voxel_id), Constant[str](term))
                    for voxel_id in voxel_ids
                }
                for term in terms
            ]
        )
    )
    return SetInstance(
        {
            Activation: frozenset(voxel_term_tuples),
            DoesActivate: frozenset(voxel_term_tuples),
            TermInStudy: frozenset([(Constant[str](term),) for term in terms]),
            DoesAppearInStudy: frozenset(
                [(Constant[str](term),) for term in terms]
            ),
            Term: frozenset(term_tuples),
            Voxel: frozenset(voxel_tuples),
        }
    )


def build_virtual_interpretations():
    voxel_term_tuples = set.union(
        *(
            [set()]
            + [
                {
                    (Constant[int](voxel_id), Constant[str](term))
                    for voxel_id in selected_voxel_ids
                }
                for term in selected_terms
            ]
        )
    )
    return [
        SetInstance(
            {
                Activation: frozenset(voxel_term_tuples),
                DoesActivate: frozenset(voxel_term_tuples),
                TermInStudy: frozenset(
                    [(Constant[str](term),) for term in selected_terms]
                ),
                DoesAppearInStudy: frozenset(
                    [(Constant[str](term),) for term in selected_terms]
                ),
                Term: frozenset(term_tuples),
                Voxel: frozenset(voxel_tuples),
            }
        ),
        SetInstance(
            {
                Activation: frozenset(),
                DoesActivate: frozenset(),
                TermInStudy: frozenset(),
                DoesAppearInStudy: frozenset(),
                TermInStudy: frozenset(),
                DoesAppearInStudy: frozenset(),
                Term: frozenset(term_tuples),
                Voxel: frozenset(voxel_tuples),
            }
        ),
    ]


class ProbDatalog(ProbDatalogProgram, ExpressionBasicEvaluator):
    pass


term_facts = [Fact(Term(Constant(term))) for term in selected_terms]
voxel_facts = [Fact(Voxel(Constant(vid))) for vid in selected_voxel_ids]
extensional_database = term_facts + voxel_facts
intensional_database = [
    Implication(
        DoesActivate(v, t), Conjunction([Voxel(v), Term(t), Activation(v, t)])
    ),
    Implication(DoesAppearInStudy(t), Conjunction([Term(t), TermInStudy(t)])),
]
voxel_term_probfacts = [
    Implication(
        ProbabilisticPredicate(
            Symbol(f"p_{voxel_id}_{term}"),
            Activation(Constant(voxel_id), Constant(term)),
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
probabilistic_database = voxel_term_probfacts + term_probfacts

program_code = ExpressionBlock(
    extensional_database + intensional_database + probabilistic_database
)

interpretations = [
    build_interpretation(study_id) for study_id in selected_study_ids
]

estimations = full_observability_parameter_estimation(
    program_code, interpretations
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
