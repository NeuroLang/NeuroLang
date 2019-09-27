# coding: utf-8
r'''
ProbDatalog Neurosynth per-term forward inference brain maps reconstruction
===========================================================================

This example reconstructs the forward inference brain maps associated with each
term in the Neurosynth [1]_ database.

.. [1] Yarkoni et al., "Large-scale automated synthesis of human functional
       neuroimaging data"

'''

import os
from collections import defaultdict
import typing

import neurosynth as ns
from neurosynth.base import imageutils
from neurosynth import Dataset
from nilearn import plotting
import numpy as np

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.expression_walker import ExpressionBasicEvaluator
from neurolang.datalog.expressions import Implication, Fact, Conjunction
from neurolang.datalog.instance import SetInstance
from neurolang.probabilistic.probdatalog import (
    ProbDatalogProgram, ProbFact, full_observability_parameter_estimation
)

if not os.path.isfile('database.txt'):
    ns.dataset.download(path='.', unpack=True)
if not os.path.isfile('dataset.pkl'):
    dataset = Dataset('database.txt')
    dataset.add_features('features.txt')
    dataset.save('dataset.pkl')
else:
    dataset = Dataset.load('dataset.pkl')

study_ids = set(dataset.feature_table.data.index)
terms_with_decent_study_count = set(
    dataset.feature_table.get_features_by_ids(
        dataset.feature_table.data.index, threshold=0.01
    )
)
n_terms = len(terms_with_decent_study_count)

selected_terms = {'reward', 'pain'}
selected_study_ids = set(
    list(dataset.feature_table.get_ids(
        features=list(selected_terms), threshold=0.5
    ))[:20]
)

image_data = dataset.get_image_data()
selected_image_data = dataset.get_image_data(ids=list(selected_study_ids))

selected_voxel_ids = set(
    list(selected_image_data.mean(axis=1).argsort()[-50:][::-1])[:5]
)

Activation = Symbol('Activation')
DoesActivate = Symbol('DoesActivate')
Voxel = Symbol('Voxel')
Term = Symbol('Term')
v = Symbol('v')
t = Symbol('t')

term_tuples = frozenset({(Term(Constant[str](term)), )
                         for term in selected_terms})
voxel_tuples = frozenset({(Voxel(Constant[int](voxel_id)), )
                          for voxel_id in selected_voxel_ids})


def study_id_to_idx(study_id):
    return np.argwhere(dataset.feature_table.data.index == study_id)[0][0]


def get_study_terms(study_id):
    mask = dataset.feature_table.data.ix[study_id] > 0.01
    return set(dataset.feature_table.data.columns[mask]) & selected_terms


def get_study_reported_voxel_ids(study_id):
    study_idx = study_id_to_idx(study_id)
    return (
        set(np.argwhere(image_data[:, study_idx] > 0).flatten()) &
        selected_voxel_ids
    )


def build_interpretation(study_id):
    terms = get_study_terms(study_id)
    voxel_ids = get_study_reported_voxel_ids(study_id)
    voxel_term_tuples = set.union(
        *([set()] + [{(Constant[int](voxel_id), Constant[str](term))
                      for voxel_id in voxel_ids}
                     for term in terms])
    )
    return SetInstance({
        Activation: frozenset(voxel_term_tuples),
        DoesActivate: frozenset(voxel_term_tuples),
        Term: frozenset(term_tuples),
        Voxel: frozenset(voxel_tuples),
    })


def build_virtual_interpretations():
    voxel_term_tuples = set.union(
        *([set()] + [{(Constant[int](voxel_id), Constant[str](term))
                      for voxel_id in selected_voxel_ids}
                     for term in selected_terms])
    )
    return [
        SetInstance({
            Activation: frozenset(voxel_term_tuples),
            DoesActivate: frozenset(voxel_term_tuples),
            Term: frozenset(term_tuples),
            Voxel: frozenset(voxel_tuples),
        }),
        SetInstance({
            Activation: frozenset(),
            DoesActivate: frozenset(),
            Term: frozenset(term_tuples),
            Voxel: frozenset(voxel_tuples),
        }),
    ]


class ProbDatalog(ProbDatalogProgram, ExpressionBasicEvaluator):
    pass


program = ProbDatalog()
for term in selected_terms:
    program.walk(Fact(Term(Constant[str](term))))
for voxel_id in selected_voxel_ids:
    program.walk(Fact(Voxel(Constant[int](voxel_id))))
program.walk(
    Implication(
        DoesActivate(v, t), Conjunction([Voxel(v),
                                         Term(t),
                                         Activation(v, t)])
    )
)

for term in selected_terms:
    for voxel_id in selected_voxel_ids:
        parameter = Symbol(f'p_{term}_{voxel_id}')
        atom = Activation(Constant[int](voxel_id), Constant[str](term))
        probfact = ProbFact(parameter, atom)
        program.walk(probfact)

interpretations = [
    build_interpretation(study_id) for study_id in selected_study_ids
] + build_virtual_interpretations()

estimations = full_observability_parameter_estimation(program, interpretations)
