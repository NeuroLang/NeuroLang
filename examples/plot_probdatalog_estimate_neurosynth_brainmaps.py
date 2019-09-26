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

import neurosynth as ns
from neurosynth import Dataset
from nilearn import plotting
import numpy as np

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.expression_walker import ExpressionBasicEvaluator
from neurolang.datalog.expressions import Implication, Fact
from neurolang.datalog.instance import SetInstance
from neurolang.probabilistic.probdatalog import ProbDatalogProgram, ProbFact

ns.dataset.download(path='.', unpack=True)
dataset = Dataset('database.txt')
dataset.add_features('features.txt')

image_data = dataset.get_image_data()
study_ids = set(dataset.feature_table.data.index)
n_voxels = image_data.shape[0]
n_studies = image_data.shape[1]
n_terms = len(dataset.feature_table.data)

terms_with_decent_study_count = set(
    dataset.feature_table.get_features_by_ids(
        dataset.feature_table.data.index, threshold=0.01
    )
)


def study_id_to_idx(study_id):
    return np.argwhere(dataset.feature_table.data.index == study_id)[0][0]


def get_study_terms(study_id):
    mask = dataset.feature_table.data.ix[study_id] > 0.01
    return (
        set(dataset.feature_table.data.columns[mask]) &
        terms_with_decent_study_count
    )


def get_study_reported_voxel_ids(study_id):
    study_idx = study_id_to_idx(study_id)
    return np.argwhere(image_data[:, study_idx] > 0).flatten()


def build_interpretation(study_id):
    study_terms = get_study_terms(study_id)
    study_reported_voxel_ids = get_study_reported_voxel_ids(study_id)
    return SetInstance({
        Activation:
        frozenset(
            set.union(
                *[{(Constant[int](voxel_id), Constant[str](term))
                   for voxel_id in study_reported_voxel_ids}
                  for term in study_terms]
            )
        )
    })


class ProbDatalog(ProbDatalogProgram, ExpressionBasicEvaluator):
    pass


Activation = Symbol('Activation')
DoesActivate = Symbol('DoesActivate')
Voxel = Symbol('Voxel')
Term = Symbol('Term')

probfacts = set.union(*[
    {
        ProbFact(
            Symbol(f'p_{term}_{voxel_id}'),
            Activation(Constant[int](voxel_id), Constant[str](term))
        )
        for voxel_id in range(n_voxels)
    }
    for term in terms_with_decent_study_count
])

block = ExpressionBlock((
))
