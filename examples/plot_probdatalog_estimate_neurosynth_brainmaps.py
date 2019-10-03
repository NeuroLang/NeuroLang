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


def get_dataset():
    if not os.path.isfile('database.txt'):
        ns.dataset.download(path='.', unpack=True)
    if not os.path.isfile('dataset.pkl'):
        dataset = Dataset('database.txt')
        dataset.add_features('features.txt')
        dataset.save('dataset.pkl')
    else:
        dataset = Dataset.load('dataset.pkl')
    return dataset


dataset = get_dataset()

selected_terms = {'memory', 'visual'}
selected_voxel_ids = {
    40558, 44484, 112166, 116221, 116314, 124573, 184775, 184872, 188711
}
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
            np.array(list(per_term_study_ids[term]))
        )
    ).flatten()
    for term in selected_terms
}
terms_per_study_id = {
    study_id:
    {term
     for term in selected_terms
     if study_id in per_term_study_ids[term]}
    for study_id in selected_study_ids
}

image_data = dataset.image_table.data

Activation = Symbol('Activation')
DoesActivate = Symbol('DoesActivate')
TermInStudy = Symbol('TermInStudy')
DoesAppearInStudy = Symbol('DoesAppearInStudy')
Voxel = Symbol('Voxel')
Term = Symbol('Term')
v = Symbol('v')
t = Symbol('t')

term_tuples = frozenset({(Constant[str](term), ) for term in selected_terms})
voxel_tuples = frozenset({(Constant[int](voxel_id), )
                          for voxel_id in selected_voxel_ids})


def study_id_to_idx(study_id):
    return np.argwhere(np.array(dataset.image_table.ids) == study_id
                       ).flatten()[0]


def get_study_reported_voxel_ids(study_id):
    study_idx = study_id_to_idx(study_id)
    return (
        set(np.argwhere(image_data[:, study_idx] > 0).flatten()) &
        selected_voxel_ids
    )


def build_interpretation(study_id):
    terms = terms_per_study_id[study_id]
    voxel_ids = get_study_reported_voxel_ids(study_id)
    voxel_term_tuples = set.union(
        *([set()] + [{(Constant[int](voxel_id), Constant[str](term))
                      for voxel_id in voxel_ids}
                     for term in terms])
    )
    return SetInstance({
        Activation:
        frozenset(voxel_term_tuples),
        DoesActivate:
        frozenset(voxel_term_tuples),
        TermInStudy:
        frozenset([(Constant[str](term), ) for term in terms]),
        DoesAppearInStudy:
        frozenset([(Constant[str](term), ) for term in terms]),
        Term:
        frozenset(term_tuples),
        Voxel:
        frozenset(voxel_tuples),
    })


def build_virtual_interpretations():
    voxel_term_tuples = set.union(
        *([set()] + [{(Constant[int](voxel_id), Constant[str](term))
                      for voxel_id in selected_voxel_ids}
                     for term in selected_terms])
    )
    return [
        SetInstance({
            Activation:
            frozenset(voxel_term_tuples),
            DoesActivate:
            frozenset(voxel_term_tuples),
            TermInStudy:
            frozenset([(Constant[str](term), ) for term in selected_terms]),
            DoesAppearInStudy:
            frozenset([(Constant[str](term), ) for term in selected_terms]),
            Term:
            frozenset(term_tuples),
            Voxel:
            frozenset(voxel_tuples),
        }),
        SetInstance({
            Activation: frozenset(),
            DoesActivate: frozenset(),
            TermInStudy: frozenset(),
            DoesAppearInStudy: frozenset(),
            TermInStudy: frozenset(),
            DoesAppearInStudy: frozenset(),
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
program.walk(
    Implication(DoesAppearInStudy(t), Conjunction([Term(t),
                                                   TermInStudy(t)]))
)

for term in selected_terms:
    program.walk(
        ProbFact(Symbol(f'p_{term}'), TermInStudy(Constant[str](term)))
    )
    for voxel_id in selected_voxel_ids:
        parameter = Symbol(f'p_{term}_{voxel_id}')
        atom = Activation(Constant[int](voxel_id), Constant[str](term))
        probfact = ProbFact(parameter, atom)
        program.walk(probfact)

interpretations = [
    build_interpretation(study_id) for study_id in selected_study_ids
] + build_virtual_interpretations()

estimations = full_observability_parameter_estimation(program, interpretations)


def count_voxel_term_in_interpretations(voxel_id, term):
    tupl = (Constant[int](voxel_id), Constant[str](term))
    count = sum(
        tupl in interpretation.elements[Activation]
        for interpretation in interpretations
    )
    return count


# compare estimations with neurosynth's meta analysis
results = dict()
for term in selected_terms:
    if term == 'memory':
        import pdb
        pdb.set_trace()
    ma = ns.meta.MetaAnalysis(
        dataset, list(per_term_study_ids[term]),
        list(set(selected_study_ids) - set(per_term_study_ids[term]))
    )
    assert set(ma.selected_ids) == set(per_term_study_ids[term])
    results[term] = ma.images['pAgF']

for term in selected_terms:
    for voxel_id in selected_voxel_ids:
        symbol = Symbol(f'p_{term}_{voxel_id}')
        actual = results[term][voxel_id]
        estimated = estimations[symbol] / estimations[Symbol(f'p_{term}')]
        print(symbol.name)
        print('actual = {}, estimated = {}'.format(actual, estimated))
