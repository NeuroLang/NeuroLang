import os
from collections import defaultdict

from neurosynth import meta, decode, network, Dataset
from nilearn import plotting
import numpy as np

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.solver_datalog_naive import Fact, Implication
from neurolang.probabilistic.ppdl import DeltaTerm
from neurolang.probabilistic.graphical_model import (
    TableCPDGraphicalModelSolver
)

base_dir = '/Users/viovene/data/neurosynth'

# dataset = Dataset(os.path.join(base_dir, 'database.txt'))
# dataset.add_features(os.path.join(base_dir, 'features.txt'))
# dataset.save(os.path.join(base_dir, 'dataset.pkl'))
dataset = Dataset.load(os.path.join(base_dir, 'dataset.pkl'))

term = 'emotion'
ids = dataset.get_studies(features=term)
feature_data = dataset.get_image_data(ids)

SelectedStudy = Symbol('SelectedStudy')
StudyActivation = Symbol('StudyActivation')
Activation = Symbol('Activation')
Magic = Symbol('Magic')
Term = Symbol('Term')
Voxel = Symbol('Voxel')
Bernoulli = Symbol('Bernoulli')
OneInK = Symbol('OneInK')
v = Symbol('v')
t = Symbol('t')
s = Symbol('s')
a = Symbol('a')

random_voxel_id = int(np.random.choice(range(feature_data.shape[0])))

program_input = {
    Fact(Magic(Constant[int](random_voxel_id), Constant[str](term)))
}

program = ExpressionBlock((
    Implication(
        SelectedStudy(DeltaTerm(OneInK, (Constant[int](len(ids)), ))),
        Constant[bool](True)
    ),
    Implication(
        StudyActivation(
            v, t, s, DeltaTerm(Bernoulli, (Constant[float](0.5), ))
        ),
        SelectedStudy(s) & Magic(v, t)
    ),
    Implication(
        Activation(s, t, a),
        SelectedStudy(s) & StudyActivation(v, t, s, a)
    ),
))

evidence = set()
for s, study_id in enumerate(ids):
    voxel_value = feature_data[random_voxel_id, s]
    evidence.add(
        Fact(
            StudyActivation(
                Constant[int](random_voxel_id), Constant[str](term),
                Constant[int](int(study_id)), Constant[int](int(voxel_value))
            )
        )
    )

solver = TableCPDGraphicalModelSolver()
solver.walk(program)
