import os
from collections import defaultdict

from neurosynth import meta, decode, network, Dataset
from nilearn import plotting
import numpy as np

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.solver_datalog_naive import (
    Fact, Implication, SolverNonRecursiveDatalogNaive, Query
)
from neurolang.probabilistic.ppdl import DeltaTerm
from neurolang.solver_datalog_extensional_db import ExtensionalDatabaseSolver
from neurolang.expression_walker import ExpressionBasicEvaluator
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

Study = Symbol('Study')
StudyGroup = Symbol('StudyGroup')
SelectedStudy = Symbol('SelectedStudy')
StudyActivation = Symbol('StudyActivation')
Activation = Symbol('Activation')
Magic = Symbol('Magic')
Term = Symbol('Term')
TermViewed = Symbol('TermViewed')
TermStudy = Symbol('TermStudy')
Voxel = Symbol('Voxel')
Bernoulli = Symbol('Bernoulli')
OneInK = Symbol('OneInK')
v = Symbol('v')
t = Symbol('t')
s = Symbol('s')
a = Symbol('a')
p = Symbol('p')
y = Symbol('y')


class DatalogSolver(
    SolverNonRecursiveDatalogNaive, ExtensionalDatabaseSolver,
    ExpressionBasicEvaluator
):
    pass


groupby_program = ExpressionBlock(
    (Implication(StudyGroup(s), Study(s)), ) +
    tuple(Fact(Study(Constant[int](int(study_id)))) for study_id in ids)
)
solver = DatalogSolver()
solver.walk(groupby_program)
groups = solver.walk(Query(s, StudyGroup(s))).value

interpretation_generator_program = ExpressionBlock((
    Implication(TermViewed(t, Constant[int](1)),
                TermStudy(s, t) & Study(s)),
))

for study_id in groups:
    program_input = ExpressionBlock((
        Fact(Study(study_id)),
        Fact(TermStudy(study_id, Constant[str](term))),
    ))
    solver = DatalogSolver()
    solver.walk(interpretation_generator_program)
    solver.walk(program_input)
    interpretation = solver.walk(Query((t, y), TermViewed(t, y)))

    ppdl_program = ExpressionBlock((
        Implication(
            TermViewed(t, DeltaTerm(Bernoulli, (p, ))),
            Term(t) & Magic(t)
        ),
        Fact(Magic(t)),
    ))
    solver = TableCPDGraphicalModelSolver()
    solver.walk(ppdl_program)

# random_voxel_id = int(np.random.choice(range(feature_data.shape[0])))

# program_input = {
# Fact(Magic(Constant[int](random_voxel_id), Constant[str](term)))
# }

# program = ExpressionBlock((
# Implication(
# SelectedStudy(DeltaTerm(OneInK, (Constant[int](len(ids)), ))),
# Constant[bool](True)
# ),
# Implication(
# StudyActivation(
# v, t, s, DeltaTerm(Bernoulli, (Constant[float](0.5), ))
# ),
# SelectedStudy(s) & Magic(v, t)
# ),
# Implication(
# Activation(s, t, a),
# SelectedStudy(s) & StudyActivation(v, t, s, a)
# ),
# ))

# evidence = set()
# for s, study_id in enumerate(ids):
# voxel_value = feature_data[random_voxel_id, s]
# evidence.add(
# Fact(
# StudyActivation(
# Constant[int](random_voxel_id), Constant[str](term),
# Constant[int](int(study_id)), Constant[int](int(voxel_value))
# )
# )
# )

# solver = TableCPDGraphicalModelSolver()
# solver.walk(program)
