import os
from collections import defaultdict

from neurosynth import meta, decode, network, Dataset
from nilearn import plotting
import numpy as np

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.solver_datalog_naive import (
    Fact, Implication, SolverNonRecursiveDatalogNaive, Query, DatalogBasic
)
from neurolang.probabilistic.ppdl import (
    DeltaTerm, concatenate_to_expression_block
)
from neurolang.solver_datalog_extensional_db import ExtensionalDatabaseSolver
from neurolang.expression_walker import ExpressionBasicEvaluator
from neurolang.probabilistic.graphical_model import (
    TableCPDGraphicalModelSolver, ConditionalProbabilityQuery, FactSet
)
from neurolang.datalog_chase import build_chase_solution

base_dir = '/Users/viovene/data/neurosynth'

# dataset = Dataset(os.path.join(base_dir, 'database.txt'))
# dataset.add_features(os.path.join(base_dir, 'features.txt'))
# dataset.save(os.path.join(base_dir, 'dataset.pkl'))
dataset = Dataset.load(os.path.join(base_dir, 'dataset.pkl'))

term = 'emotion'
ids = dataset.get_studies(features=term)
study_id_to_study_idx = {study_id: idx for idx, study_id in enumerate(ids)}
feature_data = dataset.get_image_data(ids)

Study = Symbol('Study')
StudyGroup = Symbol('StudyGroup')
SelectedStudy = Symbol('SelectedStudy')
VoxelStudy = Symbol('VoxelStudy')
VoxelActivated = Symbol('VoxelActivated')
StudyActivation = Symbol('StudyActivation')
Activation = Symbol('Activation')
Magic = Symbol('Magic')
Term = Symbol('Term')
TermViewed = Symbol('TermViewed')
TermStudy = Symbol('TermStudy')
Voxel = Symbol('Voxel')
Bernoulli = Symbol('bernoulli')
OneInK = Symbol('OneInK')
v = Symbol('v')
t = Symbol('t')
s = Symbol('s')
a = Symbol('a')
p = Symbol('p')
y = Symbol('y')


class DatalogSolver(DatalogBasic, ExpressionBasicEvaluator):
    pass


study_database = ExpressionBlock(
    tuple(Fact(Study(Constant[int](study_id))) for study_id in ids)
)

groupby_program = ExpressionBlock((Implication(StudyGroup(s), Study(s)), ) +
                                  study_database.expressions)
solver = DatalogSolver()
solver.walk(groupby_program)
solution = build_chase_solution(solver)
groups = solution[StudyGroup].value

interpretation_generator_program = ExpressionBlock((
    Implication(TermViewed(t, Constant[int](1)),
                TermStudy(s, t) & Study(s)),
    Implication(
        VoxelActivated(v, Constant[int](1)),
        VoxelStudy(s, v) & Study(s)
    ),
))


def solution_to_datalog_code(solution):
    expressions = []
    for predicate, tuple_set in solution.items():
        for t in tuple_set.value:
            expressions.append(Fact(predicate(*t.value)))
    return ExpressionBlock(expressions)


brain_map = np.zeros(feature_data.shape[0])

voxel_id = 0
interpretations = []
for study_id in groups:
    study_id = study_id.value[0]
    program_input = ExpressionBlock((
        Fact(Study(study_id)),
        Fact(TermStudy(study_id, Constant[str](term))),
    ))
    if feature_data[voxel_id, study_id_to_study_idx[study_id.value]] == 1:
        program_input = concatenate_to_expression_block(
            program_input,
            ExpressionBlock(
                (Fact(VoxelStudy(study_id, Constant[int](voxel_id))), )
            )
        )
    solver = DatalogSolver()
    solver.walk(interpretation_generator_program)
    solver.walk(program_input)
    solution = build_chase_solution(solver)
    interpretation = solution_to_datalog_code(solution)
    interpretation = ExpressionBlock([
        expression for expression in interpretation.expressions
        if expression not in program_input.expressions
    ])
    interpretations.append(interpretation)

ppdl_program = ExpressionBlock((
    Implication(
        TermViewed(t, DeltaTerm(Bernoulli, (Constant[float](0.5), ))),
        Term(t) & Magic(t)
    ),
    Implication(
        VoxelActivated(v, DeltaTerm(Bernoulli, (Constant[float](0.5), ))),
        Voxel(v) & Magic(v)
    ),
    Fact(Magic(Constant[str](term))),
    Fact(Magic(Constant[int](voxel_id))),
    Fact(Voxel(Constant[int](voxel_id))),
    Fact(Term(Constant[str](term))),
))

evidence = Constant[FactSet]({
    Fact(TermViewed(Constant[str](term), Constant[int](1)))
})
query = ConditionalProbabilityQuery(evidence)
solver = TableCPDGraphicalModelSolver()
solver.walk(ppdl_program)
solution = solver.conditional_probability_query_resolution(query)

prob_activation = np.sum([
    prob for outcome, prob in solution.value.table.items() if
    Fact(VoxelActivated(Constant[int](voxel_id), Constant[int](1))) in outcome
])

solutions = []
for interpretation in interpretations:
    solver = TableCPDGraphicalModelSolver()
    solver.walk(ppdl_program)
    evidence = Constant[FactSet](frozenset(interpretation.expressions))
    query = ConditionalProbabilityQuery(evidence)
    solution = solver.conditional_probability_query_resolution(query)
    solutions.append(solution)

expected_probability = [
    np.sum([
        int(
            Fact(VoxelActivated(Constant[int](voxel_id), Constant[int]
                                (1))) in value
        ) * prob for value, prob in solution.value.table.items()
    ]) for solution in solutions
]

brain_map[voxel_id] = expected_probability
