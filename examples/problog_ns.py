# -*- coding: utf-8 -*-
r"""
Neurosynth's Coordinate-Based Meta-Analysis database encoded in NeuroLang
=========================================================================

Raw Neurosynth data (term-to-study associations and reported voxel activations)
are represented within the program. Queries can then be run to produce forward
inference brain maps.

"""
from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend

nl = ProbabilisticFrontend()
ns_study_id = nl.load_neurosynth_study_ids(name="ns_study_id")
ns_term_in_study = nl.load_neurosynth_term_study_associations(
    name="ns_term_in_study"
)
ns_activation = nl.load_neurosynth_reported_activations(name="ns_activation")
selected_study = nl.add_uniform_probabilistic_choice_over_set(
    ns_study_id.expression.value, name="selected_study"
)
with nl.scope as e:
    e.term_association[e.term] = (
        ns_term_in_study[e.study_id, e.term] & selected_study[e.study_id]
    )
    e.activation[e.voxel_id] = (
        ns_activation[e.study_id, e.voxel_id] & selected_study[e.study_id]
    )
    res = nl.solve_all()
