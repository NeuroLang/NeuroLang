import numpy as np
from neurolang.frontend.neurosynth_utils import NeuroSynthHandler
from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend

nl = ProbabilisticFrontend()

# loading neurosynth data into the program
nsh = NeuroSynthHandler()
study_ids = nsh.ns_load_all_study_ids()
sampled_study_ids = set(
    (int(study_id),)
    for study_id in study_ids[
        np.random.randint(low=0, high=len(study_ids), size=10)
    ].flatten()
)
ns_term_in_study = nl.add_tuple_set(
    nsh.ns_load_term_study_associations(
        threshold=1e-3, study_ids=sampled_study_ids
    ),
    name="ns_term_in_study",
)
ns_activation = nl.add_tuple_set(
    nsh.ns_load_reported_activations(), name="ns_activation"
)
selected_study = nl.add_uniform_probabilistic_choice_over_set(
    sampled_study_ids, name="selected_study"
)
with nl.scope as e:
    e.term_association[e.term] = (
        ns_term_in_study[e.study_id, e.term] & selected_study[e.study_id]
    )
    e.activation[e.voxel_id] = (
        ns_activation[e.study_id, e.voxel_id] & selected_study[e.study_id]
    )
    res = nl.solve_all()
