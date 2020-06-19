from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from nilearn.datasets import utils

from neurolang.frontend import NeurolangDL
from neurolang.frontend.neurosynth_utils import NeuroSynthHandler
from neurolang.probabilistic.cplogic.program import CPLogicMixin


class ProbabilisticFrontend(NeurolangDL):
    pass


nl = ProbabilisticFrontend()

# loading neurosynth data into the program
nsh = NeuroSynthHandler()
ns_term_in_study = nl.add_tuple_set(
    nsh.ns_load_term_study_associations(threshold=1e-3),
    name="ns_term_in_study",
)
ns_activation = nl.add_tuple_set(
    nsh.ns_load_reported_activations(), name="ns_activation"
)
selected_study = nl.add_uniform_probabilistic_choice_over_set(
    nsh.ns_load_all_study_ids(), name="selected_study"
)
with nl.scope as e:
    e.term_association[e.term] = (
        ns_term_in_study[e.term, e.study_id] & selected_study[e.study_id]
    )
    e.activation[e.voxel_id] = (
        ns_activation[e.study_id, e.voxel_id] & selected_study[e.study_id]
    )
    res = nl.succ_query(
        e.activation[e.voxel_id] & e.term_association["auditory"]
    )
