# -*- coding: utf-8 -*-
r"""
NeuroLang Datalog Example based on the Destrieux Atlas and Neurosynth
=====================================================================
Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
"""
import nibabel as nib
from nilearn import datasets
from nilearn import plotting
import numpy as np

from neurolang import frontend as fe

###############################################################################
# Load the Destrieux example from nilearn
# ---------------------------------------

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux_map = nib.load(destrieux_dataset["maps"])


###############################################################################
# Initialize the NeuroLang instance and load Destrieux's cortical parcellation
# -----------------------------------------------------------------------------


nl = fe.NeurolangDL()
destrieux = nl.new_symbol(name="destrieux")
d = []
for label_number, name in destrieux_dataset["labels"]:
    if label_number == 0:
        continue
    name = name.decode()
    region = nl.create_region(destrieux_map, label=label_number)
    if region is None:
        continue
    name = name.replace("-", "_").replace(" ", "_")
    d.append((name.lower(), region))

destrieux = nl.add_tuple_set(d, name="destrieux")


###############################################################################
# Add a function to measure a region's volume
# -----------------------------------------------------------------------------


@nl.add_symbol
def region_volume(region: fe.ExplicitVBR) -> float:
    volume = len(region.voxels) * float(
        np.product(np.abs(np.linalg.eigvals(region.affine[:-1, :-1])))
    )
    return volume


###############################################################################
# Load all contiguous regions from Neurosynth that fit the term "supramarginal"
# -----------------------------------------------------------------------------


neurosynth_supramarginal = nl.load_neurosynth_term_regions(
    "supramarginal", name="neurosynth_supramarginal"
)


########################################################################
# Query all Destrieux regions that overlap with NeuroSynth supramarginal
# region having volume larger than 2500mm3 with the environment
# ----------------------------------------------------------------------

q1 = "".join("""
ans(name, region_1)
:-
destrieux(name, region_1),
neurosynth_supramarginal(region_2),
overlapping(region_1, region_2)
""".splitlines(keepends=False))

with nl.environment as e:
    nl.execute_nat_datalog_program(q1)
    q1_res = nl.solve_all()["ans"]

    # for tupl in q1_res:
        # name = tupl.value[0].value
        # region = tupl.value[1].value
        # plotting.plot_roi(region.spatial_image(), title=name)

########################################################################
# Query ids of studies related to both terms "default mode" and
# "pcc" in the Neurosynth database
# ----------------------------------------------------------------------

neurosynth_default_mode_study_id = nl.load_neurosynth_term_study_ids(
    term="default mode", name="neurosynth_default_mode_study_id"
)
neurosynth_pcc_study_id = nl.load_neurosynth_term_study_ids(
    term="pcc", name="neurosynth_pcc_study_id"
)

q2 = "".join("""
ans2(study_id)
:-
neurosynth_default_mode_study_id(study_id),
neurosynth_pcc_study_id(study_id)
""".splitlines(keepends=False))

with nl.environment as e:
    nl.execute_nat_datalog_program(q2)
    q2_res = nl.solve_all()["ans2"]
    print("matching study ids")
    for tupl in q2_res:
        study_id = tupl.value[0].value
        print(study_id)

neurosynth_study_tfidf = nl.load_neurosynth_study_tfidf_feature_for_terms(
    terms=["default mode", "pcc"], name="neurosynth_study_tfidf",
)

q3 = "".join("""
ans3(study_id, term, tfidf)
:-
neurosynth_default_mode_study_id(study_id),
neurosynth_pcc_study_id(study_id),
neurosynth_study_tfidf(study_id, term, tfidf)
""".splitlines(keepends=False))

with nl.environment as e:
    nl.execute_nat_datalog_program(q3)
    q3_res = nl.solve_all()["ans3"]
    for tupl in q3_res:
        study_id = tupl.value[0].value
        term = tupl.value[1].value
        tfidf = tupl.value[2].value
        print(study_id, term, tfidf)
