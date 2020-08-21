"""
General notebook
================

Runs the general notebook over two subjects and plots the results

TODO: Proper description
"""

###############################################################################
# Import dependencies from auxiliar files
#

import pandas as pd

from examples.general_notebook import process_subject
from examples.general_notebook_plotting import (
    # plot_surface_map_of_individual_folds,
    # plot_surf_prob_map,
    plot_all_folds_per_subject,
    plot_individual_fold_per_subject,
    plot_stat_map_of_folds,
)

###############################################################################
# Process two subjects
#

SUBJECT_IDS = ["105923", "111514"]


results = [process_subject(s) for s in SUBJECT_IDS]


###############################################################################
# Format results
#


infos = []
for i, sid in enumerate(SUBJECT_IDS):
    infos.append(results[i][0][sid])
df_all_spatial_images = pd.DataFrame(data=infos, index=SUBJECT_IDS)
df_all_spatial_images.drop(columns=["destrieux_affines", "primaries"])


frames = []
for d1, d2 in results:
    df_d2 = pd.DataFrame(d2)
    frames.append(df_d2)

df_all = pd.concat(frames)
df_all.index.names = ["number"]
df_all


q_list = ["Q_cingulate_cnl"]

# q_list = [
#     "Q_inferior_temporal",
#     "Q_olfactory",
#     "Q_precentral",
#     "Q_superior_temporal",
#     "Q_postcentral",
#     "Q_orbital_H_shaped",
#     "Q_occipitotemporal",
#     "Q_jensen",
#     "Q_inferior_frontal",
#     "Q_intraparietal",
#     "Q_anterior_occipital",
#     "Q_subparietal",
#     "Q_superior_frontal",
#     "Q_callosomarginal",
#     "Q_superior_occipital",
#     "Q_collateral",
#     "Q_intralingual",
#     "Q_lateral_occipital",
#     "Q_middle_frontal",
#     "Q_superior_rostral",
#     "Q_cingulate",
#     "Q_paracingulate",
#     "Q_inferior_occipital",
#     "Q_anterior_parolfactory",
#     "Q_lunate",
#     "Q_cuneal",
#     "Q_frontomarginal",
#     "Q_hippocampal",
#     "Q_superior_parietal",
#     "Q_rhinal",
#     "Q_temporopolar",
#     "Q_retrocalcarine",
#     "Q_paracentral",
#     "Q_angular",
#     "Q_inferior_rostral",
#     "Q_intralimbic",
# ]

###############################################################################
# Surface plots
# -------------
#

# surface_each_subject_query_result = plot_surface_map_of_individual_folds(
#     SUBJECT_IDS,
#     q_list,
#     df_all,
#     df_all_spatial_images,
#     query_name="Q_cingulate_cnl",
# )


# surface_probability_map = plot_surf_prob_map(
#     SUBJECT_IDS,
#     q_list,
#     df_all,
#     df_all_spatial_images,
#     primary_sulcus_name="L_S_parieto_occipital",
#     query_name=None,
#     plane="medial",
# )


###############################################################################
# Volume plots
# ------------
#

###############################################################################
# All folds per subject
#

each_subject_all_folds = plot_all_folds_per_subject(
    SUBJECT_IDS, df_all, df_all_spatial_images
)


###############################################################################
# L_S_parieto_occipital per subject
#

each_subject_destrieux_sulcus = plot_individual_fold_per_subject(
    SUBJECT_IDS,
    q_list,
    df_all,
    df_all_spatial_images,
    primary_sulcus_name="L_S_parieto_occipital",
    query_name=None,
)

###############################################################################
# Cingulate from query per subject
#

each_subject_query_result = plot_individual_fold_per_subject(
    SUBJECT_IDS,
    q_list,
    df_all,
    df_all_spatial_images,
    primary_sulcus_name=None,
    query_name="Q_cingulate_cnl",
)


###############################################################################
# Stat map of Cingulate from across all subjects
#

query_prob_map = plot_stat_map_of_folds(
    SUBJECT_IDS,
    q_list,
    df_all,
    df_all_spatial_images,
    primary_sulcus_name=None,
    query_name="Q_cingulate_cnl",
)
