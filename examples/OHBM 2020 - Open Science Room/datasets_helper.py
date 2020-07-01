import os
import urllib.request
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.datasets import utils

from neurolang import frontend as fe


def ns_prob_joint_term_study(nsh, term):
    studies_term = nsh.ns_study_tfidf_feature_for_terms(term)
    df = pd.DataFrame(studies_term, columns=["study", "term", "prob"])
    p_doc = 1 / len(nsh.ns_load_all_study_ids())
    df["prob"] = df["prob"] * p_doc
    return df[df.prob > 0][["prob", "term", "study"]]


def ns_prob_joint_voxel_study(nsh):
    data = nsh.ns_load_reported_activations()
    df = pd.DataFrame(data, columns=["study", "voxel"])
    p_doc = 1 / len(nsh.ns_load_all_study_ids())
    df["prob"] = p_doc
    df = df.astype({"prob": float, "voxel": int, "study": int})
    return df[["prob", "voxel", "study"]]


def load_auditory_datasets(nl):

    # Load ontology
    d_onto = utils._get_dataset_dir("ontologies", data_dir="neurolang_data")

    if not os.path.exists(d_onto + "/neurofma_fma3.0.owl"):
        print("Downloading FMA ontology")
        url = "http://data.bioontology.org/ontologies/NeuroFMA/submissions/1/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
        urllib.request.urlretrieve(url, d_onto + "/neurofma_fma3.0.owl")
        print("Dataset created in neurolang_data/ontologies")

    # Load Neurosynth
    print("Loading NeuroSynth dataset")
    nsh = fe.neurosynth_utils.NeuroSynthHandler()

    nl.add_uniform_probabilistic_choice_over_set(
        nsh.ns_load_all_study_ids(), name="p_study"
    )

    nl.add_probabilistic_facts_from_tuples(
        [
            (p, t, s)
            for p, t, s in ns_prob_joint_term_study(nsh, ["auditory"]).values
        ],
        name="p_term_study",
    )
    nl.add_probabilistic_facts_from_tuples(
        [(p, v, s) for p, v, s in ns_prob_joint_voxel_study(nsh).values],
        name="p_voxel_study",
    )

    nsh = fe.neurosynth_utils.NeuroSynthHandler()
    ns_ds = nsh.ns_load_dataset()
    it = ns_ds.image_table

    masked_ = it.masker.unmask(np.arange(it.data.shape[0]))
    nnz = masked_.nonzero()
    vox_id_MNI = np.c_[
        masked_[nnz].astype(int),
        nib.affines.apply_affine(it.masker.volume.affine, np.transpose(nnz)),
        [
            fe.ExplicitVBR(
                [v],
                affine_matrix=it.masker.volume.affine,
                image_dim=it.masker.volume.shape,
            )
            for v in zip(*nnz)
        ],
    ]

    ns_vox_id_MNI = nl.add_tuple_set(vox_id_MNI, name="ns_vox_id_MNI")

    # Load Destrieux
    print("Loading Destrieux atlas")
    dd = datasets.fetch_atlas_destrieux_2009()
    destrieux_to_ns_mni = image.resample_to_img(
        dd["maps"], it.masker.volume, interpolation="nearest"
    )
    dd_data = destrieux_to_ns_mni.get_fdata()
    dd_unmaskes = np.where(destrieux_to_ns_mni.get_fdata() > 0)

    xyz_to_dd_region = []
    for v in zip(*dd_unmaskes):
        region = dd_data[v[0]][v[1]][v[2]]
        xyz_to_dd_region.append((v, region))

    xyz_to_ns_region = []
    for n, _, _, _, region in vox_id_MNI:
        xyz_to_ns_region.append((tuple(region.voxels[0]), n))

    dd_labels = []
    for n, name in dd["labels"]:
        dd_labels.append(
            (
                n,
                name.decode("UTF-8")
                .replace(" ", "_")
                .replace("-", "_")
                .lower(),
            )
        )

    # Deterministic facts

    xyz_ns = nl.add_tuple_set(
        [(xyz[0], xyz[1], xyz[2], id_) for xyz, id_ in xyz_to_ns_region],
        name="xyz_neurosynth",
    )
    xyz_dd = nl.add_tuple_set(
        [(xyz[0], xyz[1], xyz[2], id_) for xyz, id_ in xyz_to_dd_region],
        name="xyz_destrieux",
    )
    dd_label = nl.add_tuple_set(dd_labels, name="destrieux_labels")

    ds = destrieux_name_to_fma_relations()
    nl.add_tuple_set(
        [(dsname, onto) for dsname, onto in ds], name="relation_destrieux_fma"
    )


def destrieux_name_to_fma_relations():
    return [
        ("l_g_and_s_frontomargin", "Left frontomarginal gyrus"),
        ("l_g_and_s_occipital_inf", "Left inferior occipital gyrus"),
        ("l_g_and_s_paracentral", "Left paracentral lobule"),
        ("l_g_and_s_subcentral", "Left subcentral gyrus"),
        (
            "l_g_and_s_transv_frontopol",
            "Left superior transverse frontopolar gyrus",
        ),
        ("l_g_and_s_cingul_ant", "Left anterior cingulate gyrus"),
        ("l_g_and_s_cingul_mid_ant", "Left anterior middle cingulate gyrus"),
        ("l_g_and_s_cingul_mid_post", "Left posterior middle cingulate gyrus"),
        (
            "l_g_cingul_post_dorsal",
            "Dorsal segment of left posterior middle cingulate gyrus",
        ),
        (
            "l_g_cingul_post_ventral",
            "Ventral segment of left posterior middle cingulate gyrus",
        ),
        ("l_g_cuneus", "Left cuneus"),
        (
            "l_g_front_inf_opercular",
            "Opercular part of left inferior frontal gyrus",
        ),
        (
            "l_g_front_inf_orbital",
            "Orbital part of left inferior frontal gyrus",
        ),
        (
            "l_g_front_inf_triangul",
            "Triangular part of left inferior frontal gyrus",
        ),
        ("l_g_front_middle", "Left middle frontal gyrus"),
        ("l_g_front_sup", "Left superior frontal gyrus"),
        ("l_g_ins_lg_and_s_cent_ins", "Left central insular sulcus"),
        ("l_g_ins_lg_and_s_cent_ins", "Left long insular gyrus"),
        ("l_g_insular_short", "Short insular gyrus"),
        ("l_g_occipital_middleLeft", " 	Left lateral occipital gyrus"),
        ("l_g_occipital_sup", "Left superior occipital gyrus"),
        ("l_g_oc_temp_lat_fusifor", "Left fusiform gyrus"),
        ("l_g_oc_temp_med_lingual", "Left lingual gyrus"),
        ("l_g_oc_temp_med_parahip", "Left parahippocampal gyrus"),
        ("l_g_orbital", "Left orbital gyrus"),
        ("l_g_pariet_inf_angular", "Left angular gyrus"),
        ("l_g_pariet_inf_supramar", "Left supramarginal gyrus"),
        ("l_g_parietal_sup", "Left superior parietal lobule"),
        ("l_g_postcentral", "Left postcentral gyrus"),
        ("l_g_precentral", "Left precentral gyrus"),
        ("l_g_precuneus", "Left precuneus"),
        ("l_g_rectus", "Left straight gyrus"),
        ("l_g_subcallosal", "Left paraterminal gyrus"),
        ("l_g_temp_sup_g_t_transv", "Left transverse temporal gyrus"),
        ("l_g_temp_sup_lateral", "Left superior temporal gyrus"),
        ("l_g_temp_sup_plan_polar", "Left superior temporal gyrus"),
        ("l_g_temp_sup_plan_tempo", "Left superior temporal gyrus"),
        ("l_g_temporal_inf", "Left inferior temporal gyrus"),
        ("l_g_temporal_middle", "Left middle temporal gyrus"),
        (
            "l_lat_fis_ant_horizont",
            "Anterior horizontal limb of left lateral sulcus",
        ),
        (
            "l_lat_fis_ant_vertical",
            "Anterior ascending limb of left lateral sulcus",
        ),
        ("l_lat_fis_post", "Posterior ascending limb of left lateral sulcus"),
        ("l_lat_fis_post", "Left lateral sulcus"),
        ("l_pole_occipital", "Left occipital pole"),
        ("l_pole_temporal", "Left temporal pole"),
        ("l_s_calcarine", "Left Calcarine sulcus"),
        ("l_s_central", "Left central sulcus"),
        ("l_s_cingul_marginalis", "Left marginal sulcus"),
        ("l_s_circular_insula_ant", "Circular sulcus of left insula"),
        ("l_s_circular_insula_inf", "Circular sulcus of left insula"),
        ("l_s_circular_insula_sup", "Circular sulcus of left insula"),
        ("l_s_collat_transv_ant", "Left collateral sulcus"),
        ("l_s_collat_transv_post", "Left collateral sulcus"),
        ("l_s_front_inf", "Left inferior frontal sulcus"),
        ("l_s_front_sup", "Left superior frontal sulcus"),
        ("l_s_intrapariet_and_p_trans", "Left intraparietal sulcus"),
        ("l_s_oc_middle_and_lunatus", "Left lunate sulcus"),
        ("l_s_oc_sup_and_transversal", "Left transverse occipital sulcus"),
        ("l_s_occipital_ant", "Left anterior occipital sulcus"),
        ("l_s_oc_temp_lat", "Left occipitotemporal sulcus"),
        ("l_s_oc_temp_med_and_lingual", "Left intralingual sulcus"),
        ("l_s_orbital_lateral", "Left orbital sulcus"),
        ("l_s_orbital_med_olfact", "Left olfactory sulcus"),
        ("l_s_orbital_h_shaped", "Left transverse orbital sulcus"),
        ("l_s_orbital_h_shaped", "Left orbital sulcus"),
        ("l_s_parieto_occipital", "Left parieto-occipital sulcus"),
        ("l_s_pericallosal", "Left callosal sulcus"),
        ("l_s_postcentral", "Left postcentral sulcus"),
        ("l_s_precentral_inf_part", "Left precentral sulcus"),
        ("l_s_precentral_sup_part", "Left precentral sulcus"),
        ("l_s_suborbital", "Left fronto-orbital sulcus"),
        ("l_s_subparietal", "Left subparietal sulcus"),
        ("l_s_temporal_inf", "Left inferior temporal sulcus"),
        ("l_s_temporal_sup", "Left superior temporal sulcus"),
        ("l_s_temporal_transverse", "Left transverse temporal sulcus"),
        ("r_g_and_s_frontomargin", "Right frontomarginal gyrus"),
        ("r_g_and_s_occipital_inf", "Right inferior occipital gyrus"),
        ("r_g_and_s_paracentral", "Right paracentral lobule"),
        ("r_g_and_s_subcentral", "Right subcentral gyrus"),
        (
            "r_g_and_s_transv_frontopol",
            "Right superior transverse frontopolar gyrus",
        ),
        ("r_g_and_s_cingul_ant", "Right anterior cingulate gyrus"),
        ("r_g_and_s_cingul_mid_ant", "Right anterior middle cingulate gyrus"),
        (
            "r_g_and_s_cingul_mid_post",
            "Right posterior middle cingulate gyrus",
        ),
        (
            "r_g_cingul_post_dorsal",
            "Dorsal segment of right posterior middle cingulate gyrus",
        ),
        (
            "r_g_cingul_post_ventral",
            "Ventral segment of right posterior middle cingulate gyrus",
        ),
        ("r_g_cuneus", "Right cuneus"),
        (
            "r_g_front_inf_opercular",
            "Opercular part of right inferior frontal gyrus",
        ),
        (
            "r_g_front_inf_orbital",
            "Orbital part of right inferior frontal gyrus",
        ),
        (
            "r_g_front_inf_triangul",
            "Triangular part of right inferior frontal gyrus",
        ),
        ("r_g_front_middle", "Right middle frontal gyrus"),
        ("r_g_front_sup", "Right superior frontal gyrus"),
        ("r_g_ins_lg_and_s_cent_ins", "Right central insular sulcus"),
        ("r_g_ins_lg_and_s_cent_ins", "Right long insular gyrus"),
        ("r_g_insular_short", "Right short insular gyrus"),
        ("r_g_occipital_middle", "Right lateral occipital gyrus"),
        ("r_g_occipital_sup", "Right superior occipital gyrus"),
        ("r_g_oc_temp_lat_fusifor", "Right fusiform gyrus"),
        ("r_g_oc_temp_med_lingual", "Right lingual gyrus"),
        ("r_g_oc_temp_med_parahip", "Right parahippocampal gyrus"),
        ("r_g_orbital", "Right orbital gyrus"),
        ("r_g_pariet_inf_angular", "Right angular gyrus"),
        ("r_g_pariet_inf_supramar", "Right supramarginal gyrus"),
        ("r_g_parietal_sup", "Right superior parietal lobule"),
        ("r_g_postcentral", "Right postcentral gyrus"),
        ("r_g_precentral", "Right precentral gyrus"),
        ("r_g_precuneus", "Right precuneus"),
        ("r_g_rectus", "Right straight gyrus"),
        ("r_g_subcallosal", "Right paraterminal gyrus"),
        ("r_g_temp_sup_g_t_transv", "Right transverse temporal gyrus"),
        ("r_g_temp_sup_lateral", "Right superior temporal gyrus"),
        ("r_g_temp_sup_plan_polar", "Right superior temporal gyrus"),
        ("r_g_temp_sup_plan_tempo", "Right superior temporal gyrus"),
        ("r_g_temporal_inf", "Right inferior temporal gyrus"),
        ("r_g_temporal_middle", "Right middle temporal gyrus"),
        (
            "r_lat_fis_ant_horizont",
            "Anterior horizontal limb of right lateral sulcus",
        ),
        (
            "r_lat_fis_ant_vertical",
            "Anterior ascending limb of right lateral sulcus",
        ),
        ("r_lat_fis_post", "Right lateral sulcus"),
        ("r_lat_fis_post", "Posterior ascending limb of right lateral sulcus"),
        ("r_pole_occipital", "Right occipital pole"),
        ("r_pole_temporal", "Right temporal pole"),
        ("r_s_calcarine", "Right Calcarine sulcus"),
        ("r_s_central", "Right central sulcus"),
        ("r_s_cingul_marginalis", "Right marginal sulcus"),
        ("r_s_circular_insula_ant", "Circular sulcus of Right insula"),
        ("r_s_circular_insula_inf", "Circular sulcus of Right insula"),
        ("r_s_circular_insula_sup", "Circular sulcus of Right insula"),
        ("r_s_collat_transv_ant", "Right collateral sulcus"),
        ("r_s_collat_transv_post", "Right collateral sulcus"),
        ("r_s_front_inf", "Right inferior frontal sulcus"),
        ("r_s_front_sup", "Right superior frontal sulcus"),
        ("r_s_intrapariet_and_p_trans", "Right intraparietal sulcus"),
        ("r_s_oc_middle_and_lunatus", "Right lunate sulcus"),
        ("r_s_oc_sup_and_transversal", "Right transverse occipital sulcus"),
        ("r_s_occipital_ant", "Right anterior occipital sulcus"),
        ("r_s_oc_temp_lat", "Right occipitotemporal sulcus"),
        ("r_s_oc_temp_med_and_lingual", "Right intralingual sulcus"),
        ("r_s_orbital_lateral", "Right orbital sulcus"),
        ("r_s_orbital_med_olfact", "Right olfactory sulcus"),
        ("r_s_orbital_h_shaped", "Right orbital sulcus"),
        ("r_s_orbital_h_shaped", "Right transverse orbital sulcus"),
        ("r_s_parieto_occipital", "Right parieto-occipital sulcus"),
        ("r_s_pericallosal", "Right callosal sulcus"),
        ("r_s_postcentral", "Right postcentral sulcus"),
        ("r_s_precentral_inf_part", "Right precentral sulcus"),
        ("r_s_precentral_sup_part", "Right precentral sulcus"),
        ("r_s_suborbital", "Right fronto-orbital sulcus"),
        ("r_s_subparietal", "Right subparietal sulcus"),
        ("r_s_temporal_inf", "Right inferior temporal sulcus"),
        ("r_s_temporal_sup", "Right superior temporal sulcus"),
        ("r_s_temporal_transverse", "Right transverse temporal sulcus"),
    ]


def load_pain_datasets(nl):
    d_onto = utils._get_dataset_dir("ontologies", data_dir="neurolang_data")

    if not os.path.exists(d_onto + "/IOBC_1_4_0.xrdf"):
        print("Downloading IOBC ontology")
        url = "http://data.bioontology.org/ontologies/IOBC/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf"
        urllib.request.urlretrieve(url, d_onto + "/IOBC_1_4_0.xrdf")
        print("Dataset created in neurolang_data/ontologies")

    d_neurosynth = utils._get_dataset_dir(
        "neurosynth", data_dir="neurolang_data"
    )

    f_neurosynth = utils._fetch_files(
        d_neurosynth,
        [
            (
                f,
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            )
            for f in ("database.txt", "features.txt")
        ],
        verbose=True,
    )

    database = pd.read_csv(f_neurosynth[0], sep="\t")
    features = pd.read_csv(f_neurosynth[1], sep="\t")

    features_normalised = features.melt(
        id_vars=features.columns[0],
        var_name="term",
        value_vars=features.columns[1:],
        value_name="tfidf",
    ).query("tfidf > 0")

    nsh = fe.neurosynth_utils.NeuroSynthHandler()
    ns_ds = nsh.ns_load_dataset()
    it = ns_ds.image_table
    vox_ids, study_ids_ix = it.data.nonzero()
    study_ids = ns_ds.image_table.ids[study_ids_ix]
    study_id_vox_id = np.transpose([study_ids, vox_ids])
    masked_ = it.masker.unmask(np.arange(it.data.shape[0]))
    nnz = masked_.nonzero()
    vox_id_MNI = np.c_[
        masked_[nnz].astype(int),
        nib.affines.apply_affine(it.masker.volume.affine, np.transpose(nnz)),
        [
            fe.ExplicitVBR(
                [v],
                affine_matrix=it.masker.volume.affine,
                image_dim=it.masker.volume.shape,
            )
            for v in zip(*nnz)
        ],
    ]

    ns_tfidf_pmid_term = nl.add_tuple_set(
        features_normalised[["tfidf", "pmid", "term"]].values,
        name="ns_tfidf_pmid_term",
    )
    # ns_activations = nl.add_tuple_set(database[['id', 'x', 'y', 'z', 'space']].values, name='ns_activations')
    ns_activations_by_id = nl.add_tuple_set(
        study_id_vox_id, name="ns_activations_by_id"
    )
    ns_vox_id_MNI = nl.add_tuple_set(vox_id_MNI, name="ns_vox_id_MNI")
