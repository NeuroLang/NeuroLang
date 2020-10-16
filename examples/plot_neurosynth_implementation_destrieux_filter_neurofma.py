# -*- coding: utf-8 -*-
r'''
NeuroLang Example based Implementing a NeuroSynth Query
====================================================

'''

import logging
import sys
from typing import Iterable
import warnings

from neurolang import frontend as fe
import nibabel as nib
from nilearn import datasets, image, plotting
import numpy as np
import pandas as pd

logger = logging.getLogger('neurolang.probabilistic')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))
warnings.filterwarnings("ignore")


###############################################################################
# Data preparation
# ----------------

###############################################################################
# Load the MNI template and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)


###############################################################################
# Load Destrieux's atlas
destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux = nib.load(destrieux_dataset['maps'])
destrieux_resampled = image.resample_img(
    destrieux, mni_t1_4mm.affine, interpolation='nearest'
)
destrieux_resampled_data = np.asanyarray(
    destrieux_resampled.dataobj, dtype=np.int32
)
destrieux_voxels_ijk = destrieux_resampled_data.nonzero()
destrieux_voxels_value = destrieux_resampled_data[destrieux_voxels_ijk]
destrieux_table = pd.DataFrame(
    np.transpose(destrieux_voxels_ijk), columns=['i', 'j', 'k']
)
destrieux_table['label'] = destrieux_voxels_value

destrieux_label_names = []
for label_number, name in destrieux_dataset['labels']:
    if label_number == 0:
        continue
    name = name.decode()
    name = name.replace('-', '_').replace(' ', '_')
    destrieux_label_names.append((name.lower(), label_number))


###############################################################################
# Load the NeuroSynth database

ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neurosynth'),
    [
        (
            'database.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
        (
            'features.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
    ]
)

ns_database = pd.read_csv(ns_database_fn, sep=f'\t')
ijk_positions = (
    np.round(nib.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[['x', 'y', 'z']].values.astype(float)
    )).astype(int)
)
ns_database['i'] = ijk_positions[:, 0]
ns_database['j'] = ijk_positions[:, 1]
ns_database['k'] = ijk_positions[:, 2]

ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
ns_docs = ns_features[['pmid']].drop_duplicates()
ns_terms = (
    pd.melt(
            ns_features,
            var_name='term', id_vars='pmid', value_name='TfIdf'
       )
    .query('TfIdf > 1e-3')[['pmid', 'term']]
)


###############################################################################
# Load the NeuroFMA ontology

neuroFMA = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neuroFMA'),
    [
        (
            'neurofma.xml',
            'http://data.bioontology.org/ontologies/NeuroFMA/download?'
            'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
            {'move': 'neurofma.xml'}
        )
    ]
)[0]


###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = fe.NeurolangPDL()
nl.load_ontology(neuroFMA)
for r in nl.current_program:
    print(r)

###############################################################################
# Adding new aggregation function to build a region overlay
@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> fe.ExplicitVBR:
    voxels = np.c_[i, j, k]
    return fe.ExplicitVBROverlay(
        voxels, mni_t1_4mm.affine, p,
        image_dim=mni_t1_4mm.shape
    )


@nl.add_symbol
def agg_max(i: Iterable) -> float:
    return np.max(i)


###############################################################################
# Loading the database

activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)
destrieux_image = nl.add_tuple_set(
    destrieux_table.values,
    name='destrieux_image'
)
destrieux_labels = nl.add_tuple_set(
    destrieux_label_names, name='destrieux_labels'
)

for set_symbol in (
    'activations', 'terms', 'docs', 'destrieux_image', 'destrieux_labels'
):
    print(f"#{set_symbol}: {len(nl.symbols[set_symbol].value)}")

###############################################################################
# Probabilistic program and querying


with nl.scope as e:
    e.vox_term_prob[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ] &
        e.terms[e.d, 'auditory'] &
        e.docs[e.d]
    )

    e.term_prob[e.t, e.PROB[e.t]] = (
        e.terms[e.d, e.t] &
        e.docs[e.d]
    )

    e.region_term_prob[e.region, e.PROB[e.region]] = (
        e.destrieux_labels(e.region, e.region_label)
        & e.destrieux_image(e.i, e.j, e.k, e.region_label)
        & e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ]
        & e.terms[e.d, 'auditory']
        & e.docs[e.d]
    )

    e.vox_cond_query[e.i, e.j, e.k, e.p] = (
        e.vox_term_prob(e.i, e.j, e.k, e.num_prob)
        & e.term_prob('auditory', e.denom_prob)
        & (e.p == (e.num_prob / e.denom_prob))
    )

    e.vox_cond_query_auditory[e.i, e.j, e.k, e.p] = (
        e.vox_cond_query[e.i, e.j, e.k, e.p]
    )

    e.region_cond_query[e.name, e.p] = (
        e.region_term_prob[e.name, e.num_prob]
        & e.term_prob('auditory', e.denom_prob)
        & (e.p == (e.num_prob / e.denom_prob))

    )

    e.destrieux_region_max_probability[e.region, agg_max(e.p)] = (
        e.vox_cond_query_auditory(e.i, e.j, e.k, e.p)
        & e.destrieux_image(e.i, e.j, e.k, e.region_label)
        & e.destrieux_labels(e.region, e.region_label)
    )

    e.destrieux_region_image_probability[agg_create_region_overlay(e.i, e.j, e.k, e.p)] = (
        e.region_cond_query[e.name, e.p] &
        e.destrieux_labels[e.name, e.label] &
        e.destrieux_image[e.i, e.j, e.k, e.label]
    )

    e.voxel_activation_probability[agg_create_region_overlay[e.i, e.j, e.k, e.p]] = (
        e.vox_cond_query_auditory(e.i, e.j, e.k, e.p)
    )

    res = nl.solve_all()
    img_query = res['voxel_activation_probability']
    dest_query = res['destrieux_region_image_probability']
    drcp = res['region_cond_query']
    drmp = res['destrieux_region_max_probability']

###############################################################################
# Results
# --------------------------------------------

###############################################################################
# Maximum probability per voxel in the region that the region has an activation
# when an article has the word "Auditory"
(
    drmp
    .as_pandas_dataframe()
    .sort_values(drmp.columns[-1], ascending=False)
    .head()
)


###############################################################################
# Conditional probabilityper region that the region has an activation
# when an article has the word "Auditory"
(
    drcp
    .as_pandas_dataframe()
    .sort_values(drcp.columns[-1], ascending=False)
    .head()
)

###############################################################################
# Per voxel associations to "Auditory" top 5%
result_image = (
    img_query
    .fetch_one()
    [0]
    .spatial_image()
)
img = result_image.get_fdata()
plot = plotting.plot_stat_map(
    result_image, threshold=np.percentile(img[img > 0], 95)
)
plotting.show()

###############################################################################
# Per region associations to "Auditory" top 15%


img = dest_query.fetch_one()[0].spatial_image().get_fdata()
plot = plotting.plot_stat_map(
    dest_query.fetch_one()[0].spatial_image(),
    display_mode='y',
    threshold=np.percentile(img[img > 0], 85),
    cmap='YlOrRd'
)
plotting.show()
