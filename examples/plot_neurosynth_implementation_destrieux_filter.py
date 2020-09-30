# -*- coding: utf-8 -*-
r'''
NeuroLang Example based Implementing a NeuroSynth Query
====================================================

'''


from nilearn import datasets, image, plotting
import pandas as pd
from neurolang import frontend as fe
from neurolang.frontend import probabilistic_frontend as pfe
from typing import Iterable
import nibabel as nib
import numpy as np


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
destrieux_resampled = image.resample_img(destrieux, mni_t1_4mm.affine)
destrieux_resampled_data = np.asanyarray(destrieux_resampled.dataobj, dtype=np.int32)
destrieux_voxels_ijk = destrieux_resampled_data.nonzero()
destrieux_voxels_value = destrieux_resampled_data[destrieux_voxels_ijk]
destrieux_table = pd.DataFrame(np.transpose(destrieux_voxels_ijk), columns=['i', 'j', 'k'])
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
    'neurolang',
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
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = pfe.ProbabilisticFrontend()


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
destrieux_labels = nl.add_tuple_set(destrieux_label_names, name='destrieux_labels')


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
        & e.term_prob("auditory", e.denom_prob)
        & (e.p == (e.num_prob / e.denom_prob))
    )

    e.vox_cond_query_auditory[e.i, e.j, e.k, e.p] = (
        e.vox_cond_query[e.i, e.j, e.k, e.p]
    )

    e.region_cond_query[e.name, e.p] = (
        e.region_term_prob[e.name, e.num_prob]
        & e.term_prob("auditory", e.denom_prob)
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

    e.img[agg_create_region_overlay[e.i, e.j, e.k, e.p]] = (
        e.vox_cond_query_auditory(e.i, e.j, e.k, e.p)
    )

    img_query = nl.query(
       (e.x,),
       e.img(e.x)
    )

    dest_query = nl.query(
        (e.x,),
        e.destrieux_region_image_probability(e.x)
    )

    drcp = nl.query((e.r, e.p), e.region_cond_query(e.r, e.p))
    drmp = nl.query((e.r, e.p), e.destrieux_region_max_probability(e.r, e.p))


###############################################################################
# Plotting results
# --------------------------------------------

print(drmp.as_pandas_dataframe().sort_values('p'))
print(drcp.as_pandas_dataframe().sort_values('p'))
result_image = (
    img_query
    .fetch_one()
    [0]
    .spatial_image()
)
img = result_image.get_fdata()
# plot = plotting.plot_stat_map(
#    result_image, threshold=np.percentile(img[img > 0], 95)
# )
# plotting.show()

img = dest_query.fetch_one()[0].spatial_image().get_fdata()
plot = plotting.plot_stat_map(
    dest_query.fetch_one()[0].spatial_image(),
    threshold=np.percentile(img[img > 0], 95)
)
plotting.show()
