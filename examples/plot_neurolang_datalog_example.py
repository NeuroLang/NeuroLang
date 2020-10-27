# -*- coding: utf-8 -*-
r'''
NeuroLang Datalog Example based on the Destrieux Atlas and Neurosynth
=====================================================================


Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
'''
import logging
from operator import contains as contains_
from typing import Iterable
import sys

import nibabel as nib
from nilearn import datasets
from nilearn import plotting
import numpy as np
import pandas as pd

from neurolang import frontend as fe

logger = logging.getLogger('neurolang.datalog.chase')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


###############################################################################
# Load the Destrieux example from nilearn
# ---------------------------------------

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux_map = nib.load(destrieux_dataset['maps'])


###############################################################################
# Initialize the NeuroLang instance and load Destrieux's cortical parcellation
# -----------------------------------------------------------------------------


nl = fe.NeurolangDL()
destrieux_tuples = []
for label_number, name in destrieux_dataset['labels']:
    if label_number == 0:
        continue
    name = name.decode()
    region = nl.create_region(destrieux_map, label=label_number)
    if region is None:
        continue
    name = name.replace('-', '_').replace(' ', '_')
    destrieux_tuples.append((name.lower(), region))

destrieux = nl.add_tuple_set(destrieux_tuples, name='destrieux')


###############################################################################
# Add a function to measure a region's volume
# -----------------------------------------------------------------------------

@nl.add_symbol
def region_volume(region: fe.ExplicitVBR) -> float:
    volume = (
        len(region.voxels) *
        float(np.product(np.abs(np.linalg.eigvals(region.affine[:-1, :-1]))))
    )
    return volume

contains = nl.add_symbol(contains_, name='contains')

########################################################################
# Query all Destrieux regions having volume larger than 2500mm3
# ----------------------------------------------------------------------


with nl.scope as e:

    e.anterior_to_precentral[e.name, e.region] = (
        e.destrieux(e.name, e.region) &
        e.destrieux('l_g_precentral', e.region_) &
        contains(e.region, (..., e.j, ...)) &
        contains(e.region_, (..., e.j_, ...)) &
        (e.j_ > e.j)
    )

    res = nl.query(
            (e.name, e.region),
            e.anterior_to_precentral(e.name, e.region)
            # & (region_volume(e.region) > 2500)
    )


for name, region in res:
    plotting.plot_roi(region.spatial_image(), title=name)


########################################################################
# Query all Destrieux regions that overlap with NeuroSynth activations
# present when the term "auditory" is mentioned in the document
# ----------------------------------------------------------------------

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
        np.linalg.inv(destrieux_map.affine),
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
    .query('TfIdf > 0')
    [['pmid', 'term', 'TfIdf']]
)
ns_terms.to_csv('term_documents.csv')
(
    ns_database
    [["x", "y", "z", "i", "j", "k", "id"]]
    .rename(columns={'id': 'pmid'})
    .to_csv("document_activations.csv")
)


###############################################################################
# Deterministic Logic Programming in NeuroLang
# --------------------------------------------

nl = fe.NeurolangDL()


###############################################################################
# Adding new aggregation function to build a region overlay

@nl.add_symbol
def agg_create_region(
    i: Iterable, j: Iterable, k: Iterable
) -> fe.ExplicitVBR:
    voxels = np.c_[i, j, k]
    voxels = voxels[(voxels < destrieux_map.shape).all(1)]
    return fe.ExplicitVBR(
        voxels, destrieux_map.affine,
        image_dim=destrieux_map.shape
    )


###############################################################################
# Loading the database

activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
destrieux = nl.add_tuple_set(destrieux_tuples, name='destrieux')


###############################################################################
# Query regions in Destrieux that have at least an activation reported when
# the word "auditory" is in the article


with nl.scope as e:
    e.activation_for_pain[e.i, e.j, e.k] = (
        e.activations[
            e.article, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ] &
        e.terms[e.article, 'pain', e.tfidf] &
        (e.tfidf > .7)
    )
    e.activation_for_pain[e.i, e.j, e.k] = (
        e.activations[
            e.article, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ] &
        e.terms[e.article, 'nociception', e.tfidf] &
        (e.tfidf > .7)
    )

    e.pain_region[e.agg_create_region(e.i, e.j, e.k)] = (
        e.activation_for_pain(e.i, e.j, e.k)
    )

    res = nl.query(
            (e.region,),
            e.pain_region[e.region]
    )


for region, in res:
    plotting.plot_roi(region.spatial_image())
