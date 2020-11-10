# -*- coding: utf-8 -*-
r'''
Loading and Filtering the results of the Destrieux et al. Atlas using the FMA ontology
======================================================================================

Loading the Destrieux regions and the FMA ontology into NeuroLang and
executing a simple query combining both datasets

'''

import logging
import sys
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

from neurolang import frontend as fe
from neurolang.frontend import probabilistic_frontend as pfe
from typing import Iterable
from nilearn import datasets, image, plotting
from rdflib import RDFS



"""
Data preparation
----------------
"""

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
# Load the NeuroFMA ontology

neuroFMA = datasets.utils._fetch_files(
    'neurolang',
    [
        (
            'neurofma.xml',
            'http://data.bioontology.org/ontologies/NeuroFMA/download?'
            'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
            {'move': 'neurofma.xml'}
        )
    ]
)[0]


fma_destrieux_path = datasets.utils._fetch_files(
    'neurolang',
    [
        (
            'fma_destrieux.csv',
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/fma_destrieux.csv',
            {}
        )
    ]
)[0]
fma_destrieux_rel = pd.read_csv(fma_destrieux_path, sep=';', header=None, names=['destrieux', 'fma'], dtype={'destrieux': str, 'fma': str})

###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = pfe.ProbabilisticFrontend()
nl.load_ontology(neuroFMA)


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
# Create the ontology symbols that we will use in our query

label = nl.new_symbol(name=str(RDFS.label))
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
regional_part = nl.new_symbol(name='http://sig.biostr.washington.edu/fma3.0#regional_part_of')

###############################################################################
# and load all the information within Neurolang

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

fma_destrieux = nl.add_tuple_set(
    fma_destrieux_rel.values, name='relation_destrieux_fma'
)

###############################################################################
# Probabilistic program and querying

with nl.scope as e:
    
    e.fma_related_region[e.subregion_name, e.fma_entity_name] = (
        label(e.fma_uri, e.fma_entity_name) & 
        regional_part(e.fma_region, e.fma_uri) & 
        subclass_of(e.fma_subregion, e.fma_region) &
        label(e.fma_subregion, e.subregion_name)
    )
    
    e.fma_related_region[e.recursive_name, e.fma_name] = (
        e.fma_related_region(e.fma_subregion, e.fma_name) &
        label(e.fma_uri, e.fma_subregion) &
        subclass_of(e.recursive_region, e.fma_uri) & 
        label(e.recursive_region, e.recursive_name)
    )
    
    e.destrieux_ijk[e.destrieux_name, e.i, e.j, e.k] = (
        e.destrieux_labels[e.destrieux_name, e.id_destrieux] &
        e.destrieux_image[e.i, e.j, e.k, e.id_destrieux]
    )
    
    e.region_voxels[e.i, e.j, e.k] = (
        e.fma_related_region[e.fma_subregions, 'Temporal lobe'] &
        e.relation_destrieux_fma[e.destrieux_name, e.fma_subregions] &
        e.destrieux_ijk[e.destrieux_name, e.i, e.j, e.k]
    )
    
    e.vox_term_prob[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = (
        e.region_voxels[e.i, e.j, e.k] &
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

    e.vox_cond_query[e.i, e.j, e.k, e.p] = (
        e.vox_term_prob(e.i, e.j, e.k, e.num_prob)
        & e.term_prob('auditory', e.denom_prob)
        & (e.p == (e.num_prob / e.denom_prob))
    )

    e.destrieux_region_max_probability[e.region, agg_max(e.p)] = (
       e.vox_cond_query(e.i, e.j, e.k, e.p)
       & e.destrieux_image(e.i, e.j, e.k, e.region_label)
       & e.destrieux_labels(e.region, e.region_label)
    )

    e.voxel_activation_probability[agg_create_region_overlay[e.i, e.j, e.k, e.p]] = (
        e.vox_cond_query(e.i, e.j, e.k, e.p)
    )

    res = nl.solve_all()
    img_query = res['voxel_activation_probability']
    drmp = res['destrieux_region_max_probability']
    

###############################################################################
# Results
# --------------------------------------------
# Using the ontology, we limit the results of the analysis only to regions 
# within the Temporal Lobe

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
# Per voxel associations to "Auditory" top 5%
result_image = (
    img_query
    .fetch_one()
    [0]
    .spatial_image()
)
img = result_image.get_fdata()
plot = plotting.plot_stat_map(
    result_image,
    threshold=np.percentile(img[img > 0], 95)
)
plotting.show()
