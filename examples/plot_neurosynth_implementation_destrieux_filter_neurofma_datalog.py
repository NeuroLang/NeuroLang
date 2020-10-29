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
from neurolang.frontend import probabilistic_frontend as pfe
import nibabel as nib
from nilearn import datasets, image, plotting
import numpy as np
import pandas as pd

for module_name in ('frontend', 'probabilistic'):
    logger = logging.getLogger(f'neurolang.{module_name}')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stderr))
warnings.filterwarnings("ignore")


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


fma_destrieux_path = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neuroFMA'),
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
from rdflib import RDFS

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


label = nl.new_symbol(name=str(RDFS.label))
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
regional_part = nl.new_symbol(name='http://sig.biostr.washington.edu/fma3.0#regional_part_of')

""
label

""
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

for set_symbol in (
    'activations', 'terms', 'docs', 'destrieux_image', 'destrieux_labels'
):
    print(f"#{set_symbol}: {len(nl.symbols[set_symbol].value)}")

###############################################################################
# Probabilistic program and querying

datalog_code = f'''
    fma_related_region(subregion_name, fma_entity_name) :- \
        `{str(label)}`(fma_uri, fma_entity_name), \
        `{str(regional_part)}`(fma_region, fma_uri), \
        `{str(subclass_of)}`(fma_subregion, fma_region), \
        `{str(label)}`(fma_subregion, subregion_name)
    
    fma_related_region(recursive_name, fma_name) :- \
        fma_related_region(fma_subregion, fma_name), \
        `{str(label)}`(fma_uri, fma_subregion), \
        `{str(subclass_of)}`(recursive_region, fma_uri), \
        `{str(label)}`(recursive_region, recursive_name)
    
    destrieux_ijk(destrieux_name, i, j, k) :- \
        destrieux_labels(destrieux_name, id_destrieux), \
        destrieux_image(i, j, k, id_destrieux)
    
    region_voxels(i, j, k) :- \
        fma_related_region(fma_subregions, 'Temporal lobe'), \
        relation_destrieux_fma(destrieux_name, fma_subregions), \
        destrieux_ijk(destrieux_name, i, j, k)
    
    vox_term_prob(i, j, k, PROB(i, j, k)) :- \
        region_voxels(i, j, k), \
        activations(d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ..., ..., ..., ..., i, j, k),  \
        terms(d, 'auditory'), \
        docs(d)

    term_prob(t, PROB(t)) :- \
        terms(d, t), \
        docs(d)

    region_term_prob(region, PROB(region)) :-  \
        destrieux_labels(region, region_label), \
        destrieux_image(i, j, k, region_label), \
        activations(d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,..., ..., ..., i, j, k), \
        terms(d, 'auditory'), \
        docs(d)

    vox_cond_query(i, j, k, p) :- \
        vox_term_prob(i, j, k, num_prob), \
        term_prob('auditory', denom_prob), \
        p == num_prob / denom_prob

    vox_cond_query_auditory(i, j, k, p) :- \
        vox_cond_query(i, j, k, p)

    region_cond_query(name, p) :- \
        region_term_prob(name, num_prob), \
        term_prob('auditory', denom_prob), \
        p == num_prob / denom_prob

    destrieux_region_max_probability(region, agg_max(p)) :- \
        vox_cond_query_auditory(i, j, k, p), \
        destrieux_image(i, j, k, region_label), \
        destrieux_labels(region, region_label)

    destrieux_region_image_probability(agg_create_region_overlay(i, j, k, p)) :- \
        region_cond_query(name, p), \
        destrieux_labels(name, label), \
        destrieux_image(i, j, k, label)

    voxel_activation_probability(agg_create_region_overlay(i, j, k, p)) :- \
        vox_cond_query_auditory(i, j, k, p)
'''

with nl.scope as e:
    nl.execute_datalog_program(datalog_code)
    
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
    result_image,
    # threshold=np.percentile(img[img > 0], 95)
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

""

