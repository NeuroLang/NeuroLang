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
from time import time


"""
Data preparation
----------------
"""

###############################################################################
# Load the MNI atlas and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)

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
ns_terms.to_csv('term_documents.csv')
(
    ns_database
    [["x", "y", "z", "i", "j", "k", "id"]]
    .rename(columns={'id': 'pmid'})
    .to_csv("document_activations.csv")
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
def agg_percentile(x: Iterable, q: float) -> float:
    ret = np.percentile(x, 95)
    print("THR", ret)
    return ret


###############################################################################
# Loading the database

activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)


###############################################################################
# Probabilistic program and querying


datalog_code = '''
    vox_term_prob(i, j, k, PROB(i, j, k)) :- \
        activations( \
            d, ..., ..., ..., ..., 'MNI', \
            ..., ..., ..., ..., ..., ..., ..., i, j, k \
        ), \
        terms(d, 'auditory'), \
        docs(d)

    term_prob(t, PROB(t)) :- \
        terms(d, t), docs(d)

    vox_cond_query(i, j, k, p) :- \
        vox_term_prob(i, j, k, num_prob), \
        term_prob('auditory', denom_prob), \
        p == num_prob / denom_prob

    vox_cond_query_percentile(agg_percentile(p, 99)) :- \
            vox_cond_query(..., ..., ..., p)

    img(agg_create_region_overlay(i, j, k, p), thr) :- \
        vox_cond_query(i, j, k, p), vox_cond_query_percentile(thr)
'''

with nl.scope as e:
    nl.execute_datalog_program(datalog_code)

    start = time()
    img_query = nl.query(
       (e.x, e.thr),
       e.img(e.x, e.thr)
    )
    print(f'{time() - start}')


###############################################################################
# Plotting results
# --------------------------------------------

img, thr = img_query.fetch_one()
result_image = img.spatial_image()
img = result_image.get_fdata()
plot = plotting.plot_stat_map(
    result_image, threshold=thr
)
plotting.show()

""

