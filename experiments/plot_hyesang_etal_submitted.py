# -*- coding: utf-8 -*-
r'''
Reverse Inference From Hippocampal-IPS Circuits to Cognition
============================================================

'''

# %%
# Imports

import re
import warnings
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import datasets, image
from rdflib import RDFS
from sklearn.model_selection import KFold

from neurolang import frontend as fe

warnings.filterwarnings("ignore")


# %%
# Data preparation
# ----------------


###############################################################################
# Load the MNI template and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)


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
# Load the CogAt ontology

cogAt = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('CogAt'),
    [
        (
            'cogat.xml',
            'http://data.bioontology.org/ontologies/COGAT/download?'
            'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
            {'move': 'cogat.xml'}
        )
    ]
)[0]

# %%
###############################################################################
# Load Hyesang et al's ROIs

hyesang_etal_files = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('hyesang_et_al'),
    [
        (
            'hyesang_etal_wip/' + filename,
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/hyesang_etal_wip.tar.gz',
            {'uncompress': True}
        )
        for filename in [
            'combined_BN_L_Hipp_roi_1mm_bin.nii.gz',
            'combined_BN_R_Hipp_roi_1mm_bin.nii.gz',
            'combined_BN_L_PPC_targets_roi_1mm_bin.nii.gz',
            'combined_BN_R_PPC_targets_1mmiso_roi_1mm_bin.nii.gz'
        ]
    ]
)

paths = hyesang_etal_files
region_names = {
    'combined_BN_L_Hipp_roi': 'Left Hippocampus',
    'combined_BN_R_Hipp_roi': 'Right Hippocampus',
    'combined_BN_L_PPC_targets_roi': 'Left IPS',
    'combined_BN_R_PPC_targets': 'Right IPS'
}

df_imgs = []
for path in paths:
    name = path.split('/')[-1].split('_1mm')[0]
    img = image.load_img(path)
    img = image.resample_img(
        img, mni_t1_4mm.affine, interpolation='nearest'
    )

    img_data = img.get_fdata()
    img_unmaskes = np.nonzero(img_data)

    coords = []
    for v in zip(*img_unmaskes):
        prob = img_data[v[0]][v[1]][v[2]]
        coords.append((tuple(v) + tuple([prob])))

    df_img = pd.DataFrame(coords, columns=['i', 'j', 'k', 'value'])
    df_img['name'] = region_names.get(name, name)
    df_imgs.append(df_img)

df_imgs = pd.concat(df_imgs)


# %%
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = fe.NeurolangPDL()


##############################################################################
# Adding builtin functions
# ~~~~~~~~~~~~~~~~~~~~~~~~

@nl.add_symbol
def word_lower(name: str) -> str:
    return name.lower()


@nl.add_symbol
def str_match(string: str, pattern: str) -> bool:
    return re.match(pattern, string) is not None


@nl.add_symbol
def mean(iterable: Iterable) -> float:
    return np.mean(iterable)


@nl.add_symbol
def std(iterable: Iterable) -> float:
    return np.std(iterable)

##############################################################################
# Load Data into the NeuroLang engine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the CogAt Ontology

nl.load_ontology(cogAt)
part_of = nl.new_symbol(name='http://www.obofoundry.org/ro/ro.owl#part_of')
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
label = nl.new_symbol(name=str(RDFS.label))
hasTopConcept = nl.new_symbol(name='http://www.w3.org/2004/02/skos/core#hasTopConcept')


# Load the NeuroSynth database
activations = nl.add_tuple_set(ns_database, name='activations')
terms = nl.add_tuple_set(ns_terms, name='terms')

doc_indices = ns_docs.index.values
docs = nl.add_uniform_probabilistic_choice_over_set(ns_docs, name='docs')

# Generate 20 folds to compute confidence intervals for
# meta-analytic probability estimations
kfold = KFold(n_splits=20, shuffle=True, random_state=42)

ns_doc_folds = pd.concat(
    ns_docs.iloc[train].assign(fold=[i] * len(train))
    for i, (train, _) in enumerate(kfold.split(ns_docs))
)
doc_folds = nl.add_tuple_set(ns_doc_folds, name='doc_folds')

vinod_image = nl.add_tuple_set(
    df_imgs[['name', 'i', 'j', 'k']],
    name='vinod_image'
)

##############################################################################
# Code and execute the probabilistic query in NeuroLang
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

with nl.scope as e:

    e.ontology_terms[e.onto_name] = (
        hasTopConcept[e.uri, e.cp] &
        label[e.uri, e.onto_name]
    )

    e.filtered_terms[e.lower_name] = (
        e.ontology_terms[e.term] &
        (e.lower_name == word_lower[e.term])
    )

    e.filtered_regions[e.d, e.name, e.i, e.j, e.k] = (
        e.vinod_image[e.name, e.i, e.j, e.k] &
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ]
    )

    e.term_prob[
        e.t, e.name1, e.name2, e.fold,
        e.PROB[e.t, e.name1, e.name2, e.fold]
    ] = (
        (
            e.docs[e.d] & e.doc_folds[e.d, e.fold] &
            e.filtered_terms[e.t] & e.terms[e.d, e.t]
        ) // (
            e.docs[e.d] & e.doc_folds[e.d, e.fold] &
            e.filtered_regions[e.d, e.name1, e.i1, e.j1, e.k1] &
            e.filtered_regions[e.d, e.name2, e.i2, e.j2, e.k2] &
            (e.name1 != e.name2)
        )
    )

    e.result_mean[e.term, e.name1, e.name2, e.mean(e.PROB)] = (
        e.term_prob[e.term, e.name1, e.name2, e.fold, e.PROB] &
        e.str_match(e.name1, '.*Hipp.*') &
        e.str_match(e.name2, '.*IPS.*')
    )

    e.result_std[e.term, e.name1, e.name2, e.std(e.PROB)] = (
        e.term_prob[e.term, e.name1, e.name2, e.fold, e.PROB] &
        e.str_match(e.name1, '.*Hipp.*') &
        e.str_match(e.name2, '.*IPS.*')
    )

    e.result_summary_stats[
        e.term, e.name1, e.name2, e.prob_mean, e.prob_std
    ] = (
        e.result_mean[e.term, e.name1, e.name2, e.prob_mean] &
        e.result_std[e.term, e.name1, e.name2, e.prob_std]
    )

    res = nl.solve_all()
    result_summary_stats = res['result_summary_stats'].as_pandas_dataframe()


##############################################################################
# Plotting the results
# --------------------

# %%
# Plot resulting probability term rank across all conditions

result_order = result_summary_stats['prob_mean'].argsort()
mean_prob = result_summary_stats.iloc[result_order]['prob_mean']
std_prob = result_summary_stats.iloc[result_order]['prob_std']
plt.plot(mean_prob.values)
plt.fill_between(
    np.arange(len(mean_prob)),
    mean_prob.values - 3 * std_prob.values,
    mean_prob.values + 3 * std_prob.values,
    alpha=.4
)
plt.xlabel('Term Rank')
plt.ylabel('P(Term | Hippocampal Circuit is Reported)')
plt.axhline(np.percentile(mean_prob.values, 95), c='r')


# %%
# Plot probability estimations for the top 5% most
# probable of term | hippocampal-IPS circuit combinations

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(
        index=args[1],
        columns=args[0], values=args[2]
    )

    mask = None
    if 'threshold' in kwargs:
        mask = d < kwargs['threshold']
        kwargs = kwargs.copy()
        del kwargs['threshold']
        kwargs['mask'] = mask

    return sns.heatmap(d, **kwargs)


thr = np.percentile(result_summary_stats['prob_mean'], 95)
sel_terms = (
    result_summary_stats
    .query('prob_mean >= @thr')
    .reset_index()
    .term
    .unique()
)
results_summary_stats_thr = (
    result_summary_stats
    .query('term in @sel_terms')
    .reset_index()
    .sort_values(['name1', 'name2', 'term'])
)
fg = sns.FacetGrid(
    results_summary_stats_thr,
    col='name1',
    height=4, sharex=True, sharey=True
)
fg.map_dataframe(
    draw_heatmap, 'name2', 'term', 'prob_mean',
    threshold=thr,
    vmin=thr,
    vmax=results_summary_stats_thr.prob_mean.max(),
    cmap='viridis',
    annot=True
)
fg.set_titles("{col_name}")
for ax in fg.axes.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')

# %%
# Plot dispersion of probability estimations for the top 5% most
# probable of term | hippocampal-IPS circuit combinations

fg = sns.FacetGrid(
    results_summary_stats_thr,
    col='name1',
    height=4, sharex=True, sharey=True
)
fg.map_dataframe(
    draw_heatmap, 'name2', 'term', 'prob_std',
    vmax=results_summary_stats_thr.prob_std.max(),
    cmap='plasma', annot=True
)
fg.set_titles("{col_name}")
for ax in fg.axes.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')


# %%
if False:
    # %%
    import holoviews as hv
    from holoviews import dim, opts

    hv.extension('matplotlib')
    hv.output(fig='png', size=200)

    # %%
    res['circuit'] = res.apply(lambda x: x['name1'] + ' - ' + x['name2'], axis=1)

    # %%
    nodes = pd.DataFrame(
        np.r_[res['term'].unique(), res['circuit'].unique()],
        columns=['name']
    )
    nodes.loc[:res['term'].nunique(), 'group'] = 1
    nodes.loc[res['term'].nunique():, 'group'] = 2
    nodes['group'] = nodes.group.astype(int)
    nodes = nodes.reset_index()

    # %%

    links = (
        res[['circuit', 'term', 'PROB']].rename(
            columns={
                'circuit': 'source',
                'term': 'target',
                'PROB': 'value'
            }
        )
    )
    links.loc[links.value < thr, 'value'] = 0
    links['source'] = links.merge(nodes, left_on='source', right_on='name', how='left')['index'].values
    links['target'] = links.merge(nodes, left_on='target', right_on='name', how='left')['index'].values
    print(links.head())

    # %%
    (
        hv.Chord((links, hv.Dataset(nodes, 'index')))
        #.select(value=(1, None))
        .opts(
            opts.Chord(
                cmap='Category20', 
                #edge_color=dim('source').astype(str),
                labels='name',
                node_color=dim('index').astype(str)
            )
        )
    )
