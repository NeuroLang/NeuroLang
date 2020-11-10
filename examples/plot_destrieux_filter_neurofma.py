# -*- coding: utf-8 -*-
r'''
Identifying the temporal lobe using the Destrieux et al. atlas and the FMA ontology
===================================================================================


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
# Load Destrieux's atlas
mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux = nib.load(destrieux_dataset['maps'])
destrieux_resampled = image.resample_img(
    destrieux, mni_t1.affine, interpolation='nearest'
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
# Loading and querying the information
# --------------------------------------------

###############################################################################
# Loading the intology within NeuroLang

nl = pfe.NeurolangPDL()
nl.load_ontology(neuroFMA)


###############################################################################
# Adding new aggregation function to build a region overlay

@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable
) -> fe.ExplicitVBR:
    voxels = np.c_[i, j, k]
    return fe.ExplicitVBROverlay(
        voxels, mni_t1.affine, np.ones(voxels.shape[0]),
        image_dim=mni_t1.shape
    )

###############################################################################
# Create the ontology symbols that we will use in our query

label = nl.new_symbol(name=str(RDFS.label))
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
regional_part = nl.new_symbol(name='http://sig.biostr.washington.edu/fma3.0#regional_part_of')

###############################################################################
# and load all the tuples in our database

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
# And finally, we perform the query:

with nl.scope as e:
    
    e.fma_related_region[e.subregion_name, e.fma_entity_name] = (
        label(e.fma_uri, e.fma_entity_name) & 
        regional_part(e.fma_region, e.fma_uri) & 
        subclass_of(e.fma_subregion, e.fma_region) &
        label(e.fma_subregion, e.subregion_name)
    )
    
    e.fma_related_region[e.subregion_name, e.fma_entity_name] = (
        e.fma_related_region(e.fma_subregion, e.fma_entity_name) &
        label(e.fma_uri, e.fma_subregion) &
        subclass_of(e.recursive_region, e.fma_uri) & 
        label(e.recursive_region, e.subregion_name)
    )
    
    e.destrieux_ijk[e.destrieux_name, e.i, e.j, e.k] = (
        e.destrieux_labels[e.destrieux_name, e.id_destrieux] &
        e.destrieux_image[e.i, e.j, e.k, e.id_destrieux]
    )
    
    e.region_voxels[agg_create_region_overlay[e.i, e.j, e.k]] = (
        e.fma_related_region[e.fma_subregions, 'Temporal lobe'] &
        e.relation_destrieux_fma[e.destrieux_name, e.fma_subregions] &
        e.destrieux_ijk[e.destrieux_name, e.i, e.j, e.k]
    )
    
    res = nl.solve_all()
    img_query = res['region_voxels']
    

###############################################################################
# Results
# --------------------------------------------
# Using the ontology, we limit the results of the analysis only to regions 
# within the Temporal Lobe

result_image = (
    img_query
    .fetch_one()
    [0]
    .spatial_image()
)
img = result_image.get_fdata()
plotting.plot_roi(result_image, display_mode='x', cut_coords=np.linspace(-63, 63, 5))

""

