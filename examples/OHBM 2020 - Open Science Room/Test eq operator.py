# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: neurolang
#     language: python
#     name: neurolang
# ---

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

# +
import stats_helper, datasets_helper
from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend
from rdflib import RDFS
from nilearn import plotting
import numpy as np
from matplotlib import pyplot as plt
from typing import Iterable
from neurolang import frontend as fe

nl = ProbabilisticFrontend()
datasets_helper.load_reverse_inference_dataset(nl)

path = 'neurolang_data/ontologies/cogat.xrdf'
nl.load_ontology(path)

# +
part_of = nl.new_symbol(name='http://www.obofoundry.org/ro/ro.owl#part_of')
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
label = nl.new_symbol(name=str(RDFS.label))
hasTopConcept = nl.new_symbol(name='http://www.w3.org/2004/02/skos/core#hasTopConcept')

@nl.add_symbol
def word_lower(name: str) -> str:
    print(name)
    return str(name).lower()


# +
from operator import eq

with nl.scope as e:
    #e.julich_to_neurosynth[e.julich_id, e.id_neurosynth, e.x, e.y, e.z] = (
    #    e.xyz_julich[e.x, e.y, e.z, e.julich_id] &
    #    e.xyz_neurosynth[e.x, e.y, e.z, e.id_neurosynth]
    #)
    
    #e.region_voxels[e.name, e.id_neurosynth, e.x, e.y, e.z] = (
    #    e.julich_id[e.name, e.julich_id] &
    #    e.julich_to_neurosynth[e.julich_id, e.id_neurosynth, e.x, e.y, e.z]
    #)
    
    #e.julich_id[e.name, e.id] = (
    #    e.julich_ontology[e.name, 'labelIndex', e.id]
    #)
    
    #e.julich_voxels[e.id_neurosynth, e.x, e.y, e.z] = (
    #    e.region_voxels['Area Ia (Insula)', e.id_neurosynth, e.x, e.y, e.z]
    #)
    
    #e.p_act[e.id_voxel, e.term] = (
    #    e.p_voxel_study[e.id_voxel, e.id_study] & 
    #    e.p_term_study[e.term,  e.id_study] & 
    #    e.p_study[e.id_study]
    #)
    
    e.ontology_terms[e.lower] = (
        hasTopConcept[e.uri, 'Executive-Cognitive Control'] &
        label[e.uri, e.name] &
        (e.lower == word_lower(e.name))
    )
    
    #e.probability_voxel[e.term] = (
    #    e.p_act[e.id_voxel, e.term] &
    #    e.julich_voxels[e.id_voxel, e.x, e.y, e.z] &
    #    e.ontology_terms[e.term]
    #)
    
    nl_results = nl.solve_query(e.ontology_terms[e.name])
# -

nl_results['ontology_terms']


