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
from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend
from neurolang import frontend as fe

nl = ProbabilisticFrontend()


# +
nl.add_tuple_set([('Val1', 'var'), ('Val2', 'var'), ('Val3', 'var')], name='test_var')

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
    
    e.lower[e.lower] = (
        e.test_var[e.name, 'var'] &
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


