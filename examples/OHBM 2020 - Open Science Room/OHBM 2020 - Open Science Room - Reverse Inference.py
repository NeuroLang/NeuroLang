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

import stats_helper, datasets_helper
from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend
from rdflib import RDFS
from nilearn import plotting
import numpy as np
from matplotlib import pyplot as plt
from typing import Iterable
from neurolang import frontend as fe

# We will use the FMA ontology to obtain regions of the brain included within the `Temporal lobe`. We will obtain all the entities that make up the `Temporal Lobe` and then we will convert them into regions using the information provided by the Destrieux atlas. This will allow us to perform spatial operations on these regions, allowing us to obtain those NeuroSynth regions associated with the term `auditory` that overlap our results.
#
#

nl = ProbabilisticFrontend()
datasets_helper.load_reverse_inference_dataset(nl, n=50)

paths = ['neurolang_data/ontologies/neurofma_fma3.0.owl', 'neurolang_data/ontologies/cogat.xrdf']
nl.load_ontology(paths, load_format=["xml", "xml"])

# +
label = nl.new_symbol(name=str(RDFS.label))
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
regional_part = nl.new_symbol(name='http://sig.biostr.washington.edu/fma3.0#regional_part_of')

#@nl.add_symbol
#def agg_create_region(x: Iterable, y: Iterable, z: Iterable) -> fe.ExplicitVBR:
#    mni_t1 = it.masker.volume
#    voxels = nib.affines.apply_affine(np.linalg.inv(mni_t1.affine), np.c_[x, y, z])
#    return fe.ExplicitVBR(voxels, mni_t1.affine, image_dim=mni_t1.shape)

@nl.add_symbol
def first_word(name: str) -> str:
    return name.split(" ")[0]

with nl.environment as e:    
    e.fma_related_region[e.subregion_name, e.fma_uri] = (
        label(e.xfma_entity_name, e.fma_uri) & 
        regional_part(e.fma_region, e.xfma_entity_name) & 
        subclass_of(e.fma_subregion, e.fma_region) &
        label(e.fma_subregion, e.subregion_name)
    )
    e.fma_related_region[e.recursive_region, e.fma_name] = (
        subclass_of(e.recursive_region, e.fma_subregion) & e.fma_related_region(e.fma_subregion, e.fma_name)
    )
    e.fma_to_destrieux[e.fma_name, e.destrieux_name] = (
        label(e.fma_uri, e.fma_name) & e.relation_destrieux_fma(e.destrieux_name, e.fma_name)
    )
# -

with nl.environment as e:
    e.region_voxels[e.id_neurosynth, e.x, e.y, e.z] = (
        e.fma_related_region[e.fma_subregions, 'Temporal lobe'] & 
        e.fma_to_destrieux[e.fma_subregions, e.destrieux_name] & 
        e.destrieux_to_neurosynth[e.destrieux_name, e.id_neurosynth, e.x, e.y, e.z]
    )
    
    e.destrieux_to_neurosynth[e.destrieux_name, e.id_neurosynth, e.x, e.y, e.z] = (
        e.destrieux_labels[e.id_destrieux, e.destrieux_name] &
        e.xyz_destrieux[e.x, e.y, e.z, e.id_destrieux] &
        e.xyz_neurosynth[e.x, e.y, e.z, e.id_neurosynth]
    )
    
    e.p_act[e.id_voxel, e.term, e.id_study] = (
        e.p_voxel_study[e.id_voxel, e.id_study] & 
        e.p_term_study[e.term,  e.id_study] & 
        e.p_study[e.id_study]
    )
    
    e.probability_voxel[e.id_voxel, e.x, e.y, e.z] = (
        e.p_act[e.id_voxel, e.term, e.id_study] &
        e.region_voxels[e.id_voxel, e.x, e.y, e.z]
    )
    
    #nl_results = nl.solve_all()
    nl_results = nl.solve_query(e.probability_voxel[e.id_voxel, e.x, e.y, e.z])
    
    #e.probability_voxel[nl.symbols.agg_create_region(e.x, e.y, e.z)] = (
    #    e.p_act[e.id_voxel, e.term, e.id_study] &
    #    e.region_voxels[e.id_voxel, e.x, e.y, e.z]
    #)
    
    #e.final[e.region] = e.probability_voxel[e.region]
    
    #nl_results = nl.solve_query(e.final[e.region])

t = nl_results.value._container.values
f = [(float(prob), id_voxel, x, y, z) for z, id_voxel, x, y, prob in t]
p_act_aud = nl.add_probabilistic_facts_from_tuples(tuple(f), name='p_act_aud');

prob_img_nl = datasets_helper.parse_results(nl_results)
plotting.plot_stat_map(
    prob_img_nl, 
    title='Tag "auditory" (Neurolang)', 
    cmap='PuBuGn',
    display_mode='x',
    cut_coords=np.linspace(-63, 63, 5),
)

plotting.plot_stat_map(
    prob_img_nl, title='Tag "auditory" (Neurolang)', 
    cmap='PuBuGn',
    display_mode='y',
    cut_coords=np.linspace(-30, 5, 5),
)

# +
from rdflib import RDF

part_of = nl.new_symbol(name='http://www.obofoundry.org/ro/ro.owl#part_of')

# Should found a better to imply this
triples = nl.symbol_table[nl.get_ontology_triples_symbol().name]
a = triples.value.as_numpy_array()
t = [('Auditory', str(RDF.type), 'http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00148')]

t = np.concatenate((a, t))
nl.add_extensional_predicate_from_tuples(t, name=nl.get_ontology_triples_symbol().name)
# -

with nl.scope as e:
    e.pre_part[e.x, e.y] = part_of[e.x, e.y]

    e.perception_terms[e.short_name] = (
        e.pre_part["Auditory", e.y] & 
        subclass_of[e.z, e.y] & 
        label(e.z, e.term) &
        (e.short_name == nl.symbols.first_word(e.term))
    )
    
    e.p_term_given_act[e.term, e.voxid] = (
        e.ns_reported_activations[e.study, e.voxid] &
        e.perception_terms[e.term] & 
        e.ns_term_study_associations[e.study, e.term]
    )
    
    e.p_term_g_aud_voxels[e.term] = (
        e.p_term_given_act[e.term, e.voxid] &
        e.p_act_aud[e.voxid, e.x, e.y, e.z]
    )
    
    nl_reverse = nl.solve_query(e.p_term_g_aud_voxels[e.term])

nl_reverse





#prob_terms, prob_voxels, prob_terms_voxels = stats_helper.load_neurosynth_database()
prob_img = stats_helper.parse_neurolang_result(result, prob_terms)


plotting.plot_stat_map(
    prob_img, 
    title='Tag "auditory" (Neurolang)', 
    cmap='PuBuGn',
    display_mode='x',
    cut_coords=np.linspace(-63, 63, 5),
)

plotting.plot_stat_map(
    prob_img, title='Tag "auditory" (Neurolang)', 
    cmap='PuBuGn',
    display_mode='y',
    cut_coords=np.linspace(-30, 5, 5),
)

# Now let's see the same result obtained directly from the NeuroSynth database.

prob_img_ns = stats_helper.parse_neurosynth_result(prob_terms_voxels)

plotting.plot_stat_map(
    prob_img_ns, title='Tag "auditory" (Neurosynth)', 
    cmap='PuBu',
    display_mode='x',
    cut_coords=np.linspace(-63, 63, 5),
)

plotting.plot_stat_map(
    prob_img_ns, title='Tag "auditory" (Neurosynth)', 
    cmap='PuBu',
    display_mode='y',
    cut_coords=np.linspace(-30, 5, 5),
)

# Now we can analyze the results by plotting the p-values obtained. Let's start with the NeuroLang results.

res, p_values_corrected, p_value_image = stats_helper.compute_p_values(prob_img, q=1e-25)

plt.hist(-np.log10(res))
plt.axvline(-np.log10(p_values_corrected), c='r')

plotting.plot_stat_map(
    p_value_image, 
    title=r'$-\log_{10} P$ value (Neurolang)', 
    threshold=-np.log10(p_values_corrected), 
    cmap='YlOrRd',
    display_mode='x',
    cut_coords=np.linspace(-63, 63, 5),
)

plotting.plot_stat_map(
    p_value_image, title=r'$-\log_{10} P$ value (Neurolang)', 
    threshold=-np.log10(p_values_corrected),
    cmap='YlOrRd',
    display_mode='y',
    cut_coords=np.linspace(-30, 5, 5),
)

# In the above results, we can see that the regions have a high specificity and that they focus entirely on our area of interest. Reducing the area of work in this way allows us to minimize variance, enabling us to obtain results with greater statistical power.

# And now let's do the same with the NeuroSynth results to compare. It is important to mention that the techniques used for the calculation of the p-values, make a comparison against the average of the activations. Bearing this in mind, by decreasing the region to be analyzed and focusing it on the activated region, the average of the activations increases.

res, p_values_corrected, p_value_image = stats_helper.compute_p_values(prob_img_ns, q=1e-25)

plt.hist(-np.log10(res))
plt.axvline(-np.log10(p_values_corrected), c='r')

plotting.plot_stat_map(
    p_value_image, 
    title=r'$-\log_{10} P$ value (NeuroSynth)', 
    threshold=-np.log10(p_values_corrected), 
    cmap='YlOrRd',
    display_mode='x',
    cut_coords=np.linspace(-63, 63, 5),
)

plotting.plot_stat_map(
    p_value_image, title=r'$-\log_{10} P$ value (NeuroSynth)', 
    threshold=-np.log10(p_values_corrected),
    cmap='YlOrRd',
    display_mode='y',
    cut_coords=np.linspace(-30, 5, 5),
)

# It can be seen above how despite using a restrictive threshold for the p-value ($q<10^{25}$, FDR corrected), in the Neurosynth example there are activations considered statistically significant in the motor cortex that should not be present for the `auditory` tag. Using a prior information in NeuroLang, we are able to remove these false positives and obtain a cleaner result. 

# #### References
# [1] Yarkoni, T.: Neurosynth core tools v0.3.1, DOI: 10.5281/zenodo.9925 (2014). <br/>
# [2] Yarkoni, T., Poldrack, R. A., Nichols, T. E., Van Essen, D. C. & Wager, T. D: Large-scale automated synthesis of human functional neuroimaging data. Nat. Methods 8, 665–670, DOI: 10.1038/nmeth.1635 (2011). <br/>
# [3] News. Journal of Investigative Medicine 58 (8), 929 (Dec2010). https://doi.org/10.2310/JIM.0b013e3182025955, http://jim.bmj.com/content/58/8/929.abstract <br/>
# [4] Insel, T. R., Landis, S.C., Collins, F.S.: Research priorities. The NIHBRAIN Initiative. Science (New York, N.Y.) 340 (6133), 687–688 (May  2013). https://doi.org/10.1126/science.1239276 <br/>
# [5] Markram, H.: The human brain project. Scientific American306(6), 50–55 (Jun2012). https://doi.org/10.1038/scientificamerican0612-50
# [6] Derrfuss, J. & Mar, R. A. Lost in localization: the need for a universal coordinate database. NeuroImage 48, 1–7, DOI:10.1016/j.neuroimage.2009.01.053 (2009).
