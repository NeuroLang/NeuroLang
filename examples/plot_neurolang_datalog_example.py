# -*- coding: utf-8 -*-
r'''
NeuroLang Datalog Example based on the Destrieux Atlas and Neurosynth
=====================================================================


Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
'''
import nibabel as nib
from nilearn import datasets
from nilearn import plotting
import numpy as np

from neurolang import frontend as fe

###############################################################################
# Load the Destrieux example from nilearn
# ---------------------------------------

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux_map = nib.load(destrieux_dataset['maps'])


###############################################################################
# Initialize the NeuroLang instance and load Destrieux's cortical parcellation
# -----------------------------------------------------------------------------


nl = fe.NeurolangDL()
destrieux = nl.new_symbol(name='destrieux')
d = []
for label_number, name in destrieux_dataset['labels']:
    if label_number == 0:
        continue
    name = name.decode()
    region = nl.create_region(destrieux_map, label=label_number)
    if region is None:
        continue
    name = name.replace('-', '_').replace(' ', '_')
    d.append((name.lower(), region))

destrieux = nl.add_tuple_set(d, name='destrieux')


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


###############################################################################
# Load all contiguous regions from Neurosynth that fit the term "supramarginal"
# -----------------------------------------------------------------------------


neurosynth_supramarginal = nl.load_neurosynth_term_regions(
    'supramarginal',
    name='neurosynth_supramarginal'
)


########################################################################
# Query all Destrieux regions that overlap with NeuroSynth supramarginal
# region having volume larger than 2500mm3 with the environment
# ----------------------------------------------------------------------


with nl.environment as e:
    res = nl.query(
            e.query(e.name, e.region_1),
            e.destrieux(e.name, e.region_1) &
            neurosynth_supramarginal(e.region_2) &
            (region_volume(e.region_2) > 2500) &
            nl.symbols.overlapping(e.region_1, e.region_2)
    )


for name, region in res.value:
    plotting.plot_roi(region.spatial_image(), title=name)


########################################################################
# Query all Destrieux regions that overlap with NeuroSynth supramarginal
# region having volume larger than 2500mm3
# ----------------------------------------------------------------------


region_1 = nl.new_symbol(name='region_1')
region_2 = nl.new_symbol(name='region_2')
query = nl.new_symbol(name='query')
name = nl.new_symbol(name='name')

res = nl.query(
        query(name, region_1),
        destrieux(name, region_1) & neurosynth_supramarginal(region_2) &
        (region_volume(region_2) > 2500) &
        nl.symbols.overlapping(region_1, region_2)
)


for name, region in res.value:
    plotting.plot_roi(region.spatial_image(), title=name)
