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

with nl.environment as e:
    res = nl.query(
            e.query('r_s_temporal_transverse', e.region_1),
            e.destrieux('r_s_temporal_transverse', e.region_1)
    )


for name, region in res.value:
    plotting.plot_roi(region.spatial_image(), title=name)
