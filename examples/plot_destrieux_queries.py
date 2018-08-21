# -*- coding: utf-8 -*-
r'''
NeuroLang Query Example based on the Destrieux Atlas
====================================================


Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
'''
from nilearn import datasets
from nilearn import plotting
import nibabel as nib
import numpy as np

from neurolang import frontend as fe

###############################################################################
# Load the Destrieux example from nilearn
# ---------------------------------------

destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
destrieux_map = nib.load(destrieux_dataset['maps'])

###############################################################################
# Input the left sulci into the NeuroLang interface
# -------------------------------------------------

nl = fe.RegionFrontend()
for number, name in destrieux_dataset['labels']:
    name = name.decode()
    if not name.startswith('L ') or 'S_' not in name:
        continue

    # Create a region object
    voxels = np.transpose((destrieux_map.get_data() == number).nonzero())
    region = fe.ExplicitVBR(
        voxels, destrieux_map.affine, img_dim=destrieux_map.shape
    )

    # Fine tune the symbol name
    name = 'L_' + name[2:].replace('-', '_')
    nl.add_region(region, result_symbol_name=name)


##################################################
# Plot one of the symbols

plotting.plot_roi(nl.symbols.L_S_central.value.spatial_image())


###############################################################################
# Create and run a simple query
# ----------------------------------------------

x = nl.new_region_symbol('x')
q = nl.query(x, nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central))
print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)


###############################################################################
# Create and run query which is a bit more complex
# ----------------------------------------------
x = nl.new_region_symbol('x')
q = nl.query(
    x,
    nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central) &
    nl.symbols.anatomical_superior_of(x, nl.symbols.L_S_temporal_sup)
)
print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)


###############################################################################
# Create and run query with existential quantifiers and negation
# --------------------------------------------------------
x = nl.new_region_symbol('x')
y = nl.new_region_symbol('y')
q = nl.query(
    x,
    nl.symbols.anatomical_anterior_of(x, nl.symbols.L_S_central) &
    ~nl.exists(
        y,
        nl.symbols.anatomical_anterior_of(y, nl.symbols.L_S_central) &
        nl.symbols.anatomical_anterior_of(x, y)
    )
)
print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)
