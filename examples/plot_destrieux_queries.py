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
for label_number, name in destrieux_dataset['labels']:
    name = name.decode()
    if (
        not name.startswith('L ') or
        not ('S_' in name or 'Lat_Fis' in name or 'Pole' in name)
    ):
        continue

    # Create a region object
    region = nl.create_region(destrieux_map, label=label_number)

    # Fine tune the symbol name
    name = 'L_' + name[2:].replace('-', '_')
    nl.add_region(region, name=name.lower())

##################################################
# Plot one of the symbols

plotting.plot_roi(nl.symbols.l_s_central.value.spatial_image())


###############################################################################
# Create and run a simple query
# -----------------------------

x = nl.new_region_symbol('x')
q = nl.query(x, nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_central))
print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)


###############################################################################
# Create and run query which is a bit more complex
# ------------------------------------------------
x = nl.new_region_symbol('x')
q = nl.query(
    x,
    nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_central) &
    nl.symbols.anatomical_superior_of(x, nl.symbols.l_s_temporal_sup)
)
print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)


###############################################################################
# Create and run query with existential quantifiers and negation
# --------------------------------------------------------------
x = nl.new_region_symbol('x')
y = nl.new_region_symbol('y')
q = nl.query(
    x,
    nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_central) &
    ~nl.exists(
        y,
        nl.symbols.anatomical_anterior_of(y, nl.symbols.l_s_central) &
        nl.symbols.anatomical_anterior_of(x, y)
    )
)
print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)


###############################################################################
# Use the set of results from a query in a different one
# ------------------------------------------------------
x = nl.new_region_symbol('x')
temporal_lobe_query = nl.query(
    x,
    nl.symbols.anatomical_inferior_of(x, nl.symbols.l_s_parieto_occipital) &
    nl.symbols.anatomical_anterior_of(x, nl.symbols.l_s_calcarine) &
    nl.symbols.anatomical_posterior_of(x, nl.symbols.l_lat_fis_ant_vertical)
)
temporal_lobe = q.do(name='temporal_lobe')
print(temporal_lobe)

##################################################
#

q = nl.query(
    x,
    nl.symbols.isin(x, temporal_lobe) &
    ~nl.symbols.anatomical_inferior_of(x, nl.symbols.l_s_temporal_inf)
)

print(q)

##################################################
#

res = q.do()
for r in res:
    plotting.plot_roi(r.value.spatial_image(), title=r.symbol_name)
