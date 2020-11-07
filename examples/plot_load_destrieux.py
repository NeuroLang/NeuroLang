# coding: utf-8
r'''
Loading and Querying the Destrieux et al. Atlas
========================================================================


Uploading the Destrieux regions NeuroLang and
executing a simple query.
'''

from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import datasets, plotting
from neurolang.frontend import NeurolangPDL


##################################################
# Initialise the NeuroLang probabilistic engine.

nl = NeurolangPDL()


###############################################################################
# Load the Destrieux example from nilearn as a fact list
atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_labels = {
    label: str(name.decode('utf8'))
    for label, name in atlas_destrieux['labels']
}


nl.add_atlas_set('destrieux', atlas_labels, nib.load(atlas_destrieux['maps']))

###############################################################################
# Add utility functions to separate hemispheric regions

@nl.add_symbol
def left_hemisphere_name(x: str) -> bool:
    return x.startswith('L ')


@nl.add_symbol
def right_hemisphere_name(x: str) -> bool:
    return x.startswith('R ')


###############################################################################
# Query all left hemisphere regions superior to the temporal superior sulucs
# and anterior to the central sulcus

with nl.environment as e:
    # The set `superior_sts_l` is composed by every
    # name `name` and region `r` where
    e.superior_sts_l[e.name, e.r] = (
        # the every region `name`, `r` is in Destrieux at al's atlas
        e.destrieux(e.name, e.r) &
        # the `name` corresponds to a left hemisphere name
        e.left_hemisphere_name(e.name) &
        # the region `r` is anatomically superior to the
        # left superior temporal and anterior to the central sulci
        e.anatomical_superior_of(e.r, e.superior_sts_l) &
        e.anatomical_anterior_of(e.r, e.central_l) &
        # where `superior_sts_l` and `central_l` are identified
        # by their names in Destrieux et al's atlas.
        e.destrieux('L S_temporal_sup', e.superior_sts_l) &
        e.destrieux('L S_central', e.central_l)
    )

    result = nl.query((e.name, e.r), e.superior_sts_l(e.name, e.r))


###############################################################################
# Visualise results

subplots = plt.subplots(
    nrows=len(result), ncols=1,
    figsize=(10, 5 * len(result))
)[1]
for (r, subplot) in zip(result, subplots):
    name, region = r
    subplot.set_title(name)
    plotting.plot_roi(region.spatial_image(), title=name, axes=subplot)
