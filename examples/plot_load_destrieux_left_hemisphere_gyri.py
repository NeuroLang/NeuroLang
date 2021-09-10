# coding: utf-8
r'''
Loading and Querying the Destrieux et al. Atlas' Left Hemisphere
========================================================================


Uploading the Destrieux regions NeuroLang and
executing a simple query.
'''

from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import datasets, plotting
from neurolang.frontend import NeurolangDL


##################################################
# Initialise the NeuroLang probabilistic engine.

nl = NeurolangDL()


###############################################################################
# Load the Destrieux example from nilearn as a fact list


atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_labels = {
    label: str(name.decode('utf8'))
    for label, name in atlas_destrieux['labels']
}


nl.add_atlas_set('destrieux_atlas', atlas_labels, nib.load(atlas_destrieux['maps']))

###############################################################################
# Add utility function


@nl.add_symbol
def startswith(prefix: str, s: str) -> bool:
    return s.startswith(prefix)

###############################################################################
# Query all left hemisphere regions superior to the temporal superior sulucs
# and anterior to the central sulcus


with nl.scope as e:
    e.on_left_hemisphere[e.x] = (
            e.destrieux_atlas(e.l, e.x) & e.startswith('L S', e.l)
    )

    result = nl.query((e.r,), e.on_left_hemisphere(e.r))


###############################################################################
# Visualise results

subplots = plt.subplots(
    nrows=len(result), ncols=1,
    figsize=(10, 5 * len(result))
)[1]

for (r, subplot) in zip(result, subplots):
    region = r[0]
    plotting.plot_roi(region.spatial_image(), axes=subplot)
