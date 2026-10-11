# coding: utf-8
r'''
Loading and Querying the Destrieux et al. Atlas' Left Hemisphere
========================================================================


Uploading the Destrieux regions into NeuroLang and
executing a simple query selecting the left-hemisphere sulci.
'''
import warnings

warnings.filterwarnings("ignore")


from matplotlib import pyplot as plt
import nibabel as nib
from nilearn import datasets, plotting
from neurolang.frontend import NeurolangDL


##################################################
# Initialise the NeuroLang (deterministic) Datalog engine.

nl = NeurolangDL()


###############################################################################
# Load the Destrieux example from nilearn as a fact list


atlas_destrieux = datasets.fetch_atlas_destrieux_2009()
if 'lut' in atlas_destrieux:
    # nilearn >= 0.11 provides a look-up table with label indices and names
    lut = atlas_destrieux['lut']
    atlas_labels = dict(zip(lut['index'], lut['name'].astype(str)))
else:
    atlas_labels = {
        label: str(name)
        for label, name in atlas_destrieux['labels']
    }
atlas_labels.pop(0, None)  # background


nl.add_atlas_set('destrieux_atlas', atlas_labels, nib.load(atlas_destrieux['maps']))

###############################################################################
# Add utility function


@nl.add_symbol
def startswith(prefix: str, s: str) -> bool:
    return s.startswith(prefix)

###############################################################################
# Query all regions of the left hemisphere whose Destrieux name starts with
# ``L S``, that is, the left-hemisphere sulci


with nl.scope as e:
    e.on_left_hemisphere[e.x] = (
            e.destrieux_atlas(e.l, e.x) & e.startswith('L S', e.l)
    )

    result = nl.query((e.r,), e.on_left_hemisphere(e.r))


###############################################################################
# Visualise the first few results

print(f"{len(result)} left-hemisphere sulci found")
to_plot = list(result)[:4]
subplots = plt.subplots(
    nrows=len(to_plot), ncols=1,
    figsize=(10, 3 * len(to_plot))
)[1]

for (r, subplot) in zip(to_plot, subplots):
    region = r[0]
    plotting.plot_roi(region.spatial_image(), axes=subplot)
