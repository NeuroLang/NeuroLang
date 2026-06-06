# coding: utf-8
r'''
Loading and Querying the Destrieux Atlas' Left Hemisphere Gyri
================================================================

Upload the Destrieux regions into NeuroLang and
execute a simple query, displaying a few selected regions.
'''
import warnings

warnings.filterwarnings("ignore")


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
    label: str(name)
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
# Show the first four matching regions in a concise 2×2 grid so the
# sphinx-gallery thumbnail is informative.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (ax, result_row) in enumerate(zip(axes.ravel(), result[:4])):
    region = result_row[0]
    plotting.plot_roi(region.spatial_image(), axes=ax, title=f"Region {idx+1}")
plt.tight_layout()
plt.show()

print(f"(Total {len(result)} left-hemisphere regions found)")
