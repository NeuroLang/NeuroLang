# coding: utf-8
r'''
Datalog Intermediate Representation Example based on the Destrieux Atlas
========================================================================


Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
'''

import nilearn
import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting

import nibabel as nib
from neurolang.datalog.chase import Chase
from neurolang import expression_walker as ew
from neurolang import expressions, region_solver, regions
from neurolang.datalog import DatalogProgram
from neurolang.datalog.expressions import Fact, Implication, TranslateToLogic

###############################################################################
# Set up IR shortcuts

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
Fact_ = Fact
Eb_ = expressions.ExpressionBlock


###############################################################################
# Load the Destrieux example from nilearn as a fact list
# ------------------------------------------------------
atlas_destrieux = nilearn.datasets.fetch_atlas_destrieux_2009()

image = nib.load(atlas_destrieux['maps'])
image_data = image.get_data()


##################################################
# Load the regions into Voxel-style regions
region_dict = {}
for label, name in atlas_destrieux['labels']:
    if label == 0:
        continue

    voxels = np.transpose((image_data == label).nonzero())
    if voxels.shape[0] == 0:
        continue

    r = regions.ExplicitVBR(
            voxels,
            image.affine, image_dim=image.shape
    )
    region_dict[name.decode('utf8')] = r

plotting.plot_roi(region_dict['L S_temporal_sup'].spatial_image())


##################################################
# Make the fact list
destrieux = S_('Destrieux')
destrieux_facts = [
    Fact_(destrieux(
        C_(name),
        C_(region)
    ))
    for name, region in region_dict.items()
]


###############################################################################
# Set the datalog interpreter with Region-managing builtins and adding lh, lr
# -----------------------------------------------------------------------------


class Datalog(
    region_solver.RegionSolver,
    TranslateToLogic,
    DatalogProgram,
    ew.ExpressionBasicEvaluator
):
    def function_lh(self, x: str) -> bool:
        return x.startswith('L ')

    def function_rh(self, x: str) -> bool:
        return x.startswith('R ')


###############################################################################
# Construct a query
# -----------------------------------------------------------------------------
# superior_sts_l(name, r) :- destrieux('L S_temporal_sup', superior_sts_l),
#                      anatomical_superior_of(r, superior_sts_l),
#                      lh(name), destrieux(name, r)

superior_sts_l = S_('region_l_sts')
r = S_('r')
name = S_('name')

r1 = Imp_(
    S_('superior_sts_l')(name, r),
    destrieux(C_('L S_temporal_sup'), superior_sts_l) &
    S_('anatomical_superior_of')(r, superior_sts_l) &
    S_('lh')(name) &
    destrieux(name, r)
)

print(r1)

datalog_program = Eb_(
    destrieux_facts + [
        r1,
    ]
)


###############################################################################
# Interpreting and running the query
# -----------------------------------------------------------------------------


dl = Datalog()
dl.walk(datalog_program)
solution = Chase(dl).build_chase_solution()


###############################################################################
# Extracting the results from the intermediate representation to a python set
# and plotting the first element of the result


rsbv = ew.ReplaceExpressionsByValues({})
result = rsbv.walk(solution['superior_sts_l'])

for name, region in result.unwrapped_iter():
    plt.figure()
    plotting.plot_roi(region.spatial_image(), title=name)
