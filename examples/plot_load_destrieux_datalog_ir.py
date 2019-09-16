# coding: utf-8
r'''
Datalog Intermediate Representation Example based on the Destrieux Atlas
========================================================================


Uploading the Destrieux left sulci into NeuroLang and
executing some simple queries.
'''

import nilearn
from nilearn import plotting
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np


from neurolang import regions
from neurolang import region_solver

from neurolang import expressions
from neurolang import solver_datalog_naive as sdb
from neurolang import expression_walker as ew
from neurolang import datalog_chase as dc


###############################################################################
# Set up IR shortcuts

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = sdb.Implication
Fact_ = sdb.Fact
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
    r = regions.ExplicitVBR(
            np.transpose((image_data == label).nonzero()),
            image.affine, image_dim=image.shape
    )
    if r.voxels.shape[0] > 0:
        region_dict[name.decode('utf8')] = r

plotting.plot_roi(region_dict['L S_temporal_sup'].spatial_image())


##################################################
# Make the fact list
destrieux = sdb.Symbol('Destrieux')
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
    dc.sdb.DatalogBasic,
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
solution = dc.build_chase_solution(dl)


###############################################################################
# Extracting the results from the intermediate representation to a python set
# and plotting the first element of the result


rsbv = ew.ReplaceExpressionsByValues({})
result = rsbv.walk(solution['superior_sts_l'])
r = next(iter(result))[0]

for name, region in result:
    plt.figure()
    plotting.plot_roi(region.spatial_image(), title=name)
