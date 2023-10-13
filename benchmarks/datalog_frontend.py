import numpy as np
from nilearn import datasets

import nibabel as nib
from neurolang import frontend as fe


class TimeRegionComparisons:
    params = [
        [
            'anterior_of', 'posterior_of', 'overlapping',
            'superior_of', 'inferior_of', 'left_of', 'right_of'
        ]
    ]

    param_names = ['direction']

    timeout = 10 * 60

    def setup(self, direction):
        raise NotImplementedError("Skip benchmark")
        if not hasattr(self, 'nl'):
            self.setup_()

    def setup_(self):
        nl = fe.NeurolangDL()

        destrieux_atlas = datasets.fetch_atlas_destrieux_2009()
        yeo_atlas = datasets.fetch_atlas_yeo_2011()

        img = nib.load(destrieux_atlas['maps'])
        aff = img.affine
        data = img.get_data()
        rset = []
        for label, name in destrieux_atlas['labels']:
            if label == 0:
                continue
            voxels = np.transpose((data == label).nonzero())
            if len(voxels) == 0:
                continue
            rset.append((
                name.decode('utf8'),
                fe.ExplicitVBR(
                    voxels, aff,
                    image_dim=img.shape, prebuild_tree=True
                )
            ))
        nl.add_tuple_set(rset, name='destrieux')

        img = nib.load(yeo_atlas['thick_17'])
        aff = img.affine
        data = img.get_data().squeeze()
        rset = []
        for label in range(1, 18):
            name = str(label)
            if label == 0:
                continue
            voxels = np.transpose((data == label).nonzero())
            if len(voxels) == 0:
                continue
            rset.append((
                name,
                fe.ExplicitVBR(
                    voxels, aff,
                    image_dim=data.shape,
                    prebuild_tree=True
                )
            ))
        nl.add_tuple_set(rset, name='yeo')
        self.nl = nl

    def time_spatial_relation(self, direction):
        with self.nl.scope as e:
            self.nl.query(
                (e.d,),
                e.destrieux(e.d, e.dr) & e.yeo('1', e.dy) &
                e[direction](e.dr, e.dy)
            )
