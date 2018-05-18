from .RCD_relations import *
from .exceptions import RegionException
from .utils import data_manipulation
import numpy as np
import nibabel as nib
from numpy.linalg import inv

class Region:

    def __init__(self, lb, ub) -> None:
        if not np.all([lb[i] < ub[i] for i in range(len(lb))]):
            raise RegionException('Lower bounds must be lower than upper bounds when creating rectangular regions')
        self._bounding_box = np.c_[lb, ub]
        self._bounding_box.setflags(write=False)

    def __hash__(self):
        return hash(self._bounding_box.tobytes())

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def _lb(self):
        return self.bounding_box[:, 0]

    @property
    def _ub(self):
        return self.bounding_box[:, 1]

    @property
    def center(self) -> np.array:
        return 0.5 * (self._lb + self._ub)

    @property
    def width(self) -> np.array:
        return self._ub - self._lb

    def axis_intervals(self) -> np.array:
        return np.array([tuple([self._lb[i], self._ub[i]]) for i in range(len(self._lb))])

    def __eq__(self, other) -> bool:
        return np.all(self._lb == other._lb) and np.all(self._ub == other._ub)

    def __repr__(self):
        return 'Region(lb={}, up={})'.format(tuple(self._lb), tuple(self._ub))

class VolumetricBrainRegion(Region):

    def to_xyz(self, affine):
        '''return world coordinates of the region corresponding to the affine matrix transform'''
        raise NotImplementedError()

    def to_ijk(self, affine):
        '''return ijk voxels coordinates corresponding to the affine matrix transform'''
        raise NotImplementedError()

class ExplicitVBR(VolumetricBrainRegion):

    def __init__(self, voxels, affine_matrix):
        self._voxels = voxels
        self._affine_matrix = affine_matrix
        self._bounding_box = None

    def __hash__(self):
        return hash(self.bounding_box.tobytes())

    @property
    def bounding_box(self):
        if self._bounding_box is not None:
            return self._bounding_box
        (lb, ub) = data_manipulation.region_data_limits(self.to_xyz())
        self._bounding_box = np.c_[lb, ub]
        return self._bounding_box

    def to_xyz(self):
        return nib.affines.apply_affine(self._affine_matrix, self._voxels)

    def to_ijk(self, affine):
        return nib.affines.apply_affine(
            np.linalg.solve(affine, self._affine_matrix),
            self._voxels)


class ImplicitVBR(VolumetricBrainRegion):

    def voxel_in_region(self, voxel):
        raise NotImplementedError() #todo override set __in__

    def to_ijk(self, affine):
        raise NotImplementedError()

    def to_xyz(self, affine):
        raise NotImplementedError()

    def to_explicit_region(self, affine):
        voxels_coordinates = self.to_ijk(affine)
        return ExplicitVBR(voxels_coordinates, affine)


class SphericVolume(ImplicitVBR):

    def __init__(self, center, radius):
        self._center = center
        self._radius = radius
        self._bounding_box = None

    @property
    def bounding_box(self):
        if self._bounding_box is not None:
            return self._bounding_box
        lb = tuple(np.array(self._center) - self._radius)
        ub = tuple(np.array(self._center) + self._radius)
        self._bounding_box = np.c_[lb, ub]
        return self._bounding_box

    def to_ijk(self, affine):
        bounds_voxels = nib.affines.apply_affine(np.linalg.inv(affine), np.array([self._lb, self._ub]))
        [xs, ys, zs] = [range(int(min(bounds_voxels[:, i])), int(max(bounds_voxels[:, i]))) for i in range(3)]

        #todo improve
        voxel_coordinates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    xyz_coords = nib.affines.apply_affine(affine, np.array([x, y, z]))
                    if np.linalg.norm(np.array(xyz_coords) - np.array(self._center)) <= self._radius:
                        voxel_coordinates.append([x, y, z])
        return np.array(voxel_coordinates)

    def center(self):
        return self._center

    def __eq__(self, other) -> bool:
        return np.all(self._center == other._center) and self._radius == other._radius

    def __repr__(self):
        return 'SphericVolume(Center={}, Radius={})'.format(tuple(self._center), self._radius)
