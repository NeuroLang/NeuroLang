from .exceptions import NeuroLangException
from .utils import data_manipulation
import numpy as np
import nibabel as nib
import copy
from numpy.linalg import inv


#code repetition, could be abstracted to one method (region, aff, set_op)

def region_union(regions_set, affine):
    voxels_per_regions = [set(map(tuple, elem.to_ijk(affine))) for elem in regions_set] # first convert to array of tuples
    result_voxels = set.union(*voxels_per_regions)
    return ExplicitVBR(list(map(lambda x: np.array(x), result_voxels)), affine)  # then convert back to 2d array, FIX!


def region_intersection(regions_set, affine):
    voxels_per_regions = [set(map(tuple, elem.to_ijk(affine))) for elem in regions_set]
    result_voxels = set.intersection(*voxels_per_regions)
    return ExplicitVBR(list(map(lambda x: np.array(x), result_voxels)), affine)


def region_difference(regions_set, affine):
    voxels_per_regions = [set(map(tuple, elem.to_ijk(affine))) for elem in regions_set]
    result_voxels = set.difference(*voxels_per_regions)
    return ExplicitVBR(list(map(lambda x: np.array(x), result_voxels)), affine)


class Region:

    def __init__(self, lb, ub) -> None:
        if not np.all([lb[i] < ub[i] for i in range(len(lb))]):
            raise NeuroLangException('Lower bounds must be lower than upper bounds when creating rectangular regions')
        self._bounding_box = np.array([np.c_[lb, ub]])
        self._bounding_box.setflags(write=False)
        self._dim = len(lb)

    def __hash__(self):
        return hash(self.bounding_box.tobytes())

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def dim(self):
        return self._dim

    @property
    def _lb(self):
        return min(self.bounding_box[:, :, 0], key=tuple)

    @property
    def _ub(self):
        return max(self.bounding_box[:, :, 1], key=tuple)

    @property
    def center(self) -> np.array:
        return 0.5 * (self._lb + self._ub)

    @property
    def width(self) -> np.array:
        return self._ub - self._lb

    def split_bounding_box(self):
        result = []
        for bb in self.bounding_box:
            bb1, bb2 = copy.copy(bb), copy.copy(bb)
            ax = np.argmax([rng[1] - rng[0] for rng in bb])
            middle = (bb[ax, 0] + bb[ax, 1])/2
            bb1[ax] = [bb[ax, 0],  middle]
            bb2[ax] = [middle, bb[ax, 1]]
            result.extend([bb1, bb2])
        self._bounding_box = np.array(result)

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

    def to_xyz_set(self, affine):
        raise NotImplementedError()

    def to_ijk_set(self, affine):
        raise NotImplementedError()

    def remove_empty_bounding_boxes(self):
        raise NotImplementedError()


class ExplicitVBR(VolumetricBrainRegion):

    def __init__(self, voxels, affine_matrix):
        self._voxels = voxels
        self._affine_matrix = affine_matrix
        self._bounding_box = None
        self._dim = len(voxels[0])

    @property
    def bounding_box(self):
        if self._bounding_box is not None:
            return self._bounding_box
        (lb, ub) = data_manipulation.region_data_limits(self.to_xyz())
        self._bounding_box = np.array([np.c_[lb, ub]])
        return self._bounding_box

    def to_xyz(self):
        return nib.affines.apply_affine(self._affine_matrix, self._voxels)

    def to_ijk(self, affine):
        return nib.affines.apply_affine(
            np.linalg.solve(affine, self._affine_matrix),
            self._voxels)

    def remove_empty_bounding_boxes(self):
        bbs = self.bounding_box
        bb_elems = np.zeros(len(bbs))
        for value in self.to_xyz():
            for i in range(len(bb_elems)):
                if data_manipulation.coordinate_in_limits(value, bbs[i]):
                    bb_elems[i] += 1
                    break
        self._bounding_box = bbs[(bb_elems != 0)]

    def downward_granularity(self):
        self.split_bounding_box()
        self.remove_empty_bounding_boxes()


class ImplicitVBR(VolumetricBrainRegion):

    def voxel_in_region(self, voxel):
        raise NotImplementedError() #todo override set __in__

    def to_ijk(self, affine):
        raise NotImplementedError()

    def to_xyz(self, affine):
        raise NotImplementedError()

    def to_explicit_vbr(self, affine):
        voxels_coordinates = self.to_ijk(affine)
        return ExplicitVBR(voxels_coordinates, affine)


class SphericalVolume(ImplicitVBR):

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
        self._dim = len(lb)
        self._bounding_box = np.array([np.c_[lb, ub]])
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

    def __hash__(self):
        return hash(self.bounding_box.tobytes())

    def __eq__(self, other) -> bool:
        return np.all(self._center == other._center) and self._radius == other._radius

    def __repr__(self):
        return 'SphericalVolume(Center={}, Radius={})'.format(tuple(self._center), self._radius)


class PlanarVolume(ImplicitVBR):

    def __init__(self, origin, vector, direction=1, limit=1000):
        self._origin = np.array(origin)
        if not np.any([vector[i] > 0 for i in range(len(vector))]):
            raise NeuroLangException('Vector normal to the plane must be non-zero')
        self._vector = np.array(vector) / np.linalg.norm(vector)
        self._bounding_box = None

        if direction not in [1, -1]:
            raise NeuroLangException('Direction must either be 1 (superior to) or -1 (inferior to)')
        self._dir = direction
        if limit <= 0:
            raise NeuroLangException('Limit must be a positive value')
        self._limit = limit

    def point_in_plane(self, point):
        return np.dot(self._vector, self._origin - point) == 0

    def project_point_to_plane(self, point):
        point = np.array(point)
        d = np.dot(self._vector, point)
        return point - d * self._vector

    @property
    def bounding_box(self):
        if self._bounding_box is not None:
            return self._bounding_box

        outside = (self._dir * self._limit,) * 3
        inside = self.project_point_to_plane(outside) * -1
        [lb, ub] = sorted([inside, outside], key=lambda x: x[0])
        self._bounding_box = np.array([np.c_[lb, ub]])
        self._dim = len(lb)
        return self._bounding_box

    def to_ijk(self, affine):
        bounds_voxels = nib.affines.apply_affine(np.linalg.inv(affine), np.array([self._lb, self._ub]))
        [xs, ys, zs] = [range(int(min(bounds_voxels[:, i])), int(max(bounds_voxels[:, i]))) for i in range(3)]

        #todo improve
        voxel_coordinates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    voxel_coordinates.append([x, y, z])
        return np.array(voxel_coordinates)

    def __hash__(self):
        return hash(self.bounding_box.tobytes())

    def __eq__(self, other) -> bool:
        return np.all(self._origin == other._origin) and self._vector == other._vector

    def __repr__(self):
        return 'PlanarVolume(Origin={}, Normal Vector={})'.format(tuple(self._origin), self._vector)
