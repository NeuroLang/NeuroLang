from .exceptions import NeuroLangException
from .utils import data_manipulation
from .brain_tree import AABB, Tree, aabb_from_vertices
import numpy as np
import nibabel as nib
import copy
from numpy.linalg import inv


#code repetition, could be abstracted to one method (region, aff, set_op)
def region_union(regions_set, affine):
    voxels_per_regions = [set(map(tuple, elem.to_ijk(affine))) for elem in regions_set] # first convert to array of tuples
    result_voxels = set.union(*voxels_per_regions)
    return ExplicitVBR(np.array(list(map(list, result_voxels))), affine)  # then convert back to 2d array, FIX!


def region_intersection(regions_set, affine):
    voxels_per_regions = [set(map(tuple, elem.to_ijk(affine))) for elem in regions_set]
    result_voxels = set.intersection(*voxels_per_regions)
    if len(result_voxels) == 0:
        return None
    return ExplicitVBR(np.array(list(map(list, result_voxels))), affine)


def region_difference(regions_set, affine):
    voxels_per_regions = [set(map(tuple, elem.to_ijk(affine))) for elem in regions_set]
    result_voxels = set.difference(*voxels_per_regions)
    if len(result_voxels) == 0:
        return None
    return ExplicitVBR(np.array(list(map(list, result_voxels))), affine)


class Region:

    def __init__(self, lb, ub) -> None:
        if not np.all([lb[i] < ub[i] for i in range(len(lb))]):
            raise NeuroLangException('Lower bounds must be lower than upper bounds when creating rectangular regions')
        self._aabb_tree = Tree()
        self._aabb_tree.add(AABB(lb, ub))

    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    @property
    def bounding_box(self):
        return self.aabb_tree.root.box

    @property
    def aabb_tree(self):
        return self._aabb_tree

    @property
    def center(self) -> np.array:
        return self.bounding_box.center

    @property
    def width(self) -> np.array:
        return self.bounding_box.width

    def __eq__(self, other) -> bool:
        return self.bounding_box == other.bounding_box

    def __repr__(self):
        return 'Region(AABB={})'.format(self.bounding_box)

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
        self._aabb_tree = None

    @property
    def bounding_box(self):
        return self.aabb_tree.root.box

    @property
    def aabb_tree(self):
        if self._aabb_tree is None:
            self._aabb_tree = self.build_tree()
        return self._aabb_tree

    def build_tree(self):
        box = aabb_from_vertices(self.to_xyz())
        voxels_width = abs(np.linalg.eigvals(self._affine_matrix)[:-1])

        nodes = {}
        nodes[0] = [box.lb, box.ub, self._voxels]
        tree = Tree()
        tree.add(box)
        i = 1
        make_tree = True
        while make_tree:

            parent = ((i + 1) // 2) - 1
            if parent not in nodes.keys():
                i += 2
                continue
            [lb, ub, parent_voxels] = nodes[parent]
            ax = np.argmax(ub - lb)
            middle = np.zeros(box.dim)
            middle[ax] = (lb[ax] + ub[ax]) / 2
            middle_voxel = nib.affines.apply_affine(np.linalg.inv(self._affine_matrix), middle)[
                ax]  # this only works if the affine matrix is diagonal
            b1_voxs = parent_voxels[parent_voxels.T[ax] <= middle_voxel]
            b2_voxs = parent_voxels[parent_voxels.T[ax] > middle_voxel]

            if not (len(b1_voxs) == 0 and len(b2_voxs) == 0):
                box1, box2 = aabb_from_vertices(
                    nib.affines.apply_affine(self._affine_matrix, b1_voxs)), aabb_from_vertices(
                    nib.affines.apply_affine(self._affine_matrix, b2_voxs))

                if np.any(box1.ub - box1.lb < voxels_width * 2):
                    make_tree = False
                else:
                    if len(b1_voxs) != 0:
                        tree.add(box1)
                        nodes[i] = [box1.lb, box1.ub, b1_voxs]
                    if len(b2_voxs) != 0:
                        tree.add(box2)
                        nodes[i + 1] = [box2.lb, box2.ub, b2_voxs]
            i += 2
        return tree

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

    def to_explicit_vbr(self, affine):
        voxels_coordinates = self.to_ijk(affine)
        return ExplicitVBR(voxels_coordinates, affine)


class SphericalVolume(ImplicitVBR):

    def __init__(self, center, radius):
        self._center = center
        self._radius = radius
        self._aabb_tree = None

    @property
    def bounding_box(self):
        return self.aabb_tree.root.box

    @property
    def aabb_tree(self):
        if self._aabb_tree is not None:
            return self._aabb_tree
        self._aabb_tree = Tree()
        lb = tuple(np.array(self._center) - self._radius)
        ub = tuple(np.array(self._center) + self._radius)
        self._aabb_tree.add(AABB(lb, ub))
        return self._aabb_tree

    def to_ijk(self, affine):
        bb = self.bounding_box
        bounds_voxels = nib.affines.apply_affine(np.linalg.inv(affine), np.array([bb.lb, bb.ub]))
        [xs, ys, zs] = [range(int(min(bounds_voxels[:, i])), int(max(bounds_voxels[:, i]))) for i in range(bb.dim)]

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
        return hash(self.bounding_box.limits.tobytes())

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
        self._aabb_tree = None

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
        return self.aabb_tree.root.box

    @property
    def aabb_tree(self):
        if self._aabb_tree is not None:
            return self._aabb_tree
        self._aabb_tree = Tree()
        outside = (self._dir * self._limit,) * 3
        inside = self.project_point_to_plane(outside) * -1
        [lb, ub] = sorted([inside, outside], key=lambda x: x[0])
        self._aabb_tree.add(AABB(lb, ub))
        return self._aabb_tree

    def to_ijk(self, affine):
        bb = self.bounding_box
        bounds_voxels = nib.affines.apply_affine(np.linalg.inv(affine), np.array([bb.lb, bb.ub]))
        [xs, ys, zs] = [range(int(min(bounds_voxels[:, i])), int(max(bounds_voxels[:, i]))) for i in range(3)]

        #todo improve
        voxel_coordinates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    voxel_coordinates.append([x, y, z])
        return np.array(voxel_coordinates)

    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    def __eq__(self, other) -> bool:
        return np.all(self._origin == other._origin) and self._vector == other._vector

    def __repr__(self):
        return 'PlanarVolume(Origin={}, Normal Vector={})'.format(tuple(self._origin), self._vector)
