import numpy as np
import nibabel as nib
import scipy.ndimage
from itertools import product

from .exceptions import NeuroLangException
from .brain_tree import AABB, Tree, aabb_from_vertices

__all__ = [
    'region_union', 'region_intersection', 'region_difference',
    'region_set_from_masked_data', 'take_principal_regions',
    'Region', 'VolumetricBrainRegion',
    'ImplicitVBR', 'ExplicitVBR',
    'SphericalVolume', 'PlanarVolume'
]


def region_union(region_set, affine=None):
    return region_set_algebraic_op(set.union, region_set, affine)


def region_intersection(region_set, affine=None):
    return region_set_algebraic_op(set.intersection, region_set, affine)


def region_difference(region_set, affine=None):
    return region_set_algebraic_op(set.difference, region_set, affine)


def region_set_algebraic_op(op, region_set, affine=None):

    rs = set(filter(lambda x: x is not None, region_set))
    if affine is None:
        affine = next(iter(rs)).affine

    max_dim = (0,) * 3
    for region in region_set:
        if isinstance(region, ImplicitVBR):
            region = region.to_explicit_vbr(affine, max_dim)

        if not isinstance(region, ExplicitVBR):
            raise ValueError(f'Invalid type of region in set: {region}')

        if (region.image_dim is not None and
                any(map(lambda x, y: x > y, region.image_dim, max_dim))):
            max_dim = region.image_dim

    voxels_set_of_regions = [set(map(tuple, region.to_ijk(affine))) for
                             region in rs]
    result_voxels = np.array(list(op(*voxels_set_of_regions)), dtype=list)
    return ExplicitVBR(result_voxels, affine, max_dim)


class Region:
    def __init__(self, lb, ub):
        if not np.all([lb[i] < ub[i] for i in range(len(lb))]):
            raise NeuroLangException(
                'Lower bounds must be lower'
                ' than upper bounds when creating rectangular regions')
        self._bounding_box = AABB(lb, ub)

    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def center(self):
        return self.bounding_box.center

    @property
    def width(self):
        return self.bounding_box.width

    def __eq__(self, other):
        return self.bounding_box == other.bounding_box

    def __repr__(self):
        return f'Region(AABB={self.bounding_box})'


class VolumetricBrainRegion(Region):
    def to_xyz(self, affine):
        """return world coordinates of the region
         corresponding to the affine matrix transform"""
        raise NotImplementedError()

    def to_ijk(self, affine):
        """return ijk voxels coordinates
         corresponding to the affine matrix transform"""
        raise NotImplementedError()

    def to_xyz_set(self, affine):
        raise NotImplementedError()

    def to_ijk_set(self, affine):
        raise NotImplementedError()

    def remove_empty_bounding_boxes(self):
        raise NotImplementedError()


class ExplicitVBR(VolumetricBrainRegion):
    def __init__(self, voxels, affine_matrix, img_dim=None):
        self.voxels = np.asanyarray(voxels, dtype=int)
        self.affine = affine_matrix
        self.image_dim = img_dim
        self._aabb_tree = None

    @property
    def bounding_box(self):
        return self.aabb_tree.root.box

    @property
    def aabb_tree(self):
        if self._aabb_tree is None:
            self._aabb_tree = self.build_tree()
        return self._aabb_tree

    def generate_bounding_box(self, voxels_ijk):
        voxels_xyz = nib.affines.apply_affine(self.affine, voxels_ijk)
        voxels_xyz_u = nib.affines.apply_affine(self.affine, voxels_ijk + 1)
        voxels_xyz = np.concatenate((voxels_xyz, voxels_xyz_u), axis=0)
        return aabb_from_vertices(voxels_xyz)

    def build_tree(self):
        box = self.generate_bounding_box(self.voxels)

        nodes = {}
        nodes[0] = [box.lb, box.ub, self.voxels]
        tree = Tree()
        tree.add(box)
        last_added = 0
        i = 1

        make_tree = True
        affine_matrix_inv = np.linalg.inv(self.affine)
        middle = np.zeros(box.dim)

        while make_tree:
            parent = ((i + 1) // 2) - 1

            if parent in nodes:
                lb, ub, parent_voxels = nodes[parent]
                ax = np.argmax(ub - lb)
                middle[:] = 0
                middle[ax] = (lb[ax] + ub[ax]) / 2
                middle_voxel = nib.affines.apply_affine(
                    affine_matrix_inv, middle)[ax]
                # this only works if the affine matrix is diagonal

                b1_voxs = parent_voxels[parent_voxels.T[ax] <= middle_voxel]
                if len(b1_voxs) != 0 and len(b1_voxs) != len(parent_voxels):
                    box1 = self.generate_bounding_box(b1_voxs)
                    tree.add_left(box1)
                    nodes[i] = [box1.lb, box1.ub, b1_voxs]
                    last_added = i

                b2_voxs = parent_voxels[parent_voxels.T[ax] > middle_voxel]
                if len(b2_voxs) != 0 and len(b2_voxs) != len(parent_voxels):
                    box2 = self.generate_bounding_box(b2_voxs)
                    tree.add_right(box2)
                    nodes[i + 1] = [box2.lb, box2.ub, b2_voxs]
                    last_added = i + 1

                if last_added == parent:
                    make_tree = False

            i += 2
        return tree

    def to_xyz(self):
        return nib.affines.apply_affine(self.affine, self.voxels)

    def to_ijk(self, affine):
        return nib.affines.apply_affine(
            np.linalg.solve(affine, self.affine),
            self.voxels)

    def spatial_image(self, out=None, value=1):
        if out is None:
            mask = np.zeros(self.image_dim, dtype=np.int16)
            out = nib.spatialimages.SpatialImage(mask, self.affine)
        elif (out.shape != self.image_dim and
              not np.allclose(out.affine, self.affine)):
            raise ValueError("Image data has incompatible dimensionality")
        else:
            mask = out.get_data()

        mask[tuple(self.voxels.T)] = value
        return out

    def __eq__(self, other):
        return (np.all(self.affine == other.affine) and
                np.all(self.voxels == other.voxels))

    def __repr__(self):
        return f'Region(VBR= affine:{self.affine}, voxels:{self.voxels})'

    def __hash__(self):
        return hash((self.voxels.tobytes(), self.affine.tobytes()))


def region_set_from_masked_data(data, affine, dim):
    s = scipy.ndimage.generate_binary_structure(3, 2)
    labeled_array, num_features = scipy.ndimage.label(data, structure=s)
    regions = set()
    for i in range(1, num_features):
        region_voxels = list(zip(*np.where(labeled_array == i)))
        regions.add(ExplicitVBR(region_voxels, affine, dim))

    return set(regions)


def take_principal_regions(region_set, k):
    sorted_by_size = sorted(list(region_set),
                            key=lambda x: len(x.voxels), reverse=True)
    return set(sorted_by_size[:k])


class ImplicitVBR(VolumetricBrainRegion):
    def __contains__(self, voxel):
        raise NotImplementedError()

    def to_ijk(self, affine):
        raise NotImplementedError()

    def to_xyz(self, affine):
        raise NotImplementedError()

    def to_explicit_vbr(self, affine, image_shape):
        voxels_coordinates = self.to_ijk(affine)
        return ExplicitVBR(voxels_coordinates, affine, image_shape)


class SphericalVolume(ImplicitVBR):
    def __init__(self, center, radius):
        self._center = np.asanyarray(center, dtype=int)
        self._radius = np.asanyarray(radius, dtype=int)
        lb = self._center - self._radius
        ub = self._center + self._radius
        self._bounding_box = AABB(lb, ub)

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    @property
    def bounding_box(self):
        return self._bounding_box

    def to_ijk(self, affine):
        bb = self.bounding_box
        bounds_voxels = nib.affines.apply_affine(
            np.linalg.inv(affine), np.array([bb.lb, bb.ub]))
        ranges = [range(int(min(bounds_voxels[:, i])),
                        int(max(bounds_voxels[:, i])))
                  for i in range(bb.dim)]

        voxel_coordinates = np.array([point for point in
                                      np.array(list(product(*ranges))) if
                                      nib.affines.apply_affine(affine, point)
                                      in self])
        return voxel_coordinates

    def __contains__(self, point):
        point = np.atleast_2d(point)
        return np.all(np.linalg.norm(self._center - point, axis=1) <= self._radius)

    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    def __eq__(self, other):
        return (np.all(self._center == other._center) and
                self._radius == other._radius)

    def __repr__(self):
        return (f'SphericalVolume(Center={tuple(self._center)}, '
                f'Radius={self._radius})')


class PlanarVolume(ImplicitVBR):
    def __init__(self, origin, vector, direction=1, limit=1000):
        self._origin = np.array(origin)

        if not np.any([vector[i] > 0 for i in range(len(vector))]):
            raise ValueError('Vector normal to the plane must be non-zero')
        self._vector = np.array(vector) / np.linalg.norm(vector)

        if direction not in [1, -1]:
            raise ValueError('Direction must either be'
                             ' 1 (superior to) or -1 (inferior to)')
        self._dir = direction

        if limit <= 0:
            raise ValueError('Limit must be a positive value')
        self._limit = limit

        box_limit = np.asanyarray((self._dir * self._limit,) * 3, dtype=int)
        box_limit_in_plane = np.asanyarray(
            self.project_point_to_plane(box_limit), dtype=int) * -1
        [lb, ub] = sorted([box_limit, box_limit_in_plane], key=lambda x: x[0])
        self._bounding_box = AABB(lb, ub)

    def project_point_to_plane(self, point):
        point = np.array(point)
        dist = np.dot(point - self._origin, self._vector)
        return tuple(point - dist * self._vector)

    @property
    def bounding_box(self):
        return self._bounding_box

    def to_ijk(self, affine):
        bb = self.bounding_box
        bounds_voxels = nib.affines.apply_affine(np.linalg.inv(affine),
                                                 np.array([bb.lb, bb.ub]))
        ranges = [range(int(min(bounds_voxels[:, i])),
                        int(max(bounds_voxels[:, i])))
                  for i in range(bb.dim)]

        return np.array(list(product(*ranges)))

    def __contains__(self, point):
        point = np.atleast_2d(point)
        return np.all(np.sum(self._vector * (self._origin - point), axis=1) == 0)


    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    def __eq__(self, other):
        return (np.all(self._origin == other._origin) and
                np.all(self._vector == other._vector))

    def __repr__(self):
        return (f'PlanarVolume(Origin={tuple(self._origin)},'
                f' Normal Vector={self._vector})')
