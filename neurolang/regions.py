import numpy as np
import nibabel as nib
import scipy.ndimage
from itertools import product

from .exceptions import NeuroLangException
from .aabb_tree import AABB, Tree, aabb_from_vertices, Node

__all__ = [
    'region_union', 'region_intersection', 'region_difference',
    'region_set_from_masked_data', 'take_principal_regions',
    'Region', 'VolumetricBrainRegion', 'PointSet',
    'ImplicitVBR', 'ExplicitVBR', 'ExplicitVBROverlay',
    'SphericalVolume', 'PlanarVolume'
]


def region_union(region_set, affine=None):
    region_set = (
        region for region in region_set
        if not isinstance(region, EmptyRegion)
    )
    return region_set_algebraic_op(set.union, region_set, affine)


def region_intersection(region_set, affine=None):
    if any(isinstance(region, EmptyRegion) for region in region_set):
        return EmptyRegion()    
    return region_set_algebraic_op(set.intersection, region_set, affine)


def region_difference(region_set, affine=None):
    return region_set_algebraic_op(set.difference, region_set, affine)


def region_set_algebraic_op(op, region_set, affine=None, n_dim=3):

    rs = list(filter(lambda x: x is not None, region_set))
    if affine is None:
        affine = next(iter(rs)).affine

    max_dim = (0,) * n_dim
    voxels_set_of_regions = []
    for region in rs:
        if not isinstance(region, ExplicitVBR):
            region = region.to_explicit_vbr(affine, max_dim)

        if (region.image_dim is not None and
                any(map(lambda x, y: x > y, region.image_dim, max_dim))):
            max_dim = region.image_dim
        voxels_set_of_regions.append(set(map(tuple, region.voxels)))

    result_voxels = np.array(tuple(op(*voxels_set_of_regions)))
    if len(result_voxels) == 0:
        return EmptyRegion()
    return ExplicitVBR(result_voxels, affine, max_dim)


class Region(object):
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

    @staticmethod
    def from_spatial_image_label(spatial_image, label, **kwargs):
        data = np.asanyarray(spatial_image.dataobj)
        voxels = np.transpose((data == label).nonzero())
        if 'image_dim' not in kwargs:
            kwargs['image_dim'] = spatial_image.shape

        if len(voxels) > 0:
            return ExplicitVBR(voxels, spatial_image.affine, **kwargs)
        else:
            return EmptyRegion()


class EmptyRegion(Region):
    def __init__(self):
        pass

    def __hash__(self):
        return hash(tuple())

    def __eq__(self, other):
        return isinstance(other, EmptyRegion)

    def __repr__(self):
        return 'EmptyRegion'


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

    def to_explicit_vbr(self, affine, image_shape):
        voxels_coordinates = self.to_ijk(affine)
        return ExplicitVBR(voxels_coordinates, affine, image_shape)


class PointSet(VolumetricBrainRegion):
    def __init__(
        self, points_ijk, affine_matrix, image_dim=None, prebuild_tree=False
    ):
        self.points_ijk = np.asanyarray(points_ijk, dtype=float)
        self.affine = affine_matrix
        self.affine_inv = np.linalg.inv(self.affine)
        self.image_dim = image_dim
        self._aabb_tree = None
        self._bounding_box = self.generate_bounding_box(self.points_ijk)
        if prebuild_tree:
            self.build_tree()

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def aabb_tree(self):
        if self._aabb_tree is None:
            self.build_tree()
        return self._aabb_tree

    def generate_bounding_box(self, points_ijk):
        points_xyz = nib.affines.apply_affine(
            self.affine, points_ijk
        )
        return aabb_from_vertices(points_xyz)

    def build_tree(self):
        box = self._bounding_box
        tree = Tree()
        tree.add(box)

        affine_matrix_inv = np.linalg.inv(self.affine)

        stack = list([(box.lb, box.ub, self.points_ijk, tree.root)])
        while stack:
            node = stack.pop()
            lb, ub, parent_voxels, tree_node = node

            # this only works if the affine matrix is diagonal
            middle_point_xyz = (lb + ub) / 2
            middle_point_ijk = nib.affines.apply_affine(
                affine_matrix_inv, middle_point_xyz
            )

            axes = np.argsort(ub - lb)
            for ax in axes:
                left_mask = parent_voxels.T[ax] <= middle_point_ijk[ax]
                points_left = parent_voxels[left_mask]
                if 0 < len(points_left) < len(parent_voxels):
                    break
            else:
                continue

            height = tree_node.height - 1

            box_left = self.generate_bounding_box(points_left)
            tree_node.left = Node(
                box=box_left, parent=tree_node, height=height
            )
            stack.append(
                (box_left.lb, box_left.ub, points_left, tree_node.left)
            )

            points_right = parent_voxels[~left_mask]
            box_right = self.generate_bounding_box(points_right)
            tree_node.right = Node(
                box=box_right, parent=tree_node, height=height
            )
            stack.append(
                (box_right.lb, box_right.ub, points_right, tree_node.right)
            )

            tree.height = -height

        stack = [tree.root]
        while stack:
            node = stack.pop()
            node.height = tree.height + node.height
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        self._aabb_tree = tree
        return tree

    def to_xyz(self, affine=None):
        if affine is None or np.allclose(affine, self.affine):
            affine = self.affine
        else:
            affine = np.dot(affine, np.linalg.inv(self.affine))
        return nib.affines.apply_affine(affine, self.points_ijk)

    def to_ijk(self, affine):
        return nib.affines.apply_affine(
            np.linalg.solve(affine, self.affine),
            self.points_ijk)

    def spatial_image(self, out=None, value=1):
        if out is None:
            mask = np.zeros(self.image_dim, dtype=np.int16)
            out = nib.spatialimages.SpatialImage(mask, self.affine)
        elif (out.shape != self.image_dim and
              not np.allclose(out.affine, self.affine)):
            raise ValueError("Image data has incompatible dimensionality")
        else:
            mask = np.asanyarray(out.dataobj)

        discrete_points_ijk = np.round(self.points_ijk).astype(int)
        mask[tuple(discrete_points_ijk.T)] = value
        return out

    def __eq__(self, other):
        return (np.array_equiv(self.affine, other.affine) and
                np.array_equiv(self.points_ijk, other.voxels))

    def __repr__(self):
        return (
            f'Region(PointSet= affine:{self.affine},'
            f' points_ijk:{self.points_ijk})'
        )

    def __hash__(self):
        return hash((self.points_ijk.tobytes(), self.affine.tobytes()))


class ExplicitVBR(VolumetricBrainRegion):
    def __init__(
        self, voxels, affine_matrix, image_dim=None, prebuild_tree=False
    ):
        self.voxels = np.asanyarray(voxels, dtype=int)
        self.affine = affine_matrix
        self.affine_inv = np.linalg.inv(self.affine)
        for ar in (self.voxels, self.affine, self.affine_inv):
            ar.setflags(write=False)
        self.image_dim = image_dim
        self._aabb_tree = None
        self._bounding_box = self.generate_bounding_box(self.voxels)
        if prebuild_tree:
            self.build_tree()

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def aabb_tree(self):
        if self._aabb_tree is None:
            self.build_tree()
        return self._aabb_tree

    def generate_bounding_box(self, voxels_ijk):
        voxels_xyz = nib.affines.apply_affine(
            self.affine, np.concatenate((voxels_ijk, voxels_ijk + 1), axis=0)
        )
        return aabb_from_vertices(voxels_xyz)

    def build_tree(self):
        box = self._bounding_box
        tree = Tree()
        tree.add(box)

        affine_matrix_inv = np.linalg.inv(self.affine)

        stack = list([(box.lb, box.ub, self.voxels, tree.root)])
        while stack:
            node = stack.pop()
            lb, ub, parent_voxels, tree_node = node

            # this only works if the affine matrix is diagonal
            middle_point_xyz = (lb + ub) / 2
            middle_point_ijk = nib.affines.apply_affine(
                affine_matrix_inv, middle_point_xyz
            )

            axes = np.argsort(ub - lb)
            for ax in axes:
                left_mask = parent_voxels.T[ax] <= middle_point_ijk[ax]
                voxels_left = parent_voxels[left_mask]
                if 0 < len(voxels_left) < len(parent_voxels):
                    break
            else:
                continue

            height = tree_node.height - 1

            box_left = self.generate_bounding_box(voxels_left)
            tree_node.left = Node(
                box=box_left, parent=tree_node, height=height
            )
            stack.append(
                (box_left.lb, box_left.ub, voxels_left, tree_node.left)
            )

            voxels_right = parent_voxels[~left_mask]
            box_right = self.generate_bounding_box(voxels_right)
            tree_node.right = Node(
                box=box_right, parent=tree_node, height=height
            )
            stack.append(
                (box_right.lb, box_right.ub, voxels_right, tree_node.right)
            )

            tree.height = -height

        stack = [tree.root]
        while stack:
            node = stack.pop()
            node.height = tree.height + node.height
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        self._aabb_tree = tree
        return tree

    def to_xyz(self, affine=None):
        if affine is None or np.allclose(affine, self.affine):
            affine = self.affine
        else:
            affine = np.dot(affine, np.linalg.inv(self.affine))
        return nib.affines.apply_affine(affine, self.voxels)

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
            mask = np.asanyarray(out.dataobj)

        mask[tuple(self.voxels.T)] = value
        return out

    def __eq__(self, other):
        return (np.array_equiv(self.affine, other.affine) and
                np.array_equiv(self.voxels, other.voxels))

    def __repr__(self):
        return f'Region(VBR= affine:{self.affine}, voxels:{self.voxels})'

    def __hash__(self):
        return hash((self.voxels.tobytes(), self.affine.tobytes()))


class ExplicitVBROverlay(ExplicitVBR):
    def __init__(
        self, voxels, affine_matrix, overlay,
        image_dim=None, prebuild_tree=False
    ):
        super().__init__(
            voxels, affine_matrix,
            image_dim=image_dim, prebuild_tree=prebuild_tree
        )
        self.overlay = np.atleast_2d(overlay)
        if self.overlay.shape == (1, len(self.voxels)):
            self.overlay = self.overlay.T
        self.overlay.setflags(write=False)
        if self.overlay.shape[0] != self.voxels.shape[0]:
            raise ValueError(
                'The length of the overlay must be '
                'the same as that of the voxels'
            )

    def spatial_image(self, out=None, background_value=0):
        if self.overlay.shape[1:] == (1,):
            image_dim = self.image_dim
        else:
            image_dim = self.image_dim + self.overlay.shape[1:]
        out = self._obtain_empty_spatial_image(
            out, image_dim, background_value
        )
        image_data = out.dataobj

        image_data[tuple(self.voxels.T)] = self.overlay.squeeze()
        return out

    def _obtain_empty_spatial_image(self, out, image_dim, background_value):
        if out is None:
            mask = np.zeros(
                image_dim,
                dtype=self.overlay.dtype
            )
            out = nib.spatialimages.SpatialImage(mask, self.affine)
        elif (
            out.shape != image_dim and
            not np.allclose(out.affine, self.affine) and
            self.overlay.dtype != out.dataobj.dtype
        ):
            raise ValueError(
                "Image data has incompatible dimensionality or type"
            )
        else:
            mask = np.asanyarray(out.dataobj)
            mask[:] = background_value
        return out

    def __eq__(self, other):
        return (
            isinstance(other, ExplicitVBROverlay) and
            np.array_equiv(self.affine, other.affine) and
            np.array_equiv(self.voxels, other.voxels) and
            np.array_equiv(self.overlay, other.overlay)
        )

    def __repr__(self):
        return (
            f'Region(VBR= affine:{self.affine}, '
            f'voxels:{self.voxels}, overlay:{self.overlay})'
        )

    def __hash__(self):
        return hash(
            (
                self.voxels.tobytes(),
                self.affine.tobytes(),
                self.overlay.tobytes()
            )
        )


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


class BoundigBoxSequenceElement(ImplicitVBR):
    def __init__(self, bounding_box_sequence):
        self.bounding_box_sequence = bounding_box_sequence
        self._bounding_box = self.generate_bounding_box(
            self.bounding_box_sequence
        )
        self._aabb_tree = None

    @staticmethod
    def generate_bounding_box(bounding_box_sequence):
        res = bounding_box_sequence[0]
        for bb in bounding_box_sequence[1:]:
            res = res.union(bb)
        return res

    @property
    def aabb_tree(self):
        if self._aabb_tree is None:
            self.build_tree()
        return self._aabb_tree

    def to_ijk(self, affine):
        midpoints_xyz = np.array([
            (b.ub + b.lb) / 2
            for b in self.bounding_box_sequence
        ])
        voxel_coordinates = np.round(nib.affines.apply_affine(
            np.linalg.inv(affine), midpoints_xyz
        ))
        return voxel_coordinates

    def build_tree(self):
        box = self._bounding_box
        tree = Tree()
        tree.add(box)

        stack = list([(box.lb, box.ub, self.bounding_box_sequence, tree.root)])
        while stack:
            node = stack.pop()
            lb, ub, parent_boxes, tree_node = node

            # this only works if the affine matrix is diagonal
            middle_point_xyz = (lb + ub) / 2

            axes = np.argsort(ub - lb)
            for ax in axes:
                ub_box_lower = ub.copy()
                ub_box_lower[ax] = middle_point_xyz[ax]
                box_lower = AABB(lb, ub_box_lower)

                boxes_lower = []
                boxes_higher = []
                for box in parent_boxes:
                    if box_lower.contains(box):
                        boxes_lower.append(box)
                    else:
                        boxes_higher.append(box)

                if 0 < len(boxes_lower) < len(parent_boxes):
                    break
            else:
                continue

            height = tree_node.height - 1
            box_left = self.generate_bounding_box(boxes_lower)
            tree_node.left = Node(
                box=box_left, parent=tree_node, height=height
            )
            stack.append(
                (box_left.lb, box_left.ub, boxes_lower, tree_node.left)
            )

            box_right = self.generate_bounding_box(boxes_higher)
            tree_node.right = Node(
                box=box_right, parent=tree_node, height=height
            )
            stack.append(
                (box_right.lb, box_right.ub, boxes_higher, tree_node.right)
            )
            tree.height = -height

        stack = [tree.root]
        while stack:
            node = stack.pop()
            node.height = tree.height + node.height
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        return tree


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

        if all(r.start == r.stop for r in ranges):
            voxel_coordinates = np.array([[r.start for r in ranges]])
        else:
            voxel_coordinates = np.array([
                point for point in
                np.array(list(product(*ranges))) if
                nib.affines.apply_affine(affine, point) in self
            ])

        return voxel_coordinates

    def __contains__(self, point):
        point = np.atleast_2d(point)
        return np.all(
            np.linalg.norm(self._center - point, axis=1) <= self._radius
        )

    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    def __eq__(self, other):
        return (np.array_equiv(self._center, other._center) and
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
        lb, ub = sorted([box_limit, box_limit_in_plane], key=lambda x: x[0])
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
        return np.all(
            np.sum(self._vector * (self._origin - point), axis=1) == 0
        )

    def __hash__(self):
        return hash(self.bounding_box.limits.tobytes())

    def __eq__(self, other):
        return (np.array_equiv(self._origin, other._origin) and
                np.array_equiv(self._vector, other._vector))

    def __repr__(self):
        return (f'PlanarVolume(Origin={tuple(self._origin)},'
                f' Normal Vector={self._vector})')
