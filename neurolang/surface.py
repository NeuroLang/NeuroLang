from collections.abc import Set
import typing
import operator

import nibabel as nib
import numpy as np

from .solver import GenericSolver, SetBasedSolver
from .symbols_and_types import TypedSymbol


class Surface(Set):
    def __init__(self, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles
        self.vertices.setflags(write=False)
        self.triangles.setflags(write=False)
        self.triangle_ids = set(range(len(triangles)))
        self.surface = self

    def __contains__(self, element):
        return element in self.triangles

    def __iter__(self):
        return iter(self.triangles)

    def __len__(self):
        return len(self.triangles)

    def __hash__(self):
        return hash((
            self.vertices.tobytes(),
            self.triangles.tobytes()
        ))


class Overlay(object):
    def __init__(self, overlay):
        self.overlay = overlay

    def value(self, vertex):
        return self.overlay[vertex]


class SurfaceSolver(GenericSolver):
    type_name = "surface"
    type = Surface

    def predicate_from_file(self, filename: str)->Surface:
        f = nib.load(filename)
        return Surface(f.darrays[0].data, f.darrays[1].data)


class Region(Surface):
    def __init__(self, parameter, triangle_ids=None):
        if isinstance(parameter, Surface):
            surface = parameter
        else:
            surface = parameter.surface

        self.triangles = surface.triangles
        self.vertices = surface.vertices
        self.surface = surface

        if triangle_ids is None:
            triangle_ids = frozenset(i for i in range(len(surface.triangles)))
        else:
            triangle_ids = frozenset(triangle_ids)
            if not triangle_ids.issubset(np.arange(len(self.triangles))):
                raise ValueError(
                    "Triangles given are not a subset of the surface"
                )

        self.triangle_ids = triangle_ids

        if isinstance(parameter, typing.Callable):
            mask = parameter(self)
            self.triangle_ids = frozenset((
                k for k, v in mask.items()
                if v
            ))

        if len(self.triangle_ids) > 0:
            triangle_ids = set(self.triangle_ids.copy())

            triangle_sides = {
                k: {
                        tuple(sorted(self.triangles[k][[0, 1]])),
                        tuple(sorted(self.triangles[k][[0, 2]])),
                        tuple(sorted(self.triangles[k][[1, 2]]))
                    }
                for k in triangle_ids
            }

            triangle_id = triangle_ids.pop()
            region_sides = triangle_sides[triangle_id].copy()

            while len(triangle_ids) > 0:
                for next_triangle_id in triangle_ids:
                    if len(
                        triangle_sides[next_triangle_id].
                        intersection(region_sides)
                    ) > 0:
                        break
                else:
                    raise ValueError("Region triangles are not all contiguous")

                region_sides.update(triangle_sides[next_triangle_id])
                triangle_ids.remove(next_triangle_id)

        self.triangle_side_lengths = (
            np.linalg.norm(
                np.diff(
                    self.vertices[np.c_[self.triangles, self.triangles[:, 0]]],
                    axis=1),
                axis=1
            )
        )

        semiperimeter = self.triangle_side_lengths.sum(axis=1)
        self.triangle_area = np.sqrt((
            semiperimeter[:, None] - self.triangle_side_lengths
        ).prod(axis=1) * semiperimeter)

    def __contains__(self, element):
        return element in self.triangle_ids

    def __iter__(self):
        return iter(self.triangle_ids)

    def __len__(self):
        return len(self.triangle_ids)

    def __hash__(self):
        return hash((
            self.surface,
            self.triangle_ids
        ))

    def area(self)->float:
        return self.triangle_area.sum()


class SurfaceOverlay(typing.Callable[
        [Surface], typing.Mapping[int, typing.SupportsFloat]
]):
    def __init__(self, surface, vertex_mapping, interpolation=False):
        self.surface = surface

        self.vertices = self.surface.vertices
        self.triangles = self.surface.triangles

        self.vertex_mapping = vertex_mapping
        self.triangle_side_lengths = (
            np.linalg.norm(
                np.diff(
                    self.vertices[np.c_[self.triangles, self.triangles[:, 0]]],
                    axis=1),
                axis=1
            )
        )

        self.triangle_side_proportions = (
            self.triangle_side_lengths /
            np.sum(self.triangle_side_lengths, axis=1)[:, None]
        )

        if not interpolation:
            self.triangle_side_proportions = (
                self.triangle_side_proportions ==
                self.triangle_side_proportions.max(axis=1)[:, None]
            ).astype(float)

    def __operand(
        self, operator: typing.Callable, parameter: typing.SupportsFloat
    )->'SurfaceOverlay':
        return SurfaceOverlay(
            self.surface, operator(self.vertex_mapping, parameter)
        )

    def __eq__(self, parameter: typing.SupportsFloat)->'SurfaceOverlay':
        return self.__operand(operator.eq, parameter)

    def __ne__(self, parameter: typing.SupportsFloat)->'SurfaceOverlay':
        return self.__operand(operator.ne, parameter)

    def __lt__(self, parameter: typing.SupportsFloat)->'SurfaceOverlay':
        return self.__operand(operator.lt, parameter)

    def __le__(self, parameter: typing.SupportsFloat)->'SurfaceOverlay':
        return self.__operand(operator.le, parameter)

    def __gt__(self, parameter: typing.SupportsFloat)->'SurfaceOverlay':
        return self.__operand(operator.gt, parameter)

    def __ge__(self, parameter: typing.SupportsFloat)->'SurfaceOverlay':
        return self.__operand(operator.ge, parameter)

    def contiguous_regions(self)->typing.Iterator[Region]:
        triangle_ids_map = self(self.surface)
        triangle_ids = set(
            k for k, v in triangle_ids_map.items()
            if v
        )

        if len(triangle_ids) > 0:
            triangle_sides = {
                k: {
                        tuple(sorted(self.triangles[k][[0, 1]])),
                        tuple(sorted(self.triangles[k][[0, 2]])),
                        tuple(sorted(self.triangles[k][[1, 2]]))
                    }
                for k in triangle_ids
            }

            triangle_id = triangle_ids.pop()
            current_region = [triangle_id]
            region_sides = triangle_sides[triangle_id].copy()

            while len(triangle_ids) > 0:
                for next_triangle_id in triangle_ids:
                    if len(
                        triangle_sides[next_triangle_id].
                        intersection(region_sides)
                    ) > 0:
                        break
                else:
                    yield Region(self.surface, triangle_ids=current_region)
                    current_region = []
                    region_sides = set()

                region_sides.update(triangle_sides[next_triangle_id])
                current_region.append(next_triangle_id)
                triangle_ids.remove(next_triangle_id)

            if len(current_region) > 0:
                yield Region(self.surface, triangle_ids=current_region)

    def __call__(
        self, parameter: Surface
    )->typing.Mapping[int, typing.SupportsFloat]:
        if not (
            not isinstance(parameter, Surface)
            or parameter.surface is self.surface
        ):
            raise ValueError("Overlay defined on a different surface")

        triangle_ids = list(parameter.triangle_ids)
        return {
            k: v for k, v in
            zip(
                triangle_ids,
                (
                    self.vertex_mapping[self.surface.triangles[triangle_ids]] *
                    self.triangle_side_proportions[triangle_ids]
                ).sum(axis=-1).squeeze()
            )
        }


class RegionSolver(SetBasedSolver):
    type_name = "region"
    type = Region

    def predicate_on_surface(
        self, surface: Surface
    )->Region:
        return self.type(surface)

    def comparison_default(
        self, comparison: str, *operands: typing.Any
    )->typing.Union[Region, typing.AbstractSet[Region]]:
        comparison_operator = getattr(operator, comparison)
        result = operands[0]
        for operand in operands[1:]:
            result = comparison_operator(result, operand)

        if self.is_plural_evaluation:
            if hasattr(result, 'contiguous_regions'):
                return TypedSymbol(
                    typing.AbstractSet[self.type],
                    frozenset(result.contiguous_regions())
                )
        else:
            return TypedSymbol(
                self.type,
                self.type(result)
            )


class SurfaceOverlaySolver(GenericSolver):
    type_name = "surface_overlay"
    type = SurfaceOverlay

    def predicate_from_file(self, filename: str)->bool:
        f = nib.load(filename)

        self.symbol_table[self.identifier] = TypedSymbol(
            SurfaceOverlay,
            SurfaceOverlay(
                self.symbol_table[self.identifier['surface']].value,
                f.darrays[0].data
            )
        )
        return True

    def predicate_on_surface(self, surface: Surface)->bool:
        self.symbol_table[self.identifier['surface']] = TypedSymbol(
            Surface,
            surface
        )
        return True

    def execute(self, ast, plural=False, identifier=None):
        self.set_symbol_table(self.symbol_table.create_scope())
        self.identifier = identifier
        self.evaluate(ast)
        result = self.symbol_table[self.identifier]
        self.set_symbol_table(self.symbol_table.enclosing_scope)
        return result
