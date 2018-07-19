import typing
import re

from . import neurolang as nl
from .CD_relations import (cardinal_relation,
                           direction_from_relation,
                           directions_dim_space)
from .regions import Region, region_union
from .solver import GenericSolver, DatalogSolver
from .expressions import Constant, Symbol, FunctionApplication
from .expression_walker import add_match
from .brain_tree import Tree


class RegionSolver(DatalogSolver[Region]):
    type_name = 'Region'

    def __new__(cls, *args, **kwargs):
        cardinal_operations = {
            'inferior_of': 'I', 'superior_of': 'S',
            'posterior_of': 'P', 'anterior_of': 'A',
            'left_of': 'L', 'right_of': 'R',
            'overlapping': 'O'
        }

        def build_function(relation):
            def f(self, x: Region, y: Region) -> bool:
                return bool(cardinal_relation(
                    x, y, relation,
                    refine_overlapping=False,
                    stop_at=None
                ))
            return f

        for key, value in cardinal_operations.items():
            setattr(cls, f'predicate_{key}', build_function(value))

        return DatalogSolver.__new__(cls)

    def function_regexp(
        self, regexp: typing.Text
    ) -> typing.AbstractSet[Region]:
        regions = []
        for k, v in self.symbol_table.symbols_by_type(Region).items():
            if re.search(regexp, k.name):
                regions.append(k)

        return frozenset(regions)

    def function_region_union(
        self, region_set: typing.AbstractSet[Region]
    ) -> Region:

        new_region_set = []
        for region in region_set:
            region = self.walk(region)
            if not isinstance(region, Constant):
                raise ValueError(
                    "Region union can only be evaluated on resolved regions"
                )

            new_region_set.append(region.value)

        return region_union(new_region_set)


class IndexRegionSolver(GenericSolver[Region]):

    def initialize_region_index(self):
        self.index = Tree()

    def add_region_to_index(self, region):
        self.index.add(region.bounding_box, regions={region})

    def __new__(cls, *args, **kwargs):
        cardinal_operations = {
            'inferior_of': 'I', 'superior_of': 'S',
            'posterior_of': 'P', 'anterior_of': 'A',
            'left_of': 'L', 'right_of': 'R',
        }

        def build_function(relation):
            direction = direction_from_relation.get(relation)
            axis = directions_dim_space.get(relation)[0]
            def f(self, x: Region, y: Region) -> bool:
                matching = self.index.query_regions_axdir(
                    y, axis=axis, direction=direction
                )
                return (x in matching)
            return f

        for key, value in cardinal_operations.items():
            setattr(cls, f'predicate_{key}', build_function(value))

        return GenericSolver[Region].__new__(cls)
