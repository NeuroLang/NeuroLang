import typing
import re

from .CD_relations import cardinal_relation
from .regions import Region, region_union
from .solver import DatalogSolver
from .expressions import Constant


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
