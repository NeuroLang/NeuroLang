import typing
import re

from .CD_relations import cardinal_relation, inverse_directions
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

        def anatomical_direction_function(relation):

            def func(self, x: Region, y: Region) -> bool:
                def cardinal_directionality(rel):
                    return cardinal_relation(
                        x, y, rel,
                        refine_overlapping=False,
                        stop_at=None)

                return bool(
                        cardinal_directionality(relation) and not
                        cardinal_directionality(
                            inverse_directions[relation]) and not
                        cardinal_directionality('O'))
            return func

        for key, value in cardinal_operations.items():
            setattr(cls, f'predicate_{key}', build_function(value))

        anatomical_correct_operations = {
            k: cardinal_operations[k] for k in (
                'inferior_of', 'superior_of',
                'posterior_of', 'anterior_of'
                )
        }
        for key, value in anatomical_correct_operations.items():
            setattr(cls, f'predicate_anatomical_{key}',
                    anatomical_direction_function(value))

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
