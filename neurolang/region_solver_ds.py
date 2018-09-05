import typing
import re

from .CD_relations import cardinal_relation, inverse_directions
from .regions import Region, region_union
from .expression_walker import PatternWalker
from .expressions import Constant


class RegionSolver(PatternWalker[Region]):
    type_name = 'Region'

    def __new__(cls, *args, **kwargs):
        cardinal_operations = {
            'inferior_of': 'I', 'superior_of': 'S',
            'posterior_of': 'P', 'anterior_of': 'A',
            'left_of': 'L', 'right_of': 'R',
            'overlapping': 'O'
        }

        def build_function(relation, refine_overlapping=False):
            def f(self, x: Region, y: Region) -> bool:
                return bool(cardinal_relation(
                    x, y, relation,
                    refine_overlapping=refine_overlapping,
                    stop_at=None
                ))
            return f

        def anatomical_direction_function(relation, refine_overlapping=False):

            def func(self, x: Region, y: Region) -> bool:

                is_in_direction = cardinal_relation(
                    x, y, relation,
                    refine_overlapping=refine_overlapping,
                    stop_at=None
                )

                is_in_inverse_direction = cardinal_relation(
                    x, y, inverse_directions[relation],
                    refine_overlapping=refine_overlapping,
                    stop_at=None
                )

                is_overlapping = cardinal_relation(
                    x, y, cardinal_operations['overlapping'],
                    refine_overlapping=refine_overlapping,
                    stop_at=None
                )

                return bool(
                    is_in_direction and
                    not is_in_inverse_direction and
                    not is_overlapping)

            return func

        refine_overlapping = kwargs.get('refine_overlapping', False)

        for key, value in cardinal_operations.items():
            setattr(
                cls, f'function_{key}',
                build_function(value, refine_overlapping=refine_overlapping)
            )

        anatomical_correct_operations = {
            k: cardinal_operations[k] for k in (
                'inferior_of', 'superior_of',
                'posterior_of', 'anterior_of'
                )
        }
        for key, value in anatomical_correct_operations.items():
            setattr(
                cls, f'function_anatomical_{key}',
                anatomical_direction_function(
                    value, refine_overlapping=refine_overlapping
                )
            )

        return PatternWalker.__new__(cls)

    def function_regexp(
        self, regexp: typing.Text
    ) -> typing.AbstractSet[Region]:
        regions = []
        for k in self.symbol_table.symbols_by_type(Region):
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
