import typing
import itertools
import re

from . import neurolang as nl
from .CD_relations import (cardinal_relation,
                           direction_from_relation,
                           directions_dim_space)
from .regions import Region, region_union
from .solver import GenericSolver, DatalogSolver, is_conjunctive_expression
from .expressions import (
    Query, Expression, Constant, Symbol, FunctionApplication, is_subtype
)
from .expression_walker import (
    add_match, ExpressionWalker, ReplaceSymbolWalker, ReplaceSymbolsByConstants
)
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


class GetFunctionApplicationsWalker(ExpressionWalker):
    def __init__(self):
        self.function_applications = []

    @add_match(FunctionApplication)
    def function_application(self, expression):
        self.function_applications.append(expression)
        return super().function(expression)


class SpatialIndexRegionSolver(RegionSolver):

    def initialize_region_index(self):
        self.index = Tree()

    def add_region_to_index(self, region):
        self.index.add(region.bounding_box, regions={region})

    @add_match(
        Query(Symbol[Region], ...),
        guard=lambda expression: (
            expression.head._symbols == expression.body._symbols and
            is_conjunctive_expression(expression.body)
        )
    )
    def spatial_query_resolution(self, expression):

        cardinal_predicates = {
            self.included_predicates[relation]: relation for relation in (
                'inferior_of', 'superior_of',
                'posterior_of', 'anterior_of',
                'left_of', 'right_of',
            )
        }

        cardinal_operations = {
            'inferior_of': 'I', 'superior_of': 'S',
            'posterior_of': 'P', 'anterior_of': 'A',
            'left_of': 'L', 'right_of': 'R',
        }

        out_query_type = Region

        # rsw = ReplaceSymbolsByConstants(self.symbol_table)
        body = expression.body

        result = []

        if (
            expression.head not in body._symbols or
            len(body._symbols) > 1
        ):
            raise NotImplementedError(
                "All free symbols in the body must be in the head"
            )

        # retrieve all function application in the expression
        get_af_walker = GetFunctionApplicationsWalker()
        get_af_walker.walk(body)
        function_applications = get_af_walker.function_applications

        # retrieve all constant regions in the symbol table
        region_to_constant_and_symbol = {
            region.value: (region, symbol)
            for symbol, region
            in self.symbol_table.symbols_by_type(Region).items()
            if isinstance(region, Constant)
        }

        all_regions = set(region_to_constant_and_symbol.keys())

        # we start with all regions in the symbol table
        reduced_regions = all_regions

        # and reduce this set accordingly if we encounter any spatial relation
        for function_application in function_applications:
            if (
                function_application.functor in cardinal_predicates and
                function_application.args[0] is expression.head and
                isinstance(function_application.args[1], Constant)
            ):
                relation = cardinal_predicates[function_application.functor]
                anatomical_direction = cardinal_operations[relation]
                direction = direction_from_relation[anatomical_direction]
                axis = directions_dim_space[anatomical_direction][0]
                relative_region = function_application.args[1].value
                matching_regions = self.index.query_regions_axdir(
                    relative_region, axis=axis, direction=direction
                )
                reduced_regions.intersection_update(matching_regions)

        for region in reduced_regions:
            constant, symbol = region_to_constant_and_symbol[region]
            rsw = ReplaceSymbolWalker(expression.head, constant)
            rsw_body = rsw.walk(body)

            res = self.walk(rsw_body)
            if isinstance(res, Constant) and res.value:
                result.append(symbol)

        return Constant[typing.AbstractSet[out_query_type]](frozenset(result))
