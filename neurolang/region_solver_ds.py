import typing
import itertools
import re
from collections import OrderedDict

from . import neurolang as nl
from .CD_relations import (
    cardinal_relation, direction_from_relation, directions_dim_space,
    inverse_directions
)
from .regions import Region, region_union
from .solver import GenericSolver, DatalogSolver, is_conjunctive_expression
from .expressions import (
    Query, Expression, Constant, Symbol, FunctionApplication, is_subtype,
    ToBeInferred
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

        def anatomical_direction_function(relation):

            def func(self, x: Region, y: Region) -> bool:

                is_in_direction = cardinal_relation(
                    x, y, relation,
                    refine_overlapping=False,
                    stop_at=None
                )

                is_in_inverse_direction = cardinal_relation(
                    x, y, inverse_directions[relation],
                    refine_overlapping=False,
                    stop_at=None
                )

                is_overlapping = cardinal_relation(
                    x, y, cardinal_operations['overlapping'],
                    refine_overlapping=False,
                    stop_at=None
                )

                return bool(
                    is_in_direction and
                    not is_in_inverse_direction and
                    not is_overlapping)

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


class GetFunctionApplicationsWalker(ExpressionWalker):
    def __init__(self):
        self.function_applications = []

    @add_match(FunctionApplication)
    def function_application(self, expression):
        self.function_applications.append(expression)
        return super().function(expression)


def is_region_symbol(expression):
    return isinstance(expression, Symbol) and expression.type is Region


def is_tuple_of_region_symbols(expression):
    return (
        isinstance(expression, Constant) and
        is_subtype(expression.type, typing.Tuple) and
        all(is_region_symbol(x) for x in expression.value)
    )


class SpatialIndexRegionSolver(RegionSolver):

    def initialize_region_index(self):
        self.index = Tree()

    def add_region_to_index(self, region):
        self.index.add(region.bounding_box, regions={region})

    @add_match(
        Query,
        guard=lambda expression: (
            (
                is_region_symbol(expression.head) or
                is_tuple_of_region_symbols(expression.head)
            ) and
            expression.head._symbols == expression.body._symbols and
            is_conjunctive_expression(expression.body)
        )
    )
    def spatial_query_resolution(self, expression):

        cardinal_predicates = {
            self.symbol_table[relation]: relation for relation in (
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

        out_query_type = expression.type
        if out_query_type is ToBeInferred:
            out_query_type = typing.AbstractSet[expression.head.type]

        # rsw = ReplaceSymbolsByConstants(self.symbol_table)
        body = expression.body

        result = []

        # retrieve all function application in the expression
        get_af_walker = GetFunctionApplicationsWalker()
        get_af_walker.walk(body)
        function_applications = get_af_walker.function_applications

        # retrieve all constant regions in the symbol table
        region_to_symb_const = {
            maybe_constant.value: (symbol, maybe_constant)
            for symbol, maybe_constant
            in self.symbol_table.symbols_by_type(Region).items()
            if isinstance(maybe_constant, Constant)
        }

        all_regions = set(region_to_symb_const.keys())

        # we start with all regions in the symbol table
        reduced_regions = all_regions

        # and reduce this set accordingly
        # if we encounter any cardinal predicate
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

        if (
            isinstance(expression.head, Constant) and
            is_subtype(expression.head.type, typing.Tuple) and
            all(isinstance(a, Symbol) for a in expression.head.value)
        ):
            symbols_in_head = expression.head.value
        else:
            symbols_in_head = (expression.head,)


        symbol_domains = OrderedDict(
            zip(
                symbols_in_head,
                tuple((
                    (
                        region_to_symb_const[region]
                        for region in reduced_regions
                    )
                    for _ in range(len(symbols_in_head))
                ))
            )
        )

        for symbol_values in itertools.product(*symbol_domains.values()):
            rsw = ReplaceSymbolWalker(
                dict(
                    zip(
                        symbol_domains.keys(),
                        (s[1] for s in symbol_values)
                    )
                )
            )
            res = self.walk(rsw.walk(body))
            if isinstance(res, Constant) and res.value:
                if isinstance(expression.head, Symbol):
                    result.append(symbol_values[0][0])
                else:
                    result.append(tuple(zip(*symbol_values))[0])

        return Constant[out_query_type](frozenset(result))
