'''
Example in adding a new type with set semantics to the NeuroLang language
=========================================================================

A type with :code:`set` semantics has the following operations

- A predicate :code:`in <set>` which results in the :code:`<set>` given as parameter
- :code:`and` and :code:`or` operations between sets which are disjunction and conjunction

From the script, the resulting symbol table is
'''

import neurolang as nl
from neurolang.solver import SetBasedSolver
import typing
import operator
from neurolang.regions import *
from neurolang.RCD_relations import *


###################################################################
# Make a class to handle sets of regions. The typename is Region
# automatically, the regions set name is Regions ('Region' + 's')
class RegionsSetType(SetBasedSolver):
    type = Region
    type_name = 'Region'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pred_type = typing.Callable[
            [typing.AbstractSet[self.type], ],
            typing.AbstractSet[self.type]
        ]

        for direction_function in [self.north_of, self.south_of, self.east_of, self.west_of, self.overlapping]:
            self.symbol_table[
                nl.Symbol[pred_type](direction_function.__name__)
            ] = nl.Constant[pred_type](direction_function)

    def north_of(self, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        return self.direction('N', reference_region)

    def south_of(self, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        return self.direction('S', reference_region)

    def west_of(self, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        return self.direction('W', reference_region)

    def east_of(self, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        return self.direction('E', reference_region)

    def overlapping(self, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        return self.direction('O', reference_region)

    # add a match for the predicate "singleton" with a region as parameter
    # that will produce a set with just that region as a result
    @nl.add_match(nl.Predicate(nl.Symbol('singleton'), (nl.Constant[typing.Tuple[int, int, int, int]],)))
    def singleton(self, expression):
        a = expression.args[0].value
        r = Region((a[0].value, a[1].value), (a[2].value, a[3].value))
        res = nl.Constant[typing.AbstractSet[self.type]](
            frozenset((r,))
        )
        return res

    def direction(self, direction, reference_region: typing.AbstractSet[Region]) -> typing.AbstractSet[Region]:
        result, visited = frozenset(), frozenset()
        for symbol in self.symbol_table.symbols_by_type(typing.AbstractSet[Region]).values():
            regions_set = symbol.value
            if not regions_set <= visited:
                for region in regions_set:
                    is_north = True
                    for elem in reference_region:
                        mat = direction_matrix(region, elem)
                        if not is_in_direction(mat, direction) or (region in reference_region):
                            is_north = False
                            break
                    if is_north:
                        result = result.union(frozenset((region,)))
                visited = visited.union(regions_set)

        return self.walk(nl.Constant[typing.AbstractSet[Region]](result))

    @nl.add_match(nl.FunctionApplication(nl.Constant(operator.invert), (nl.Constant[typing.AbstractSet],)))
    def rewrite_finite_domain_inversion(self, expression):
        set_constant = expression.args[0]
        set_type, set_value = nl.get_type_and_value(set_constant)
        all_regions = frozenset(
            (
                v.value for v in
                self.symbol_table.symbols_by_type(
                    set_type.__args__[0]
                ).values()
            )
        )
        for v in self.symbol_table.symbols_by_type(set_type).values():
            all_regions = all_regions.union(v.value)

        result = all_regions - set_value
        return self.walk(nl.Constant[set_type](result))


##################################################################
class NeuroLangCompiler(
    RegionsSetType,
    nl.NeuroLangIntermediateRepresentationCompiler
):
    pass

# Generate a compiler that adds the previously defined type to the
# language. This works using the mixin pattern.


nlc = NeuroLangCompiler()

#####################
# Run three queries
nlc.compile('''
    unit_region are Regions singleton (0, 0, 1, 1)
    box5 are Regions singleton (0, 0, 5, 5)
    a_rectangle are Regions singleton (2, 1, 3, 2)
    another_rectangle are Regions singleton (-5, -5, 4, -4)
    rect_q4 are Regions singleton (0, -8, 4, -7)
    negative_unit are Regions singleton (-1, -1, 0, 0)
    test are Regions not in unit_region

    foo are Regions north_of another_rectangle
    bar are Regions north_of rect_q4
    two_rectangles are Regions in another_rectangle or in rect_q4
    func are Regions north_of two_rectangles
    both_limits are Regions in rect_q4 or in box5
    empty are Regions north_of both_limits

    southern are Regions south_of unit_region

    within are Regions overlapping box5

''')

#####################################
# Print the resulting symbol table
for k, v in nlc.symbol_table.items():
   print(k, ':', v)