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
# Make a class to handle sets of integers. The typename is Integer
# automatically, the integer set name is Integers ('Integer' + 's')
class IntSetType(SetBasedSolver):
    type = int
    type_name = 'Integer'

    # add a match for the predicate "singleton" with an int as parameter
    # that will produce a set with just that int as a result
    @nl.add_match(nl.Predicate(nl.Symbol('singleton'), (nl.Constant[int],)))
    def singleton(self, expression):
        a = expression.args[0]
        res = nl.Constant[typing.AbstractSet[self.type]](
            set((a,))
        )
        return res

    @nl.add_match(nl.Predicate(nl.Symbol('evens_until'), (nl.Constant[int],)))
    def evens_until(self, expression):
        until = expression.args[0].value
        res = nl.Constant[typing.AbstractSet[self.type]](
            set((nl.Constant[int](i) for i in range(0, until, 2)))
        )
        return res


##################################################################
class RegionsSetType(SetBasedSolver):
    type = Region
    type_name = 'Region'

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


    @nl.add_match(
        nl.Predicate(
            nl.Symbol('north_of'),
            (nl.Constant[typing.Tuple[int, int, int, int]],)
        )
    )
    def north_of(self, expression):
        a = expression.args[0].value
        r2 = Region((a[0].value, a[1].value), (a[2].value, a[3].value))     #todo region as arg, dont create new one from tuple

        result, visited = frozenset(), frozenset()
        for symb in self.symbol_table.symbols_by_type(typing.AbstractSet[Region]).values():
            elem = symb.value
            if not elem <= visited:
                for k in elem:
                    dir = direction_matrix(k, r2)
                    if dir[0, 1] == 1:
                        result = result.union(elem)
                visited = visited.union(elem)

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
    negative_unit are Regions singleton (-1, -1, 0, 0)
    both are Regions in box5 or in unit_region
    test are Regions not in unit_region
    northern_regions are Regions north_of (2, 1, 3, 2)
    foo are Regions north_of (-5, -5, 4, -4)
    bar are Regions north_of (0, -8, 4, -7)
''')

#####################################
# Print the resulting symbol table
for k, v in nlc.symbol_table.items():
   print(k, ':', v)