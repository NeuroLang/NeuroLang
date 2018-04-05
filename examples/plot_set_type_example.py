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
        r = Region(a[:2], a[2:])
        res = nl.Constant[typing.AbstractSet[self.type]](
            frozenset((r,))
        )
        return res

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
    origin are Regions singleton (0, 0, 0, 0)
    unit_region are Regions singleton (0, 0, 1, 1)
    both are Regions in origin or in unit_region
    test are Regions not in unit_region
''')


#####################################
# Print the resulting symbol table
for k, v in nlc.symbol_table.items():
    print(k, ':', v)


#######################################
# Print the intermediate representation
# of a two statements
ir = nlc.get_intermediate_representation('''
    one are Regions in both
''')
print(ir[0])

