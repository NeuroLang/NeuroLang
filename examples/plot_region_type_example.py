'''
Example in adding a new type with set semantics to the NeuroLang language
=========================================================================

A type with :code:`set` semantics has the following operations

- A predicate :code:`in <set>` which results in the :code:`<set>` given as parameter
- :code:`and` and :code:`or` operations between sets which are disjunction and conjunction

From the script, the resulting symbol table is
'''

from neurolang.region_solver import RegionsSetSolver
import neurolang as nl



class NeuroLangCompiler(
    RegionsSetSolver,
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
    empty are Regions north_of both_limits or in rect_q4

    southern are Regions south_of unit_region

    within are Regions overlapping box5
''')

#####################################
# Print the resulting symbol table
for k, v in nlc.symbol_table.items():
   print(k, ':', v)

print('*' * 20)
p1 = nl.Predicate(nl.Symbol('north_of'), (nl.Symbol('two_rectangles'),))
p2 = nl.Predicate(nl.Symbol('north_of'), (nl.Symbol('rect_q4'),))

print(p1, p2)
print('res1', nlc.walk(p1))
print('res2', nlc.walk(p2))
print('*' * 20)
print(~p2)
print(nlc.walk(p1 & p2))
print(nlc.walk(p1 | p2))


ir = nlc.get_intermediate_representation('''
    foobar are Regions north_of unit_region
''')
print(ir[0])
print('-' * 10)
