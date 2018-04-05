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
# Generate a compiler that adds the previously defined type to the
# language. This works using the mixin pattern.
class NeuroLangCompiler(
    IntSetType,
    nl.NeuroLangIntermediateRepresentationCompiler
):
    pass


nlc = NeuroLangCompiler()

#####################
# Run three queries
nlc.compile('''
    oneset are Integers singleton 1
    twoset are Integers singleton 2
    onetwoset are Integers in oneset or in twoset
    evens_until_ten are Integers evens_until 10
    empty are Integers in evens_until_ten and in oneset
    two_again are Integers in evens_until_ten and in onetwoset
''')


#####################################
# Print the resulting symbol table
for k, v in nlc.symbol_table.items():
    print(k, ':', v)


#######################################
# Print the intermediate representation
# of a two statements
ir = nlc.get_intermediate_representation('''
    otherset are Integers in onetwoset
    a = 5
''')
print(ir[0])
print('-' * 10)
print(ir[1])
