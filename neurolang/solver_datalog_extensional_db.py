from typing import AbstractSet, Any, Tuple
from itertools import product
from operator import contains

from .expressions import (
    Expression, NullConstant, FunctionApplication, Constant,
    NeuroLangTypeException, is_leq_informative
)
from .expression_walker import (
    add_match, PatternWalker, ReplaceSymbolsByConstants
)

__all__ = ['ExtensionalDatabaseSolver']


class ExtensionalDatabaseSolver(PatternWalker):
    '''Mixin to add sets as extensional databases on the DatalogSolver'''
    @add_match(
        FunctionApplication(Constant[AbstractSet], ...),
    )
    def functionapplication_abstract_set(self, expression):
        '''
        This pattern enables using intermediate representation objects
        as relations in a datalog-compatible syntax. This means that
        the set `R = {(1, 2), (3, 4)}` will behave such that
        `R(1, 2)` (i.e. `FunctionApplication[bool](functor, args)`
        with `functor = Constant[AbstractSet[Tuple[int, int]]](R)` and
        `args = (Constant[int](1), Constant[int](2))`)
        and `R(3, 4)` are `True` and any other tuple
        given to `R` used as a function will be `False`.
        '''

        if len(expression.args) == 1:
            element = expression.args[0]
        elif isinstance(expression.args, Constant):
            element = expression.args
        else:
            element = Constant(expression.args)

        return_type = expression.functor.type.__args__[0]
        if not is_leq_informative(element.type, return_type):
            raise NeuroLangTypeException(
                'Element type {element.type} does not '
                'correspond with set type {functor.type}'
            )

        rsc = ReplaceSymbolsByConstants(self.symbol_table)
        predset = rsc.walk(expression.functor).value
        ret = element in predset
        return Constant[bool](ret)

    @add_match(FunctionApplication(
        Constant(contains), (..., Expression[AbstractSet])
    ))
    def expression_contains(self, expression):
        return self.walk(expression.args[1](expression.args[0]))

    def function_isin(self, element: Any, set: AbstractSet) -> bool:
        '''Function for checking that an element is in a set'''
        return element in set

    def function_join(
        self, set1: AbstractSet, elem1: int, set2: AbstractSet, elem2: int
    ) -> AbstractSet[Tuple]:
        '''function to join two sets based on one item element'''

        if len(set1) > 1 and not isinstance(next(iter(set1)), tuple):
            set1 = ((e, ) for e in set1)

        if len(set2) > 1 and not isinstance(next(iter(set2)), tuple):
            set2 = ((e, ) for e in set2)

        new_set = frozenset(
            t1 + t2 for t1, t2 in product(set1, set2) if t1[elem1] == t2[elem2]
        )

        return new_set
