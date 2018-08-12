from operator import and_, or_, invert

from . import expression_walker as ew
from .neurolang import NeuroLangException


__all__ = [
    'undefined',
    'SafeRangeVariablesWalker',
    'NeuroLangException'
]


class Undefined:
    def __init__(self):
        return

    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __and__(self, other):
        return self

    def __rand__(self, other):
        return self

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


undefined = Undefined()


def application_is_not_logical_op(expression):
    return expression.functor.value not in (
        and_, or_, invert
    )


class Intersection(frozenset):
    '''
    Set of restrictions to be intersected
    to obtain the restricted domain.
    '''
    pass


class Union(frozenset):
    '''
    Set of restrictions to be merged
    to obtain the restricted domain.
    '''
    pass


class SafeRangeVariablesWalker(ew.PatternWalker):
    '''
    Obtains the safe range free variables from an expression in
    safe-range normal form (SRNF).

    For each variable it also obtains a set of intersection or
    union of predicates to be used to compute the range of each
    variable.
    the domain.
    '''

    @ew.add_match(ew.FunctionApplication[bool](ew.Constant(and_), ...))
    def conjunction(self, expression):
        restrictors = dict()
        for arg in expression.args:
            arg_rest = self.walk(arg)
            for k, v in arg_rest.items():
                restrictors[k] = v | restrictors.get(k, Intersection())

        return restrictors

    @ew.add_match(ew.FunctionApplication[bool](ew.Constant(or_), ...))
    def disjunction(self, expression):
        restrictors = None
        for arg in expression.args:
            arg_rest = self.walk(arg)
            if restrictors is None:
                restrictors = arg_rest.copy()
            else:
                for k in set(restrictors.keys()):
                    if k not in arg_rest:
                        del restrictors[k]
                    else:
                        restrictors[k] = {
                            Union((
                                Intersection(arg_rest[k]),
                                Intersection(restrictors[k])
                            ))
                        }

        return restrictors

    @ew.add_match(
        ew.FunctionApplication[bool](
            ew.Constant(invert),
            (ew.FunctionApplication[bool](ew.Constant, ...),)
        ),
        lambda expression: application_is_not_logical_op(expression.args[0])
    )
    def inversion(self, expression):
        return dict()

    @ew.add_match(
        ew.FunctionApplication[bool](ew.Constant, ...),
        application_is_not_logical_op
    )
    def fa(self, expression):
        restrictors = {
            s: Intersection((expression,))
            for s in expression._symbols
        }
        return restrictors

    @ew.add_match(ew.ExistentialPredicate[bool])
    def ex(self, expression):
        restrictors = self.walk(expression.body)
        if expression.head._symbols.issubset(restrictors.keys()):
            for s in expression.head._symbols:
                if s in restrictors:
                    del restrictors[s]
            return restrictors
        else:
            return undefined

    @ew.add_match(...)
    def _(self, expression):
        raise NeuroLangException('Expression not in safe-range normal form')
