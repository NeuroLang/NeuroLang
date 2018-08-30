from operator import and_, or_, invert, eq
import typing
import logging

from . import expression_walker as ew
from .expressions import (
    Expression,
    Constant, FunctionApplication, Symbol,
    Definition, Quantifier,
    ExistentialPredicate, UniversalPredicate
)

from .neurolang import NeuroLangException


logger = logging.getLogger()

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

    def __invert__(self):
        return self


undefined = Undefined()


def atleast_one_argument_is_application(expression):
    return any(
        isinstance(a, FunctionApplication)
        for a in expression.args
    )


def no_argument_is_application(expression):
    return all(
        not isinstance(a, FunctionApplication)
        for a in expression.args
    )


def application_is_not_logical_op(expression):
    return expression.functor.value not in (
        and_, or_, invert
    )


class Intersection(frozenset):
    '''
    Set of restrictions to be intersected
    to obtain the restricted domain.
    '''
    def __or__(self, other):
        if isinstance(other, Intersection):
            return Intersection(super().__or__(other))
        else:
            raise ValueError('parameter must be Intersection')

    def __ror__(self, other):
        return self.__or__(other)


class Union(frozenset):
    '''
    Set of restrictions to be merged
    to obtain the restricted domain.
    '''
    def __or__(self, other):
        if isinstance(other, Union):
            return Union(super().__or__(other))
        else:
            raise ValueError('parameter must be Union')

    def __ror__(self, other):
        return self.__or__(other)


class SafeRangeVariablesWalker(ew.PatternWalker):
    '''
    Obtains the safe range free variables from an expression in
    safe-range normal form (SRNF).

    For each variable it also obtains a set of intersection or
    union of predicates to be used to compute the range of each
    variable.
    '''

    @ew.add_match(FunctionApplication[bool](Constant(and_), ...))
    def conjunction(self, expression):
        restrictors = dict()
        for arg in expression.args:
            arg_rest = self.walk(arg)
            if arg_rest is undefined:
                return undefined
            for k, v in arg_rest.items():
                restrictor = v | restrictors.get(k, Intersection())
                restrictors[k] = restrictor

        return restrictors

    @ew.add_match(FunctionApplication[bool](Constant(or_), ...))
    def disjunction(self, expression):
        args = expression.args
        restrictors = self.walk(args[0])
        if restrictors is undefined:
            return undefined
        for arg in args:
            arg_rest = self.walk(arg)
            if arg_rest is undefined:
                return undefined
            for k in list(restrictors.keys()):
                if k not in arg_rest:
                    del restrictors[k]
                else:
                    r1 = restrictors[k]
                    r2 = arg_rest[k]

                    if not isinstance(r1, Union):
                        if len(r1) == 1:
                            r1 = Union(r1)
                        else:
                            r1 = Union({r1})

                    if not isinstance(r2, Union):
                        if len(r2) == 1:
                            r2 = Union(r2)
                        else:
                            r2 = Union({r2})
                    restrictors[k] = r1 | r2

        return restrictors

    @ew.add_match(
        FunctionApplication[bool](
            Constant(invert),
            (Expression[bool],)
        ),
        lambda e: (
            not isinstance(e.args[0], FunctionApplication[bool]) or
            no_argument_is_application(e.args[0])
        )
    )
    def inversion(self, expression):
        return dict()

    @ew.add_match(
        FunctionApplication[bool](Constant(eq), ...),
        lambda e: (
            no_argument_is_application(e) and
            any(
                isinstance(a, Constant)
                for a in e.args
            )
        )
    )
    def equality(self, expression):
        args = expression.args
        restrictions = Intersection({
            arg for arg in args
            if isinstance(arg, Constant)
        })

        return {
            s: restrictions
            for s in expression._symbols
        }

    @ew.add_match(
        FunctionApplication[bool](Constant, ...),
        lambda e: (
            no_argument_is_application(e) and
            e.functor != Constant(eq)
        )
    )
    def fa(self, expression):
        restrictors = {
            s: Intersection((expression,))
            for s in expression._symbols
        }
        return restrictors

    @ew.add_match(ExistentialPredicate[bool])
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


class VariableSubstitutionWalker(ew.PatternWalker):
    def __init__(self):
        self.seen_variables = set()

    @ew.add_match(
        FunctionApplication[bool](Constant, ...),
        lambda e: e.functor.value in (and_, or_, invert)
    )
    def logical_application(self, expression):
        new_args = []
        changed = False
        for arg in expression.args:
            new_arg = self.walk(arg)
            new_args.append(new_arg)
            changed |= new_arg is not arg

        if changed:
            return FunctionApplication[bool](
                expression.functor, tuple(new_args)
            )
        else:
            return expression

    @ew.add_match(
        FunctionApplication[bool](Constant, ...),
        no_argument_is_application
    )
    def fa(self, expression):
        self.seen_variables |= expression._symbols
        return expression

    @ew.add_match(Quantifier[bool])
    def quantifier(self, expression):
        replacement_symbols = dict()
        for s in expression.head._symbols:
            new_s = s
            while new_s in self.seen_variables:
                new_s = Symbol[new_s.type](new_s.name + '_')
            if new_s is not s:
                replacement_symbols[s] = new_s

        if len(replacement_symbols) > 0:
            rsw = ew.ReplaceSymbolWalker(replacement_symbols)
            expression = type(expression)(
                rsw.walk(expression.head),
                rsw.walk(expression.body)
            )

        self.seen_variables |= expression.body._symbols
        self.seen_variables |= expression.head._symbols

        return expression


class ConvertToSNRFWalker(ew.ExpressionWalker):
    @ew.add_match(FunctionApplication[bool](
        Constant(invert),
        (FunctionApplication[bool](Constant(invert), ...),)
    ))
    def push_neg_double_neg(self, expression):
        return self.walk(expression.args[0].args[0])

    @ew.add_match(
        FunctionApplication[bool](
            Constant(invert),
            (FunctionApplication[bool](Constant(and_), ...),)
        )
    )
    def push_neg_and(self, expression):
        args = expression.args[0].args
        return self.walk(FunctionApplication[bool](
            Constant[typing.Callable[[bool] * len(args), bool]](or_),
            tuple(
                FunctionApplication[bool](
                    Constant[typing.Callable[[bool], bool]](invert),
                    (a,)
                )
                for a in args
            )
        ))

    @ew.add_match(
        FunctionApplication[bool](
            Constant(invert),
            (FunctionApplication[bool](Constant(or_), ...),)
        )
    )
    def push_neg_or(self, expression):
        args = expression.args[0].args
        return self.walk(FunctionApplication[bool](
            Constant[typing.Callable[[bool] * len(args), bool]](and_),
            tuple(
                FunctionApplication[bool](
                    Constant[typing.Callable[[bool], bool]](invert),
                    (a,)
                )
                for a in args
            )
        ))

    @ew.add_match(UniversalPredicate[bool])
    def universal_to_existential(self, expression):
        return self.walk(
            FunctionApplication[bool](
                Constant[typing.Callable[[bool], bool]](invert),
                (ExistentialPredicate[bool](
                    expression.head,
                    FunctionApplication[bool](
                        Constant[typing.Callable[[bool], bool]](invert),
                        (expression.body,)
                    )
                ),)
            )
        )


class FlattenMultipleLogicalOperators(ew.ExpressionWalker):
    @ew.add_match(
        FunctionApplication[bool](Constant(and_), ...),
        lambda e: any(
            isinstance(a, FunctionApplication[bool]) and
            a.functor == Constant(and_)
            for a in e.args
        )
    )
    def flatten_and(self, expression):
        new_args = []
        for a in expression.args:
            if (
                isinstance(a, FunctionApplication[bool]) and
                a.functor == Constant(and_)
            ):
                new_args.extend(a.args)
            else:
                new_args.append(a)

        functor = Constant[typing.Callable[[bool] * len(new_args), bool]](and_)
        return self.walk(FunctionApplication[bool](functor, tuple(new_args)))

    @ew.add_match(
        FunctionApplication[bool](Constant(or_), ...),
        lambda e: any(
            isinstance(a, FunctionApplication[bool]) and
            a.functor == Constant(or_)
            for a in e.args
        )
    )
    def flatten_or(self, expression):
        new_args = []
        for a in expression.args:
            if (
                isinstance(a, FunctionApplication[bool]) and
                a.functor == Constant(or_)
            ):
                new_args.extend(a.args)
            else:
                new_args.append(a)
        functor = Constant[typing.Callable[[bool] * len(new_args), bool]](or_)
        return self.walk(FunctionApplication[bool](functor, tuple(new_args)))


def expression_to_SRNF_and_range(expression):
    '''
    Converts a first-order logic expression to SRNF
    and produces the range
    '''

    for _, e in ew.expression_iterator(expression):
        if isinstance(e, FunctionApplication[bool]):
            if (
                e.functor.value in (and_, or_, invert) and
                all(e.type is bool for e in e.args)
            ):
                pass
            elif all(not isinstance(a, Definition) for a in e.args):
                pass
            else:
                raise ValueError(
                    f'{expression} is not a valid datalog expression'
                )
        elif isinstance(e, (
            ExistentialPredicate[bool], UniversalPredicate[bool],
            Symbol, Constant
        )):
            pass
        else:
                raise ValueError(
                    f'{expression} is not a valid datalog expression'
                )

        logger.info('Variable substitution')
        vsw = VariableSubstitutionWalker()
        expression = vsw.walk(expression)

        logger.info('Flatten logical ops')
        fw = FlattenMultipleLogicalOperators()
        expression = fw.walk(expression)

        logger.info('Convert to SNRF')
        csnrf = ConvertToSNRFWalker()
        expression = csnrf.walk(expression)

        logger.info('Compute range')
        srvw = SafeRangeVariablesWalker()
        var_range = srvw.walk(expression)

        return expression, var_range
