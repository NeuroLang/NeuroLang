import logging
import inspect
import typing

from .exceptions import NeuroLangException
from .symbols_and_types import (
    Expression, Symbol, Constant, Predicate, FunctionApplication,
    type_validation_value,
    NeuroLangTypeException,
    get_type_and_value, replace_type_variable,
    ToBeInferred
)
from operator import invert, and_, or_
from .expression_walker import (
    add_match, ExpressionBasicEvaluator
)


T = typing.TypeVar('T')


class NeuroLangPredicateException(NeuroLangException):
    pass


class FiniteDomain(object):
    pass


class FiniteDomainSet(frozenset):
    pass


class GenericSolver(ExpressionBasicEvaluator):
    @property
    def plural_type_name(self):
        return self.type_name + 's'

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Predicate)
    def predicate(self, expression):
        logging.debug(str(self.__class__.__name__) + " evaluating predicate")

        if isinstance(expression.functor, Symbol):
            identifier = expression.functor
            predicate_method = 'predicate_' + identifier.name
            if hasattr(self, predicate_method):
                method = getattr(self, predicate_method)
            elif self.symbol_table[expression.functor]:
                method = self.symbol_table[expression.functor]
            else:
                raise NeuroLangException(
                    "Predicate %s not implemented" % identifier
                )

        signature = inspect.signature(method)
        type_hints = typing.get_type_hints(method)

        argument = self.walk(expression.args[0])
        if len(signature.parameters) != 1:
            raise NeuroLangPredicateException(
                "Predicates take exactly one parameter"
            )
        else:
            parameter_type = type_hints[
                next(iter(signature.parameters.keys()))
            ]

        type_, value = get_type_and_value(
            argument, symbol_table=self.symbol_table
        )

        if not type_validation_value(
            value,
            replace_type_variable(
                self.type,
                parameter_type,
                type_var=T
             ),
        ):
            raise NeuroLangTypeException("argument of wrong type")

        # type_, value = get_type_and_value(
        #    result, symbol_table=self.symbol_table
        # )

        return_type = type_hints['return']
        return_type = replace_type_variable(
            self.type,
            return_type,
            type_var=T
         )
        result = method(value)
        if not isinstance(result, Expression):
            result = Constant[return_type](method(value))

        # if not is_subtype(type_, return_type):
        #    raise NeuroLangTypeException(
        #        "Value returned doesn't have the right type"
        #    )

        # result = Expression(
        #    value,
        #    type_=return_type,
        #    symbol_table=self.symbol_table
        # )

        return result


class SetBasedSolver(GenericSolver):
    '''
    A predicate `in <set>` which results in the `<set>` given as parameter
    `and` and `or` operations between sets which are disjunction and
    conjunction.
    '''
    def predicate_in(
        self, argument: typing.AbstractSet[T]
    )->typing.AbstractSet[T]:
        return argument

    @add_match(
        FunctionApplication(Constant(invert), (Constant[typing.AbstractSet],)),
        lambda expression: isinstance(
            get_type_and_value(expression.args[0])[1],
            FiniteDomainSet
        )
    )
    def rewrite_finite_domain_inversion(self, expression):
        set_constant = expression.args[0]
        set_type, set_value = get_type_and_value(set_constant)
        result = FiniteDomainSet(
            (
                v.value for v in
                self.symbol_table.symbols_by_type(
                    set_type.__args__[0]
                ).values()
            ),
            type_=set_type,
        )
        return self.walk(Constant[set_type](result))

    @add_match(
        FunctionApplication(
            Constant(and_),
            (Constant[typing.AbstractSet], Constant[typing.AbstractSet])
        )
    )
    @add_match(
        FunctionApplication[ToBeInferred](
            Constant(or_),
            (Constant[typing.AbstractSet], Constant[typing.AbstractSet])
        )
    )
    def rewrite_and_or(self, expression):
        f = expression.functor.value
        a_type, a = get_type_and_value(expression.args[0])
        b_type, b = get_type_and_value(expression.args[1])
        e = Constant[a_type](
            f(a, b)
        )
        return e
