import logging
import typing
import inspect

from .exceptions import NeuroLangException
from .expressions import (
    Symbol, Constant, Predicate, FunctionApplication,
    Query,
    get_type_and_value,
)
from .symbols_and_types import (ExistentialPredicate, replace_type_variable)
from operator import invert, and_, or_
from .expression_walker import (
    add_match, ExpressionBasicEvaluator, ReplaceSymbolWalker
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

    @add_match(Predicate(Symbol, ...))
    def predicate(self, expression):
        logging.debug(str(self.__class__.__name__) + " evaluating predicate")

        functor = expression.functor

        new_functor = self.walk(functor)
        if new_functor is not functor:
            res = Predicate[expression.type](new_functor, expression.args)
            return self.walk(res)
        elif hasattr(self, f'predicate_{functor.name}'):
            method = getattr(self, f'predicate_{functor.name}')
            signature = inspect.signature(method)
            type_hints = typing.get_type_hints(method)

            parameter_type = type_hints[
                next(iter(signature.parameters.keys()))
            ]

            parameter_type = replace_type_variable(
                self.type,
                parameter_type,
                type_var=T
            )

            return_type = type_hints['return']
            return_type = replace_type_variable(
                self.type,
                return_type,
                type_var=T
             )
            functor_type = typing.Callable[[parameter_type], return_type]
            functor = Constant[functor_type](method)
            res = Predicate[expression.type](functor, expression.args)
            return self.walk(res)
        else:
            res = Predicate[expression.type](
                functor,
                self.walk(expression.args)
            )
            return res

    @add_match(Symbol[typing.Callable])
    def callable_symbol(self, expression):
        logging.debug(
            str(self.__class__.__name__) + " evaluating callable symbol"
        )

        functor = self.symbol_table.get(expression, expression)
        if (
            functor is expression and
            hasattr(self, f'function_{expression.name}')
        ):
            method = getattr(self, f'function_{expression.name}')
            signature = inspect.signature(method)
            type_hints = typing.get_type_hints(method)

            parameter_type = type_hints[
                next(iter(signature.parameters.keys()))
            ]

            parameter_type = replace_type_variable(
                self.type,
                parameter_type,
                type_var=T
            )

            return_type = type_hints['return']
            return_type = replace_type_variable(
                self.type,
                return_type,
                type_var=T
             )
            functor_type = typing.Callable[[parameter_type], return_type]
            functor = Constant[functor_type](method)

        return functor

    @property
    def included_predicates(self):
        predicate_constants = dict()
        for predicate in dir(self):
            if predicate.startswith('predicate_'):
                c = Constant(getattr(self, predicate))
                new_type = replace_type_variable(
                    self.type,
                    c.type,
                    type_var=T
                )
                c = c.cast(new_type)
                predicate_constants[predicate[len('predicate_'):]] = c
        return predicate_constants

    @property
    def included_functions(self):
        function_constants = dict()
        for function in dir(self):
            if function.startswith('function_'):
                c = Constant(getattr(self, function))
                new_type = replace_type_variable(
                    self.type,
                    c.type,
                    type_var=T
                )
                c = c.cast(new_type)
                function_constants[function[len('function_'):]] = c
        return function_constants


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
                if v not in set_value
            ),
            type_=set_type,
        )
        return self.walk(Constant[set_type](result))

    @add_match(
        FunctionApplication(
            Constant(...),
            (Constant[typing.AbstractSet], Constant[typing.AbstractSet])
        ),
        lambda expression: expression.functor.value in (or_, and_)
    )
    def rewrite_and_or(self, expression):
        f = expression.functor.value
        a_type, a = get_type_and_value(expression.args[0])
        b_type, b = get_type_and_value(expression.args[1])
        e = Constant[a_type](
            f(a, b)
        )
        return e

    @add_match(ExistentialPredicate)
    def existential_predicate(self, expression):

        free_variable_symbol = expression.symbol
        if free_variable_symbol in self.symbol_table._symbols:
            return self.symbol_table._symbols[free_variable_symbol]

        predicate = expression.predicate
        partially_evaluated_predicate = self.walk(predicate)
        results = frozenset()

        for elem_set in self.symbol_table.symbols_by_type(
            free_variable_symbol.type
        ).values():
            for elem in elem_set.value:
                elem = Constant[free_variable_symbol.type](frozenset([elem]))
                rsw = ReplaceSymbolWalker(free_variable_symbol, elem)
                rsw_walk = rsw.walk(partially_evaluated_predicate)
                pred = self.walk(rsw_walk)
                if pred.value != frozenset():
                    results = results.union(elem.value)
        return Constant[free_variable_symbol.type](results)


class DatalogSolver(GenericSolver):
    '''
    WIP Solver with queries having the semantics of Datalog.
    For now predicates work only on constants on the symbols table
    '''

    @add_match(FunctionApplication(Constant(invert), (Constant[bool],)))
    def rewrite_boolean_inversion(self, expression):
        return Constant(not expression.args[0].value)

    @add_match(
        FunctionApplication(Constant(and_), (Constant[bool], Constant[bool]))
    )
    def rewrite_boolean_and(self, expression):
        return Constant(expression.args[0].value and expression.args[1].value)

    @add_match(
        FunctionApplication(Constant(or_), (Constant[bool], Constant[bool]))
    )
    def rewrite_boolean_or(self, expression):
        return Constant(expression.args[0].value or expression.args[1].value)

    @add_match(Query)
    def query_resolution(self, expression):
        out_query_type = expression.head.type

        result = []
        for symbol, value in self.symbol_table.symbols_by_type(
            out_query_type
        ).items():
            if not isinstance(value, Constant):
                continue
            if symbol in expression.free_variable_symbol:
                rsw = ReplaceSymbolWalker(expression.head, value)
                body = rsw.walk(expression.body)
            else:
                body = expression.body
            res = self.walk(body)
            if res.value:
                result.append(symbol)

        return Constant[typing.AbstractSet[out_query_type]](
            frozenset(result)
        )
