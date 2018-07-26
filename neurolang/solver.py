import logging
import typing
import inspect
import itertools

from .exceptions import NeuroLangException
from .expressions import (
    Expression, NonConstant, ExistentialPredicate,
    Symbol, Constant, Predicate, FunctionApplication,
    Query,
    get_type_and_value, is_subtype, unify_types,
    ToBeInferred
)
from operator import (
    invert, and_, or_,
    add, sub, mul, truediv, pos, neg
)
from .expression_walker import (
    add_match, ExpressionBasicEvaluator, ReplaceSymbolWalker,
    PatternWalker
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

            return_type = type_hints['return']
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

            return_type = type_hints['return']
            functor_type = typing.Callable[[parameter_type], return_type]
            functor = Constant[functor_type](method)

        return functor

    @property
    def included_predicates(self):
        predicate_constants = dict()
        for predicate in dir(self):
            if predicate.startswith('predicate_'):
                c = Constant(getattr(self, predicate))
                predicate_constants[predicate[len('predicate_'):]] = c
        return predicate_constants

    @property
    def included_functions(self):
        function_constants = dict()
        for function in dir(self):
            if function.startswith('function_'):
                c = Constant(getattr(self, function))
                function_constants[function[len('function_'):]] = c
        return function_constants

    def add_functions_and_predicates_to_symbol_table(self):
        for k, v in self.included_predicates:
            self.symbol_table[k] = v
        for k, v in self.included_functions:
            self.symbol_table[k] = v
        self.symbol_table = self.symbol_table.create_scope()


class SetBasedSolver(GenericSolver[T]):
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

    @add_match(
        ExistentialPredicate,
        lambda expression: expression.head._symbols == expression.body._symbols
    )
    def existential_predicate_process(self, expression):
        free_variable_symbol = expression.head
        body = expression.body
        results = frozenset()

        for elem_set in self.symbol_table.symbols_by_type(
            free_variable_symbol.type
        ).values():
            for elem in elem_set.value:
                elem = Constant[free_variable_symbol.type](frozenset([elem]))
                rsw = ReplaceSymbolWalker(free_variable_symbol, elem)
                rsw_walk = rsw.walk(body)
                pred = self.walk(rsw_walk)
                if pred.value != frozenset():
                    results = results.union(elem.value)
        return Constant[free_variable_symbol.type](results)

    @add_match(ExistentialPredicate)
    def existential_predicate_no_process(self, expression):
        body = self.walk(expression.body)
        if body.type is not ToBeInferred:
            return_type = unify_types(expression.type, body.type)
        else:
            return_type = expression.type

        if (
            isinstance(body, Constant) and
            is_subtype(body.type, typing.AbstractSet)
        ):
            body = body.cast(return_type)
            self.symbol_table[expression.head] = body
            return body
        elif (
            body is expression.body and
            return_type is expression.type
        ):
            return expression
        else:
            return self.walk(
                ExistentialPredicate[return_type](expression.head, body)
            )

    @add_match(Query)
    def query(self, expression):
        body = self.walk(expression.body)
        return_type = unify_types(expression.type, body.type)
        body.change_type(return_type)
        expression.head.change_type(return_type)
        if body is expression.body:
            if isinstance(body, Constant):
                self.symbol_table[expression.head] = body
            else:
                self.symbol_table[expression.head] = expression
            return expression
        else:
            return self.walk(
                Query[expression.type](expression.head, body)
            )



class BooleanRewriteSolver(PatternWalker):
    # @add_match(
    #    FunctionApplication(Constant, (Expression[bool],) * 2),
    #    lambda expression:
    #        expression.functor.value in (or_, and_) and
    #        expression.type is not bool
    # )
    # def cast_binary(self, expression):
    #    if expression.type is not bool:
    #        return self.walk(expression.cast(bool))
    #    else:
    #        return expression

    @add_match(
        FunctionApplication(
            Constant(...),
            (NonConstant[bool], Constant[bool])
        ),
        lambda expression: (
            expression.functor.value in (or_, and_)
        )
    )
    def dual_operator(self, expression):
        return self.walk(
            FunctionApplication[bool](
                expression.functor,
                expression.args[::-1]
            )
        )

    @add_match(
        FunctionApplication(Constant(...), (
            NonConstant[bool],
            FunctionApplication(Constant(...), (Constant[bool], ...))
        )),
        lambda expression: (
            expression.functor.value in (or_, and_)
            and expression.args[1].functor.value is expression.functor.value
        )
    )
    def bring_constants_up_left(self, expression):
        return self.walk(
            FunctionApplication[bool](
                expression.functor,
                (
                    expression.args[1].args[0],
                    FunctionApplication[bool](
                        expression.functor,
                        (
                            expression.args[0],
                            expression.args[1].args[1]
                        )
                    )
                )
            )
        )

    @add_match(
        FunctionApplication[bool](
            Constant(invert),
            (FunctionApplication[bool](
                Constant(or_),
                (Expression[bool], Expression[bool])
            ),)
        )
    )
    def neg_disj_to_conj(self, expression):
        return self.walk(
            FunctionApplication[bool](
                Constant(and_), (
                    (~expression.args[0].args[0]).cast(bool),
                    (~expression.args[0].args[1]).cast(bool)
                )
            )
        )


class BooleanOperationsSolver(PatternWalker):
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

    @add_match(
        FunctionApplication(Constant(or_), (True, Expression[bool]))
    )
    def rewrite_boolean_or_l(self, expression):
        return Constant(True)

    @add_match(
        FunctionApplication(Constant(or_), (Expression[bool], True))
    )
    def rewrite_boolean_or_r(self, expression):
        return Constant(True)

    @add_match(
        FunctionApplication(Constant(and_), (False, Expression[bool]))
    )
    def rewrite_boolean_and_l(self, expression):
        return Constant(False)

    @add_match(
        FunctionApplication(Constant(and_), (Expression[bool], False))
    )
    def rewrite_boolean_and_r(self, expression):
        return Constant(False)


class NumericOperationsSolver(PatternWalker[T]):
    @add_match(
        FunctionApplication(Constant, (Expression[T],) * 2),
        lambda expression: expression.functor.value in (add, sub, mul, truediv)
    )
    def cast_binary(self, expression):
        return expression.cast(expression.args[0].type)

    @add_match(
        FunctionApplication(Constant, (Expression[T],)),
        lambda expression: expression.functor.value in (pos, neg)
    )
    def cast_unary(self, expression):
        return expression.cast(expression.args[0].type)


class DatalogSolver(
        BooleanRewriteSolver,
        BooleanOperationsSolver,
        NumericOperationsSolver[int],
        NumericOperationsSolver[float],
        GenericSolver
):
    '''
    WIP Solver with queries having the semantics of Datalog.
    For now predicates work only on constants on the symbols table
    '''

    @add_match(Query)
    def query_resolution(self, expression):
        out_query_type = expression.head.type

        result = []

        if not is_subtype(out_query_type, typing.Tuple):
            symbols_in_head = (expression.head,)
        else:
            symbols_in_head = expression.head.value

        if any(s not in symbols_in_head for s in expression.body._symbols):
            raise NotImplementedError(
                "All free symbols in the body must be in the head"
            )

        constants = tuple((
            (
                (k, v)
                for k, v in self.symbol_table.symbols_by_type(
                    sym.type
                ).items()
                if isinstance(v, Constant)
            )
            for sym in symbols_in_head
        ))

        constant_cross_prod = itertools.product(*constants)

        for symbol_values in constant_cross_prod:
            body = expression.body
            for i, s in enumerate(symbols_in_head):
                if s in body._symbols:
                    rsw = ReplaceSymbolWalker(s, symbol_values[i][1])
                    body = rsw.walk(body)

            res = self.walk(body)
            if res.value:
                if not is_subtype(out_query_type, typing.Tuple):
                    result.append(symbol_values[0][0])
                else:
                    result.append(tuple(zip(*symbol_values))[0])

        return Constant[typing.AbstractSet[out_query_type]](
            frozenset(result)
        )

    @add_match(
        ExistentialPredicate,
        lambda expression: expression.head._symbols == expression.body._symbols
    )
    def existential_predicate(self, expression):
        out_query_type = expression.head.type
        if not is_subtype(out_query_type, typing.Tuple):
            symbols_in_head = (expression.head,)
        else:
            symbols_in_head = expression.head.value

        constants = tuple((
            (
                (k, v)
                for k, v in self.symbol_table.symbols_by_type(
                    sym.type
                ).items()
                if isinstance(v, Constant)
            )
            for sym in symbols_in_head
        ))

        constant_cross_prod = itertools.product(*constants)

        for symbol_values in constant_cross_prod:
            body = expression.body
            for i, s in enumerate(symbols_in_head):
                if s in body._symbols:
                    rsw = ReplaceSymbolWalker(s, symbol_values[i][1])
                    body = rsw.walk(body)

            res = self.walk(body)
            if res.value:
                return Constant(True)

        return Constant(False)
