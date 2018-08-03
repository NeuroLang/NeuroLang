from collections import deque
from itertools import chain, product
import logging
import typing

from .symbols_and_types import TypedSymbolTable
from .expressions import (
    FunctionApplication, Statement, Query, Projection, Constant,
    Symbol, ExistentialPredicate, Expression,
    get_type_and_value, ToBeInferred, is_subtype, NeuroLangTypeException,
    unify_types
)

from .expression_pattern_matching import add_match, PatternMatcher


def expression_iterator(expression, include_level=False, dfs=True):
    """
    Iterate traversing expression tree.

    Iterates over elements `(parameter_name, parameter_object)` when
    `include_level` is `False`.

    If `include_level` is `True` then the iterated elements are
    `(parameter_name, parameter_object, depth_level)`.

    If `dfs` is true the iteration is in DFS order else is in BFS.
    """
    if include_level:
        current_level = 0
        stack = deque([(None, expression, current_level)])
    else:
        stack = deque([(None, expression)])

    if dfs:
        pop = stack.pop
        extend = stack.extend
    else:
        pop = stack.popleft
        extend = stack.extend

    while stack:
        current_element = pop()

        if isinstance(current_element[1], Symbol):
            children = []
        elif isinstance(current_element[1], Constant):
            if is_subtype(Constant.type, typing.Tuple):
                c = current_element[1].value
                children = product((None,), c)
            elif is_subtype(Constant.type, typing.AbstractSet):
                children = product((None,), current_element[1].value)
            else:
                children = []
        elif isinstance(current_element[1], tuple):
            c = current_element[1]
            children = product((None,), c)
        elif isinstance(current_element[1], Expression):
            c = current_element[1].__children__

            children = (
                (name, getattr(current_element[1], name))
                for name in c
            )

        if include_level:
            current_level = current_element[-1] + 1
            children = [
                (name, value, current_level)
                for name, value in children
            ]

        if (
            dfs and
            not (
                isinstance(expression, Constant) and
                is_subtype(expression, typing.AbstractSet)
            )
        ):
            try:
                children = reversed(children)
            except TypeError:
                children = list(children)
                children.reverse()

        extend(children)

        yield current_element


class PatternWalker(PatternMatcher):
    def walk(self, expression):
        logging.debug(f"walking {expression}")
        if isinstance(expression, (list, tuple)):
            result = [
                self.walk(e)
                for e in expression
            ]
            if isinstance(expression, tuple):
                result = tuple(result)
            return result
        return self.match(expression)


class ExpressionWalker(PatternWalker):
    @add_match(Statement)
    def statement(self, expression):
        return Statement[expression.type](
            expression.symbol, self.walk(expression.value)
        )

    @add_match(FunctionApplication)
    def function(self, expression):
        functor = self.walk(expression.functor)
        args = tuple(self.walk(e) for e in expression.args)
        kwargs = {k: self.walk(v) for k, v in expression.kwargs}

        if (
            functor is not expression.functor or
            any(
                arg is not new_arg
                for arg, new_arg in zip(expression.args, args)
            ) or any(
                kwargs[k] is not expression.kwargs[k]
                for k in expression.kwargs
            )
        ):
            functor_type, functor_value = get_type_and_value(functor)

            if functor_type is not ToBeInferred:
                if not is_subtype(functor_type, typing.Callable):
                    raise NeuroLangTypeException(
                        f'Function {functor} is not of callable type'
                    )
            else:
                if (
                    isinstance(functor, Constant) and
                    not callable(functor_value)
                ):
                    raise NeuroLangTypeException(
                        f'Function {functor} is not of callable type'
                    )

            result = functor(*args, **kwargs)
            return self.walk(result)
        else:
            return expression

    @add_match(Query)
    def query(self, expression):
        body = self.walk(expression.body)

        if body is not expression.body:
            return self.walk(Query[expression.type](
                expression.head, body
            ))
        else:
            return expression

    @add_match(ExistentialPredicate)
    def existential_predicate(self, expression):
        body = self.walk(expression.body)

        if body is not expression.body:
            return self.walk(ExistentialPredicate[expression.type](
                expression.head, body
            ))
        else:
            return expression

    @add_match(Projection)
    def projection(self, expression):
        collection = self.walk(expression.collection)
        item = self.walk(expression.item)

        if (
            collection is expression.collection and
            item is expression.item
        ):
            return expression
        else:
            result = Projection(collection, item)
            return self.walk(result)

    @add_match(Constant)
    def constant(self, expression):
        return expression

    @add_match(Symbol)
    def symbol(self, expression):
        return expression


class ReplaceSymbolWalker(ExpressionWalker):
    def __init__(self, symbol, value):
        self.symbol = symbol
        self.value = value

    @add_match(Symbol)
    def replace_free_variable(self, expression):
        if expression.name == self.symbol.name:
            value_type = unify_types(self.symbol.type, self.value.type)
            return self.value.cast(value_type)
        else:
            return expression


class ReplaceSymbolsByConstants(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Symbol)
    def symbol(self, expression):
        new_expression = self.symbol_table.get(expression, expression)
        if isinstance(new_expression, Constant):
            return new_expression
        else:
            return expression


class SymbolTableEvaluator(ExpressionWalker):
    def __init__(self, symbol_table=None):
        if symbol_table is None:
            symbol_table = TypedSymbolTable()
        self.symbol_table = symbol_table
        self.simplify_mode = False
        self.add_functions_and_predicates_to_symbol_table()

    @add_match(Symbol)
    def symbol_from_table(self, expression):
        try:
            return self.symbol_table.get(expression, expression)
        except KeyError:
            if self.simplify_mode:
                return expression
            else:
                raise ValueError(f'{expression} not in symbol table')

    @property
    def included_predicates(self):
        predicate_constants = dict()
        for attribute in dir(self):
            if attribute.startswith('predicate_'):
                c = Constant(getattr(self, attribute))
                predicate_constants[attribute[len('predicate_'):]] = c
        return predicate_constants

    @property
    def included_functions(self):
        function_constants = dict()
        for attribute in dir(self):
            if attribute.startswith('function_'):
                c = Constant(getattr(self, attribute))
                function_constants[attribute[len('function_'):]] = c
        return function_constants

    def add_functions_and_predicates_to_symbol_table(self):
        for k, v in chain(
            self.included_predicates.items(), self.included_functions.items()
        ):
            self.symbol_table[k] = v
        self.symbol_table.set_readonly(True)
        self.symbol_table = self.symbol_table.create_scope()

    @add_match(Statement)
    def statement(self, expression):
        value = self.walk(expression.value)
        return_type = unify_types(expression.type, value.type)
        value.change_type(return_type)
        expression.symbol.change_type(return_type)
        if value is expression.value:
            self.symbol_table[expression.symbol] = value
            return expression
        else:
            return self.walk(
                Statement[expression.type](expression.symbol, value)
            )


class ExpressionBasicEvaluator(SymbolTableEvaluator):
    @add_match(Projection(Constant(...), Constant(...)))
    def evaluate_projection(self, expression):
        return (
            expression.collection.value[int(expression.item.value)]
        )

    @add_match(
        FunctionApplication(Constant(...), ...),
        lambda expression:
            all(
                isinstance(arg, Constant)
                for arg in expression.args
            )
    )
    def evaluate_function(self, expression):
        functor = expression.functor
        functor_type, functor_value = get_type_and_value(functor)

        if functor_type is not ToBeInferred:
            if not is_subtype(functor_type, typing.Callable):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = functor_type.__args__[-1]
        else:
            if not callable(functor_value):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = ToBeInferred

        args = tuple(a.value for a in expression.args)
        kwargs = {k: v.value for k, v in expression.kwargs.items()}
        result = Constant[result_type](
            functor_value(*args, **kwargs)
        )
        return result
