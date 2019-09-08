from collections import deque
from itertools import product
import logging
import typing

from .expressions import TypedSymbolTable
from .expressions import (
    ExpressionBlock, FunctionApplication, Statement, Query, Projection,
    Constant, Symbol, ExistentialPredicate, UniversalPredicate, Expression,
    Lambda, Unknown, is_leq_informative, NeuroLangTypeException, unify_types,
    NeuroLangException
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
            children = expression_iterator_constant(current_element)
        elif isinstance(current_element[1], tuple):
            c = current_element[1]
            children = product((None, ), c)
        elif isinstance(current_element[1], Expression):
            c = current_element[1].__children__

            children = ((name, getattr(current_element[1], name))
                        for name in c)

        if include_level:
            current_level = current_element[-1] + 1
            children = [(name, value, current_level)
                        for name, value in children]

        children = fix_children_order(dfs, expression, children)

        extend(children)

        yield current_element


def fix_children_order(dfs, expression, children):
    if (
        dfs and not (
            isinstance(expression, Constant) and
            is_leq_informative(expression.type, typing.AbstractSet)
        )
    ):
        try:
            children = reversed(children)
        except TypeError:
            children = list(children)
            children.reverse()
    return children


def expression_iterator_constant(current_element):
    if is_leq_informative(current_element[1].type, typing.Tuple):
        c = current_element[1].value
        children = product((None, ), c)
    elif is_leq_informative(current_element[1].type, typing.AbstractSet):
        children = product((None, ), current_element[1].value)
    else:
        children = []
    return children


class PatternWalker(PatternMatcher):
    def walk(self, expression):
        logging.debug("walking %(expression)s", {'expression': expression})
        if isinstance(expression, tuple):
            result = [self.walk(e) for e in expression]
            result = tuple(result)
            return result
        return self.match(expression)


class ExpressionWalker(PatternWalker):
    @add_match(Expression)
    def process_expression(self, expression):
        args = expression.unapply()
        new_args = tuple()
        changed = False
        for arg in args:
            if isinstance(arg, Expression):
                new_arg = self.walk(arg)
                changed |= new_arg is not arg
            elif isinstance(arg, (tuple, list)):
                new_arg = list()
                for sub_arg in arg:
                    new_arg.append(self.walk(sub_arg))
                    changed |= new_arg[-1] is not sub_arg
                new_arg = type(arg)(new_arg)
            elif arg is Ellipsis:
                raise NeuroLangException(
                    '... is not a valid Expression argument'
                )
            else:
                new_arg = arg
            new_args += (new_arg,)

        if changed:
            new_expression = expression.apply(*new_args)
            return self.walk(new_expression)
        else:
            return expression


class ReplaceSymbolWalker(ExpressionWalker):
    def __init__(self, symbol_replacements):
        self.symbol_replacements = symbol_replacements

    @add_match(Symbol)
    def replace_free_variable(self, symbol):
        if symbol.name in self.symbol_replacements:
            replacement = self.symbol_replacements[symbol.name]
            replacement_type = unify_types(symbol.type, replacement.type)
            return replacement.cast(replacement_type)
        else:
            return symbol


class ReplaceSymbolsByConstants(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Symbol)
    def symbol(self, symbol):
        new_expression = self.symbol_table.get(symbol, symbol)
        if isinstance(new_expression, Constant):
            return new_expression
        else:
            return symbol

    @add_match(Constant[typing.AbstractSet])
    def constant_abstract_set(self, constant_abstract_set):
        return constant_abstract_set.__class__(
            type(constant_abstract_set.value)(
                self.walk(expression)
                for expression in constant_abstract_set.value
            )
        )

    @add_match(Constant[typing.Tuple])
    def constant_tuple(self, constant_tuple):
        return constant_tuple.__class__(
            tuple(
                self.walk(expression) for expression in constant_tuple.value
            )
        )


class ReplaceExpressionsByValues(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Symbol)
    def symbol(self, symbol):
        new_expression = self.symbol_table.get(symbol, symbol)
        if isinstance(new_expression, Constant):
            return self.walk(new_expression)
        else:
            raise NeuroLangException(
                f'{symbol} could not be evaluated '
                'to a constant'
            )

    @add_match(Constant[typing.AbstractSet])
    def constant_abstract_set(self, constant_abstract_set):
        return frozenset(
            self.walk(expression) for expression in constant_abstract_set.value
        )

    @add_match(Constant[typing.Tuple])
    def constant_tuple(self, constant_tuple):
        return tuple(
            self.walk(expression) for expression in constant_tuple.value
        )

    @add_match(Constant)
    def constant(self, constant):
        return constant.value


class SymbolTableEvaluator(ExpressionWalker):
    def __init__(self, symbol_table=None):
        if symbol_table is None:
            symbol_table = TypedSymbolTable()
        self.symbol_table = symbol_table
        self.simplify_mode = False
        self.add_functions_to_symbol_table()

    @add_match(Symbol)
    def symbol_from_table(self, symbol):
        try:
            return self.symbol_table.get(symbol, symbol)
        except KeyError:
            if self.simplify_mode:
                return symbol
            else:
                raise ValueError(f'{symbol} not in symbol table')

    @property
    def included_functions(self):
        function_constants = dict()
        for attribute in dir(self):
            if attribute.startswith('function_'):
                c = Constant(getattr(self, attribute))
                function_constants[attribute[len('function_'):]] = c
        return function_constants

    def add_functions_to_symbol_table(self):
        keyword_symbol_table = TypedSymbolTable()
        for k, v in self.included_functions.items():
            keyword_symbol_table[Symbol[v.type](k)] = v
        keyword_symbol_table.set_readonly(True)
        top_scope = self.symbol_table
        while top_scope.enclosing_scope is not None:
            top_scope = top_scope.enclosing_scope
        top_scope.enclosing_scope = keyword_symbol_table

    @add_match(Statement)
    def statement(self, statement):
        rhs = self.walk(statement.rhs)
        return_type = unify_types(statement.type, rhs.type)
        rhs.change_type(return_type)
        statement.lhs.change_type(return_type)
        if rhs is statement.rhs:
            self.symbol_table[statement.lhs] = rhs
            return statement
        else:
            return self.walk(Statement[statement.type](statement.lhs, rhs))

    def push_scope(self):
        self.symbol_table = self.symbol_table.create_scope()

    def pop_scope(self):
        es = self.symbol_table.enclosing_scope
        if es is None:
            raise NeuroLangException('No enclosing scope')
        self.symbol_table = self.symbol_table.enclosing_scope


class ExpressionBasicEvaluator(SymbolTableEvaluator):
    @add_match(Projection(Constant(...), Constant(...)))
    def evaluate_projection(self, projection):
        return (projection.collection.value[int(projection.item.value)])

    @add_match(
        FunctionApplication(Constant(...), ...),
        lambda e: all(
            not isinstance(arg, Expression) or isinstance(arg, Constant)
            for _, arg in expression_iterator(e.args)
        )
    )
    def evaluate_function(self, function_application):
        functor = function_application.functor
        functor_type = functor.type
        if isinstance(functor, Constant):
            functor_value = functor.value

        if functor_type is not Unknown:
            if not is_leq_informative(functor_type, typing.Callable):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = functor_type.__args__[-1]
        else:
            if not callable(functor_value):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(functor)
                )
            result_type = Unknown

        rebv = ReplaceExpressionsByValues(self.symbol_table)
        args = rebv.walk(function_application.args)
        kwargs = {
            k: rebv.walk(v)
            for k, v in function_application.kwargs.items()
        }
        result = Constant[result_type](functor_value(*args, **kwargs))
        return result

    @add_match(FunctionApplication(Lambda, ...))
    def eval_lambda(self, function_application):
        lambda_ = function_application.functor
        args = function_application.args
        lambda_args = lambda_.args
        if (
            len(args) != len(lambda_args) or not all(
                is_leq_informative(l.type, a.type)
                for l, a in zip(lambda_args, args)
            )
        ):
            raise NeuroLangTypeException(
                f'{args} is not the appropriate '
                f'argument tuple for {lambda_args}'
            )

        if len(lambda_.function_expression._symbols.intersection(lambda_args)
               ) > 0:
            rsw = ReplaceSymbolWalker(dict(zip(lambda_args, args)))
            return self.walk(rsw.walk(lambda_.function_expression))
        else:
            return lambda_.function_expression
