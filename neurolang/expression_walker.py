import logging
import typing

from .expressions import (
    Expression, Function, Definition, Query, Projection, Constant,
    get_type_and_value, ToBeInferred, is_subtype, NeuroLangTypeException
)


class ExpressionWalker(object):
    @staticmethod
    def get_class_name(expr):
        return expr.__class__.__name__.lower()

    def walk(self, expression):
        logging.debug("evaluating {}".format(expression))
        if isinstance(expression, list) or isinstance(expression, tuple):
            result = [
                self.walk(e)
                for e in expression
            ]
            if isinstance(expression, tuple):
                result = tuple(result)
            return result
        else:
            if not isinstance(expression, Expression):
                raise ValueError('{} is not an expression'.format(expression))
            class_name = self.get_class_name(expression)
            if hasattr(self, class_name):
                return getattr(self, class_name)(expression)
            else:
                logging.debug("\tevaluating with default method")
                return self._default(expression)

    def expression(self, expression):
        return Expression(
            self.walk(expression.value),
            type_=expression.type
        )

    def definition(self, expression):
        return Definition(
            expression.symbol, self.walk(expression.value),
            type_=expression.type
        )

    def function(self, expression):
        return Function(
            self.walk(expression.function),
            args=[self.walk(e) for e in expression.args],
            kwargs={k: self.walk(v) for k, v in expression.kwargs},
            type_=expression.type_
        )

    def query(self, expression):
        return Query(
            expression.symbol,
            self.walk(expression.value),
            type_=expression.type
        )

    def _default(self, expression):
        return expression


class ExpressionBasicEvaluator(ExpressionWalker):
    def __init__(self, symbol_table=None):
        if symbol_table is None:
            symbol_table = dict()
        self.symbol_table = symbol_table
        self.simplify_mode = False

    def constant(self, expression):
        return expression

    def symbol(self, expression):
        try:
            return self.symbol_table[expression]
        except KeyError:
            if self.simplify_mode:
                return expression
            else:
                raise ValueError('{} not in symbol table'.format(expression))

    def definition(self, expression):
        value = self.walk(expression.value)
        self.symbol_table[expression.symbol] = value
        return Definition(expression.symbol, value, type_=expression.type)

    def projection(self, expression):
        collection = self.walk(expression.collection)
        item = self.walk(expression.item)

        result = Projection(collection, item)

        if isinstance(collection, Constant) and isinstance(item, Constant):
            result = Constant(
                collection.value[int(item.value)],
                type_=result.type
            )

        return result

    def function(self, expression):
        function = self.walk(expression.function)
        function_type, function_value = get_type_and_value(function)
        if function_type != ToBeInferred:
            if not is_subtype(function_type, typing.Callable):
                raise NeuroLangTypeException(
                    'Function {} is not of callable type'.format(function)
                )
            result_type = function_type.__args__[-1]
        else:
            result_type = ToBeInferred

        if expression.args is None and expression.kwargs is None:
            return Function(function, type_=function_type)

        we_should_evaluate = True
        new_args = []
        for arg in expression.args:
            arg = self.walk(arg)
            new_args.append(arg)
            we_should_evaluate &= isinstance(arg, Constant)

        new_kwargs = dict()
        for k, arg in expression.kwargs.items():
            arg = self.walk(arg)
            new_kwargs[k] = arg
            we_should_evaluate &= isinstance(arg, Constant)

        we_should_evaluate &= isinstance(function, Constant)

        if we_should_evaluate:
            new_args = [a.value for a in new_args]
            new_kwargs = {k: v.value for k, v in new_kwargs.items()}
            result = Constant(
                function_value(*new_args, **new_kwargs),
                type_=result_type
            )
            return result
        else:
            return Function(function)(*new_args, **new_kwargs)


class ExpressionReplacement(ExpressionWalker):
    def __init__(self, replacements):
        self.replacements = replacements

    def walk(self, expression):
        if isinstance(expression, list):
            return super().walk(expression)
        else:
            return self.replacements.get(expression, super().walk(expression))
