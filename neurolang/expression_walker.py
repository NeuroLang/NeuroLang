import logging
import typing

from .expressions import (
    Expression, FunctionApplication, Statement, Query, Projection, Constant,
    Symbol,
    get_type_and_value, ToBeInferred, is_subtype, NeuroLangTypeException
)

from .expression_pattern_matching import add_match, PatternMatcher


class ExpressionWalker(PatternMatcher):
    @add_match(Statement)
    def statement(self, expression):
        return Statement[expression.type](
            expression.symbol, self.walk(expression.value)
        )

    @add_match(FunctionApplication)
    def function(self, expression):
        return FunctionApplication[expression.type](
            self.match(expression.functor),
            args=[self.walk(e) for e in expression.args],
            kwargs={k: self.walk(v) for k, v in expression.kwargs},
        )

    @add_match(Query)
    def query(self, expression):
        return Query[expression.type](
            expression.symbol,
            self.walk(expression.value)
        )

    @add_match(...)
    def default(self, expression):
        return expression

    def walk(self, expression):
        logging.debug("walking {}".format(expression))
        if isinstance(expression, list) or isinstance(expression, tuple):
            result = [
                self.walk(e)
                for e in expression
            ]
            if isinstance(expression, tuple):
                result = tuple(result)
            return result
        return self.match(expression)


class ExpressionBasicEvaluator(ExpressionWalker):
    def __init__(self, symbol_table=None):
        if symbol_table is None:
            symbol_table = dict()
        self.symbol_table = symbol_table
        self.simplify_mode = False

    @add_match(Constant)
    def constant(self, expression):
        return expression

    @add_match(Symbol)
    def symbol(self, expression):
        try:
            return self.symbol_table.get(expression, expression)
        except KeyError:
            if self.simplify_mode:
                return expression
            else:
                raise ValueError('{} not in symbol table'.format(expression))

    @add_match(Statement)
    def statement(self, expression):
        value = self.walk(expression.value)
        self.symbol_table[expression.symbol] = value
        if value is expression.value:
            return expression
        else:
            return self.walk(
                Statement[expression.type](expression.symbol, value)
            )

    @add_match(Query)
    def query(self, expression):
        value = self.walk(expression.value)
        self.symbol_table[expression.symbol] = value
        if value is expression.value:
            return expression
        else:
            return self.walk(
                Query[expression.type](expression.symbol, value)
            )

    @add_match(Projection(Constant(...), Constant(...)))
    def evaluate_projection(self, expression):
        return (
            expression.collection.value[int(expression.item.value)]
        )

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

    @add_match(
        FunctionApplication(Constant(...), ...),
        lambda expression:
            expression.args is not None and
            all(
                isinstance(arg, Constant)
                for arg in expression.args
            )
    )
    def evaluate_function(self, expression):
        functor = expression.functor
        functor_type, functor_value = get_type_and_value(functor)

        if functor_type != ToBeInferred:
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

        new_args = [a.value for a in expression.args]
        new_kwargs = {k: v.value for k, v in expression.kwargs.items()}
        result = Constant[result_type](
            functor_value(*new_args, **new_kwargs)
        )
        return result

    @add_match(FunctionApplication)
    def function(self, expression):
        changed = False
        functor = self.walk(expression.functor)
        changed |= functor is not expression.functor
        functor_type, functor_value = get_type_and_value(functor)

        if expression.args is None and expression.kwargs is None:
            if changed:
                result = FunctionApplication[functor_type](functor)
                return self.walk(result)
            else:
                return expression

        new_args = []
        for arg in expression.args:
            new_arg = self.walk(arg)
            new_args.append(new_arg)
            changed |= new_arg is not arg

        new_kwargs = dict()
        for k, arg in expression.kwargs.items():
            new_arg = self.walk(arg)
            new_kwargs[k] = new_arg
            changed |= new_arg is not arg

        if changed:
            functor_type, functor_value = get_type_and_value(functor)

            if functor_type != ToBeInferred:
                if not is_subtype(functor_type, typing.Callable):
                    raise NeuroLangTypeException(
                        'Function {} is not of callable type'.format(functor)
                    )
            else:
                if (
                    isinstance(functor, Constant) and
                    not callable(functor_value)
                ):
                    raise NeuroLangTypeException(
                        'Function {} is not of callable type'.format(functor)
                    )

            result = functor(*new_args, **new_kwargs)
            return self.walk(result)
        else:
            return expression


class ExpressionWalkerGraph(object):
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
        return Expression[expression.type](
            self.walk(expression.value)
        )

    def definition(self, expression):
        return Statement[expression.type](
            expression.symbol, self.walk(expression.value)
        )

    def function(self, expression):
        return FunctionApplication[expression.type](
            self.walk(expression.functor),
            args=[self.walk(e) for e in expression.args],
            kwargs={k: self.walk(v) for k, v in expression.kwargs}
        )

    def query(self, expression):
        return Query[expression.type](
            expression.symbol,
            self.walk(expression.value)
        )

    def _default(self, expression):
        return expression


class ExpressionReplacement(ExpressionWalker):
    def __init__(self, replacements):
        self.replacements = replacements

    def walk(self, expression):
        if isinstance(expression, list):
            return super().walk(expression)
        else:
            return self.replacements.get(expression, super().walk(expression))
