import operator as op
from functools import wraps
from typing import AbstractSet, Callable, Tuple

from .. import datalog as dl
from .. import expressions as exp
from .. import neurolang as nl
from ..expression_pattern_matching import NeuroLangPatternMatchingNoMatch
from ..expression_walker import (ExpressionWalker, ReplaceExpressionsByValues,
                                 add_match)
from ..type_system import is_leq_informative
from ..utils import RelationalAlgebraFrozenSet


class Expression(object):
    def __init__(self, query_builder, expression):
        self.query_builder = query_builder
        self.expression = expression

    @property
    def type(self):
        return self.expression.type

    def do(self, name=None):
        return self.query_builder.execute_expression(
            self.expression,
            name=name
        )

    def __call__(self, *args, **kwargs):
        new_args = tuple(
            a.expression if isinstance(a, Expression)
            else nl.Constant(a)
            for a in args
        )

        if (
            self.query_builder.logic_programming and
            isinstance(self, Symbol)
        ):
            functor = self.neurolang_symbol
        else:
            functor = self.expression

        new_expression = exp.FunctionApplication(functor, new_args)
        return Operation(
                self.query_builder, new_expression, self, args)

    def __setitem__(self, key, value):
        if not isinstance(value, Expression):
            value = Expression(self.query_builder, nl.Constant(value))

        if self.query_builder.logic_programming:
            if not isinstance(key, tuple):
                key = (key,)
            self.query_builder.assign(self(*key), value)
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if self.query_builder.logic_programming:
            if isinstance(key, tuple):
                return self(*key)
            else:
                return self(key)
        else:
            super().__getitem__(key)

    def __repr__(self):
        if isinstance(self.expression, nl.Constant):
            return repr(self.expression.value)
        elif isinstance(self.expression, dl.magic_sets.AdornedExpression):
            name = f'{self.expression.expression.name}'
            if self.expression.adornment:
                name += f'^{self.expression.adornment}'
            if self.expression.number:
                name += f'_{self.expression.number}'
            return name
        elif isinstance(self.expression, nl.Symbol):
            return f'{self.expression.name}'
        else:
            return object.__repr__(self)

    def __getattr__(self, name):
        if isinstance(name, Expression):
            name_ = name.expression
        else:
            name_ = nl.Constant[str](name)
        new_expression = exp.FunctionApplication(
            nl.Constant(getattr), (self.expression, name_,)
        )
        return Operation(
            self.query_builder, new_expression, self, (name,)
        )

    def help(self):
        expression = self.expression
        if isinstance(expression, nl.Constant):
            if is_leq_informative(expression.type, Callable):
                return help(expression.value)
            elif is_leq_informative(expression.type, AbstractSet):
                return "Set of tuples"
            else:
                return "Constant value"
        elif isinstance(expression, nl.FunctionApplication):
            return "Evaluation of function to parameters"
        elif isinstance(expression, nl.Symbol):
            return "Unlinked symbol"
        else:
            return "Help not defined yet"


binary_operations = (
    op.add, op.sub, op.mul, op.ge, op.le, op.gt, op.lt, op.eq,
    op.contains
)


def op_bind(op):
    @wraps(op)
    def fun(self, *args):
        new_args = tuple((
            arg.expression if isinstance(arg, Expression)
            else nl.Constant(arg)
            for arg in (self,) + args
        ))
        arg_types = [a.type for a in new_args]
        functor = nl.Constant[Callable[arg_types, nl.Unknown]](
            op, auto_infer_type=False
        )
        new_expression = functor(*new_args)
        res = Operation(
            self.query_builder, new_expression, op,
            (self,) + args, infix=len(args) > 0
        )
        return res

    return fun


def rop_bind(op):
    @wraps(op)
    def fun(self, value):
        raise NotImplementedError()
        original_value = value
        if isinstance(value, Expression):
            value = value.expression
        else:
            value = nl.Constant(value)

        return Operation(
            self.query_builder, op(self.expression, value),
            op, (self, original_value), infix=True
        )

    return fun


force_linking = [op.eq, op.ne, op.gt, op.lt, op.ge, op.le]

for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = f'__{operator_name}__'
    if name.endswith('___'):
        name = name[:-1]

    if operator in force_linking or not hasattr(Expression, name):
        setattr(Expression, name, op_bind(operator))


for operator in [
    op.add, op.sub, op.mul, op.matmul, op.truediv, op.floordiv,
    op.mod,  # op.divmod,
    op.pow, op.lshift, op.rshift, op.and_, op.xor,
    op.or_
]:
    name = f'__r{operator.__name__}__'
    if name.endswith('___'):
        name = name[:-1]

    setattr(Expression, name, rop_bind(operator))


class Operation(Expression):
    operator_repr = {
        op.and_: '\u2227',
        op.or_: '\u2228',
        op.invert: '\u00ac',
    }

    def __init__(
        self, query_builder, expression,
        operator, arguments, infix=False
    ):
        self.query_builder = query_builder
        self.expression = expression
        self.operator = operator
        self.arguments = arguments
        self.infix = infix

    def __repr__(self):
        if isinstance(self.operator, Symbol):
            op_repr = self.operator.symbol_name
        elif isinstance(self.operator, Operation):
            op_repr = '({})'.format(repr(self.operator))
        elif self.operator in self.operator_repr:
            op_repr = self.operator_repr[self.operator]
        elif isinstance(self.operator, Expression):
            op_repr = repr(self.operator)
        elif hasattr(self.operator, '__qualname__'):
            op_repr = self.operator.__qualname__
        else:
            op_repr = repr(self.operator)

        return self.__repr_arguments(op_repr)

    def __repr_arguments(self, op_repr):
        arguments_repr = []
        for a in self.arguments:
            arg_repr = self.__repr_arguments_arg(a)
            arguments_repr.append(arg_repr)
        if self.infix:
            return ' {} '.format(op_repr).join(arguments_repr)
        else:
            return '{}({})'.format(
                op_repr,
                ', '.join(arguments_repr)
            )

    def __repr_arguments_arg(self, a):
        if isinstance(a, Operation):
            arg_repr = '( {} )'.format(repr(a))
        elif isinstance(a, Symbol):
            arg_repr = a.symbol_name
        else:
            arg_repr = repr(a)
        return arg_repr


class Symbol(Expression):
    def __init__(self, query_builder, symbol_name):
        self.symbol_name = symbol_name
        self.query_builder = query_builder
        self._rsbv = ReplaceExpressionsByValues(
            self.query_builder.solver.symbol_table
        )

    def __repr__(self):
        symbol = self.symbol
        if isinstance(symbol, Symbol):
            return(f'{self.symbol_name}: {symbol.type}')
        elif isinstance(symbol, nl.Constant):
            if exp.is_leq_informative(symbol.type, AbstractSet):
                value = list(self)
            else:
                value = symbol.value

            return f'{self.symbol_name}: {symbol.type} = {value}'
        else:
            return f'{self.symbol_name}: {symbol.type}'

    def _repr_iterable_value(self, symbol):
        contained = []
        for v in self:
            contained.append(repr(v))
        return contained

    def __iter__(self):
        symbol = self.symbol
        if not (
            isinstance(symbol, nl.Constant) and (
                exp.is_leq_informative(symbol.type, AbstractSet) or
                exp.is_leq_informative(symbol.type, Tuple)
            )
        ):
            raise TypeError(
                f'Symbol of type {self.symbol.type} is not iterable'
            )

        if self.query_builder.logic_programming:
            return self.__iter_logic_programming(symbol)
        else:
            return self.__iter_non_logic_programming(symbol)

    def __iter_logic_programming(self, symbol):
        for v in symbol.value:
            if isinstance(v, nl.Constant):
                yield self._rsbv.walk(v.value)
            elif isinstance(v, nl.Symbol):
                yield Symbol(self.query_builder, v)
            else:
                raise nl.NeuroLangException(f'element {v} invalid in set')

    def __iter_non_logic_programming(self, symbol):
        all_symbols = (
            self.query_builder
            .solver.symbol_table.symbols_by_type(
                symbol.type.__args__[0]
            )
        )

        for s in symbol.value:
            if not isinstance(s, nl.Constant):
                yield Symbol(self.query_builder, s.name)
                continue
            for k, v in all_symbols.items():
                if isinstance(v, nl.Constant) and s is v.value:
                    yield Symbol(self.query_builder, k.name)
                    break
                yield Expression(self.query_builder, nl.Constant(s))

    def __len__(self):
        symbol = self.symbol
        if (
            isinstance(symbol, nl.Constant) and (
                exp.is_leq_informative(symbol.type, AbstractSet) or
                exp.is_leq_informative(symbol.type, Tuple)
            )
        ):
            return len(symbol.value)

    def __eq__(self, other):
        if isinstance(other, Expression):
            return self.expression == other.expression
        else:
            return self.expression == other

    def __hash__(self):
        return hash(self.expression)

    @property
    def symbol(self):
        return self.query_builder.solver.symbol_table[self.symbol_name]

    @property
    def neurolang_symbol(self):
        return nl.Symbol[self.type](self.symbol_name)

    @property
    def expression(self):
        return self.symbol

    @property
    def value(self):
        constant = self.query_builder.solver.symbol_table[self.symbol_name]
        if (
            isinstance(constant, exp.Constant) and
            isinstance(constant.value, RelationalAlgebraFrozenSet)
        ):
            return RelationalAlgebraFrozenSet(constant.value)
        else:
            try:
                return self._rsbv.walk(constant)
            except NeuroLangPatternMatchingNoMatch:
                raise ValueError("Expression doesn't have a python value")

    @property
    def parameter_names(self):
        return self.query_builder.parameter_names(self)


class Query(Expression):
    def __init__(self, query_builder, expression, symbol, predicate):
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self):
        return u'{{{s} | {p}}}'.format(
            s=repr(self.symbol),
            p=repr(self.predicate)
        )


class Exists(Expression):
    def __init__(self, query_builder, expression, symbol, predicate):
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self):
        return u'\u2203{s}: {p}'.format(
            s=repr(self.symbol),
            p=repr(self.predicate)
        )


class All(Expression):
    def __init__(self, query_builder, expression, symbol, predicate):
        self.query_builder = query_builder
        self.expression = expression
        self.symbol = symbol
        self.predicate = predicate

    def __repr__(self):
        return u'\u2200{s}: {p}'.format(
            s=repr(self.symbol),
            p=repr(self.predicate)
        )


class Implication(Expression):
    def __init__(self, query_builder, expression, consequent, antecedent):
        self.expression = expression
        self.query_builder = query_builder
        self.antecedent = antecedent
        self.consequent = consequent

    def __repr__(self):
        return u'{c} \u2190 {a}'.format(
            a=repr(self.antecedent),
            c=repr(self.consequent)
        )


class Fact(Expression):
    def __init__(self, query_builder, expression, consequent):
        self.expression = expression
        self.query_builder = query_builder
        self.consequent = consequent

    def __repr__(self):
        return u'{c}'.format(
            c=repr(self.consequent),
        )


class TranslateExpressionToFrontEndExpression(ExpressionWalker):
    def __init__(self, query_builder):
        self.query_builder = query_builder

    @add_match(exp.Symbol)
    def symbol(self, expression):
        return Expression(self.query_builder, expression)

    @add_match(exp.Constant)
    def constant(self, expression):
        return expression.value

    @add_match(exp.FunctionApplication)
    def function_application(self, expression):
        functor = self.walk(expression.functor)
        args = tuple(self.walk(arg) for arg in expression.args)
        return functor(*args)

    @add_match(dl.Implication(..., True))
    def fact(self, expression):
        return Fact(
            self.query_builder,
            expression, self.walk(expression.consequent)
        )

    @add_match(dl.Implication)
    def implication(self, expression):
        return Implication(
            self.query_builder,
            expression,
            self.walk(expression.consequent),
            self.walk(expression.antecedent)
        )

    @add_match(dl.Conjunction)
    def conjunction(self, expression):
        formulas = list(expression.formulas[::-1])
        current_expression = self.walk(formulas.pop())
        while len(formulas) > 0:
            current_expression = (
                current_expression &
                self.walk(formulas.pop())
            )
        return current_expression
