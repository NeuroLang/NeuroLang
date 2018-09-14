from functools import wraps
import operator as op
from typing import AbstractSet, Tuple
from .. import neurolang as nl
from ..symbols_and_types import is_leq_informative
from ..expression_walker import ReplaceExpressionsByValues
from ..expression_pattern_matching import NeuroLangPatternMatchingNoMatch


class Expression:
    def __init__(self, query_builder, expression):
        self.query_builder = query_builder
        self.expression = expression

    @property
    def type(self):
        return self.expression.type

    def do(self, result_symbol_name=None):
        return self.query_builder.execute_expression(
            self.expression,
            result_symbol_name=result_symbol_name
        )

    def __call__(self, *args, **kwargs):
        new_args = [
            a.expression if isinstance(a, Expression)
            else nl.Constant(a)
            for a in args
        ]

        new_kwargs = {
            k: (
                v.expression if isinstance(v, Expression)
                else nl.Constant(v)
            )
            for k, v in kwargs.items()
        }
        new_expression = self.expression(*new_args, **new_kwargs)
        return Operation(
                self.query_builder, new_expression, self, args)

    def __repr__(self):
        if isinstance(self.expression, nl.Constant):
            return repr(self.expression.value)
        elif isinstance(self.expression, nl.Symbol):
            return f'{self.expression.name}'
        else:
            return object.__repr__(self)


binary_opeations = (
    op.add, op.sub, op.mul
)


def op_bind(op):
    @wraps(op)
    def f(self, *args):
        new_args = [
            arg.expression if isinstance(arg, Expression)
            else nl.Constant(arg)
            for arg in args
        ]
        new_expression = op(self.expression, *new_args)
        return Operation(
            self.query_builder, new_expression, op,
            (self,) + args, infix=len(args) > 0
        )

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        original_value = value
        if isinstance(value, Expression):
            value = value.expression
        else:
            value = nl.Constant(value)

        return Operation(
            self.query_builder, op(self.expression, value),
            op, (self, original_value), infix=True
        )

    return f


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = f'__{operator_name}__'
    if name.endswith('___'):
        name = name[:-1]

    if not hasattr(Expression, name):
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
        elif hasattr(self.operator, '__qualname__'):
            op_repr = self.operator.__qualname__
        else:
            op_repr = repr(self.operator)

        arguments_repr = []
        for a in self.arguments:
            if isinstance(a, Operation):
                arguments_repr.append(
                    '( {} )'.format(repr(a))
                )
            elif isinstance(a, Symbol):
                arguments_repr.append(a.symbol_name)
            else:
                arguments_repr.append(repr(a))

        if self.infix:
            return ' {} '.format(op_repr).join(arguments_repr)
        else:
            return '{}({})'.format(
                op_repr,
                ', '.join(arguments_repr)
            )


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
            if is_leq_informative(symbol.type, AbstractSet):
                contained = []
                all_symbols = (
                    self.query_builder.solver.symbol_table.symbols_by_type(
                        symbol.type.__args__[0]
                    )
                )
                for s in symbol.value:
                    if isinstance(s, nl.Constant):
                        for k, v in all_symbols.items():
                            if isinstance(v, nl.Constant) and s is v.value:
                                contained.append(k.name)
                                break
                    if isinstance(s, nl.Symbol):
                        contained.append(s.name)
                    if isinstance(s, tuple):
                        t = '('
                        for e in s:
                            t += (e.name + ', ')
                        contained.append(t[:-2] + ')')
                return (f'{self.symbol_name}: {symbol.type} = {contained}')
            else:
                return (f'{self.symbol_name}: {symbol.type} = {symbol.value}')
        else:
            raise ValueError('...')

    def __iter__(self):
        symbol = self.symbol
        if (
            isinstance(symbol, nl.Constant) and (
                is_leq_informative(symbol.type, AbstractSet) or
                is_leq_informative(symbol.type, Tuple)
            )
        ):
            all_symbols = (
                self.query_builder.solver.symbol_table.symbols_by_type(
                    symbol.type.__args__[0]
                )
            )
            for s in symbol.value:
                if isinstance(s, nl.Constant):
                    for k, v in all_symbols.items():
                        if isinstance(v, nl.Constant) and s is v.value:
                            yield Symbol(self.query_builder, k.name)
                            break
                if isinstance(s, nl.Symbol):
                    yield Symbol(self.query_builder, s.name)
        else:
            raise TypeError(
                f'Symbol of type {self.symbol.type} is not iterable'
            )

    def __len__(self):
        symbol = self.symbol
        if (
            isinstance(symbol, nl.Constant) and (
                is_leq_informative(symbol.type, AbstractSet) or
                is_leq_informative(symbol.type, Tuple)
            )
        ):
            return len(symbol.value)

    def __eq__(self, other):
        if isinstance(other, Expression):
            return self.expression == other.expression
        else:
            return self.expression == other

    @property
    def symbol(self):
        return self.query_builder.solver.symbol_table[self.symbol_name]

    @property
    def expression(self):
        return self.symbol

    @property
    def value(self):
        constant = self.query_builder.solver.symbol_table[self.symbol_name]
        try:
            return self._rsbv.walk(constant)
        except NeuroLangPatternMatchingNoMatch:
            raise ValueError("Expression doesn't have a python value")


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
