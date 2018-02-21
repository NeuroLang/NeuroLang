from itertools import chain
import operator as op
import typing
import inspect
from functools import wraps, WRAPPER_ASSIGNMENTS
# from types import FunctionType, BuiltinFunctionType

__all__ = [
    'Symbol', 'SymbolApplication', 'evaluate',
    'typing_callable_from_annotated_function'
]


def typing_callable_from_annotated_function(function):
    signature = inspect.signature(function)
    parameter_types = [
        v.annotation for v in signature.parameters.values()
    ]
    return typing.Callable[
        parameter_types,
        signature.return_annotation
    ]


class Expression(object):
    _symbols = {}
    super_attributes = WRAPPER_ASSIGNMENTS + ('__signature__', 'mro', 'type')
    local_attributes = tuple()
    type = typing.Any

    def __getattr__(self, attr):
        if attr in self.super_attributes or attr in self.local_attributes:
            return getattr(super(), attr)
        else:
            return SymbolApplication(getattr, args=(self, attr))


class Symbol(Expression):
    def __init__(self, name, type_=typing.Any):
        self.name = name
        self.type = type_
        self._symbols = {self}

    def __eq__(self, other):
        return (
            (isinstance(other, Symbol) or isinstance(other, str)) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return 'S{%s}' % self.name


class Constant(Expression):
    def __init__(self, value, type_=typing.Any):
        self.value = value
        self.type = type_
        self._symbols = {}

    def __eq__(self, other):
        return (
            (isinstance(other, Constant) or isinstance(other, self.type)) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return 'C{%s}' % self.value


class SymbolApplication(Expression):
    def __init__(
        self, object_, args=None, kwargs=None,
        type_=typing.Any
    ):
        self.__wrapped__ = object_

        if args is None and kwargs is None:
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(self.__wrapped__, attr):
                    setattr(self, attr, getattr(self.__wrapped__, attr))
            if hasattr(self.__wrapped__, '__annotations__'):
                setattr(
                    self, '__annotations__',
                    self.__wrapped__.__annotations__
                )
                self.type = typing_callable_from_annotated_function(
                    self.__wrapped__
                )
            self.is_function_type = True
        else:
            self.is_function_type = False
            if (
                hasattr(self.__wrapped__, '__signature__') or
                hasattr(self.__wrapped__, '__annotations__')
            ):
                self.type = inspect.signature(
                    self.__wrapped__
                ).return_annotation

        if isinstance(object_, Symbol):
            self._symbols = {object_}
        elif isinstance(object_, SymbolApplication):
            self._symbols = object_._symbols.copy()
        else:
            self._symbols = set()

        self.args = args
        self.kwargs = kwargs

        if self.args is not None or self.kwargs is not None:

            if self.args is None:
                self.args = tuple()
            if self.kwargs is None:
                self.kwargs = dict()

            for arg in chain(self.args, self.kwargs.values()):
                if isinstance(arg, Symbol):
                    self._symbols.add(arg)
                elif isinstance(arg, SymbolApplication):
                    self._symbols |= arg._symbols

    def __call__(self, *args, **kwargs):
        symbols = self._symbols.copy()
        for arg in chain(args, kwargs.values()):
            if isinstance(arg, Expression):
                symbols |= arg._symbols

        if hasattr(self, '__annotations__'):
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = None

        return SymbolApplication(
            self, args, kwargs,
            type_=variable_type,
         )

    def __repr__(self):
        if hasattr(self.__wrapped__, '__name__'):
            fname = self.__wrapped__.__name__
        else:
            fname = repr(self.__wrapped__)
        r = 'EA{%s}' % fname
        if self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args) +
                ', '.join(
                    repr(k) + '=' + repr(v)
                    for k, v in self.kwargs.items()
                ) + ')')

        return r


def op_bind(op):
    @wraps(op)
    def f(*args):
        return SymbolApplication(op, args=args)

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        return SymbolApplication(op, args=(value, self))

    return f


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = '__%s__' % operator_name
    if name.endswith('___'):
        name = name[:-1]

    for c in (Constant, Symbol, SymbolApplication):
        if not hasattr(c, name):
            setattr(c, name, op_bind(operator))


for operator in [
    op.add, op.sub, op.mul, op.matmul, op.truediv, op.floordiv,
    op.mod,  # op.divmod,
    op.pow, op.lshift, op.rshift, op.and_, op.xor,
    op.or_
]:
    name = '__r%s__' % operator.__name__
    if name.endswith('___'):
        name = name[:-1]

    for c in (Constant, Symbol, SymbolApplication):
        setattr(c, name, rop_bind(operator))


def evaluate(expression, **kwargs):
    '''
    Replace free variables and evaluate the function
    '''
    if isinstance(expression, Constant):
        return expression.value
    if isinstance(expression, Symbol):
        if expression in kwargs:
            return kwargs[expression]
        else:
            return expression
    elif isinstance(expression, SymbolApplication):
        if (
                isinstance(expression.__wrapped__, Symbol) and
                expression.__wrapped__ in kwargs
        ):
            function = kwargs[expression.__wrapped__]
        elif isinstance(expression.__wrapped__, SymbolApplication):
            function = evaluate(expression.__wrapped__, **kwargs)
        else:
            function = expression.__wrapped__

        if expression.args is None:
            return function

        we_should_evaluate = True
        new_args = []
        for arg in expression.args:
            arg = evaluate(arg, **kwargs)
            new_args.append(arg)
            we_should_evaluate &= not isinstance(arg, Expression)

        new_kwargs = dict()
        for k, arg in expression.kwargs.items():
            arg = evaluate(arg, **kwargs)
            new_kwargs[k] = arg
            we_should_evaluate &= not isinstance(arg, Expression)

        function = evaluate(function, **kwargs)
        if isinstance(function, Expression):
            we_should_evaluate = False

        if we_should_evaluate:
            return function(*new_args, **new_kwargs)
        else:
            return SymbolApplication(function)(*new_args, **new_kwargs)
    else:
        return expression
