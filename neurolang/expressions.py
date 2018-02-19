from itertools import chain
import operator as op
import typing
from functools import wraps, WRAPPER_ASSIGNMENTS
# from types import FunctionType, BuiltinFunctionType

__all__ = ['Symbol', 'SymbolApplication', 'evaluate']


class Symbol(object):
    def __init__(self, name, type_=typing.Any):
        self.name = name
        self.type = type_

    def __eq__(self, other):
        return (
            (isinstance(other, Symbol) or isinstance(other, str)) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return 'S{%s}' % self.name

    def __getattr__(self, name):
        if name in WRAPPER_ASSIGNMENTS:
            return super().__getattr__(name)
        else:
            return SymbolApplication(getattr, args=(self, name))


class SymbolApplication(object):
    def __init__(
        self, object_, args=None, kwargs=None,
        type_=typing.Any
    ):
        self.__wrapped__ = object_

        if args is None and kwargs is None:
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(self.__wrapped__, attr):
                    setattr(self, attr, getattr(self.__wrapped__, attr))
            self.type = None
            self.is_function_type = True
        else:
            self.is_function_type = False
            self.type = type_

        if isinstance(object_, Symbol):
            self._free_variables = {object_}
        elif isinstance(object_, SymbolApplication):
            self._free_variables = object_._free_variables.copy()
        else:
            self._free_variables = set()

        self.args = args
        self.kwargs = kwargs

        if self.args is not None or self.kwargs is not None:

            if self.args is None:
                self.args = tuple()
            if self.kwargs is None:
                self.kwargs = dict()

            for arg in chain(self.args, self.kwargs.values()):
                if isinstance(arg, Symbol):
                    self._free_variables.add(arg)
                elif isinstance(arg, SymbolApplication):
                    self._free_variables |= arg._free_variables

    def __call__(self, *args, **kwargs):
        free_variables = self._free_variables.copy()

        for arg in chain(args, kwargs.values()):
            if isinstance(arg, Symbol):
                free_variables.add(arg)
            elif isinstance(arg, SymbolApplication):
                free_variables |= arg._free_variables

        if hasattr(self, '__annotations__'):
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = None

        if len(free_variables) > 0:
            return SymbolApplication(
                self, args, kwargs,
                type_=variable_type
            )
        else:
            return self.__wrapped__(*args, **kwargs)

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
                ) +
                ')'
            )
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

    for c in (Symbol, SymbolApplication):
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

    for c in (Symbol, SymbolApplication):
        setattr(c, name, rop_bind(operator))


def evaluate(expression, **kwargs):
    '''
    Replace free variables and evaluate the function
    '''
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

    new_args = []
    for arg in expression.args:
        if isinstance(arg, Symbol) and arg in kwargs:
            new_args.append(kwargs[arg])
        elif isinstance(arg, SymbolApplication):
            new_args.append(evaluate(arg, **kwargs))
        else:
            new_args.append(arg)

    new_kwargs = dict()
    for k, arg in expression.kwargs.items():
        if isinstance(arg, Symbol) and arg in kwargs:
            new_kwargs[k] = kwargs[arg]
        elif isinstance(arg, SymbolApplication):
            new_kwargs[k] = evaluate(arg, **kwargs)
        else:
            new_kwargs[k] = arg

    return SymbolApplication(function)(*new_args, **new_kwargs)
