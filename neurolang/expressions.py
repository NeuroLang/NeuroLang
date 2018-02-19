from itertools import chain
import operator as op
import typing
from functools import wraps, WRAPPER_ASSIGNMENTS
# from types import FunctionType, BuiltinFunctionType

__all__ = ['FreeVariable', 'FreeVariableApplication', 'evaluate']


class FreeVariable(object):
    def __init__(self, name, variable_type=typing.Any):
        self.name = name
        self.variable_type = variable_type

    def __eq__(self, other):
        return (
            (isinstance(other, FreeVariable) or isinstance(other, str)) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return 'FV{%s}' % self.name

    def __getattr__(self, name):
        if name in WRAPPER_ASSIGNMENTS:
            return super().__getattr__(name)
        else:
            return FreeVariableApplication(getattr, args=(self, name))


class FreeVariableApplication(object):
    def __init__(
        self, object_, args=None, kwargs=None,
        variable_type=typing.Any
    ):
        self.__wrapped__ = object_

        if args is None and kwargs is None:
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(self.__wrapped__, attr):
                    setattr(self, attr, getattr(self.__wrapped__, attr))
            self.variable_type = None
            self.is_function_type = True
        else:
            self.is_function_type = False
            self.variable_type = variable_type

        if isinstance(object_, FreeVariable):
            self._free_variables = {object_}
        elif isinstance(object_, FreeVariableApplication):
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
                if isinstance(arg, FreeVariable):
                    self._free_variables.add(arg)
                elif isinstance(arg, FreeVariableApplication):
                    self._free_variables |= arg._free_variables

    def __call__(self, *args, **kwargs):
        free_variables = self._free_variables.copy()

        for arg in chain(args, kwargs.values()):
            if isinstance(arg, FreeVariable):
                free_variables.add(arg)
            elif isinstance(arg, FreeVariableApplication):
                free_variables |= arg._free_variables

        if hasattr(self, '__annotations__'):
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = None

        if len(free_variables) > 0:
            return FreeVariableApplication(
                self, args, kwargs,
                variable_type=variable_type
            )
        else:
            return self.__wrapped__(*args, **kwargs)

    def __repr__(self):
        if hasattr(self.__wrapped__, '__name__'):
            fname = self.__wrapped__.__name__
        else:
            fname = repr(self.__wrapped__)
        r = 'FVA{%s}' % fname
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


def evaluate(free_variable_symbol, **kwargs):
    '''
    Replace free variables and evaluate the function
    '''
    if (
            isinstance(free_variable_symbol.__wrapped__, FreeVariable) and
            free_variable_symbol.__wrapped__ in kwargs
    ):
        function = kwargs[free_variable_symbol.__wrapped__]
    elif isinstance(free_variable_symbol.__wrapped__, FreeVariableApplication):
        function = evaluate(free_variable_symbol.__wrapped__, **kwargs)
    else:
        function = free_variable_symbol.__wrapped__

    if free_variable_symbol.args is None:
        return function

    new_args = []
    for arg in free_variable_symbol.args:
        if isinstance(arg, FreeVariable) and arg in kwargs:
            new_args.append(kwargs[arg])
        elif isinstance(arg, FreeVariableApplication):
            new_args.append(evaluate(arg, **kwargs))
        else:
            new_args.append(arg)

    new_kwargs = dict()
    for k, arg in free_variable_symbol.kwargs.items():
        if isinstance(arg, FreeVariable) and arg in kwargs:
            new_kwargs[k] = kwargs[arg]
        elif isinstance(arg, FreeVariableApplication):
            new_kwargs[k] = evaluate(arg, **kwargs)
        else:
            new_kwargs[k] = arg

    return FreeVariableApplication(function)(*new_args, **new_kwargs)


def op_bind(op):
    @wraps(op)
    def f(*args):
        return FreeVariableApplication(op, args=args)

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        return FreeVariableApplication(op, args=(value, self))

    return f


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = '__%s__' % operator_name
    if name.endswith('___'):
        name = name[:-1]

    if not hasattr(FreeVariable, name):
        setattr(FreeVariable, name, op_bind(operator))

    if not hasattr(FreeVariableApplication, name):
        setattr(FreeVariableApplication, name, op_bind(operator))

for operator in [
    op.add, op.sub, op.mul, op.matmul, op.truediv, op.floordiv,
    op.mod,  # op.divmod,
    op.pow, op.lshift, op.rshift, op.and_, op.xor,
    op.or_
]:
    name = '__r%s__' % operator.__name__
    if name.endswith('___'):
        name = name[:-1]

    setattr(FreeVariable, name, rop_bind(operator))
    setattr(FreeVariableApplication, name, rop_bind(operator))
