from itertools import chain
import operator as op
# from types import FunctionType, BuiltinFunctionType

__all__ = ['FreeVariable', 'FreeVariableApplication', 'evaluate']


class FreeVariable(object):
    def __init__(self, name):
        self.name = name

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
        return FreeVariableApplication(getattr, args=(self, name))


class FreeVariableApplication(object):
    def __init__(self, object_, args=None, kwargs=None):
        self.function = object_
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

        if len(free_variables) > 0:
            return FreeVariableApplication(self, args, kwargs)
        else:
            return self.function(*args, **kwargs)

    def __repr__(self):
        if hasattr(self.function, '__name__'):
            fname = self.function.__name__
        else:
            fname = repr(self.function)
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
            isinstance(free_variable_symbol.function, FreeVariable) and
            free_variable_symbol.function in kwargs
    ):
        function = kwargs[free_variable_symbol.function]
    elif isinstance(free_variable_symbol.function, FreeVariableApplication):
        function = evaluate(free_variable_symbol.function, **kwargs)
    else:
        function = free_variable_symbol.function

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
    def f(*args):
        return FreeVariableApplication(op, args=args)

    return f


def rop_bind(op):
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

for operator in [op.add, op.sub]:
    name = '__r%s__' % operator.__name__
    setattr(FreeVariable, name, rop_bind(operator))
    setattr(FreeVariableApplication, name, rop_bind(operator))
