from itertools import chain
import operator as op
import typing
import inspect
from functools import wraps, WRAPPER_ASSIGNMENTS
import types
from warnings import warn
from .exceptions import NeuroLangException


__all__ = [
    'Symbol', 'Function', 'Definition', 'evaluate',
    'Projection',
    'ToBeInferred',
    'typing_callable_from_annotated_function'
]


ToBeInferred = typing.TypeVar('ToBeInferred', covariant=True)


class NeuroLangTypeException(NeuroLangException):
    pass


def typing_callable_from_annotated_function(function):
    signature = inspect.signature(function)
    parameter_types = [
        v.annotation for v in signature.parameters.values()
    ]
    return typing.Callable[
        parameter_types,
        signature.return_annotation
    ]


def get_type_args(type_):
    if hasattr(type_, '__args__') and type_.__args__ is not None:
        return type_.__args__
    else:
        return tuple()


def get_type_and_value(value, symbol_table=None):
    if symbol_table is not None and isinstance(value, Symbol):
        value = symbol_table.get(value, value)

    if isinstance(value, Expression):
        return value.type, value.value
    else:
        if isinstance(value, types.FunctionType):
            return (
                typing_callable_from_annotated_function(value),
                value
            )
        else:
            return type(value), value


def is_subtype(left, right):
    if left == right:
        return True
    if right == typing.Any:
        return True
    elif left == typing.Any:
        return right == typing.Any
    elif left == ToBeInferred:
        return True
    elif hasattr(right, '__origin__') and right.__origin__ is not None:
        if right.__origin__ == typing.Union:
            return any(
                is_subtype(left, r)
                for r in right.__args__
            )
        elif issubclass(right, typing.Callable):
            if issubclass(left, typing.Callable):
                left_args = get_type_args(left)
                right_args = get_type_args(right)

                if len(left_args) != len(right_args):
                    False

                return all((
                    is_subtype(left_arg, right_arg)
                    for left_arg, right_arg in zip(left_args, right_args)
                ))
            else:
                return False
        elif (any(
            issubclass(right, T) and
            issubclass(left, T)
            for T in (
                typing.AbstractSet, typing.List, typing.Tuple,
                typing.Mapping, typing.Iterable
            )
        )):
            return all(
                is_subtype(l, r) for l, r in zip(
                    get_type_args(left), get_type_args(right)
                )
            )
        else:
            raise ValueError("typing Generic not supported")
    else:
        if left == int:
            return right in (float, complex, typing.SupportsInt)
        elif left == float:
            return right in (complex, typing.SupportsFloat)
        elif right == int:
            right = typing.SupportsInt
        elif right == float:
            right = typing.SupportsFloat
        elif right == str:
            right = typing.Text

        return issubclass(left, right)


def unify_types(t1, t2):
    if is_subtype(t1, t2):
        return t2
    elif is_subtype(t2, t1):
        return t1


def type_validation_value(value, type_, symbol_table=None):
    if type_ == typing.Any or type_ == ToBeInferred:
        return True

    if (
        (symbol_table is not None) and
        isinstance(value, Symbol)
    ):
        value = symbol_table[value].value
    elif isinstance(value, Expression):
        value = value.value

    if hasattr(type_, '__origin__') and type_.__origin__ is not None:
        if type_.__origin__ == typing.Union:
            return any(
                type_validation_value(value, t, symbol_table=symbol_table)
                for t in type_.__args__
            )
        elif issubclass(type_, typing.Callable):
            value_type, _ = get_type_and_value(value)
            return is_subtype(value_type, type_)
        elif issubclass(type_, typing.Mapping):
            return (
                issubclass(type(value), type_.__origin__) and
                ((type_.__args__ is None) or all((
                    type_validation_value(
                        k, type_.__args__[0], symbol_table=symbol_table
                    ) and
                    type_validation_value(
                        v, type_.__args__[1], symbol_table=symbol_table
                    )
                    for k, v in value.items()
                )))
            )
        elif issubclass(type_, typing.Tuple):
            return (
                issubclass(type(value), type_.__origin__) and
                all((
                    type_validation_value(
                        v, t, symbol_table=symbol_table
                    )
                    for v, t in zip(value, type_.__args__)
                ))
            )
        elif any(
            issubclass(type_, t)
            for t in (typing.AbstractSet, typing.Sequence, typing.Iterable)
        ):
            return (
                issubclass(type(value), type_.__origin__) and
                ((type_.__args__ is None) or all((
                    type_validation_value(
                        i, type_.__args__[0], symbol_table=symbol_table
                    )
                    for i in value
                )))
            )
        else:
            raise ValueError("Type %s not implemented in the checker" % type_)
    elif isinstance(value, Function):
        return is_subtype(value.type, type_)
    else:
        return isinstance(
            value, type_
        )


class Expression(object):
    _symbols = set()
    super_attributes = WRAPPER_ASSIGNMENTS + ('__signature__', 'mro', 'type')
    type = ToBeInferred

    def __init__(self, value, type_=ToBeInferred, symbol_table=None):
        if not type_validation_value(
            value, type_, symbol_table=symbol_table
        ):
            raise NeuroLangTypeException(
               "The value %s does not correspond to the type %s" %
               (value, type_)
            )
        self.type = type_
        self.value = value

        if not type_validation_value(
            value, self.type, symbol_table=symbol_table
        ):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (value, type_)
            )

        if isinstance(value, Expression):
            self._symbols |= value._symbols

    def __call__(self, *args, **kwargs):
        if hasattr(self, '__annotations__'):
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = ToBeInferred

        return Function(
            self, args, kwargs,
            type_=variable_type,
         )

    def __repr__(self):
        return 'E{{{}: {}}}'.format(self.value, self.type)

    def __getattr__(self, attr):
        if (
            attr in dir(self) or
            attr in self.super_attributes
        ):
            return getattr(super(), attr)
        else:
            return Function(
                Constant(getattr),
                args=(self, Constant(attr, type_=str))
            )


class Symbol(Expression):
    def __init__(self, name, type_=ToBeInferred, symbol_table=None):
        self.name = name
        self.type = type_
        self._symbols = {self}
        self._symbol_table = None
        self.value = self

    def __eq__(self, other):
        return (
            (isinstance(other, Symbol) or isinstance(other, str)) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return 'S{{{}: {}}}'.format(self.name, self.type)


class Constant(Expression):
    def __init__(
        self, value, type_=ToBeInferred,
        auto_infer_annotated_functions_type=True
    ):
        if callable(value):
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(value, attr):
                    setattr(self, attr, getattr(value, attr))

            if (
                auto_infer_annotated_functions_type and
                hasattr(value, '__annotations__') and type_ == ToBeInferred
            ):
                type_ = typing_callable_from_annotated_function(
                    value
                )

        self.value = value
        self.type = type_

        if not type_validation_value(
            value, self.type
        ):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (value, type_)
            )

    def __eq__(self, other):
        if self.type == ToBeInferred:
            warn('Making a comparison with types needed to be inferred')
        return (
            (
                isinstance(other, Constant) or
                self.type == ToBeInferred or isinstance(other, self.type)
            ) and
            hash(self) == hash(other)
        )

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return 'C{{{}: {}}}'.format(self.value, self.type)


class Function(Expression):
    def __init__(
        self, object_, args=None, kwargs=None,
        type_=ToBeInferred
    ):
        self.__wrapped__ = object_
        self.value = self
        if args is None and kwargs is None:
            if isinstance(self.__wrapped__, Constant):
                function = self.__wrapped__.value
            else:
                function = self.__wrapped__
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(function, attr):
                    setattr(self, attr, getattr(function, attr))
            if hasattr(function, '__annotations__'):
                self.type = typing_callable_from_annotated_function(
                    function
                )
            else:
                self.type = type_
            self.has_been_applied = False
        else:
            self.has_been_applied = True
            if isinstance(self.__wrapped__, Expression):
                if self.__wrapped__.type == ToBeInferred:
                    self.type = ToBeInferred
                elif isinstance(self.__wrapped__.type, typing.Callable):
                    self.type = self.__wrapped__.type.__args__[-1]
                else:
                    NeuroLangTypeException
            elif (
                hasattr(self.__wrapped__, '__signature__') or
                hasattr(self.__wrapped__, '__annotations__')
            ):
                self.type = inspect.signature(
                    self.__wrapped__
                ).return_annotation

        if isinstance(object_, Symbol):
            self._symbols = {object_}
        elif isinstance(object_, Function):
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
                elif isinstance(arg, Function):
                    self._symbols |= arg._symbols

    @property
    def function(self):
        return self.__wrapped__

    def __call__(self, *args, **kwargs):
        symbols = self._symbols.copy()
        for arg in chain(args, kwargs.values()):
            if isinstance(arg, Expression):
                symbols |= arg._symbols

        if hasattr(self, '__annotations__'):
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = None

        return Function(
            self, args, kwargs,
            type_=variable_type,
         )

    def __repr__(self):
        if hasattr(self.__wrapped__, '__name__'):
            fname = self.__wrapped__.__name__
        else:
            fname = repr(self.__wrapped__)
        r = 'F{{{}: {}}}'.format(fname, self.type)
        if self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args) +
                ', '.join(
                    repr(k) + '=' + repr(v)
                    for k, v in self.kwargs.items()
                ) + ')')

        return r


class Projection(Expression):
    def __init__(
        self, collection, item,
        type_=ToBeInferred,
        auto_infer_projection_type=True
    ):
        if type_ == ToBeInferred and auto_infer_projection_type:
            if not collection.type == ToBeInferred:
                if is_subtype(collection.type, typing.Tuple):
                    if (
                        isinstance(item, Constant) and
                        is_subtype(item.type, typing.SupportsInt) and
                        len(collection.type.__args__) > int(item.value)
                    ):
                        type_ = collection.type.__args__[int(item.value)]
                    else:
                        raise NeuroLangTypeException(
                            "Not {} elements in tuple".format(int(item.value))
                        )
                if is_subtype(collection.type, typing.Mapping):
                    type_ = collection.type.__args__[1]

        self._symbol = collection._symbols | item._symbols
        self.collection = collection
        self.item = item
        self.type = type_

    def __repr__(self):
        return 'P{{{}[{}]: {}}}'.format(self.collection, self.item, self.type)


class Definition(Expression):
    def __init__(
        self, symbol, expression,
        type_=ToBeInferred, symbol_table=None
    ):
        self.symbol = symbol
        self.value = expression
        self.type = type_

    def __repr__(self):
        return 'Def{{{}: {} <- {}}}'.format(
            self.symbol.name, self.type, self.value
        )


class Query(Definition):

    def __repr__(self):
        return 'Query{{{}: {} <- {}}}'.format(
            self.symbol.name, self.type, self.value
        )


def op_bind(op):
    @wraps(op)
    def f(*args):
        return Function(Constant(op), args=args)

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        return Function(Constant(op), args=(value, self))

    return f


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = '__{}__'.format(operator_name)
    if name.endswith('___'):
        name = name[:-1]

    for c in (Constant, Symbol, Function, Definition):
        if not hasattr(c, name):
            setattr(c, name, op_bind(operator))


for operator in [
    op.add, op.sub, op.mul, op.matmul, op.truediv, op.floordiv,
    op.mod,  # op.divmod,
    op.pow, op.lshift, op.rshift, op.and_, op.xor,
    op.or_
]:
    name = '__r{}__'.format(operator.__name__)
    if name.endswith('___'):
        name = name[:-1]

    for c in (Constant, Symbol, Function, Definition):
        setattr(c, name, rop_bind(operator))


def evaluate(expression, **kwargs):
    '''
    Replace free variables and evaluate the function
    '''
    if isinstance(expression, list) or isinstance(expression, tuple):
        evaluated = []
        for e in expression:
            evaluated.append(evaluate(e, **kwargs))

        return evaluated
    elif isinstance(expression, Constant):
        return expression
    elif isinstance(expression, Symbol):
        if expression in kwargs:
            return kwargs[expression]
        else:
            return expression
    elif isinstance(expression, Definition):
        result = Definition(
            expression.symbol,
            evaluate(expression.value, **kwargs),
            type_=expression.type,
        )
    elif isinstance(expression, Function):
        if isinstance(expression.function, Expression):
            function = evaluate(expression.function, **kwargs)
        else:
            raise ValueError

        if expression.args is None:
            return function

        we_should_evaluate = True
        new_args = []
        for arg in expression.args:
            arg = evaluate(arg, **kwargs)
            new_args.append(arg)
            we_should_evaluate &= isinstance(arg, Constant)

        new_kwargs = dict()
        for k, arg in expression.kwargs.items():
            arg = evaluate(arg, **kwargs)
            new_kwargs[k] = arg
            we_should_evaluate &= isinstance(arg, Constant)

        we_should_evaluate &= isinstance(function, Constant)
        if we_should_evaluate:
            function = function.value
            new_args = tuple(
                a.value for a in new_args
            )
            new_kwargs = {
                k: v.value
                for k, v in new_kwargs.items()
            }
            result = Constant(
                function(*new_args, **new_kwargs),
                type_=expression.type
            )
            return result
        else:
            return Function(function)(*new_args, **new_kwargs)
    else:
        return expression
