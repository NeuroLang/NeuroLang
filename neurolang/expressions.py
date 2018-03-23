from itertools import chain
import operator as op
import typing
import inspect
from functools import wraps, WRAPPER_ASSIGNMENTS, lru_cache
import types
from warnings import warn
from .exceptions import NeuroLangException


__all__ = [
    'Symbol', 'FunctionApplication', 'Statement',
    'Projection', 'Predicate',
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
        type_ = value.type
        if isinstance(value, (Constant, Statement)):
            value = value.value
        return type_, value
    elif isinstance(value, types.FunctionType):
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
        return right == typing.Any or right == ToBeInferred
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

    if isinstance(value, Expression):
        value_type, value = get_type_and_value(value)

        if value is ...:
            return is_subtype(value_type, type_)

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
    elif isinstance(value, FunctionApplication):
        return is_subtype(value.type, type_)
    else:
        return isinstance(
            value, type_
        )


class ExpressionMeta(type):

    def __new__(cls, *args, **kwargs):
        __no_explicit_type__ = 'type' not in args[2]
        obj = super().__new__(cls, *args, **kwargs)
        obj.__no_explicit_type__ = __no_explicit_type__
        if obj.__no_explicit_type__:
            obj.type = typing.Any
        orig_init = obj.__init__

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            if self.__no_explicit_type__:
                self.type = ToBeInferred
            return orig_init(self, *args, **kwargs)

        obj.__init__ = new_init
        return obj

    @lru_cache(maxsize=None)
    def __getitem__(cls, type_):
        d = dict(cls.__dict__)
        d['type'] = type_
        d['__generic_class__'] = cls
        d['__no_explicit_type__'] = False
        return cls.__class__(
            cls.__name__, cls.__bases__,
            d
        )

    def __repr__(cls):
        r = cls.__name__
        if hasattr(cls, 'type'):
            if isinstance(cls.type, type):
                c = cls.type.__name__
            else:
                c = repr(cls.type)
            r += '[{}]'.format(c)
        return r

    def __subclasscheck__(cls, other):
        return (
            super().__subclasscheck__(other) or
            (
                hasattr(other, '__generic_class__') and
                issubclass(other.__generic_class__, cls)
            ) or
            (
                hasattr(other, '__generic_class__') and
                hasattr(cls, '__generic_class__') and
                issubclass(
                    other.__generic_class__,
                    cls.__generic_class__
                ) and
                is_subtype(other.type, cls.type)
            )
        )

    def __instancecheck__(cls, other):
        return (
            super().__instancecheck__(other) or
            issubclass(other.__class__, cls)
        )


class Expression(metaclass=ExpressionMeta):
    super_attributes = WRAPPER_ASSIGNMENTS + ('__signature__', 'mro', 'type')

    def __init__(self):
        raise TypeError("Expression can not be instantiated")

    def __getitem__(self, index):
        return Projection(self, index)

    def __call__(self, *args, **kwargs):
        if hasattr(self, '__annotations__'):
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = ToBeInferred

        return FunctionApplication[variable_type](
            self, args, kwargs,
         )

    def __getattr__(self, attr):
        if (
            attr in dir(self) or
            attr in self.super_attributes
        ):
            return getattr(super(), attr)
        else:
            return FunctionApplication(
                Constant[typing.Callable[[self.type, str], ToBeInferred]](
                    getattr,
                ),
                args=(self, Constant[str](attr))
            )


class Definition(Expression):
    '''
    Parent class for all composite operations
    such as A + B or F(A)
    '''


class Symbol(Expression):
    def __init__(self, name):
        self.name = name
        self._symbols = {self}

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
        self, value,
        auto_infer_type=True
    ):
        self.value = value
        self.__wrapped__ = None

        if self.value is ...:
            pass
        elif callable(self.value):
            self.__wrapped__ = value
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(value, attr):
                    setattr(self, attr, getattr(value, attr))

            if (auto_infer_type) and self.type == ToBeInferred:
                if hasattr(value, '__annotations__'):
                    self.type = typing_callable_from_annotated_function(value)
        elif auto_infer_type and self.type == ToBeInferred:
            if isinstance(self.value, tuple):
                self.type = typing.Tuple[tuple(
                    a.type if a is not ... else typing.Any
                    for a in self.value
                )]
            else:
                self.type = type(value)

        if not (
            self.value is ... or
            type_validation_value(self.value, self.type)
        ):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (self.value, self.type)
            )

    def __eq__(self, other):
        if self.type == ToBeInferred:
            warn('Making a comparison with types needed to be inferred')

        if isinstance(other, Expression):
            types_equal = (
                is_subtype(self.type, other.type) or
                is_subtype(other.type, self.type)
            )
            if types_equal:
                if isinstance(other, Constant):
                    return other.value == self.value
                else:
                    return hash(other) == hash(self)
            else:
                return False
        else:
            return (
                type_validation_value(other, self.type) and
                hash(other) == hash(self)
            )

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        if self.value is ...:
            value_str = '...'
        else:
            value_str = repr(self.value)
        return 'C{{{}: {}}}'.format(value_str, self.type)


class FunctionApplication(Definition):
    def __init__(
        self, functor, args, kwargs=None,
    ):
        self.functor = functor
        self.args = args
        self.kwargs = kwargs

        if isinstance(self.functor, Expression):
            if self.functor.type in (ToBeInferred, typing.Any):
                self.type = self.functor.type
            elif isinstance(self.functor.type, typing.Callable):
                if self.type == ToBeInferred:
                    self.type = self.functor.type.__args__[-1]
            else:
                raise NeuroLangTypeException("Functor is not an expression")

            if isinstance(functor, Symbol):
                self._symbols = {functor}
            elif isinstance(functor, FunctionApplication):
                self._symbols = functor._symbols.copy()
            else:
                self._symbols = set()

        elif (
            hasattr(self.functor, '__signature__') or
            hasattr(self.functor, '__annotations__')
        ):
            self.type = inspect.signature(
                self.functor
            ).return_annotation

        if self.kwargs is None:
            self.kwargs = dict()

        if self.args is not ...:
            if self.args is None:
                self.args = tuple()
            for arg in chain(self.args, self.kwargs.values()):
                if isinstance(arg, Symbol):
                    self._symbols.add(arg)
                elif isinstance(arg, FunctionApplication):
                    self._symbols |= arg._symbols

    @property
    def function(self):
        return self.functor

    def __repr__(self):
        r = u'\u03BB{{{}: {}}}'.format(self.functor, self.type)
        if self.args is ...:
            r += '(...)'
        elif self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args) +
                ', '.join(
                    repr(k) + '=' + repr(v)
                    for k, v in self.kwargs.items()
                ) + ')')

        return r


class Projection(Definition):
    def __init__(
        self, collection, item,
        auto_infer_projection_type=True
    ):
        if self.type == ToBeInferred and auto_infer_projection_type:
            if collection is not ...:
                if not collection.type == ToBeInferred:
                    if is_subtype(collection.type, typing.Tuple):
                        if (
                            isinstance(item, Constant) and
                            is_subtype(item.type, typing.SupportsInt) and
                            len(collection.type.__args__) > int(item.value)
                        ):
                            self.type = collection.type.__args__[
                                int(item.value)
                            ]
                        else:
                            raise NeuroLangTypeException(
                                "Not {} elements in tuple".format(
                                    int(item.value)
                                )
                            )
                    if is_subtype(collection.type, typing.Mapping):
                        self.type = collection.type.__args__[1]

        if collection is not ...:
            self._symbol = collection._symbols
        if item is not ...:
            self._symbol |= item._symbols

        self.collection = collection
        self.item = item

    def __repr__(self):
        return u"\u03C3{{{}[{}]: {}}}".format(
            self.collection, self.item, self.type
        )


class Predicate(FunctionApplication):
    def __repr__(self):
        r = 'P{{{}: {}}}'.format(self.functor, self.type)
        if self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args) +
                ', '.join(
                    repr(k) + '=' + repr(v)
                    for k, v in self.kwargs.items()
                ) + ')')
        return r


class Statement(Expression):
    def __init__(
        self, symbol, value,
    ):
        self.symbol = symbol
        self.value = value

    def __repr__(self):
        if self.symbol is ...:
            name = '...'
        else:
            name = self.symbol.name
        return 'Statement{{{}: {} <- {}}}'.format(
            name, self.type, self.value
        )


class Query(Statement):
    def __repr__(self):
        if self.symbol is ...:
            name = '...'
        else:
            name = self.symbol.name

        return 'Query{{{}: {} <- {}}}'.format(
            name, self.type, self.value
        )


def op_bind(op):
    @wraps(op)
    def f(*args):
        arg_types = [get_type_and_value(a)[0] for a in args]
        return FunctionApplication(
            Constant[typing.Callable[arg_types, ToBeInferred]](op),
            args,
        )

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        arg_types = [get_type_and_value(a)[0] for a in (value, self)]
        return FunctionApplication(
            Constant[typing.Callable[arg_types, ToBeInferred]](op, ),
            args=(value, self),
        )

    return f


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = '__{}__'.format(operator_name)
    if name.endswith('___'):
        name = name[:-1]

    for c in (Constant, Symbol, FunctionApplication, Statement):
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

    for c in (Constant, Symbol, FunctionApplication, Statement):
        setattr(c, name, rop_bind(operator))
