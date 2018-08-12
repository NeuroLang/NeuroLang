"""Expressions for the intermediate representation and auxiliary functions."""
from itertools import chain
import operator as op
import typing
import inspect
from functools import wraps, WRAPPER_ASSIGNMENTS, lru_cache
import types
import threading
from warnings import warn
import logging
from contextlib import contextmanager
from .exceptions import NeuroLangException


__all__ = [
    'Symbol', 'FunctionApplication', 'Statement',
    'Projection', 'ExistentialPredicate', 'UniversalPredicate',
    'ToBeInferred',
    'typing_callable_from_annotated_function'
]


ToBeInferred = typing.TypeVar('ToBeInferred', covariant=True)


_lock = threading.RLock()

_expressions_behave_as_python_objects = dict()


@contextmanager
def expressions_behave_as_objects():
    global _lock
    global _expressions_behave_as_python_objects
    thread_id = threading.get_ident()

    with _lock:
        _expressions_behave_as_python_objects[thread_id] = True
    yield
    with _lock:
        del _expressions_behave_as_python_objects[thread_id]


class NeuroLangTypeException(NeuroLangException):
    pass


def typing_callable_from_annotated_function(function):
    """Get typing.Callable type representing the annotated function type."""
    signature = inspect.signature(function)
    parameter_types = [
        v.annotation if v.annotation is not inspect.Parameter.empty
        else ToBeInferred
        for v in signature.parameters.values()
    ]

    if signature.return_annotation is inspect.Parameter.empty:
        return_annotation = ToBeInferred
    else:
        return_annotation = signature.return_annotation
    return typing.Callable[
        parameter_types,
        return_annotation
    ]


def get_type_args(type_):
    if hasattr(type_, '__args__') and type_.__args__ is not None:
        if is_subtype(type_, typing.Callable):
            return list((list(type_.__args__[:-1]), type_.__args__[-1]))
        else:
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
    elif isinstance(value, (types.FunctionType, types.MethodType)):
        return (
            typing_callable_from_annotated_function(value),
            value
        )
    else:
        return type(value), value


def is_subtype(left, right):
    if (left is right) or (left == right):
        return True
    if right is typing.Any:
        return True
    elif left is typing.Any:
        return right is typing.Any or right is ToBeInferred
    elif left is ToBeInferred:
        return True
    elif right is ToBeInferred:
        return left is ToBeInferred or left is typing.Any
    elif hasattr(right, '__origin__') and right.__origin__ is not None:
        if right.__origin__ == typing.Union:
            return any(
                is_subtype(left, r)
                for r in right.__args__
            )
        elif issubclass(right, typing.Callable):
            if issubclass(left, typing.Callable):
                left_args, left_return = get_type_args(left)
                right_args, right_return = get_type_args(right)

                if len(left_args) != len(right_args):
                    return False

                return len(left_args) == 0 or (
                    is_subtype(left_return, right_return) and all((
                        is_subtype(left_arg, right_arg)
                        for left_arg, right_arg in zip(left_args, right_args)
                    ))
                )
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
        elif right.__origin__ == typing.Generic:
            raise ValueError("typing Generic not supported")
        else:
            return False
    else:
        if left is int:
            return right in (float, complex, typing.SupportsInt)
        elif left is float:
            return right in (complex, typing.SupportsFloat)
        elif right is int:
            right = typing.SupportsInt
        elif right is float:
            right = typing.SupportsFloat
        elif right is str:
            right = typing.Text

        return issubclass(left, right)


def unify_types(t1, t2):
    if t1 is ToBeInferred:
        return t2
    elif t2 is ToBeInferred:
        return t1
    elif is_subtype(t1, t2):
        return t2
    elif is_subtype(t2, t1):
        return t1
    else:
        raise NeuroLangTypeException(
            "The types {} and {} can't be unified".format(
                t1, t2
            )
        )


def type_validation_value(value, type_, symbol_table=None):
    if type_ is typing.Any or type_ is ToBeInferred:
        return True

    if isinstance(value, Symbol):
        if (symbol_table is not None):
            value = symbol_table[value].value
        else:
            return is_subtype(value.type, type_)

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
        return is_subtype(value.type, type_.type)
    else:
        return isinstance(
            value, type_
        )


class ParametricTypeClassMeta(type):
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


class ExpressionMeta(ParametricTypeClassMeta):
    """
    Metaclass for expressions. It guarantees a set of properties for every.

    class and instance:
    * Classes have an attribute `type` which is set through the syntax
      ClassName[TypeName]. If type is not specified then the `type`
      attribute defaults to `typing.Any` for the class and
      `ToBeInferred` for the instance. In this case the `__no_explicit_type__`
      attribute is set to `True`
    * Instances have an attribute for every argument in the `__init__`
      constructor method and it's set by default to the value passed
      during construction.
    * Instances have a `_symbols` attribute which defaults to the empty set.
      In other cases it must contain the set of free variables in the
      expression.
    * Constructor arguments can be given a value of `...` this is a
      wildcard pattern for pattern-matching expressing that any value can go in
      this parameter. In this case, the class constructor is not executed
      and the `__is_pattern__` attribute is set to `True`.
    """

    def __new__(cls, *args, **kwargs):
        __no_explicit_type__ = 'type' not in args[2]
        obj = super().__new__(cls, *args, **kwargs)
        obj.__no_explicit_type__ = __no_explicit_type__
        if obj.__no_explicit_type__:
            obj.type = typing.Any
        orig_init = obj.__init__
        obj.__children__ = [
            name for name, parameter
            in inspect.signature(orig_init).parameters.items()
            if parameter.default is inspect.Parameter.empty
        ][1:]

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            generic_pattern_match = any(
                a is ... or (isinstance(a, tuple) and ... in a) or
                (inspect.isclass(a) and issubclass(a, Expression))
                for a in args
            )
            self.__is_pattern__ = generic_pattern_match
            self._symbols = set()

            if self.__no_explicit_type__:
                self.type = ToBeInferred

            parameters = inspect.signature(self.__class__).parameters
            for parameter, value in zip(parameters.items(), args):
                argname, arg = parameter
                if arg.default is not inspect.Parameter.empty:
                    continue
                setattr(self, argname, value)

            if not self.__is_pattern__:
                return orig_init(self, *args, **kwargs)

        obj.__init__ = new_init
        return obj


class Expression(metaclass=ExpressionMeta):
    __super_attributes__ = WRAPPER_ASSIGNMENTS + (
        '__signature__', 'mro', 'type', '_symbols',
        '__code__', '__defaults__', '__kwdefaults__', '__no_type_check__'
    )

    def __init__(self, *args, **kwargs):
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
        global _expressions_behave_as_python_objects
        thread_id = threading.get_ident()

        if (
            thread_id not in _expressions_behave_as_python_objects or
            attr in dir(self) or
            attr in self.__super_attributes__ or
            self.__is_pattern__
        ):
            return object.__getattribute__(self, attr)
        else:
            logging.debug("Getting wrapped attribute {}".format(
                attr
            ))
            return self.get_wrapped_attribute(attr)

    def get_wrapped_attribute(self, attr):
        return FunctionApplication(
            Constant[typing.Callable[[self.type, str], ToBeInferred]](
                getattr,
            ),
            args=(self, Constant[str](attr))
        )

    def change_type(self, type_):
        self.__class__ = self.__class__[type_]

    def cast(self, type_):
        if type_ == self.type:
            return self
        parameters = inspect.signature(self.__class__).parameters
        args = (
            getattr(self, argname)
            for argname, arg in parameters.items()
            if arg.default is inspect.Parameter.empty
        )
        if hasattr(self.__class__, '__generic_class__'):
            ret = self.__class__.__generic_class__[type_](*args)
        else:
            ret = self.__class__[type_](*args)

        assert ret.type is type_
        return ret

    @property
    def __type_repr__(self):
        if (
            hasattr(self.type, '__qualname__') and
            not hasattr(self.type, '__args__')
        ):
            return self.type.__qualname__
        else:
            return repr(self.type)

    def __eq__(self, other):
        if self.__is_pattern__ or not isinstance(other, Expression):
            return super().__eq__(other)

        if not isinstance(other, type(self)):
            return False

        for child in self.__children__:
            val = getattr(self, child)
            val_other = getattr(self, child)

            if isinstance(val, (list, tuple)):
                if not all(v == o for v, o in zip(val, val_other)):
                    break
            elif not(val == val_other):
                break
        else:
            return True
        return False

    def __hash__(self):
        return hash(tuple(getattr(self, c) for c in self.__children__))


class ExpressionBlock(Expression):
    def __init__(self, expressions):
        self.expressions = expressions

    def __repr__(self):
        return '\\n'.join(
            repr(e) for e in self.expressions
        )


class NonConstant(Expression):
    """Any expression which is not a constant."""


class Definition(NonConstant):
    """
    Parent class for all composite operations.

    such as A + B or F(A)
    """


class Symbol(NonConstant):
    """Symbol of a certain type."""

    def __init__(self, name):
        """Initialize symbol with it's name."""
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
        return 'S{{{}: {}}}'.format(self.name, self.__type_repr__)


class Constant(Expression):
    def __init__(
        self, value,
        auto_infer_type=True,
        verify_type=True
    ):
        self.value = value
        self.__wrapped__ = None
        self.auto_infer_type = auto_infer_type
        self.verify_type = verify_type

        if callable(self.value):
            self.__wrapped__ = value
            for attr in WRAPPER_ASSIGNMENTS:
                if hasattr(value, attr):
                    setattr(self, attr, getattr(value, attr))

            if auto_infer_type and self.type is ToBeInferred:
                if hasattr(value, '__annotations__'):
                    self.type = typing_callable_from_annotated_function(value)
        elif auto_infer_type and self.type is ToBeInferred:
            if isinstance(self.value, tuple):
                self.type = typing.Tuple[tuple(
                    a.type
                    for a in self.value
                )]
                self._symbols = set()
                for a in self.value:
                    try:
                        self._symbols |= a._symbols
                    except AttributeError:
                        pass
            elif isinstance(self.value, frozenset):
                current_type = None
                self._symbols = set()
                for a in self.value:
                    try:
                        self._symbols |= a._symbols
                    except AttributeError:
                        pass
                    if isinstance(a, Expression):
                        new_type = a.type
                    else:
                        new_type = type(a)
                    if current_type is None:
                        current_type = new_type
                    else:
                        current_type = unify_types(current_type, new_type)
                self.type = typing.AbstractSet[current_type]
            else:
                self.type = type(value)

        if not self.__verify_type__(self.value, self.type):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (self.value, self.type)
            )

        if auto_infer_type and self.type is not ToBeInferred:
            self.change_type(self.type)

    def __verify_type__(self, value, type_):
        return (
            isinstance(
                value,
                (types.BuiltinFunctionType, types.BuiltinMethodType)
            ) or (
                self.verify_type and
                type_validation_value(value, type_)
            )
        )

    def __eq__(self, other):
        if self.type is ToBeInferred:
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
        elif callable(self.value) and not isinstance(self.value, Expression):
            value_str = self.value.__qualname__
        else:
            value_str = repr(self.value)
        return 'C{{{}: {}}}'.format(value_str, self.__type_repr__)

    def change_type(self, type_):
        self.__class__ = self.__class__[type_]
        if not self.__verify_type__(self.value, self.type):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (self.value, self.type)
            )


class FunctionApplication(Definition):
    def __init__(
        self, functor, args, kwargs=None,
    ):
        self.functor = functor
        self.args = args
        self.kwargs = kwargs

        if self.type in (ToBeInferred, typing.Any):
            if self.functor.type in (ToBeInferred, typing.Any):
                pass
            elif isinstance(self.functor.type, typing.Callable):
                self.type = self.functor.type.__args__[-1]
            else:
                raise NeuroLangTypeException("Functor is not an expression")
        else:
            if not (
                self.functor.type in (ToBeInferred, typing.Any)
                or is_subtype(self.functor.type.__args__[-1], self.type)
            ):
                raise NeuroLangTypeException(
                    "Functor return type not unifiable with application type"
                )

        if isinstance(functor, Symbol):
            self._symbols = {functor}
        elif isinstance(functor, FunctionApplication):
            self._symbols = functor._symbols.copy()
        else:
            self._symbols = set()

        if self.kwargs is None:
            self.kwargs = dict()

        if self.args is None:
            self.args = tuple()
        elif not isinstance(self.args, tuple):
            raise ValueError('args parameter must be a tuple')

        for arg in chain(self.args, self.kwargs.values()):
            if isinstance(arg, Symbol):
                self._symbols.add(arg)
            elif isinstance(arg, Expression):
                self._symbols |= arg._symbols

    @property
    def function(self):
        return self.functor

    def __repr__(self):
        r = u'\u03BB{{{}: {}}}'.format(self.functor, self.__type_repr__)
        if self.args is ...:
            r += '(...)'
        elif self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args)
                + ')'
                )

        return r


class Projection(Definition):
    def __init__(
        self, collection, item,
        auto_infer_projection_type=True
    ):
        if self.type is ToBeInferred and auto_infer_projection_type:
            if collection.type is not ToBeInferred:
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

        self._symbols = collection._symbols
        self._symbols |= item._symbols

        self.collection = collection
        self.item = item

    def __repr__(self):
        return u"\u03C3{{{}[{}]: {}}}".format(
            self.collection, self.item, self.__type_repr__
        )


class Quantifier(Definition):
    pass


class ExistentialPredicate(Quantifier):
    def __init__(self, head, body):

        if not isinstance(head, Symbol):
            raise NeuroLangException(
                'A symbol should be provided for the '
                'existential quantifier expression'
            )
        if not isinstance(body, FunctionApplication):
            raise NeuroLangException(
                'A function application over '
                'predicates should be associated to the quantifier'
            )

        if head not in body._symbols:
            raise NeuroLangException(
                'Symbol should be a free '
                'variable on the predicate'
            )
        self.head = head
        self.body = body
        self._symbols = body._symbols - {head}

    def __repr__(self):
        r = (
            u'\u2203{{{}: {} st {}}}'
            .format(self.head, self.__type_repr__, self.body)
        )
        return r


class UniversalPredicate(Quantifier):
    def __init__(self, head, body):

        if not isinstance(head, Symbol):
            raise NeuroLangException(
                'A symbol should be provided for the '
                'universal quantifier expression'
            )
        if not isinstance(body, FunctionApplication):
            raise NeuroLangException(
                'A function application over '
                'predicates should be associated to the quantifier'
            )

        if head not in body._symbols:
            raise NeuroLangException(
                'Symbol should be a free '
                'variable on the predicate'
            )
        self.head = head
        self.body = body
        self._symbols = body._symbols - {head}

    def __repr__(self):
        r = (
            u'\u2200{{{}: {} st {}}}'
            .format(self.head, self.__type_repr__, self.body)
        )
        return r


class Statement(Definition):
    def __init__(
        self, symbol, value,
    ):
        self.symbol = symbol
        self.value = value

    def reflect(self):
        return self.value

    def __repr__(self):
        if self.symbol is ...:
            name = '...'
        else:
            name = self.symbol.name
        return 'Statement{{{}: {} <- {}}}'.format(
            name, self.__type_repr__, self.value
        )


class Query(Definition):
    def __init__(
        self, head, body,
    ):
        self.head = head
        self.body = body

    def reflect(self):
        return self.body

    def __repr__(self):
        if self.head is ...:
            name = '...'
        elif isinstance(self.head, Symbol):
            name = self.head.name
        else:
            name = repr(self.head)

        return 'Query{{{}: {} <- {}}}'.format(
            name, self.__type_repr__, self.body
        )


binary_opeations = (
    op.add, op.sub, op.mul
)


def op_bind(op):
    @wraps(op)
    def f(*args):
        arg_types = [get_type_and_value(a)[0] for a in args]
        return FunctionApplication(
            Constant[typing.Callable[arg_types, ToBeInferred]](
                op, auto_infer_type=False
            ),
            args,
        )

    return f


def rop_bind(op):
    @wraps(op)
    def f(self, value):
        arg_types = [get_type_and_value(a)[0] for a in (value, self)]
        return FunctionApplication(
            Constant[typing.Callable[arg_types, ToBeInferred]](
                op, auto_infer_type=False
            ),
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

    for c in (Constant, Symbol, ExistentialPredicate, FunctionApplication,
              Statement, Query):
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

    for c in (Constant, Symbol, FunctionApplication, Statement, Query):
        setattr(c, name, rop_bind(operator))
