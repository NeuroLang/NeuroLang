"""Expressions for the intermediate representation and auxiliary functions."""
import inspect
import logging
import operator as op
import threading
import types
import typing
from contextlib import contextmanager
from functools import WRAPPER_ASSIGNMENTS, lru_cache, wraps
from itertools import chain
from warnings import warn

from .exceptions import NeuroLangException
from .type_system import NeuroLangTypeException, Unknown
from .type_system import get_args as get_type_args
from .type_system import infer_type as _infer_type
from .type_system import infer_type_builtins, is_leq_informative, unify_types
from .typed_symbol_table import TypedSymbolTable

__all__ = [
    'Symbol', 'FunctionApplication', 'Statement',
    'Projection', 'Unknown', 'get_type_args',
    'TypedSymbolTable', 'TypedSymbolTableMixin'
]


_lock = threading.RLock()

_expressions_behave_as_python_objects = dict()
_sure_is_not_pattern = dict()


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


@contextmanager
def sure_is_not_pattern():
    global _lock
    global _sure_is_not_pattern
    thread_id = threading.get_ident()

    with _lock:
        n = _sure_is_not_pattern.get(thread_id, 0)
        _sure_is_not_pattern[thread_id] = n + 1
    yield
    with _lock:
        _sure_is_not_pattern[thread_id] -= 1
        if _sure_is_not_pattern[thread_id] == 0:
            del _sure_is_not_pattern[thread_id]


@contextmanager
def sure_is_not_pattern_():
    global _lock
    global _sure_is_not_pattern
    thread_id = threading.get_ident()

    with _lock:
        _sure_is_not_pattern[thread_id] = True
    yield
    with _lock:
        del _sure_is_not_pattern[thread_id]


def type_validation_value(value, type_):
    if type_ is typing.Any or type_ is Unknown:
        return True

    value_type = infer_type(value)
    return is_leq_informative(value_type, type_)


class ParametricTypeClassMeta(type):
    @lru_cache(maxsize=None)
    def __getitem__(self, type_):
        d = dict(self.__dict__)
        d['type'] = type_
        d['__generic_class__'] = self
        d['__no_explicit_type__'] = False
        d['__parameterized__'] = True
        return self.__class__(
            self.__name__, self.__bases__,
            d
        )

    def __new__(cls, name, bases, attributes, **kwargs):
        attributes['__parameterized__'] = attributes.get(
            '__parameterized__', False
        )
        obj = super().__new__(cls, name, bases, attributes, **kwargs)
        return obj

    def __repr__(self):
        r = self.__name__
        if hasattr(self, 'type'):
            if isinstance(self.type, type):
                c = self.type.__name__
            else:
                c = repr(self.type)
            r += '[{}]'.format(c)
        return r

    @lru_cache(maxsize=128)
    def __subclasscheck__(self, other):
        if other is self:
            return True
        elif (
            isinstance(other, ParametricTypeClassMeta) and
            other.__parameterized__
        ):
            return self.__subclasscheck__parameterized(other)
        else:
            return super().__subclasscheck__(other)

    def __subclasscheck__parameterized(self, other):
        if self.__parameterized__:
            return issubclass(
                other.__generic_class__,
                self.__generic_class__
            ) and is_leq_informative(other.type, self.type)
        else:
            return issubclass(
                other.__generic_class__, self
            )

    def __instancecheck__(self, other):
        return (
            super().__instancecheck__(other) or
            issubclass(other.__class__, self)
        )


def _check_expression_is_pattern(expression):
    '''
    Checks whether the Expression is a pattern for
    pattern matching instead of an instance representing
    an intermediate representation object
    '''
    return (
        expression is ... or
        (isinstance(expression, Expression) and expression.__is_pattern__) or
        (inspect.isclass(expression) and issubclass(expression, Expression))
    )


class ExpressionMeta(ParametricTypeClassMeta):
    """
    Metaclass for expressions. It guarantees a set of properties for every.

    class and instance:
    * Classes have an attribute `type` which is set through the syntax
      ClassName[TypeName]. If type is not specified then the `type`
      attribute defaults to `typing.Any` for the class and
      `Unknown` for the instance. In this case the `__no_explicit_type__`
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

        def init_process_pattern(self, args):
            parameters = inspect.signature(self.__class__).parameters
            cls_argnames = [
                argname for argname, arg in parameters.items()
                if arg.default is inspect.Parameter.empty
            ]
            if len(cls_argnames) != len(args):
                raise TypeError(
                    f'Pattern {self.__class__} with '
                    'wrong number of parameters. '
                    f'Parameters are {cls_argnames}'
                )

            for argname, value in zip(cls_argnames, args):
                setattr(self, argname, value)

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            global _sure_is_not_pattern
            thread_id = threading.get_ident()

            if (
                thread_id not in _sure_is_not_pattern
            ):
                generic_pattern_match = True
                for arg in args:
                    if (
                        _check_expression_is_pattern(arg) or
                        (
                            isinstance(arg, (tuple, list)) and
                            any(
                                _check_expression_is_pattern(a)
                                for a in arg
                            )
                        )
                    ):
                        break
                else:
                    generic_pattern_match = False
            else:
                generic_pattern_match = False

            self.__is_pattern__ = generic_pattern_match
            self._symbols = set()

            if self.__no_explicit_type__:
                self.type = Unknown

            if self.__is_pattern__:
                init_process_pattern(self, args)
            else:
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
        global _expressions_behave_as_python_objects
        thread_id = threading.get_ident()

        if (
            thread_id in _expressions_behave_as_python_objects or
            self.__is_pattern__
        ):
            return Projection(self, index)
        else:
            super().__getitem__(index)

    def __call__(self, *args, **kwargs):
        if hasattr(self, '__annotations__') and len(self.__annotations__) > 0:
            variable_type = self.__annotations__.get('return', None)
        else:
            variable_type = Unknown

        return FunctionApplication[variable_type](
            self, args, kwargs=kwargs,
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
            Constant[typing.Callable[[self.type, str], Unknown]](
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

        # TODO: this should be ret.type is not type_
        #       but there is an issue with the type caching of Python
        #       that makes them different ( id(res.type) != id(type_) )
        if ret.type != type_:
            raise NeuroLangTypeException('Cast impossible')
        return ret

    def unapply(self):
        '''Returns a tuple of parameters used to build the expression.'''
        return tuple(
            getattr(self, child)
            for child in self.__children__
        )

    @classmethod
    def apply(cls, *args):
        '''Builds a new expression using a tuple of its parameters'''
        return cls(*args)

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
            val_other = getattr(other, child)

            if isinstance(val, (list, tuple)):
                if (
                    len(val) != len(val_other) or
                    not all(v == o for v, o in zip(val, val_other))
                ):
                    break
            elif not val == val_other:
                break
        else:
            return True
        return False

    def __hash__(self):
        return hash(tuple(getattr(self, c) for c in self.__children__))


class ExpressionBlock(Expression):
    def __init__(self, expressions):
        self.expressions = tuple(expressions)
        self._symbols = set()
        for exp in expressions:
            self._symbols |= exp._symbols

    def __repr__(self):
        return 'BLOCK START\n' + '\n    '.join(
            repr(e) for e in self.expressions
        ) + '\nBLOCK END'


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
        self.is_fresh = False

    def __eq__(self, other):
        return (
            hash(self) == hash(other) and
            (isinstance(other, Symbol) or isinstance(other, str))
         )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return 'S{{{}: {}}}'.format(self.name, self.__type_repr__)

    @staticmethod
    def _fresh_generator():
        lock = threading.RLock()
        i = 0
        while True:
            with lock:
                fresh = f'fresh_{i:08}'
                i += 1
            yield fresh

    @classmethod
    def fresh(cls):
        if not hasattr(Symbol, '_fresh_generator_'):
            Symbol._fresh_generator_ = Symbol._fresh_generator()
        new_symbol = cls(next(Symbol._fresh_generator_))
        if cls.type is not typing.Any:
            new_symbol = new_symbol.cast(cls.type)
        new_symbol.is_fresh = True
        return new_symbol


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
            self.__init_callable_formula__(value, auto_infer_type)
        elif auto_infer_type and self.type is Unknown:
            self.__auto_infer_type__()
        if not self.__verify_type__(self.value, self.type):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (self.value, self.type)
            )

        if auto_infer_type and self.type is not Unknown:
            self.change_type(self.type)

    def __init_callable_formula__(self, value, auto_infer_type):
        self.__wrapped__ = value
        for attr in WRAPPER_ASSIGNMENTS:
            if hasattr(value, attr):
                setattr(self, attr, getattr(value, attr))

        if auto_infer_type and self.type is Unknown:
            if hasattr(value, '__annotations__'):
                self.type = infer_type(value)
            elif isinstance(value, types.BuiltinFunctionType):
                self.type = infer_type_builtins(value)

    def __auto_infer_type__(self):
        self.type = infer_type(self.value)
        self._symbols = set()
        if is_leq_informative(self.type, typing.Mapping):
            self._auto_build_mapping_()
        elif (
            not is_leq_informative(self.type, typing.Text) and
            is_leq_informative(self.type, typing.Iterable)
        ):
            self._auto_build_iterable_()

    def _auto_build_iterable_(self):
        new_content = []
        for a in self.value:
            if not isinstance(a, Expression):
                a = Constant(a)
            self._symbols |= a._symbols
            new_content.append(a)
        try:
            self.value = type(self.value)(new_content)
        except TypeError:
            self.value = type(self.value)(*new_content)

    def _auto_build_mapping_(self):
        new_content = dict()
        for k, v in self.value.items():
            if not isinstance(k, Expression):
                k = Constant(k)
            if not isinstance(v, Expression):
                v = Constant(v)
            self._symbols |= k._symbols | v._symbols
            new_content[k] = v
        self.value = type(self.value)(new_content)

    def __verify_type__(self, value, type_):
        return (
            isinstance(
                value,
                (types.BuiltinFunctionType, types.BuiltinMethodType)
            ) or (
                not self.verify_type or
                type_validation_value(value, type_)
            )
        )

    def __eq__(self, other):
        if self.type is Unknown:
            warn('Making a comparison with types needed to be inferred')

        if isinstance(other, Expression):
            if isinstance(other, Constant):
                value_equal = other.value == self.value
            else:
                value_equal = hash(other) == hash(self)

            return value_equal and (
                is_leq_informative(self.type, other.type) or
                is_leq_informative(other.type, self.type)
            )
        else:
            return (
                hash(other) == hash(self) and
                type_validation_value(other, self.type)
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


class Lambda(Definition):
    def __init__(self, args, function_expression):
        self.args = args
        self.function_expression = function_expression

        if self.args is None:
            self.arg = tuple()

        if (
            isinstance(args, tuple) and
            any(not isinstance(arg, Symbol) for arg in args)
        ):
            raise NeuroLangException(
                "All arguments need to be a tuple of symbols"
            )

        src_type = [arg.type for arg in self.args]
        dst_type = self.function_expression.type
        self.type = unify_types(
            typing.Callable[src_type, dst_type],
            self.type
        )

        self._symbols = self.function_expression._symbols - set(self.args)

    def __repr__(self):
        r = u'\u03BB {} -> {}: {}'.format(
            self.args, self.function_expression, self.__type_repr__
        )
        return r


class FunctionApplication(Definition):
    def __init__(
        self, functor, args,
        kwargs=None, validate_arguments=False, verify_type=True
    ):
        self.functor = functor
        self.args = args
        self.kwargs = kwargs

        if verify_type:
            self._verify_and_set_type()

        if self.kwargs is None:
            self.kwargs = dict()

        if self.args is None:
            self.args = tuple()
        elif validate_arguments:
            self._validate_arguments()
        self.__symbols = None

    def _verify_and_set_type(self):
        if self.type in (Unknown, typing.Any):
            if self.functor.type in (Unknown, typing.Any):
                pass
            elif isinstance(self.functor.type, typing.Callable):
                self.type = self.functor.type.__args__[-1]
            else:
                if not (
                    self.functor.type in (Unknown, typing.Any)
                    or is_leq_informative(
                        self.functor.type.__args__[-1],
                        self.type
                    )
                ):
                    raise NeuroLangTypeException(
                        "Functor return type not unifiable "
                        "with application type"
                    )

    def _validate_arguments(self):
        if not (
            isinstance(self.args, tuple) and
            all(isinstance(a, Expression) for a in self.args)
        ):
            raise ValueError(
                'args parameter must be a tuple of expressions'
            )

    @property
    def _symbols(self):
        if self.__symbols is None:
            self.__symbols = self.functor._symbols.copy()
            for arg in chain(self.args, self.kwargs.values()):
                self.__symbols |= arg._symbols
        return self.__symbols

    @_symbols.setter
    def _symbols(self, value):
        self.__symbols = value

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
        if (
            self.type is Unknown and auto_infer_projection_type and
            collection.type is not Unknown
        ):
            self._auto_infer_type(collection, item)

        self._symbols = collection._symbols
        self._symbols |= item._symbols

        self.collection = collection
        self.item = item

    def __repr__(self):
        return u"\u03C3{{{}[{}]: {}}}".format(
            self.collection, self.item, self.__type_repr__
        )

    def _auto_infer_type(self, collection, item):
        if is_leq_informative(collection.type, typing.Tuple):
            if (
                isinstance(item, Constant) and
                is_leq_informative(item.type, typing.SupportsInt) and
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
        if is_leq_informative(collection.type, typing.Mapping):
            self.type = collection.type.__args__[1]


class Statement(Definition):
    def __init__(
        self, lhs, rhs,
    ):
        self.lhs = lhs
        self.rhs = rhs

    def reflect(self):
        return self.rhs

    def __repr__(self):
        return 'Statement{{{}: {} <- {}}}'.format(
            repr(self.lhs), self.__type_repr__, repr(self.rhs)
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


def infer_type(value, deep=False):
    if isinstance(value, Expression):
        return value.type
    else:
        return _infer_type(
            value, deep=deep, recursive_callback=infer_type
        )


binary_opeations = (
    op.add, op.sub, op.mul
)


def op_bind(op):
    @wraps(op)
    def fun(*args):
        arg_types = [a.type for a in args]
        return FunctionApplication(
            Constant[typing.Callable[arg_types, Unknown]](
                op, auto_infer_type=False
            ),
            args,
        )

    return fun


def rop_bind(op):
    @wraps(op)
    def fun(self, value):
        arg_types = [a.type for a in (value, self)]
        return FunctionApplication(
            Constant[typing.Callable[arg_types, Unknown]](
                op, auto_infer_type=False
            ),
            args=(value, self),
        )

    return fun


for operator_name in dir(op):
    operator = getattr(op, operator_name)
    if operator_name.startswith('_'):
        continue

    name = '__{}__'.format(operator_name)
    if name.endswith('___'):
        name = name[:-1]

    for c in (Constant, Symbol, FunctionApplication,
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


class TypedSymbolTableMixin:
    """Add capabilities to deal with a symbol table.
    """
    def __init__(self, symbol_table=None):
        if symbol_table is None:
            symbol_table = TypedSymbolTable()
        self.symbol_table = symbol_table
        self.simplify_mode = False
        self.add_functions_to_symbol_table()

    @property
    def included_functions(self):
        function_constants = dict()
        for attribute in dir(self):
            if attribute.startswith('function_'):
                c = Constant(getattr(self, attribute))
                function_constants[attribute[len('function_'):]] = c
        return function_constants

    def add_functions_to_symbol_table(self):
        keyword_symbol_table = TypedSymbolTable()
        for k, v in self.included_functions.items():
            keyword_symbol_table[Symbol[v.type](k)] = v
        keyword_symbol_table.set_readonly(True)
        top_scope = self.symbol_table
        while top_scope.enclosing_scope is not None:
            top_scope = top_scope.enclosing_scope
        top_scope.enclosing_scope = keyword_symbol_table

    def push_scope(self):
        self.symbol_table = self.symbol_table.create_scope()

    def pop_scope(self):
        es = self.symbol_table.enclosing_scope
        if es is None:
            raise NeuroLangException('No enclosing scope')
        self.symbol_table = self.symbol_table.enclosing_scope
