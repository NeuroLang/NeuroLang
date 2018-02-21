import typing
import types
import collections
from itertools import chain

from .exceptions import NeuroLangException
from . import expressions
from .expressions import (
    typing_callable_from_annotated_function,
    ToBeInferred,
    Symbol,
    Function,
    Definition,
    evaluate
)


__all__ = [
    'ToBeInferred',
    'Symbol', 'Expression', 'Function', 'Definition', 'TypedSymbolTable',
    'typing_callable_from_annotated_function',
    'NeuroLangTypeException', 'is_subtype',
    'get_Callable_arguments_and_return', 'get_type_and_value', 'evaluate'
]


class NeuroLangTypeException(NeuroLangException):
    pass


def get_Callable_arguments_and_return(callable):
    return callable.__args__[:-1], callable.__args__[-1]


def get_type_args(type_):
    if hasattr(type_, '__args__') and type_.__args__ is not None:
        return type_.__args__
    else:
        return tuple()


def is_subtype(left, right):
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
        if right == int:
            right = typing.SupportsInt
        elif right == float:
            right = typing.SupportsFloat
        elif right == str:
            right = typing.Text

        return issubclass(left, right)


def replace_type_variable(type_, type_hint, type_var=None):
    if (
        isinstance(type_hint, typing.TypeVar) and
        type_hint == type_var
    ):
        return type_
    elif hasattr(type_hint, '__args__') and type_hint.__args__ is not None:
        new_args = []
        for arg in get_type_args(type_hint):
            new_args.append(
                replace_type_variable(type_, arg, type_var=type_var)
            )
        return type_hint.__origin__[tuple(new_args)]
    else:
        return type_hint


def get_type_and_value(value, symbol_table=None):
    if symbol_table is not None and isinstance(value, Symbol):
        value = symbol_table[value]

    if isinstance(value, Expression):
        return value.type, value.value
    elif isinstance(value, expressions.Symbol):
        return value.type, value
    elif isinstance(value, expressions.Function):
        return value.type, value
    else:
        if isinstance(value, types.FunctionType):
            return (
                expressions.typing_callable_from_annotated_function(value),
                value
            )
        else:
            return type(value), value


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


class Expression(expressions.Constant):
    def __init__(self, type_, value, symbol_table=None):
        if not type_validation_value(
            value, type_, symbol_table=symbol_table
        ):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (value, type_)
            )
        self.type = type_
        self.value = value


class TypedSymbolTable(collections.MutableMapping):
    def __init__(self, enclosing_scope=None):
        self._symbols = collections.OrderedDict()
        self._symbols_by_type = collections.defaultdict(
            lambda: set()
        )
        self.enclosing_scope = enclosing_scope

    def __len__(self):
        return len(self._symbols)

    def __getitem__(self, key):
        try:
            return self._symbols[key]
        except KeyError:
            if self.enclosing_scope is not None:
                return self.enclosing_scope[key]
            else:
                raise KeyError("Expression %s not in the table" % key)

    def __setitem__(self, key, value):
        if isinstance(value, expressions.Expression):
            self._symbols[key] = value
            if value.type not in self._symbols_by_type:
                self._symbols_by_type[value.type] = dict()
            self._symbols_by_type[value.type][key] = value
        elif value is None:
            self._symbols[key] = None
        else:
            raise ValueError("Wrong assignment %s" % str(value))

    def __delitem__(self, key):
        value = self._symbols[key]
        del self._symbols_by_type[value.type][key]
        del self._symbols[key]

    def __iter__(self):
        keys = iter(self._symbols.keys())
        if self.enclosing_scope is not None:
            keys = chain(keys, iter(self.enclosing_scope))

        return keys

    def __repr__(self):
        return '{%s}' % (
            ', '.join([
                '%s: (%s)' % (k, v)
                for k, v in self._symbols.items()
            ])
        )

    def types(self):
        ret = self._symbols_by_type.keys()
        if self.enclosing_scope is not None:
            ret = ret | self.enclosing_scope.types()
        return ret

    def symbols_by_type(self, type_):
        if self.enclosing_scope is not None:
            ret = self.enclosing_scope.symbols_by_type(type_)
        else:
            ret = dict()

        ret.update(self._symbols_by_type[type_])
        return ret

    def create_scope(self):
        subscope = TypedSymbolTable(enclosing_scope=self)
        return subscope
