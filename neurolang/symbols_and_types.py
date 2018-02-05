import typing
import types
import inspect
import collections

from .exceptions import NeuroLangException


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
        return []


def is_subtype(left, right):
    if (
        issubclass(right, typing.Callable) and
        issubclass(left, typing.Callable)
    ):
        left_args = get_type_args(left)
        right_args = get_type_args(right)

        if len(left_args) != len(right_args):
            False

        return all((
            is_subtype(left_arg, right_arg)
            for left_arg, right_arg in zip(left_args, right_args)
        ))
    elif (any(
        issubclass(right, T) and
        issubclass(left, T)
        for T in (typing.Set, typing.List, typing.Tuple)
    )):
        return all(
            is_subtype(l, r) for l, r in zip(
                get_type_args(left), get_type_args(right)
            )
        )
    else:
        if right == int:
            right = typing.SupportsInt
        elif right == float:
            right = typing.SupportsFloat
        elif right == complex:
            right = typing.SupportsComplex
        elif right == str:
            right = typing.Text

        return issubclass(left, right)


def resolve_forward_references(type_, type_hint, type_name=None):
    if type_name is None:
        type_name = type_.__name__
    if (
        isinstance(type_hint, str) and
        type_hint == type_name
    ):
        return type_
    elif (
        isinstance(type_hint, typing._ForwardRef) and
        type_hint.__forward_arg__ == type_name
    ):
        return type_
    elif hasattr(type_hint, '__args__'):
        new_args = []
        for arg in type_hint.__args__:
            if isinstance(arg, list):
                new_arg = []
                for subarg in arg:
                    new_arg.append(
                        resolve_forward_references(
                            type_, arg, type_name=type_name
                        )
                    )
                new_args.append(new_arg)
            else:
                new_args.append(
                    resolve_forward_references(type_, arg, type_name=type_name)
                )
        return type_hint.__base__[tuple(new_args)]
    else:
        return type_hint


def get_type(value):
    if isinstance(value, typing.Callable):
        return typing_callable_from_annotated_function(value)
    else:
        return type(value)


def type_validation_value(value, type_, value_mapping=None):
    if type_ == typing.Any:
        return True
    elif issubclass(type_, typing.Callable):
        if isinstance(value, types.FunctionType):
            symbol_type = typing_callable_from_annotated_function(value)
        else:
            symbol_type = type(value)
        return is_subtype(symbol_type, type_)
    elif issubclass(type_, typing.Mapping):
        return (
            issubclass(type(value), type_.__base__) and
            ((type_.args is None) or all((
                type_validation_value(
                    k, type_.__args__[0], value_mapping=value_mapping
                ) and
                type_validation_value(
                    v, type_.__args__[1], value_mapping=value_mapping
                )
                for k, v in value.items()
            )))
        )
    elif issubclass(type_, typing.Set):
        return (
            issubclass(type(value), type_.__base__) and
            ((type_.__args__ is None) or all((
                type_validation_value(
                    i, type_.__args__[0], value_mapping=value_mapping
                )
                for i in value
            )))
        )
    elif issubclass(type_, typing.Tuple):
        return (
            issubclass(type(value), type_.__base__) and
            ((type_.__args__ is None) or all((
                type_validation_value(
                    v, t, value_mapping=value_mapping
                )
                for v, t in zip(value, type_.__args__)
            )))
        )
    else:
        if (
            (value_mapping is None) or
            not isinstance(value, Identifier)
        ):
            if isinstance(value, Symbol):
                value = value.value
            return is_subtype(type(value), type_)
        else:
            value = value_mapping[value]
            if isinstance(value, Symbol):
                value = value.value
            return is_subtype(type(value), type_)


class Symbol(object):
    def __init__(self, type_, value, value_mapping=None):
        if not type_validation_value(
            value, type_, value_mapping=value_mapping
        ):
            raise NeuroLangTypeException(
                "The value %s does not correspond to the type %s" %
                (value, type_)
            )
        self.type = type_
        self.value = value

    def __repr__(self):
        return '%s: %s' % (self.value, self.type)


class Identifier(object):
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return 'Id(%s)' % repr(self.value)


class SymbolTable(collections.MutableMapping):
    def __init__(self, enclosing_scope=None):
        self._symbols = collections.OrderedDict()
        self._symbols_by_type = dict()
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
                raise NeuroLangException("Symbol %s not in the table" % key)

    def __setitem__(self, key, value):
        if isinstance(value, Symbol):
            self._symbols[key] = value
            if value.type not in self._symbols_by_type:
                self._symbols_by_type[value.type] = dict()
            self._symbols_by_type[value.type][key] = value
        else:
            raise ValueError("Wrong assignement %s" % str(value))

    def __delitem__(self, key):
        value = self._symbols[key]
        del self._symbols_by_type[value.type][key]
        del self._symbols[key]

    def __iter__(self):
        keys = self._symbols.keys()
        act = self.enclosing_scope
        while act is not None:
            keys = act.keys() or keys
            act = act.enclosing_scope
        return iter(keys)

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
            ret = ret or self._symbols_by_type.keys()
        return ret

    def symbols_by_type(self, type_):
        if self.enclosing_scope is not None:
            ret = self.enclosing_scope.symbols_by_type(type_)
        else:
            ret = dict()

        ret.update(self._symbols_by_type[type_])
        return ret

    def create_scope(self):
        subscope = SymbolTable(enclosing_scope=self)
        return subscope


class FiniteDomain(object):
    pass
