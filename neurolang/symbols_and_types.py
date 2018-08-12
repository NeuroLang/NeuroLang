import typing
import collections
from itertools import chain

from . import expressions
from .expressions import (
    typing_callable_from_annotated_function,
    ExpressionBlock,
    ToBeInferred,
    Constant, Expression,
    Symbol,
    Lambda,
    FunctionApplication,
    Statement,
    ExistentialPredicate,
    UniversalPredicate,
    Query,
    Projection,
    is_subtype,
    get_type_args,
    get_type_and_value,
    type_validation_value,
    unify_types,
    NeuroLangTypeException,
    NeuroLangException
)


__all__ = [
    'ToBeInferred',
    'Symbol', 'Constant', 'Expression', 'FunctionApplication', 'Statement',
    'Projection', 'ExistentialPredicate', 'UniversalPredicate', 'Lambda',
    'Query',
    'TypedSymbolTable', 'ExpressionBlock',
    'typing_callable_from_annotated_function',
    'NeuroLangTypeException', 'is_subtype', 'type_validation_value',
    'unify_types',
    'get_Callable_arguments_and_return', 'get_type_and_value'
]


def get_Callable_arguments_and_return(callable):
    return callable.__args__[:-1], callable.__args__[-1]


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
    elif isinstance(type_hint, typing.Iterable):
        return [
            replace_type_variable(type_, arg, type_var=type_var)
            for arg in type_hint
        ]
    else:
        return type_hint


class TypedSymbolTable(collections.MutableMapping):
    def __init__(self, enclosing_scope=None, readonly=False):
        self._symbols = collections.OrderedDict()
        self._values_to_symbol = collections.defaultdict(set)
        self._symbols_by_type = collections.defaultdict(dict)
        self.enclosing_scope = enclosing_scope
        self.readonly = readonly

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
        if self.readonly:
            raise NeuroLangException("This symbol table is readonly")
        if isinstance(value, expressions.Expression):
            if not isinstance(key, expressions.Expression):
                key = expressions.Symbol[value.type](key)
            self._symbols[key] = value
            self._values_to_symbol[value].add(key)
            if value.type not in self._symbols_by_type:
                self._symbols_by_type[value.type] = dict()
            self._symbols_by_type[value.type][key] = value
        elif value is None:
            self._symbols[key] = None
        else:
            raise ValueError("Wrong assignment %s" % str(value))

    def __delitem__(self, key):
        if self.readonly:
            raise NeuroLangException("This symbol table is readonly")
        value = self._symbols[key]
        del self._symbols_by_type[value.type][key]
        del self._symbols[key]
        self._values_to_symbol[value].remove(key)

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

    def symbols_for_value(self, value):
        return self._values_to_symbol[value]

    def create_scope(self):
        subscope = TypedSymbolTable(enclosing_scope=self)
        return subscope

    def set_readonly(self, readonly):
        self.readonly = readonly
