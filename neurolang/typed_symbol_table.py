import collections
from itertools import chain

from . import expressions
from .expressions import (
    Constant, Symbol, Expression, Unknown, is_leq_informative
)
from .exceptions import NeuroLangException

__all__ = ['TypedSymbolTable', 'TypedSymbolTableMixin']


class TypedSymbolTable(collections.abc.MutableMapping):
    def __init__(self, enclosing_scope=None, readonly=False):
        self._symbols = collections.OrderedDict()

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
        if isinstance(value, Expression):
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
            ', '.join(['%s: (%s)' % (k, v) for k, v in self._symbols.items()])
        )

    def types(self):
        ret = self._symbols_by_type.keys()
        if self.enclosing_scope is not None:
            ret = ret | self.enclosing_scope.types()
        return ret

    def symbols_by_type(self, type_, include_subtypes=True):
        ret = dict()
        if self.enclosing_scope is not None:
            ret.update(self.enclosing_scope.symbols_by_type(type_))

        for t in self.types():
            if (
                t is type_ or (
                    include_subtypes and
                    t is not Unknown and
                    is_leq_informative(t, type_)
                )
            ):
                ret.update(self._symbols_by_type[t])

        return ret

    def create_scope(self):
        subscope = TypedSymbolTable(enclosing_scope=self)
        return subscope

    def set_readonly(self, readonly):
        self.readonly = readonly


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
