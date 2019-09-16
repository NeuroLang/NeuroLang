import collections
from itertools import chain

from . import expressions
from .exceptions import NeuroLangException


__all__ = ['TypedSymbolTable']


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

    def symbols_by_type(self, type_, include_subtypes=True):
        ret = dict()
        if self.enclosing_scope is not None:
            ret.update(self.enclosing_scope.symbols_by_type(type_))

        for t in self.types():
            if (
                t is type_ or (
                    include_subtypes and
                    t is not expressions.Unknown and
                    expressions.is_leq_informative(t, type_)
                )
            ):
                ret.update(self._symbols_by_type[t])

        return ret

    def create_scope(self):
        subscope = TypedSymbolTable(enclosing_scope=self)
        return subscope

    def set_readonly(self, readonly):
        self.readonly = readonly
