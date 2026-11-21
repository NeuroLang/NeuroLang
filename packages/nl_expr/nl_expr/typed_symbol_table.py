import collections
from itertools import chain
from types import MappingProxyType

from .exceptions import NeuroLangException
from .type_system import Unknown, is_leq_informative


__all__ = ['TypedSymbolTable']


class TypedSymbolTable(collections.abc.MutableMapping):
    def __init__(self, enclosing_scope=None, readonly=False):
        self._symbols = collections.OrderedDict()

        self._symbols_by_type = collections.defaultdict(dict)
        self.enclosing_scope = enclosing_scope
        self.readonly = readonly
        # changed with each assignment or deletion: the symbols of a type,
        # and what else ``computed`` keeps, until the table or a scope
        # enclosing it changes
        self._version = 0
        self._by_type_cache = {}

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
        self._version += 1
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
        self._version += 1
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

    def _versions(self):
        """The versions of the table and of the scopes enclosing it."""
        scope, versions = self, []
        while scope is not None:
            versions.append((id(scope), scope._version))
            scope = scope.enclosing_scope
        return tuple(versions)

    def symbols_by_type(self, type_, include_subtypes=True):
        """
        Return a read-only mapping of the symbols of type *type_* (or of
        a type more informative, if *include_subtypes*), of the table and
        the scopes enclosing it. It is computed once, and the same mapping
        returned, until the table or one of those scopes changes: a
        magic-sets rewrite asks for the relations of the extensional
        database thousands of times (15 000 for a query of tracts), and
        comparing every type, or copying the result, each time was most
        of it.
        """
        return self.computed(
            ("symbols_by_type", type_, include_subtypes),
            lambda: MappingProxyType(
                self._symbols_by_type_now(type_, include_subtypes)
            )
        )

    def computed(self, key, compute):
        """
        Return ``compute()``, computed once for *key* and returned again
        until the table or one of the scopes enclosing it changes: the
        result must depend on their symbols alone.
        """
        versions = self._versions()
        cached = self._by_type_cache.get(key)
        if cached is None or cached[0] != versions:
            cached = self._by_type_cache[key] = (versions, compute())
        return cached[1]

    def _symbols_by_type_now(self, type_, include_subtypes):
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


# Imported last: ``expressions`` imports ``TypedSymbolTable`` from this module.
from . import expressions  # noqa: E402
