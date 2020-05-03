from abc import abstractmethod, abstractproperty
from collections.abc import MutableSet, Set


class RelationalAlgebraFrozenSet(Set):
    def __init__(self, columns):
        raise NotImplementedError()

    @classmethod
    def create_view_from(cls, other):
        raise NotImplementedError()

    @classmethod
    def dee(cls):
        raise NotImplementedError()

    @classmethod
    def dum(cls):
        raise NotImplementedError()

    @abstractmethod
    def is_empty(self):
        pass

    def is_dum(self):
        return (
            self.arity == 0 and
            self.is_empty()
        )

    def is_dee(self):
        return (
            self.arity == 0 and
            not self.is_empty()
        )

    @abstractmethod
    def __contains__(self, element):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def fetch_one(self):
        pass

    @abstractproperty
    def arity(self):
        pass

    @abstractproperty
    def columns(self):
        pass

    @abstractmethod
    def projection(self, *columns):
        pass

    @abstractmethod
    def selection(self, select_criteria):
        pass

    @abstractmethod
    def selection_columns(self, select_criteria):
        pass

    @abstractmethod
    def equijoin(self, other, join_indices, return_mappings=False):
        pass

    @abstractmethod
    def cross_product(self, other):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def groupby(self, columns):
        pass

    @abstractmethod
    def itervalues(self):
        pass

    @abstractmethod
    def as_numpy_array():
        pass

    def _dee_dum_product(self, other):
        res = None
        if self.is_dum():
            res = self
        elif other.is_dum():
            res = other
        elif self.is_dee():
            res = other
        elif other.is_dee():
            res = self
        return res

    def _dee_dum_sum(self, other):
        res = None
        if self.is_dum():
            res = other
        elif other.is_dum():
            res = self
        elif self.is_dee() and other.is_dee():
            return self
        return res


class NamedRelationalAlgebraFrozenSet(RelationalAlgebraFrozenSet):
    def __init__(self, columns, iterable=None):
        raise NotImplementedError()

    @classmethod
    def dee(cls):
        raise NotImplementedError()

    @classmethod
    def dum(cls):
        raise NotImplementedError()

    @abstractproperty
    def columns(self):
        pass

    @abstractproperty
    def arity(self):
        pass

    @abstractmethod
    def __contains__(self, element):
        pass

    @abstractmethod
    def projection(self, *columns):
        pass

    def equijoin(self, other, join_indices, return_mappings=False):
        raise NotImplementedError()

    @abstractmethod
    def naturaljoin(self, other):
        pass

    @abstractmethod
    def cross_product(self, other):
        pass

    @abstractmethod
    def rename_column(self, src, dst):
        pass

    @abstractmethod
    def rename_columns(self, renames):
        pass

    @abstractmethod
    def groupby(self, columns):
        pass

    @abstractmethod
    def aggregate(self, group_columns, aggregate_function):
        pass

    @abstractmethod
    def extended_projection(self, eval_expressions):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def fetch_one(self):
        pass

    @abstractmethod
    def to_unnamed(self):
        pass

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()


class RelationalAlgebraSet(RelationalAlgebraFrozenSet, MutableSet):

    @abstractmethod
    def add(self, value):
        pass

    @abstractmethod
    def discard(self, value):
        pass
