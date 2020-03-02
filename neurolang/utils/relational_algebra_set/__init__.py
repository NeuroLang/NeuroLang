from collections import Set, MutableSet
from abc import abstractmethod


class RelationalAlgebraFrozenSet(Set):
    @abstractmethod
    def columns(self):
        pass

    @abstractmethod
    def arity(self):
        pass

    @abstractmethod
    def projection(self, *columns):
        pass

    @abstractmethod
    def selection(self, select_criteria):
        pass

    @abstractmethod
    def equijoin(self, other, join_indices=None):
        pass

    def cross_product(self, other):
        return self.equijoin(other)

    @abstractmethod
    def groupby(self, columns):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class NamedRelationalAlgebraFrozenSet(RelationalAlgebraFrozenSet):
    @abstractmethod
    def naturaljoin(self, other):
        pass

    @abstractmethod
    def to_unnamed(self):
        pass

    @abstractmethod
    def rename_column(self, column_src, column_dst):
        pass


class RelationalAlgebraSet(RelationalAlgebraFrozenSet, MutableSet):
    pass
