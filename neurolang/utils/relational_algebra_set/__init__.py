from collections.abc import Set, MutableSet
from abc import abstractmethod


class RelationalAlgebraFrozenSet(Set):
    @abstractmethod
    def columns(self):
        pass

    @abstractmethod
    def arity(self):
        pass

    def is_null(self):
        return self.arity == 0 or len(self) == 0

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

    @abstractmethod
    def aggregate(self, group_columns, aggregate_function):
        pass

    @abstractmethod
    def extended_projection(self, eval_expressions):
        pass

    def __repr__(self):
        out = '{'
        for i, v in enumerate(self):
            if i == 10:
                out += '...'
            out += repr(v) + ', '
        out += '}'
        return out

    def __str__(self):
        return repr(self)


class RelationalAlgebraSet(RelationalAlgebraFrozenSet, MutableSet):
    pass
