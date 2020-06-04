import types
from collections import namedtuple
from collections.abc import Iterable, Set
from itertools import chain
from uuid import uuid4

import pandas as pd
import numpy as np
import sqlalchemy

from .. import OrderedSet, relational_algebra_set
from .sqlalchemy_helpers import CreateView

engine = sqlalchemy.create_engine("sqlite:///", echo=False)


column_operators = sqlalchemy.func


class RelationalAlgebraStringExpression(str):
    def __repr__(self):
        return "{}{{ {} }}".format(self.__class__.__name__, super().__repr__())


class RelationalAlgebraFrozenSet(
    relational_algebra_set.abstract.RelationalAlgebraFrozenSet
 ):
    def __init__(
        self,
        iterable=None,
        engine=engine,
    ):
        self.engine = engine
        self.is_view = False
        self.parents = []
        self._len = None
        self._hash = None
        self._name = self._new_name()
        self._created = False

        if iterable is not None:
            self._create_table_from_iterable(iterable)
        else:
            self._arity = 0
            self._len = 0
            self._columns = tuple()
        self._create_queries()

    @classmethod
    def create_view_from(cls, other):
        if not isinstance(other, cls):
            raise ValueError(
                "View can only be created from an object of the same class"
            )
        output = cls()
        output._initialize_from_instance_same_class(other)
        return output

    def _create_table_from_iterable(self, iterable, column_names=None):
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._name = iterable._name
            if iterable.arity > 0 and column_names is None:
                self._columns = iterable._columns
                self._arity = iterable._arity
            self._len = iterable._len
            self.parents = iterable.parents + [iterable]
            self._created = iterable._created
            self.is_view = iterable.is_view
        else:
            df = pd.DataFrame(iterable, columns=column_names)
            self._columns = tuple(df.columns)
            if len(self._columns) > 0:
                df.to_sql(self._name, self.engine, index=False)
                self._len = None
                self._created = True
            else:
                self._len = len(df)
            self._arity = len(df.columns)

    @classmethod
    def create_from_table_or_view(
        cls,
        name=None,
        engine=engine,
        is_view=False,
        columns=None,
        indices=None,
        parents=None,
        length=None
    ):
        new_set = cls()
        new_set._name = name
        new_set.is_view = is_view

        inspector = sqlalchemy.inspect(new_set.engine)

        if parents is not None:
            new_set.parents = parents
        if columns is None:
            new_set._columns = tuple(
                column['name']
                for column in
                inspector.get_columns(new_set._name)
            )
        else:
            new_set._columns = columns

        new_set._arity = len(new_set._columns)
        new_set._len = length
        view_names = set(inspector.get_view_names())
        table_names = set(inspector.get_table_names())
        all_names = view_names | table_names
        if is_view:
            new_set._created = (
                (name in view_names) and
                all(parent._name in all_names for parent in new_set.parents)
            )
        else:
            new_set._created = (name in table_names)
        new_set._create_queries()
        return new_set

    def _initialize_from_instance_same_class(
        self, other, override_columns=True
    ):
        self.engine = other.engine
        if override_columns:
            self._columns = other.columns
        if other._created and other.arity > 0:
            query = CreateView(
                self._name,
                sqlalchemy.select(
                    [
                        c.label(str(new_c))
                        for c, new_c in zip(other._sql_columns, self.columns)
                    ],
                    from_obj=other._table
                )
            )
            conn = self.engine.connect()
            conn.execute(query)
            self._created = True
        else:
            self._created = False
        self._arity = len(self.columns)
        self._len = other._len
        self.is_view = True
        if other.is_view:
            self.parents = other.parents + [other]
        else:
            self.parents = [other]
        self._create_queries()

    def _create_queries(self):
        self._sql_columns = [
            sqlalchemy.column(str(c))
            for c in self.columns
        ]
        self._table = sqlalchemy.sql.table(self._name, *self._sql_columns)
        self._contains_query = sqlalchemy.text(
            f"select * from {self._name}"
            + " where "
            + " and ".join(f"`{i}` == :t{i}" for i in self.columns)
        )

    @staticmethod
    def _new_name():
        return "table_" + str(uuid4()).replace("-", "_")

    def _new_index_name(self):
        return f'idx_{self._name}'

    def __del__(self):
        if self._created:
            if self.is_view:
                self.engine.execute(f"drop view {self._name}")
            elif len(self.parents) == 0:
                self.engine.execute(f"drop table {self._name}")

    def _normalise_element(self, element):
        if isinstance(element, dict):
            pass
        elif hasattr(element, '__iter__') and not isinstance(element, str):
            element = dict(zip(self.columns, element))
        else:
            element = dict(zip(self.columns, (element,)))
        return element

    @classmethod
    def dee(cls):
        output = cls()
        output._len = 1
        return output

    @classmethod
    def dum(cls):
        return cls()

    @property
    def columns(self):
        return self._columns

    @property
    def arity(self):
        return self._arity

    def __contains__(self, element):
        if self._arity == 0:
            return False
        element = self._normalise_element(element)
        conn = self.engine.connect()
        res = conn.execute(
            self._contains_query,
            **{f"t{i}": e for i, e in element.items()}
        )
        return res.first() is not None

    def __iter__(self):
        if self.arity > 0 and len(self) > 0:
            no_dups = self.eliminate_duplicates()
            query = sqlalchemy.sql.select([
                sqlalchemy.sql.column(str(c))
                for c in self.columns
            ], from_obj=no_dups._table)
            conn = self.engine.connect()
            res = conn.execute(query)
            for t in res:
                yield tuple(t)
        elif self.arity == 0 and len(self) > 0:
            yield tuple()

    def fetch_one(self):
        query = sqlalchemy.sql.select([
            sqlalchemy.sql.column(str(c))
            for c in self.columns
        ], from_obj=self._table, limit=1)
        conn = self.engine.connect()
        res = conn.execute(query)
        return tuple(res.fetchone())

    def itervalues(self):
        raise NotImplementedError()

    def __len__(self):
        if self._len is None:
            if self._created:
                no_dups = self.eliminate_duplicates()
                query = sqlalchemy.sql.select(
                    [sqlalchemy.sql.text('count(*)')],
                    from_obj=no_dups._table
                )
                conn = self.engine.connect()
                res = conn.execute(query).scalar()
                self._len = res
            else:
                self._len = 0
        return self._len

    def is_empty(self):
        return len(self) == 0

    def projection(self, *columns):
        if len(columns) == 0 or self.arity == 0:
            new = type(self)()
            if len(self) > 0:
                new._len = 1
            return new

        new_name = self._new_name()
        query = CreateView(
            new_name,
            sqlalchemy.select(
                [
                    sqlalchemy.column(str(c)).label(str(i))
                    for i, c in enumerate(columns)
                ],
                from_obj=self._table
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=tuple(range(len(columns))),
            parents=[self],
            length=None
        )

    def selection(self, select_criteria):
        if self.is_empty():
            return self._empty_set_same_structure()
        new_name = self._new_name()
        query = sqlalchemy.sql.select(['*'], from_obj=self._table)

        if callable(select_criteria):
            raise NotImplementedError(
                "Arbitrary python expressions not allowed"
            )
        elif isinstance(select_criteria, RelationalAlgebraStringExpression):
            query.append_whereclause(sqlalchemy.text(select_criteria))
        else:
            for k, v in select_criteria.items():
                query.append_whereclause(sqlalchemy.sql.column(str(k)) == v)
        query = CreateView(new_name, query)
        conn = self.engine.connect()
        conn.execute(query)
        result = type(self).create_from_table_or_view(
            name=new_name, engine=self.engine, is_view=True,
            columns=self.columns, parents=[self]
        )
        return result

    def selection_columns(self, select_criteria):
        if self.is_empty():
            return self._empty_set_same_structure()

        new_name = self._new_name()
        query = sqlalchemy.sql.select(['*'], from_obj=self._table)
        for k, v in select_criteria.items():
            query.append_whereclause(
                sqlalchemy.sql.column(str(k)) == sqlalchemy.sql.column(str(v))
            )
        query = CreateView(new_name, query)
        conn = self.engine.connect()
        conn.execute(query)
        result = type(self).create_from_table_or_view(
            name=new_name, engine=self.engine, is_view=True,
            columns=self.columns, parents=[self]
        )
        return result

    def equijoin(self, other, join_indices=None):
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if self.is_empty() or other.is_empty():
            return type(self)()

        if other is self:
            other = self.copy()
        new_name = self._new_name()

        select = sqlalchemy.select(
            [self._table.c[str(i)] for i in range(self.arity)] +
            [
                other._table.c[str(i)].label(str(i + self.arity))
                for i in range(other.arity)
            ]
        ).select_from(self._table)

        if join_indices is None:
            join = other._table
        else:
            join_clause = True
            for i, j in join_indices:
                jc = (
                    self._table.c[str(i)] == other._table.c[str(j)]
                )
                join_clause = jc & join_clause
            join = sqlalchemy.join(
                self._table, other._table, join_clause
            )
        select = select.select_from(join)
        query = CreateView(new_name, select)
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=tuple(range(self.arity + other.arity)),
            is_view=True,
            parents=[self, other],
        )

    def cross_product(self, other):
        return self.equijoin(other)

    def __and__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__and__(other)
        res = self._dee_dum_product(other)
        if res is not None:
            return res
        if not self._equal_sets_structure(other):
            raise ValueError(
                "Relational algebra set columns should be the same"
            )

        columns = [sqlalchemy.column(str(c)) for c in self.columns]
        new_name = self._new_name()
        query = CreateView(
            new_name,
            sqlalchemy.intersect(
                sqlalchemy.select(columns, from_obj=self._table),
                sqlalchemy.select(columns, from_obj=other._table)
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=self.columns,
            is_view=True,
            parents=[self, other],
        )

    def _equal_sets_structure(self, other):
        return set(self.columns) == set(other.columns)

    def __or__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__sub__(other)
        res = self._dee_dum_sum(other)
        if res is not None:
            return res
        if not self._equal_sets_structure(other):
            raise ValueError("Sets do not have the same columns")

        new_name = self._new_name()
        columns = [
            sqlalchemy.column(str(c))
            for c in self.columns
        ]
        query = CreateView(
            new_name,
            sqlalchemy.union(
                sqlalchemy.select(columns, self._table),
                sqlalchemy.select(columns, other._table)
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=self.columns,
            is_view=True,
            parents=[self, other],
        )

    def __sub__(self, other):
        if not isinstance(other, RelationalAlgebraFrozenSet):
            return super().__sub__(other)
        if self.is_empty() or other.is_empty():
            return self.copy()
        if self.is_dee():
            if other.is_dee():
                return self.dum()
            return self.dee()
        if not self._equal_sets_structure(other):
            raise ValueError("Sets do not have the same columns")

        new_name = self._new_name()
        columns = [
            sqlalchemy.column(str(c))
            for c in self.columns
        ]
        query = CreateView(
            new_name,
            sqlalchemy.except_(
                sqlalchemy.select(columns, self._table),
                sqlalchemy.select(columns, other._table)
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=self.columns,
            is_view=True,
            parents=[self, other],
        )

    def deepcopy(self):
        if self.is_dee():
            return self.dee()
        if len(self) > 0:
            new_name = self._new_name()
            query = sqlalchemy.text(
                f"CREATE TABLE {new_name} AS SELECT * FROM {self._name}"
            )
            conn = self.engine.connect()
            conn.execute(query)
            return type(self).create_from_table_or_view(
                name=new_name, engine=self.engine, is_view=False,
                columns=self.columns,
            )
        else:
            return type(self)()

    def _create_view(self, src_table_name, dst_table_name):
        select = sqlalchemy.select(
            [sqlalchemy.text('*')],
            from_obj=sqlalchemy.table(src_table_name)
        )
        query = CreateView(dst_table_name, select)
        conn = self.engine.connect()
        conn.execute(query)

    def copy(self):
        if self.is_dee():
            return self.dee()
        elif self.is_dum():
            return self.dum()
        new_name = self._new_name()
        self._create_view(self._name, new_name),
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=self.columns,
            parents=[self],
        )

    def eliminate_duplicates(self):
        new_name = type(self)._new_name()
        query = CreateView(
            new_name,
            sqlalchemy.select(
                self._sql_columns,
                from_obj=self._table,
                distinct=True
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=self.columns,
            parents=[self],
        )

    def __eq__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            if self.is_dee() or other.is_dee():
                res = self.is_dee() and other.is_dee()
            elif self.is_dum() or other.is_dum():
                res = self.is_dum() and other.is_dum()
            elif self._name == other._name:
                res = True
            elif not self._equal_sets_structure(other):
                res = False
            else:
                columns = [
                    sqlalchemy.column(str(c)) for c in self.columns
                ]
                select_left = sqlalchemy.select(
                    columns=columns,
                    from_obj=self._table
                )
                select_right = sqlalchemy.select(
                    columns=columns,
                    from_obj=other._table
                )
                diff_left = select_left.except_(select_right)
                diff_right = select_right.except_(select_left)
                conn = self.engine.connect()
                if conn.execute(diff_left).fetchone() is not None:
                    res = False
                elif conn.execute(diff_right).fetchone() is not None:
                    res = False
                else:
                    res = True
            return res
        else:
            return super().__eq__(other)

    def groupby(self, columns):
        if self.arity > 0:
            single_column = False
            if not isinstance(columns, Iterable):
                single_column = True
                columns = (columns,)

            sql_columns = tuple(
                sqlalchemy.sql.column(str(c))
                for c in columns
            )
            query = sqlalchemy.sql.select(
                columns=sql_columns,
                from_obj=self._table,
                group_by=sql_columns
            )

            conn = self.engine.connect()
            r = conn.execute(query)
            for t in r:
                g = self.selection(dict(zip(columns, t)))
                if single_column:
                    t_out = t[0]
                else:
                    t_out = tuple(t)
                yield t_out, g

    def aggregate(self, group_columns, aggregate_function):
            if (
                isinstance(group_columns, str) or
                not isinstance(group_columns, Iterable)
            ):
                group_columns = (group_columns,)

            result_cols = []
            for k, v in aggregate_function.items():
                if isinstance(v, types.BuiltinFunctionType):
                    v = v.__name__

                if callable(v):
                    f = v
                elif isinstance(v, str):
                    f = getattr(sqlalchemy.func, v)
                else:
                    raise ValueError(
                        f"aggregate function for {k} needs "
                        "to be callable or a string"
                    )
                c_ = sqlalchemy.column(str(k))
                v = sqlalchemy.sql.label(str(k), f(c_))
                result_cols.append(v)

            group_columns_sql = [
                sqlalchemy.column(str(c))
                for c in group_columns
            ]

            new_name = type(self)._new_name()
            select = sqlalchemy.select(
                columns=group_columns_sql + result_cols,
                from_obj=self._table,
            ).group_by(*group_columns_sql)

            query = CreateView(new_name, select)
            conn = self.engine.connect()
            conn.execute(query)

            return type(self).create_from_table_or_view(
                name=new_name,
                engine=self.engine,
                is_view=True,
                columns=list(chain(group_columns, aggregate_function.keys())),
                parents=[self],
            )

    def __hash__(self):
        if self._hash is None:
            query = sqlalchemy.sql.select(
                columns=['*'],
                distinct=True,
                from_obj=self._table
            )
            conn = self.engine.connect()
            r = conn.execute(query)
            ts = tuple(tuple(t) for t in r.fetchall())
            self._hash = hash(ts)
        return self._hash

    def as_numpy_array(self):
        if self.arity > 0 and len(self) > 0:
            conn = self.engine.connect()
            res = conn.execute(sqlalchemy.select(
                self._sql_columns, from_obj=self._table,
                distinct=True
            ))
            return np.array(res.fetchall())
        else:
            return np.array([])


class NamedRelationalAlgebraFrozenSet(
    RelationalAlgebraFrozenSet,
    relational_algebra_set.abstract.NamedRelationalAlgebraFrozenSet
):
    def __init__(
        self, columns=None, iterable=None, engine=engine
    ):
        if isinstance(columns, NamedRelationalAlgebraFrozenSet):
            iterable = columns
            columns = columns.columns
 
        self.engine = engine
        if columns is None:
            columns = tuple()

        self._check_for_duplicated_columns(columns)
        self._columns = tuple(columns)
        self._name = self._new_name()
        self.parents = []
        self._created = False

        if (
            isinstance(iterable, RelationalAlgebraFrozenSet) and
            iterable.arity > 0
        ):
            self._initialize_from_instance_same_class(
                iterable, override_columns=False
            )
        elif iterable is not None:
            self._create_table_from_iterable(
                iterable, column_names=self.columns
            )
            self.is_view = False
        else:
            self._len = 0
            self._arity = len(self.columns)
            self.is_view = False
        self._create_queries()
        self.named_tuple_type = None

    def to_unnamed(self):
        new_name = self._new_name()
        query = CreateView(
            new_name,
            sqlalchemy.select(
                [c.label(str(i)) for i, c in enumerate(self._sql_columns)],
                from_obj=self._table
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return RelationalAlgebraFrozenSet.create_from_table_or_view(
            engine=self.engine,
            name=new_name,
            is_view=True,
            columns=tuple(range(self.arity)),
            parents=[self],
            length=self._len
        )

    def projection(self, *columns):
        if len(columns) == 0:
            new = type(self)()
            if len(self) > 0:
                new._len = 1
            return new

        if self.is_dee() or self.is_dum():
            return self.copy()

        new_name = self._new_name()
        query = CreateView(
            new_name,
            sqlalchemy.select(
                [sqlalchemy.column(c) for c in columns],
                from_obj=self._table
            )
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=columns,
            parents=[self],
            length=self._len
        )

    def __iter__(self):
        if self.named_tuple_type is None:
            self.named_tuple_type = namedtuple('tuple', self.columns)

        if self.arity > 0 and len(self) > 0:
            conn = self.engine.connect()
            res = conn.execute(sqlalchemy.select(
                self._sql_columns, from_obj=self._table,
                distinct=True
            ))
            for t in res:
                yield self.named_tuple_type(**t)

    def naturaljoin(self, other):
        res = self._dee_dum_product(other)
        if res is not None:
            return res

        if other is self:
            other = self.copy()

        columns = OrderedSet(self.columns)
        other_columns = OrderedSet(other.columns)
        remainder_columns = (
            (other_columns - columns) |
            (columns - other_columns)
        )
        join_columns = columns & other_columns

        new_name = self._new_name()

        select = sqlalchemy.select(
            [sqlalchemy.column(c) for c in remainder_columns] +
            [self._table.c[str(c)] for c in join_columns]
        ).select_from(self._table)

        join_clause = None
        if len(join_columns) > 0:
            join_clause = True
            for c in join_columns:
                jc = (
                    self._table.c[str(c)] == other._table.c[str(c)]
                )
                join_clause = jc & join_clause
            join = sqlalchemy.join(
                self._table, other._table, join_clause
            )
        else:
            join = other._table
        select = select.select_from(join)
        query = CreateView(new_name, select)

        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=tuple(join_columns | remainder_columns),
            is_view=True,
            parents=[self, other],
        )

    def cross_product(self, other):
        return self.naturaljoin(other)

    def groupby(self, columns):
        if self.arity > 0:
            single_column = False
            if isinstance(columns, (str, int)):
                single_column = True
                columns = (columns,)

            columns = tuple(
                sqlalchemy.sql.column(str(c))
                for c in columns
            )

            query = sqlalchemy.sql.select(
                columns=columns,
                from_obj=self._table,
                group_by=columns
            )

            conn = self.engine.connect()
            r = conn.execute(query)
            for t in r:
                g = self.selection(t)
                if single_column:
                    t_out = t[0]
                else:
                    t_out = tuple(t)
                yield t_out, g

    def rename_column(self, column_src, column_dst):
        new_name = self._new_name()
        columns = []
        result_columns = []
        for column in self.columns:
            col = sqlalchemy.column(str(column))
            rc = column
            if column == column_src:
                col = col.label(str(column_dst))
                rc = column_dst
            columns.append(col)
            result_columns.append(rc)

        query = CreateView(
            new_name,
            sqlalchemy.select(columns, from_obj=self._table)
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=result_columns,
            is_view=True,
            parents=[self],
        )

    def rename_columns(self, renames):
        # prevent duplicated destination columns
        self._check_for_duplicated_columns(renames.values())
        if not set(renames).issubset(self.columns):
            # get the missing source columns
            # for a more convenient error message
            not_found_cols = set(c for c in renames if c not in self.columns)
            raise ValueError(
                f"Cannot rename non-existing columns: {not_found_cols}"
            )

        new_name = self._new_name()
        columns = []
        result_columns = []
        for column in self.columns:
            col = sqlalchemy.column(str(column))
            rc = column
            if column in renames:
                column_dst = renames[column]
                col = col.label(str(column_dst))
                rc = column_dst
            columns.append(col)
            result_columns.append(rc)

        query = CreateView(
            new_name,
            sqlalchemy.select(columns, from_obj=self._table)
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=result_columns,
            is_view=True,
            parents=[self],
        )

    def extended_projection(self, eval_expressions):
        T = namedtuple('T', self.columns)
        t = T(**{c: sqlalchemy.column(c) for c in self.columns})
        result_cols = []
        for k, v in eval_expressions.items():
            if callable(v):
                v = v(t)
            elif isinstance(v, RelationalAlgebraStringExpression):
                v = sqlalchemy.text(v)
            else:
                v = sqlalchemy.literal(v)
            v = sqlalchemy.sql.label(str(k), v)
            result_cols.append(v)
        result_cols = [
            c for c in self._sql_columns
            if c.name not in eval_expressions
        ] + result_cols
        select = sqlalchemy.select(
            result_cols, from_obj=self._table
        )
        new_name = type(self)._new_name()
        query = CreateView(new_name, select)
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=[c.name for c in result_cols],
            parents=[self],
        )

    @staticmethod
    def _check_for_duplicated_columns(columns):
        if len(set(columns)) != len(columns):
            columns = list(columns)
            dup_cols = set(c for c in columns if columns.count(c) > 1)
            raise ValueError(
                "Duplicated column names are not allowed. "
                f"Found the following duplicated columns: {dup_cols}"
            )


class RelationalAlgebraSet(
    RelationalAlgebraFrozenSet,
    relational_algebra_set.abstract.RelationalAlgebraSet
):
    def __init__(self, iterable=None, engine=engine):
        if isinstance(iterable, RelationalAlgebraFrozenSet):
            iterable = iterable.deepcopy()
        super().__init__(iterable=iterable, engine=engine)
        self._generate_mutable_queries()

    def copy(self):
        res = self.deepcopy()
        res._generate_mutable_queries()
        return res

    def _generate_mutable_queries(self):
        comma_fields = ', '.join(f'`{i}`' for i in range(self.arity))
        self.add_query = (
            f'INSERT INTO {self._name} '
            + f'( {comma_fields} )'
            + ' VALUES '
            + '(' + ', '.join(f':t{i}' for i in range(self.arity)) + ')'
        )
        self.ior_query = (
            f'INSERT INTO {self._name} '
            + f'( {comma_fields} )'
            + f' SELECT {comma_fields} ' + ' FROM {other._name} '
        )
        self.discard_query = sqlalchemy.text(
            f"DELETE FROM {self._name}"
            + " WHERE "
            + " AND ".join(f"`{i}` == :t{i}" for i in range(self.arity))
        )

    def add(self, value):
        if len(self.columns) == 0:
            self._columns = list(range(len(value)))
        value = self._normalise_element(value)

        if self._arity == 0:
            self._create_table_from_iterable((value,))
            self._create_queries()
            self._generate_mutable_queries()
        else:
            conn = self.engine.connect()
            conn.execute(
                self.add_query,
                {f't{i}': e for i, e in value.items()}
            )
            if self._len is not None:
                self._len = None

    def discard(self, value):
        value = self._normalise_element(value)
        conn = self.engine.connect()
        conn.execute(
            self.discard_query,
            {f"t{i}": e for i, e in value.items()}
        )
        self._len = None

    def __ior__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            if self.is_dee():
                self._initialize_from_instance_same_class(other)
                return self
            elif other.is_dee():
                return self
            elif other.is_empty():
                return self
            elif self.is_empty():
                self._create_table_from_iterable(other)
                self._create_queries()
                self._generate_mutable_queries()
                return self
            query = self.ior_query.format(other=other)
            conn = self.engine.connect()
            conn.execute(query)
            self._len = None
            return self
        else:
            return super().__ior__(other)

    def __isub__(self, other):
        if other.is_empty() or self.is_empty():
            return self
        if self.is_dee() and other.is_dee():
            return self.dum()
        if isinstance(other, RelationalAlgebraFrozenSet):
            query = (
                self._table
                .delete()
                .where(
                    sqlalchemy.tuple_(*self._table.columns)
                    .in_(
                        sqlalchemy.select(
                            (
                                other._table.c[c.name]
                                for c in self._table.columns
                            ),
                            from_obj=other._table
                        )
                    )
                )
            )
            conn = self.engine.connect()
            conn.execute(query)
            self._len = None
            return self
        else:
            return super().__isub__(other)

    def __del__(self):
        if self._created:
            if self.is_view:
                self.engine.execute(f"drop view {self._name}")
            elif len(self.parents) == 0:
                pass
                #  self.engine.execute(f"drop table {self._name}")
