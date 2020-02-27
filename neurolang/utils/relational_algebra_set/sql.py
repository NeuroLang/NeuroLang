from collections import namedtuple
from collections.abc import Iterable, MutableSet, Set
from uuid import uuid4

import pandas as pd
import sqlalchemy

from .. import OrderedSet

engine = sqlalchemy.create_engine("sqlite:///", echo=False)


class RelationalAlgebraFrozenSet(Set):
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

        if iterable is not None:
            self._create_table_from_iterable(iterable)
        else:
            self.arity = 0
            self._len = 0
            self.columns = tuple()

    @classmethod
    def create_from_table_or_view(
        cls,
        name=None,
        engine=engine,
        is_view=False,
        columns=None,
        parents=None,
    ):
        new_set = cls()
        new_set._name = name
        new_set.is_view = is_view
        new_set.parents = parents
        if columns is None:
            new_set.columns = tuple(
                column['name']
                for column in
                sqlalchemy.inspect(new_set.engine).get_columns(new_set._name)
            )
        else:
            new_set.columns = columns
        new_set.arity = len(new_set.columns)
        new_set._len = None
        new_set._create_queries()
        return new_set

    def _create_table_from_iterable(self, iterable, column_names=None):
        df = pd.DataFrame(list(iterable), columns=column_names)
        df.to_sql(self._name, self.engine, index=False)
        self.arity = len(df.columns)
        self._len = None
        self.columns = tuple(df.columns)
        self._create_queries()

    def _create_queries(self):
        self._contains_query = sqlalchemy.text(
            f"select * from {self._name}"
            + " where "
            + " and ".join(f"`{i}` == :t{i}" for i in self.columns)
        )

    @staticmethod
    def _new_name():
        return "table_" + str(uuid4()).replace("-", "_")

    def __del__(self):
        if self.arity == 0:
            return
        if self.is_view:
            self.engine.execute(f"drop view {self._name}")
        else:
            self.engine.execute(f"drop table {self._name}")

    def __contains__(self, element):
        if self.arity == 0:
            return False
        if not isinstance(element, Iterable):
            element = (element,)
        conn = self.engine.connect()

        if isinstance(element, dict):
            res = conn.execute(
                self._contains_query,
                **{f"t{i}": e for i, e in element.items()}
            )
        else:
            res = conn.execute(
                self._contains_query,
                **{f"t{i}": e for i, e in zip(self.columns, element)}
            )
        return res.first() is not None

    def __iter__(self):
        if self.arity > 0 and len(self) > 0:
            conn = self.engine.connect()
            res = conn.execute(f"SELECT DISTINCT * FROM {self._name}")
            for t in res:
                yield tuple(t)

    def __len__(self):
        if self._len is None:
            no_dups = self.eliminate_duplicates()
            conn = self.engine.connect()
            res = conn.execute(f"SELECT COUNT(*) FROM {no_dups._name}")
            r = next(res)
            self._len = r[0]
        return self._len

    def projection(self, *columns):
        new_name = self._new_name()
        query = (
            f"CREATE VIEW {new_name} as select "
            + ", ".join(f"`{c}` as `{i}`" for i, c in enumerate(columns))
            + f" from {self._name}"
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=tuple(range(len(columns))),
            parents=[self],
        )

    def selection(self, select_criteria):
        new_name = self._new_name()
        table = sqlalchemy.sql.table(self._name)
        query = sqlalchemy.sql.select(['*'])
        query.append_from(table)
        for k, v in select_criteria.items():
            query.append_whereclause(sqlalchemy.sql.column(str(k)) == v)
        query = query.compile(compile_kwargs={"literal_binds": True})
        query = f'CREATE VIEW {new_name} AS {query}'
        conn = self.engine.connect()
        conn.execute(query)
        result = type(self).create_from_table_or_view(
            name=new_name, engine=self.engine, is_view=True,
            columns=self.columns, parents=[self]
        )
        return result

    def equijoin(self, other, join_indices=None):
        if other is self:
            other = self.copy()
        new_name = self._new_name()
        result = (
            ", ".join(
                f"{self._name}.`{i}` as `{i}`" for i in range(self.arity)
            )
            + ", "
            + ", ".join(
                f"{other._name}.`{i}` as `{i + self.arity}`"
                for i in range(other.arity)
            )
        )
        query = (
            f"CREATE VIEW {new_name} AS SELECT {result} "
            + f"FROM {self._name} INNER JOIN {other._name} "
        )
        if join_indices is not None:
            query += " ON " + " AND ".join(
                f"{self._name}.`{i}` = {other._name}.`{j}`"
                for i, j in join_indices
            )

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
        new_name = self._new_name()
        query = (
            f"CREATE VIEW {new_name} AS SELECT * from "
            f"{self._name} INTERSECT SELECT * from {other._name}"
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

    def __or__(self, other):
        new_name = self._new_name()
        query = (
            f"CREATE VIEW {new_name} AS SELECT * "
            "from {self._name} UNION SELECT * from {other._name}"
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
        new_name = self._new_name()
        query = (
            f"CREATE VIEW {new_name} AS SELECT * from "
            f"{self._name} EXCEPT SELECT * from {other._name}"
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

    def copy(self):
        new_name = self._new_name()
        query = sqlalchemy.text(
            f"CREATE VIEW {new_name} AS SELECT * FROM {self._name}"
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

    def eliminate_duplicates(self):
        new_name = self._new_name()
        query = sqlalchemy.text(
            f"CREATE VIEW {new_name} AS SELECT DISTINCT *"
            + f" FROM {self._name}"
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
            return not (self._name != other._name) or len(self & other) == len(
                self
            )
        else:
            return super().__eq__(other)

    def groupby(self, columns):
        single_column = False
        if not isinstance(columns, Iterable):
            single_column = True
            columns = (columns,)

        str_columns = ', '.join(f'`{i}`' for i in columns)

        query = (
            f'SELECT {str_columns} FROM {self._name}'
            f' GROUP BY {str_columns}'
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

    def __hash__(self):
        if self._hash is None:
            conn = self.engine.connect()
            r = conn.execute(f'select DISTINCT * from {self._name}')
            ts = tuple(tuple(t) for t in r.fetchall())
            self._hash = hash(ts)
        return self._hash


class NamedRelationalAlgebraFrozenSet(RelationalAlgebraFrozenSet):
    def __init__(
        self, columns=None, iterable=None, engine=engine
    ):
        self.engine = engine
        if columns is None:
            columns = tuple()
        self.columns = columns
        self._name = self._new_name()

        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_instance_same_class(iterable)
        elif iterable is not None:
            self._create_table_from_iterable(
                iterable, column_names=self.columns
            )
            self.is_view = False
        else:
            self._len = 0
            self.arity = len(self.columns)
            self.is_view = False

        self.named_tuple_type = None

    def _initialize_from_instance_same_class(self, other):
        self.engine = other.engine
        query = sqlalchemy.sql.text(
            f'CREATE VIEW {self._name} AS SELECT ' +
            ', '.join(
                f'`{src_c}` as {dst_c}'
                for src_c, dst_c in
                zip(self.columns, other.columns)
            ) +
            f' FROM {other._name}'
        )
        conn = self.engine.connect()
        conn.execute(query)
        self.arity = len(self.columns)
        self.is_view = True
        if other.is_view:
            self.parents = other.parents
        else:
            self.parents = [other]

    def to_unnamed(self):
        new_name = self._new_name()
        query = sqlalchemy.sql.text(
            f'CREATE VIEW {new_name} AS SELECT ' +
            ', '.join(
                f'{src_c} as `{dst_c}`'
                for dst_c, src_c in
                enumerate(self.columns)
            ) +
            f' FROM {self._name}'
        )
        conn = self.engine.connect()
        conn.execute(query)
        return RelationalAlgebraFrozenSet.create_from_table_or_view(
            engine=self.engine,
            name=new_name,
            is_view=True,
            columns=tuple(range(self.arity)),
            parents=[self]
        )

    def projection(self, *columns):
        new_name = self._new_name()
        query = (
            f"CREATE VIEW {new_name} as select "
            + ", ".join(f"{c}" for c in columns)
            + f" from {self._name}"
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            is_view=True,
            columns=columns,
            parents=[self],
        )

    def __iter__(self):
        if self.named_tuple_type is None:
            self.named_tuple_type = namedtuple('tuple', self.columns)

        if self.arity > 0 and len(self) > 0:
            conn = self.engine.connect()
            res = conn.execute(
                f"SELECT DISTINCT {', '.join(self.columns)} FROM {self._name}"
            )
            for t in res:
                yield self.named_tuple_type(**t)

    def naturaljoin(self, other):
        if other is self:
            other = self.copy()

        columns = OrderedSet(self.columns)
        other_columns = OrderedSet(other.columns)
        remainder_columns = other_columns - columns
        join_columns = columns & other_columns

        new_name = self._new_name()

        result = (
            ', '.join(f'{self._name}.{c} as {c}' for c in columns)
        )
        if len(remainder_columns) > 0:
            result += (
                ', ' +
                ', '.join(
                        f'{other._name}.{c} as {c}'
                        for c in remainder_columns
                    )
            )
        query = (
            f"CREATE VIEW {new_name} AS SELECT {result} "
            + f"FROM {self._name} INNER JOIN {other._name} "
        )
        if len(join_columns) > 0:
            query += " ON " + " AND ".join(
                f"{self._name}.{c} = {other._name}.{c}"
                for c in join_columns
            )

        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=tuple(result),
            is_view=True,
            parents=[self, other],
        )

    def cross_product(self, other):
        return self.naturaljoin(other)

    def groupby(self, columns):
        single_column = False
        if isinstance(columns, str):
            single_column = True
            columns = (columns,)

        str_columns = ', '.join(f'`{i}`' for i in columns)

        query = (
            f'SELECT {str_columns} FROM {self._name}'
            f' GROUP BY {str_columns}'
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
        result_columns = tuple()
        for column in self.columns:
            if column == column_src:
                c_dst = column_dst
            else:
                c_dst = column

            col_str = f'{self._name}.{column} as {c_dst}'
            result_columns += (c_dst,)
            columns.append(col_str)

        str_columns = ', '.join(columns)
        query = (
            f'CREATE VIEW {new_name} AS SELECT {str_columns} FROM {self._name}'
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self).create_from_table_or_view(
            name=new_name,
            engine=self.engine,
            columns=tuple(result_columns),
            is_view=True,
            parents=[self],
        )


class RelationalAlgebraSet(RelationalAlgebraFrozenSet, MutableSet):
    def __init__(self, *args, **kwargs):
        if kwargs.get('is_view', False):
            name = self._new_name()
            old_name = kwargs['name']
            engine = kwargs['engine']
            conn = engine.connect()
            conn.execute(
                f'CREATE TABLE {name} AS SELECT * FROM {old_name}'
            )
            kwargs['is_view'] = False
            kwargs['name'] = name
        super().__init__(*args, **kwargs)
        self._generate_mutable_queries()

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
        if self.arity == 0:
            self._create_table_from_iterable([value])
            self._generate_mutable_queries()
        else:
            if not isinstance(value, Iterable):
                value = (value,)
            if self._len is None:
                self._len = 0
            conn = self.engine.connect()
            conn.execute(
                self.add_query,
                {f't{i}': e for i, e in enumerate(value)}
            )
            self._len += 1

    def discard(self, value):
        if not isinstance(value, Iterable):
            value = (value,)
        conn = self.engine.connect()
        conn.execute(
            self.discard_query,
            {f"t{i}": e for i, e in enumerate(value)}
        )
        self._len = None

    def __ior__(self, other):
        if isinstance(other, RelationalAlgebraFrozenSet):
            query = self.ior_query.format(other=other)
            conn = self.engine.connect()
            conn.execute(query)
            self._len = None
            return self
        else:
            return super().__ior__(other)

    def __isub__(self, other):
        if False and isinstance(other, RelationalAlgebraSet):
            diff_ix = ~self._container.index.isin(other._container.index)
            self._container = self._container.loc[diff_ix]
            return self
        else:
            return super().__isub__(other)

    def copy(self):
        return self.deepcopy()
