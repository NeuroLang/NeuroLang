from collections.abc import Set, MutableSet, Iterable
from uuid import uuid1

import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("sqlite:///", echo=False)


class RelationalAlgebraFrozenSet(Set):
    def __init__(
        self,
        iterable=None,
        engine=engine,
        name=None,
        is_view=False,
        arity=None,
        parents=None,
    ):
        self.engine = engine
        self.is_view = is_view
        self.parents = parents
        self._len = None
        self._hash = None
        if iterable is not None and name is not None:
            raise ValueError()
        elif iterable is not None:
            self._name = self._new_name()
            self._create_table_from_iterable(iterable)
        elif name is not None:
            self._name = name
            self.arity = arity
            self._create_queries()
        else:
            self._name = self._new_name()
            self.arity = 0
            self._len = 0

    def _create_table_from_iterable(self, iterable, column_names=None):
        df = pd.DataFrame(list(iterable), columns=column_names)
        df.to_sql(self._name, self.engine, index=False)
        self.arity = len(df.columns)
        self._len = None
        self._column_names = df.columns
        self._create_queries()

    def _create_queries(self):
        self._contains_query = sqlalchemy.text(
            f"select * from {self._name}"
            + " where "
            + " and ".join(f"`{i}` == :t{i}" for i in range(self.arity))
        )

    @staticmethod
    def _new_name():
        return "table_" + str(uuid1()).replace("-", "_")

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
        query = self._contains_query.bindparams(
            **{f"t{i}": e for i, e in enumerate(element)}
        )
        conn = self.engine.connect()
        res = conn.execute(query)
        return res.first() is not None

    def __iter__(self):
        if self.arity > 0:
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
        return type(self)(
            name=new_name,
            engine=self.engine,
            is_view=True,
            arity=len(columns),
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
        return type(self)(
            name=new_name, engine=self.engine, is_view=True,
            arity=self.arity, parents=[self]
        )

    def equijoin(self, other, join_indices=None):
        if other is self:
            other = self.view()
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
        return type(self)(
            name=new_name,
            engine=self.engine,
            arity=self.arity + other.arity,
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
        return type(self)(
            name=new_name,
            engine=self.engine,
            arity=self.arity,
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
        return type(self)(
            name=new_name,
            engine=self.engine,
            arity=self.arity,
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
        return type(self)(
            name=new_name,
            engine=self.engine,
            arity=self.arity,
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
        return type(self)(
            name=new_name, engine=self.engine, is_view=False, arity=self.arity
        )

    def copy(self):
        new_name = self._new_name()
        query = sqlalchemy.text(
            f"CREATE VIEW {new_name} AS SELECT * FROM {self._name}"
        )
        conn = self.engine.connect()
        conn.execute(query)
        return type(self)(
            name=new_name,
            engine=self.engine,
            is_view=True,
            arity=self.arity,
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
        return type(self)(
            name=new_name,
            engine=self.engine,
            is_view=True,
            arity=self.arity,
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
    def __init__(self, columns, iterable=None):
        self._columns = tuple(columns)
        self._columns_sort = tuple(pd.Index(columns).argsort())
        if iterable is None:
            iterable = []

        if isinstance(iterable, RelationalAlgebraFrozenSet):
            self._initialize_from_instance_same_class(iterable)
        else:
            self._container = pd.DataFrame(
                list(iterable),
                columns=self._columns
            )


class RelationalAlgebraSet(RelationalAlgebraFrozenSet, MutableSet):
    def __init__(self, *args, **kwargs):
        if kwargs['is_view']:
            name = self._new_name()
            engine = kwargs['engine']
            conn = engine.connect()
            conn.execute(
                f'CREATE TABLE {name} FROM SELECT * FROM {kwargs["name"]}'
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
            f"delete from {self._name}"
            + " where "
            + " and ".join(f"`{i}` == :t{i}" for i in range(self.arity))
        )

    def add(self, value):
        if self.arity == 0:
            self._create_table_from_iterable([value])
            self._generate_mutable_queries()
        else:
            if not isinstance(value, Iterable):
                value = (value,)
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
