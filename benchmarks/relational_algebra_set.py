import numpy as np
import pandas as pd
from functools import reduce
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
    NamedSQLARelationalAlgebraFrozenSet,
)
import sqlalchemy


class TimeNamedLeftNaturalJoins:
    params = [[10 ** 4, 10 ** 5, 10 ** 6], [10], [3], [2, 8], [0.75]]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r):
        """
        Generate njoins NamedRelationalAlgebraFrozenSets each with N elements
        & ncols arity. The first njoin_columns columns are identical to
        perform joins on.

        Parameters
        ----------
        N : int
            number of elements per set
        ncols : int
            number of columns per set
        njoin_columns : int
            number of identical columns for all sets
        njoins : int
            number of joins to chain
        distinct_r: int
            ratio of distinct elements in the set
        """
        join_columns = [hex(x) for x in range(njoin_columns)]
        keys = pd.DataFrame(
            np.random.randint(0, N * distinct_r, size=(N, njoin_columns)),
            columns=join_columns,
        )
        self.sets = []
        for i in range(njoins):
            cols = [hex(ncols * i + x) for x in range(njoin_columns, ncols)]
            df = pd.concat(
                [
                    keys,
                    pd.DataFrame(
                        np.random.randint(
                            0, N * distinct_r, size=(N, ncols - njoin_columns)
                        ),
                        columns=cols,
                    ),
                ],
                axis=1,
            )
            self.sets.append(
                NamedRelationalAlgebraFrozenSet(join_columns + cols, df)
            )

    def time_ra_left_naturaljoin(
        self, N, ncols, njoin_columns, njoins, distinct_r
    ):
        reduce(lambda a, b: a.left_naturaljoin(b), self.sets)


engine = sqlalchemy.create_engine("sqlite:///neurolang.db", echo=False)


class TimeChainedNaturalJoins:
    params = [
        [10 ** 5],
        # [10 ** 4, 10 ** 5, 10 ** 6],
        [10],
        [3],
        [6],
        [0.75],
        [NamedRelationalAlgebraFrozenSet, NamedSQLARelationalAlgebraFrozenSet],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "engine class",
    ]

    # timeout = 60 * 3

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, set_class):
        """
        Generate NamedRelationalAlgebraFrozenSets to test performance impact
        of chain order on natural joins. All sets share the same key columns,
        but the first ones have a large portion of the keys repeated
        multiple times, while the last ones have only a small subset of
        distinct keys and are smaller in size.
        Hence, chaining the joins in reverse order of the sets should be
        much more efficient than chaining them in the direct order.

        Parameters
        ----------
        N : int
            number of elements per set
        ncols : int
            number of columns per set
        njoin_columns : int
            number of identical columns for all sets
        njoins : int
            number of joins to chain
        distinct_r: int
            ratio of distinct elements in the sets
        """
        join_columns = [hex(x) for x in range(njoin_columns)]
        keys = pd.DataFrame(
            np.random.randint(0, N * distinct_r, size=(N, njoin_columns)),
            columns=join_columns,
        )
        self.sets = []
        for i in range(njoins):
            # Take a sample of the default keys.
            skeys = keys.sample(frac=1 / (i + 1))
            skeys = pd.DataFrame(
                np.tile(skeys.to_numpy(), (njoins - i, 1)),
                columns=join_columns,
            )
            # Generate random data for the rest of the set
            cols = [hex(ncols * i + x) for x in range(njoin_columns, ncols)]
            df = pd.concat(
                [
                    skeys,
                    pd.DataFrame(
                        np.random.randint(
                            0,
                            N * distinct_r,
                            size=(skeys.shape[0], ncols - njoin_columns),
                        ),
                        columns=cols,
                    ),
                ],
                axis=1,
            )
            if set_class == NamedSQLARelationalAlgebraFrozenSet:
                s = set_class(engine, join_columns + cols, df)
                # Place an index on the key columns
                i = sqlalchemy.Index(
                    "idx_{}".format(s._table_name),
                    *[s._table.c.get(c) for c in join_columns]
                )
                i.create(engine)
                # Analyze the DB
                with engine.connect() as conn:
                    conn.execute("ANALYZE")
            else:
                s = set_class(join_columns + cols, df)
            self.sets.append(s)

    def time_ra_naturaljoin_hard(
        self, N, ncols, njoin_columns, njoins, distinct_r, set_class
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets)
        self.post_process_result(res)

    def time_ra_naturaljoin_easy(
        self, N, ncols, njoin_columns, njoins, distinct_r, set_class
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets[::-1])
        self.post_process_result(res)

    def post_process_result(self, result):
        if isinstance(result, NamedSQLARelationalAlgebraFrozenSet):
            # Fetch the results from the view to execute the join query
            query = (
                sqlalchemy.select(result._table.c)
                .select_from(result._table)
                .distinct()
            )
            with engine.connect() as conn:
                conn.execute(query).fetchall()

            # Analyze the query plan
            qp = "EXPLAIN QUERY PLAN %s" % (str(query))
            with engine.connect() as conn:
                eqp = conn.execute(qp).fetchall()
                for i, s in enumerate(self.sets):
                    print("s{} : {}".format(i, s._table_name))
                print("Query Plan: ")
                for r in eqp:
                    print(r)
        else:
            print(result._container.info())


class TimeRelationalAlgebraFrozenSets:
    params = [[10 ** 4, 10 ** 5, 10 ** 6], [10], [3], [2, 8], [0.75]]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r):
        """
        Generate njoins RelationalAlgebraFrozenSets each with N elements
        & ncols arity. The first njoin_columns columns are identical to
        perform joins on.

        Parameters
        ----------
        N : int
            number of elements per set
        ncols : int
            number of columns per set
        njoin_columns : int
            number of identical columns for all sets
        njoins : int
            number of joins to chain
        distinct_r: int
            ratio of distinct elements in the set
        """
        df1 = pd.DataFrame(
            np.random.randint(0, N * distinct_r, size=(N, njoin_columns))
        )
        self.sets = []
        for _ in range(njoins):
            df = pd.DataFrame(
                np.random.randint(
                    0, N * distinct_r, size=(N, ncols - njoin_columns)
                ),
            )
            self.sets.append(
                RelationalAlgebraFrozenSet(
                    pd.concat([df1, df], axis=1, ignore_index=True)
                )
            )

    def time_ra_equijoin(self, N, ncols, njoin_columns, njoins, distinct_r):
        reduce(
            lambda a, b: a.equijoin(b, [(i, i) for i in range(njoin_columns)]),
            self.sets,
        )
