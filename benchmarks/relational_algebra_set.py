import logging
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from functools import reduce
from sqlalchemy import table, column, and_, select
from sqlalchemy.dialects import postgresql
from dask_sql import Context
from neurolang.utils.relational_algebra_set import (
    pandas,
    dask_sql,
)

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
        "neurolang.utils.relational_algebra_set": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "dask_sql.context": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)


class TimeLeftNaturalJoins:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6, 12],
        [0.75],
        [pandas, dask_sql],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "RAS module to test",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, module):
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = [
            module.NamedRelationalAlgebraFrozenSet(df.columns, df)
            for df in dfs
        ]

    def time_ra_left_naturaljoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.left_naturaljoin(b), self.sets)
        post_process_result(self.sets, res)


class TimeChainedNaturalJoins:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6],
        [0.75],
        [pandas, dask_sql],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "RAS module to test",
    ]

    timeout = 60 * 3

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, module):
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = [
            module.NamedRelationalAlgebraFrozenSet(df.columns, df)
            for df in dfs
        ]

    def time_ra_naturaljoin_hard(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets)
        post_process_result(self.sets, res)

    def time_ra_naturaljoin_easy(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets[::-1])
        post_process_result(self.sets, res)


class TimeRawMerge:
    params = [[10 ** 4, 10 ** 5], [10], [3], [6, 12], [0.75], [dd, pd]]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "lib",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, lib):
        self.dfs = _generate_dataframes(
            N, ncols, njoin_columns, njoins, distinct_r
        )
        if lib == dd:
            self.dfs = [dd.from_pandas(d, npartitions=1) for d in self.dfs]

    def time_merge(self, N, ncols, njoin_columns, njoins, distinct_r, lib):
        on = list(self.dfs[0].columns[:njoin_columns])
        res = reduce(lambda a, b: lib.merge(a, b, on=on), self.dfs)
        if lib == dd:
            df = res.compute()


class TimeDaskSQLJoins:
    params = [[10 ** 4, 10 ** 5], [10], [3], [6], [0.75]]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r):
        self.dfs = _generate_dataframes(
            N, ncols, njoin_columns, njoins, distinct_r
        )
        self.dfs = [dd.from_pandas(d, npartitions=1) for d in self.dfs]
        self.join_cols = [
            c for c in self.dfs[0].columns if c in self.dfs[1].columns
        ]
        self.ctx = Context()
        self._create_tables()
        self._create_sql_query()

    def _create_tables(self):
        self.tables = []
        for i, df in enumerate(self.dfs):
            _table_name = f"table_{i:03}"
            self.ctx.create_table(_table_name, df)
            _table = table(_table_name, *[column(c) for c in df.columns])
            self.tables.append(_table)

    def _create_sql_query(self):
        left = self.tables[0]
        joinq = left
        select_cols = list(left.c)
        for right in self.tables[1:]:
            on = and_(
                left.c.get(col) == right.c.get(col) for col in self.join_cols
            )
            joinq = joinq.join(right, on)
            select_cols += [c for c in right.c if c.name not in self.join_cols]
        query = select(*select_cols).select_from(joinq)
        self.sql_query = str(
            query.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )

    def time_joins(self, N, ncols, njoin_columns, njoins, distinct_r):
        start = time.perf_counter()
        print(f"Processing SQL query: {self.sql_query}")
        res = self.ctx.sql(self.sql_query)
        stop = time.perf_counter()
        print(f"Processing SQL query took {stop-start:0.4f} s.")
        start = time.perf_counter()
        print("Computing dask dataframe")
        res.compute()
        stop = time.perf_counter()
        print(f"Computing dask dataframe took {stop-start:0.4f} s.")
        # Visualize task graph
        # res.visualize('taskgraph.png')
        return res


class TimeEquiJoin:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6, 12],
        [0.75],
        [pandas, dask_sql],
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "RAS module to test",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, module):
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        for d in dfs:
            d.columns = pd.RangeIndex(ncols)
        self.sets = [module.RelationalAlgebraFrozenSet(df) for df in dfs]

    def time_ra_equijoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, module
    ):
        res = reduce(
            lambda a, b: a.equijoin(b, [(i, i) for i in range(njoin_columns)]),
            self.sets,
        )

        post_process_result(self.sets, res)


def post_process_result(sets, result):
    if isinstance(result, dask_sql.RelationalAlgebraFrozenSet):
        # print(result)
        # Fetch one seems slower than _fetchall. Need to investigate.
        result._fetchall()
        # result.fetch_one()


def _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r):
    """
    Generate njoins dataframes of decreasing size. The first dataframe has
    shape N x ncols, while each dataframe after that has shape 
    (N / (i + 1) x ncols).
    The first njoin_columns cols of each dataframe have identical rows to
    perform joins on.
    """
    join_columns = [hex(x) for x in range(njoin_columns)]
    rstate = np.random.RandomState(0)
    keys = pd.DataFrame(
        rstate.randint(0, N * distinct_r, size=(N, njoin_columns)),
        columns=join_columns,
    )
    sets = []
    for i in range(njoins):
        # Take a sample of the default keys.
        skeys = keys.sample(frac=1 / (i + 1), random_state=rstate)
        skeys = pd.DataFrame(
            np.tile(skeys.to_numpy(), (njoins - i, 1)), columns=join_columns,
        )
        # Generate random data for the rest of the set
        cols = [hex(ncols * i + x) for x in range(njoin_columns, ncols)]
        df = pd.concat(
            [
                skeys,
                pd.DataFrame(
                    rstate.randint(
                        0,
                        N * distinct_r,
                        size=(skeys.shape[0], ncols - njoin_columns),
                    ),
                    columns=cols,
                ),
            ],
            axis=1,
        )
        sets.append(df)
    return sets
