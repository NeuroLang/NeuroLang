import importlib
import logging
import logging.config
from functools import reduce

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None

BACKENDS = ["pandas", "polars"]

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
    },
}
logging.config.dictConfig(LOGGING_CONFIG)


def _ras_module(backend):
    """Import the relational algebra set module of ``backend``."""
    if backend == "polars" and pl is None:
        raise NotImplementedError("polars is not installed")
    return importlib.import_module(
        f"neurolang.utils.relational_algebra_set.{backend}"
    )


def _named_sets(module, backend, dfs):
    """Build named relational algebra sets of ``backend`` from ``dfs``."""
    if backend == "polars":
        return [
            module.NamedRelationalAlgebraFrozenSet(
                tuple(df.columns), pl.from_pandas(df).lazy()
            )
            for df in dfs
        ]
    return [
        module.NamedRelationalAlgebraFrozenSet(df.columns, df) for df in dfs
    ]


def _unnamed_sets(module, backend, dfs):
    """Build unnamed relational algebra sets of ``backend`` from ``dfs``."""
    sets = []
    for df in dfs:
        if backend == "polars":
            df = df.set_axis([str(i) for i in range(df.shape[1])], axis=1)
            sets.append(module.RelationalAlgebraFrozenSet(pl.from_pandas(df)))
        else:
            df = df.set_axis(pd.RangeIndex(df.shape[1]), axis=1)
            sets.append(module.RelationalAlgebraFrozenSet(df))
    return sets


def _post_process_result(result):
    """
    Materialise ``result``.

    The polars backend evaluates lazily, so computing the number of
    (deduplicated) tuples forces both backends to finish the work.
    """
    len(result)


class TimeLeftNaturalJoins:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6],
        [0.75],
        BACKENDS,
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "backend",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, backend):
        module = _ras_module(backend)
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = _named_sets(module, backend, dfs)

    def time_ra_left_naturaljoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, backend
    ):
        res = reduce(lambda a, b: a.left_naturaljoin(b), self.sets)
        _post_process_result(res)


class TimeChainedNaturalJoins:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6],
        [0.75],
        BACKENDS,
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "backend",
    ]

    timeout = 60 * 3

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, backend):
        module = _ras_module(backend)
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = _named_sets(module, backend, dfs)

    def time_ra_naturaljoin_hard(
        self, N, ncols, njoin_columns, njoins, distinct_r, backend
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets)
        _post_process_result(res)

    def time_ra_naturaljoin_easy(
        self, N, ncols, njoin_columns, njoins, distinct_r, backend
    ):
        res = reduce(lambda a, b: a.naturaljoin(b), self.sets[::-1])
        _post_process_result(res)


class TimeRawMerge:

    """Baseline: the same chained joins on bare pandas / polars frames."""

    params = [[10 ** 4, 10 ** 5], [10], [3], [6], [0.75], BACKENDS]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "lib",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, lib):
        if lib == "polars" and pl is None:
            raise NotImplementedError("polars is not installed")
        self.dfs = _generate_dataframes(
            N, ncols, njoin_columns, njoins, distinct_r
        )
        if lib == "polars":
            self.dfs = [pl.from_pandas(d) for d in self.dfs]

    def time_merge(self, N, ncols, njoin_columns, njoins, distinct_r, lib):
        on = list(self.dfs[0].columns[:njoin_columns])
        if lib == "polars":
            reduce(lambda a, b: a.join(b, on=on), self.dfs)
        else:
            reduce(lambda a, b: pd.merge(a, b, on=on), self.dfs)


class TimeEquiJoin:
    params = [
        [10 ** 4, 10 ** 5],
        [10],
        [3],
        [6],
        [0.75],
        BACKENDS,
    ]

    param_names = [
        "rows",
        "cols",
        "number of join columns",
        "number of chained joins",
        "ratio of dictinct elements",
        "backend",
    ]

    def setup(self, N, ncols, njoin_columns, njoins, distinct_r, backend):
        module = _ras_module(backend)
        dfs = _generate_dataframes(N, ncols, njoin_columns, njoins, distinct_r)
        self.sets = _unnamed_sets(module, backend, dfs)

    def time_ra_equijoin(
        self, N, ncols, njoin_columns, njoins, distinct_r, backend
    ):
        res = reduce(
            lambda a, b: a.equijoin(b, [(i, i) for i in range(njoin_columns)]),
            self.sets,
        )
        _post_process_result(res)


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
