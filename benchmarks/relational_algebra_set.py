import numpy as np
import pandas as pd
from functools import reduce
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)


class TimeNamedRelationalAlgebraFrozenSets:
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
        same_cols = [hex(x) for x in range(njoin_columns)]
        df1 = pd.DataFrame(
            np.random.randint(0, N * distinct_r, size=(N, njoin_columns)),
            columns=same_cols,
        )
        self.sets = []
        for i in range(njoins):
            cols = [hex(ncols * i + x) for x in range(njoin_columns, ncols)]
            df = pd.DataFrame(
                np.random.randint(
                    0, N * distinct_r, size=(N, ncols - njoin_columns)
                ),
                columns=cols,
            )
            self.sets.append(
                NamedRelationalAlgebraFrozenSet(
                    same_cols + cols, pd.concat([df1, df], axis=1)
                )
            )

    def time_ra_naturaljoin(self, N, ncols, njoin_columns, njoins, distinct_r):
        reduce(lambda a, b: a.naturaljoin(b), self.sets)

    def time_ra_left_naturaljoin(
        self, N, ncols, njoin_columns, njoins, distinct_r
    ):
        reduce(lambda a, b: a.left_naturaljoin(b), self.sets)


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
