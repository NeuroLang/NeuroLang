import numpy as np
import pandas as pd
from neurolang.utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)


class TimeNamedRelationalAlgebraFrozenSets:
    params = [
        [10 ** 5, 10 ** 6, 10 ** 7],
    ]

    param_names = [
        "rows",
    ]

    def setup(self, N):
        """
        Generate 2 NamedRelationalAlgebraFrozenSets with N elements &
        ncols arity. The first ncols//3 columns are identical to perform
        joins on.
        """
        self.ncols = 10
        self.same_cols = self.ncols // 3
        cols1 = [hex(x) for x in range(self.ncols)]
        cols2 = cols1[-self.same_cols :] + [
            hex(self.ncols + x) for x in range(self.ncols - self.same_cols)
        ]
        df1 = pd.DataFrame(
            np.random.randint(0, N, size=(N, self.ncols)), columns=cols1
        )
        df2 = pd.concat(
            [
                df1[cols2[: self.same_cols]],
                pd.DataFrame(
                    np.random.randint(
                        0, N, size=(N, self.ncols - self.same_cols)
                    ),
                    columns=cols2[self.same_cols :],
                ),
            ],
            axis=1,
        )
        self.ns1 = NamedRelationalAlgebraFrozenSet(cols1, df1)
        self.ns2 = NamedRelationalAlgebraFrozenSet(cols2, df2)

    def time_ra_naturaljoin(self, N):
        self.ns1.naturaljoin(self.ns2)

    def time_ra_left_naturaljoin(self, N):
        self.ns2.left_naturaljoin(self.ns1)


class TimeRelationalAlgebraFrozenSets:
    params = [
        [10 ** 5, 10 ** 6, 10 ** 7],
    ]

    param_names = [
        "rows",
    ]

    def setup(self, N):
        """
        Generate 2 RelationalAlgebraFrozenSets with N elements &
        ncols arity. The first ncols//3 columns are identical to perform
        joins on.
        """
        self.ncols = 10
        self.same_cols = self.ncols // 3
        df1 = pd.DataFrame(np.random.randint(0, N, size=(N, self.ncols)))
        df2 = pd.concat(
            [
                df1.loc[:, : self.same_cols - 1],
                pd.DataFrame(
                    np.random.randint(
                        0, N, size=(N, self.ncols - self.same_cols)
                    ),
                ),
            ],
            axis=1,
            ignore_index=True,
        )
        self.ns1 = RelationalAlgebraFrozenSet(df1)
        self.ns2 = RelationalAlgebraFrozenSet(df2)

    def time_ra_equijoin(self, N):
        self.ns2.equijoin(self.ns2, [(self.same_cols - 1, self.same_cols - 1)])
