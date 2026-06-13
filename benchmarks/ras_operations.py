import importlib
import gc

from neurolang.config import config


class TimeRASOperations:
    params = (["pandas", "polars"], [1000, 10000, 100000, 1000000])
    param_names = ["backend", "n"]
    timeout = 300

    def setup(self, backend, n):
        config.set_query_backend(backend)

        import neurolang.utils.relational_algebra_set as ras_mod
        importlib.reload(ras_mod)
        self.RAS = ras_mod

        half = n // 2
        self.tuples_a = [(i, i * 2, f"a_{i % 100}") for i in range(n)]
        self.tuples_b = [
            (i, i * 3, f"b_{i % 100}") for i in range(half, n + half)
        ]
        self.tuples_c = [(i, i // 2, f"c_{i % 50}") for i in range(n)]
        self.n_small = max(1, min(500, n // 20))
        self.n = n

    def teardown(self, backend, n):
        gc.collect()

    def time_projection(self, backend, n):
        ras = self.RAS
        s = ras.RelationalAlgebraFrozenSet(self.tuples_a)
        s.projection(0, 2)

    def time_selection_callable(self, backend, n):
        ras = self.RAS
        s = ras.RelationalAlgebraFrozenSet(self.tuples_a)
        s.selection(lambda t: t[0] % 2 == 0)

    def time_cross_product(self, backend, n):
        ras = self.RAS
        sa = ras.RelationalAlgebraFrozenSet(self.tuples_a[: self.n_small])
        sb = ras.RelationalAlgebraFrozenSet(self.tuples_b[: self.n_small])
        sa.cross_product(sb)

    def _named_xy_label(self, ras, data):
        return ras.NamedRelationalAlgebraFrozenSet(
            columns=("x", "y", "label"), iterable=data
        )

    def _named_xz_tag(self, ras):
        return ras.NamedRelationalAlgebraFrozenSet(
            columns=("x", "z", "tag"), iterable=self.tuples_b
        )

    def time_naturaljoin(self, backend, n):
        ras = self.RAS
        sa = self._named_xy_label(ras, self.tuples_a)
        sb = self._named_xz_tag(ras)
        sa.naturaljoin(sb)

    def time_union(self, backend, n):
        ras = self.RAS
        sa = self._named_xy_label(ras, self.tuples_a)
        sb = self._named_xy_label(ras, self.tuples_b)
        _ = sa | sb

    def time_difference(self, backend, n):
        ras = self.RAS
        sa = self._named_xy_label(ras, self.tuples_a)
        sb = self._named_xy_label(ras, self.tuples_b)
        _ = sa - sb

    def time_groupby_agg(self, backend, n):
        ras = self.RAS
        sc = self._named_xy_label(ras, self.tuples_c)
        sc.aggregate(["label"], {"x": sum})

    def time_extended_projection(self, backend, n):
        ras = self.RAS
        sa = self._named_xy_label(ras, self.tuples_a)
        sa.extended_projection({"x": "x", "double_x": "x * 2"})
