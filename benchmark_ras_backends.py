"""
Benchmark: NeuroLang Polars RAS backend vs Pandas RAS backend.

Usage:
    python3 benchmark_ras_backends.py
"""
from __future__ import annotations

import gc
import timeit
from typing import Callable, Tuple

from neurolang.config import config


def _with_backend(backend: str):
    import importlib
    import neurolang.utils.relational_algebra_set
    config.set("RAS", "backend", backend)
    config.switch_backend()
    importlib.reload(neurolang.utils.relational_algebra_set)
    mod = neurolang.utils.relational_algebra_set
    return (
        mod.RelationalAlgebraFrozenSet,
        mod.NamedRelationalAlgebraFrozenSet,
        mod.RelationalAlgebraSet,
        mod.RelationalAlgebraStringExpression,
    )


def _make_data(n: int) -> Tuple[list, list, list]:
    half = n // 2
    tuples_a = [(i, i * 2, f"a_{i % 100}") for i in range(n)]
    tuples_b = [
        (i, i * 3, f"b_{i % 100}") for i in range(half, n + half)
    ]
    tuples_c = [(i, i // 2, f"c_{i % 50}") for i in range(n)]
    return tuples_a, tuples_b, tuples_c


def _bench(
    label: str,
    n: int,
    setup_fn: str,
    stmt_fn: str,
    number: int = 1,
    repeat: int = 5,
) -> Tuple[float, float]:
    def _inner(backend: str) -> float:
        print(f"   · [{backend:8s}] {label} n={n:_}", end="", flush=True)
        timer = timeit.Timer(
            stmt=stmt_fn,
            setup=f"from __main__ import _with_backend, _make_data\n"
            f"RelationalAlgebraFrozenSet, NamedRelationalAlgebraFrozenSet, "+
            f"RelationalAlgebraSet, RelationalAlgebraStringExpression = "+
            f"_with_backend('{backend}')\n"
            + setup_fn,
            globals={"_with_backend": _with_backend, "_make_data": _make_data},
        )
        try:
            timer.timeit(number=1)
        except Exception as exc:
            print(f" -> FAILED ({exc})")
            raise
        gc.collect()
        results = timer.repeat(repeat=repeat, number=number)
        median = sorted(results)[repeat // 2]
        print(f" -> {median:.4f}s")
        return median

    pd_median = _inner("pandas")
    pl_median = _inner("polars")
    return pd_median, pl_median


def _op_project_unnamed(n: int) -> Tuple[float, float]:
    setup = (
        f"tuples_a, _, _ = _make_data({n})\n"
        "set1 = RelationalAlgebraFrozenSet(tuples_a)\n"
    )
    return _bench("projection unnamed", n, setup, "set1.projection(0, 2)", number=3)


def _op_select_callable_unnamed(n: int) -> Tuple[float, float]:
    setup = (
        f"tuples_a, _, _ = _make_data({n})\n"
        "set1 = RelationalAlgebraFrozenSet(tuples_a)\n"
    )
    return _bench("selection callable", n, setup, "set1.selection(lambda t: t[0] % 2 == 0)", number=3)


def _op_naturaljoin_named(n: int) -> Tuple[float, float]:
    setup = (
        f"tuples_a, tuples_b, _ = _make_data({n})\n"
        "sa = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_a)\n"
        "sb = NamedRelationalAlgebraFrozenSet(columns=('x', 'z', 'tag'), iterable=tuples_b)\n"
    )
    return _bench("naturaljoin named", n, setup, "sa.naturaljoin(sb)", number=2)


def _op_cross_product(n: int) -> Tuple[float, float]:
    n_small = max(1, min(500, n // 20))
    setup = (
        f"tuples_a, tuples_b, _ = _make_data({n})\n"
        f"sa = RelationalAlgebraFrozenSet(tuples_a[:{n_small}])\n"
        f"sb = RelationalAlgebraFrozenSet(tuples_b[:{n_small}])\n"
    )
    return _bench("cross product", n, setup, "sa.cross_product(sb)", number=1)


def _op_union(n: int) -> Tuple[float, float]:
    setup = (
        f"tuples_a, tuples_b, _ = _make_data({n})\n"
        "sa = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_a)\n"
        "sb = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_b)\n"
    )
    return _bench("union", n, setup, "sa | sb", number=3)


def _op_difference(n: int) -> Tuple[float, float]:
    setup = (
        f"tuples_a, tuples_b, _ = _make_data({n})\n"
        "sa = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_a)\n"
        "sb = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_b)\n"
    )
    return _bench("difference", n, setup, "sa - sb", number=3)


def _op_groupby_agg(n: int) -> Tuple[float, float]:
    setup = (
        f"_, _, tuples_c = _make_data({n})\n"
        "sc = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_c)\n"
    )
    return _bench("groupby+agg", n, setup, "sc.aggregate(['label'], sum)", number=2)


def _op_extended_projection(n: int) -> Tuple[float, float]:
    setup = (
        f"tuples_a, _, _ = _make_data({n})\n"
        "sa = NamedRelationalAlgebraFrozenSet(columns=('x', 'y', 'label'), iterable=tuples_a)\n"
    )
    stmt = "sa.extended_projection({'x': 'x', 'double_x': 'x * 2'})\n"
    return _bench("extended proj", n, setup, stmt, number=3)


def run() -> None:
    print("=" * 72)
    print("RAS Backend Benchmark: Pandas vs Polars")
    print("=" * 72)
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    operations = (
        ("Projection (unnamed)", _op_project_unnamed),
        ("Selection (callable)", _op_select_callable_unnamed),
        ("NaturalJoin (named)", _op_naturaljoin_named),
        ("CrossProduct", _op_cross_product),
        ("Union", _op_union),
        ("Difference", _op_difference),
        ("GroupBy+Agg", _op_groupby_agg),
        ("ExtendedProjection", _op_extended_projection),
    )

    for n in sizes:
        print(f"\n### Dataset size: {n:_} rows")
        for label, fn in operations:
            try:
                pd_t, pl_t = fn(n)
                speedup = pd_t / pl_t if pl_t > 0 else float("inf")
                print(f"   -> speedup: {speedup:.2f}x")
            except Exception as exc:
                print(f"   -> FAILED: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    run()
