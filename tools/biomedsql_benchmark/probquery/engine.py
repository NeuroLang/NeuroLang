"""The NeuroLang engine of arm S2: the database of the SQL arm.

The five BiomedSQL tables, read by the ``biomedsql`` engine's reader (same
names and columns, booleans as the text 'True'/'False'), the three
deterministic tables of ``data/`` as relations, and the four uncertain tables
of ``data/`` as tuple-independent probabilistic facts (probability first).
Nothing else: not the derived relations of the ``biomedsql`` engine
(qtl_tested, unit_evidence, ...), which the SQL arm does not see.

Probabilistic queries are solved by the lifted solvers only
(small dichotomy theorem, then Dalvi-Suciu lifting), as in
``../replication``; a query they refuse fails.
"""

from pathlib import Path

import pandas as pd

from build_tables import DATA, DB, DETERMINISTIC, UNCERTAIN

BUILTINS = ["lcase", "str_contains"]


def columns(with_biomedsql=True):
    """The columns of every relation of the engine, by name."""
    cols = {n: [c for c in cs if c != "prob"] for n, cs in UNCERTAIN.items()}
    cols.update(DETERMINISTIC)
    if with_biomedsql:
        for name, frame in biomedsql_frames(only_columns=True).items():
            cols[name] = list(frame.columns)
    return cols


def biomedsql_frames(only_columns=False):
    import pyarrow.parquet as pq

    from neurolang_apps.utils.engines.biomedsql.init import TABLES, _table

    if only_columns:
        return {
            name: pd.DataFrame(
                columns=pq.read_schema(DB / f"{src}.parquet").names
            )
            for name, src in TABLES.items()
        }
    return {name: _table(DB, src) for name, src in TABLES.items()}


def build(with_biomedsql=True):
    from neurolang.frontend import NeurolangPDL
    from neurolang.probabilistic import (
        dalvi_suciu_lift,
        small_dichotomy_theorem_based_solver,
    )
    from neurolang_apps.utils import engine_registry

    nl = NeurolangPDL(
        probabilistic_solvers=(
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
        ),
        probabilistic_marg_solvers=(
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
        ),
    )
    engine_registry._phase_builtins(nl, {"builtins": BUILTINS})
    if with_biomedsql:
        for name, df in biomedsql_frames().items():
            nl.add_tuple_set(df, name=name)
    for name, cols in DETERMINISTIC.items():
        df = pd.read_parquet(DATA / f"{name}.parquet")[cols]
        nl.add_tuple_set(df.itertuples(index=False, name=None), name=name)
    for name, cols in UNCERTAIN.items():
        df = pd.read_parquet(DATA / f"{name}.parquet")
        df = df[["prob"] + [c for c in cols if c != "prob"]]
        nl.add_probabilistic_facts_from_tuples(
            df.itertuples(index=False, name=None), name=name
        )
    return nl
