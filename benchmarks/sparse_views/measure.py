"""
Measure the sparse views an engine builds, and the answers of queries,
for comparing a branch with master (Linux: reads /proc).

    python measure.py ENGINE OUT.json [--queries QUERY.dl ...]
        [--examples] [--answers DIR] [--views] [--light]

Builds ENGINE at 2 mm from ~/neurolang_data and records the build time
and peak memory; with --views, the engine declares the sparse views of
the comment of ``engines.yaml`` (activated, the atlas activations), and
with --light it is the lightweight engine of the browser REPL. When the
engine has ``sparse_views`` (the branch), the
time, peak memory and bytes of building them, an exact check of the
round trip of each view to its relation's rows, and the private memory a
session forked from the built engine adds while it reads a whole view.
With --queries / --examples, runs each query (the files, and the Datalog
examples of coordinate_meta_analysis) and writes its answer, sorted, to
DIR/<name>.pkl.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# The views of the comment of the coordinate meta-analysis in engines.yaml.
VIEWS = {
    "activated": ["voxel", "study"],
    "schaefer_activated": ["label", "study"],
    "schaefer400_activated": ["schaefer400_label", "study"],
    "destrieux_activated": ["destrieux_label", "study"],
    "tian_activated": ["tian_label", "study"],
}


def status(field):
    """Return the *field* (VmHWM, VmRSS) of this process, in MiB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024
    return float("nan")


def reset_peak():
    """Reset VmHWM to the current RSS (Linux >= 4.0)."""
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def private_mib(pid):
    """Return the private memory (USS) of *pid*, in MiB."""
    kb = {}
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                kb[parts[0].rstrip(":")] = int(parts[1])
    return (kb["Private_Clean"] + kb["Private_Dirty"]) / 1024


def nbytes(view):
    """Return the bytes of the arrays of *view* and of its indices."""
    matrix = view.matrix
    out = {
        "data": matrix.data.nbytes, "indices": matrix.indices.nbytes,
        "indptr": matrix.indptr.nbytes,
        "dtypes": [str(matrix.data.dtype), str(matrix.indices.dtype),
                   str(matrix.indptr.dtype)],
    }
    for name, index in (("row_index", view.row_index),
                        ("column_index", view.column_index)):
        out[name] = {
            "entities": len(index),
            "bytes": int(index._entities.memory_usage(deep=True) + sum(
                d.memory_usage(deep=True) for d in index._dictionaries
            )),
        }
    return out


def row_hashes(frame):
    """Return the sorted 64-bit hashes of the rows of *frame*."""
    frame = frame.reset_index(drop=True)
    frame.columns = range(frame.shape[1])
    return np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy())


def round_trip(nl, name, view):
    """Check the view's tuples are the rows of its stored relation."""
    from nl_core.relational_algebra.linear_algebra import stored_relation
    from nl_expr.expressions import Symbol

    relation, positions = stored_relation(
        nl.program_ir.symbol_table, Symbol(name)
    )
    began = time.perf_counter()
    rows = relation.as_pandas_dataframe()
    back = view.to_frame()[list(view.rows) + list(view.columns) + (
        [] if view.value is None else [view.value]
    )]
    rows = rows[list(back.columns)]
    equal = len(rows) == len(back) and np.array_equal(
        row_hashes(rows), row_hashes(back)
    )
    return {"rows": len(rows), "view_tuples": len(back),
            "equal_row_hash_multisets": bool(equal),
            "seconds": round(time.perf_counter() - began, 1)}


def fork_reads(view):
    """Return the private MiB a forked process adds reading the view."""
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        before = private_mib(os.getpid())
        total = float(view.matrix.sum()) + float(view.matrix.indices.sum())
        after = private_mib(os.getpid())
        os.write(writer, json.dumps(
            {"before": before, "after": after, "checksum": total}
        ).encode())
        os._exit(0)
    os.close(writer)
    out = b""
    while chunk := os.read(reader, 4096):
        out += chunk
    os.waitpid(pid, 0)
    result = json.loads(out)
    result["added_mib"] = round(result["after"] - result["before"], 1)
    return result


def coactivation(nl, name):
    """
    Compare P(both regions active) over the uniform choice of a study, by
    the engine's query and by R R^T / N over the weighted view R of
    *name* (N the studies of ``study``, the source of ``selected_study``).
    """
    view = nl.sparse_views[name]
    query = (
        f"ans(l1, l2, PROB) :- {name}(l1, s) & {name}(l2, s) "
        "& selected_study(s)"
    )
    out = {}
    with nl.scope:
        studies = len(nl.execute_datalog_program("ans(s) :- study(s)"))
    reset_peak()
    rss, began = status("VmRSS"), time.perf_counter()
    with nl.scope:
        engine = nl.execute_datalog_program(query).as_pandas_dataframe()
    out["engine_seconds"] = round(time.perf_counter() - began, 2)
    out["engine_peak_above_rss_mib"] = round(status("VmHWM") - rss, 1)
    gc.collect()
    reset_peak()
    rss, began = status("VmRSS"), time.perf_counter()
    weighted = view.weighted()
    product = (weighted @ weighted.T).tocoo()
    out["spgemm_seconds"] = round(time.perf_counter() - began, 3)
    out["spgemm_peak_above_rss_mib"] = round(status("VmHWM") - rss, 1)
    labels = view.row_index.decode(np.arange(len(view.row_index)))[0]
    mine = pd.DataFrame({
        "l1": labels.to_numpy()[product.row],
        "l2": labels.to_numpy()[product.col],
        "p": product.data / studies,
    })
    engine.columns = ["l1", "l2", "p"]
    merged = engine.merge(mine, on=["l1", "l2"], how="outer", indicator=True)
    out["pairs_engine"], out["pairs_spgemm"] = len(engine), len(mine)
    out["pairs_only_one_side"] = int((merged["_merge"] != "both").sum())
    both = merged[merged["_merge"] == "both"]
    out["max_abs_diff"] = float((both["p_x"] - both["p_y"]).abs().max())
    out["studies"] = studies
    return out


def answer_frame(result):
    """Return the answer *result* as a sorted DataFrame."""
    if hasattr(result, "as_pandas_dataframe"):
        frame = result.as_pandas_dataframe()
    else:
        frame = pd.DataFrame(list(result))
    frame = frame.reset_index(drop=True)
    columns = list(frame.columns)
    try:
        return frame.sort_values(columns).reset_index(drop=True)
    except TypeError:  # unorderable values (regions): by their text
        order = frame.astype(str).sort_values(columns).index
        return frame.loc[order].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("out")
    parser.add_argument("--queries", nargs="*", default=[])
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--answers")
    parser.add_argument("--coactivation", help="a region x study view")
    parser.add_argument("--views", action="store_true")
    parser.add_argument("--light", action="store_true")
    args = parser.parse_args()

    from neurolang_apps.utils import engine_registry

    record = {
        "engine": args.engine, "declared_views": args.views,
        "light": args.light,
    }
    yaml_path = None
    if args.light:
        import tempfile

        from neurolang_apps.utils.tui.web import lightweight_engines_yaml

        yaml_path = lightweight_engines_yaml(tempfile.mkdtemp())
    if args.views:
        config = engine_registry.get_engine_config

        def get_config(name, yaml_path=None):
            return {**config(name, yaml_path), "sparse_views": VIEWS}

        engine_registry.get_engine_config = get_config
    views_phase = getattr(engine_registry, "_phase_sparse_views", None)
    if views_phase is not None:
        def timed_views(nl, cfg):
            gc.collect()
            rss = status("VmRSS")
            record["peak_before_views_mib"] = status("VmHWM")
            reset_peak()
            began = time.perf_counter()
            views_phase(nl, cfg)
            record["views_seconds"] = round(time.perf_counter() - began, 2)
            record["views_peak_above_rss_mib"] = round(
                status("VmHWM") - rss, 1
            )
            gc.collect()
            record["views_rss_added_mib"] = round(status("VmRSS") - rss, 1)

        engine_registry._phase_sparse_views = timed_views
    began = time.perf_counter()
    nl = engine_registry.build_engine(
        args.engine, Path.home() / "neurolang_data", resolution=2,
        yaml_path=yaml_path,
    )
    record["build_seconds"] = round(time.perf_counter() - began, 1)
    # the whole build's peak, that before the views reset it included
    record["build_peak_mib"] = max(
        status("VmHWM"), record.get("peak_before_views_mib", 0)
    )
    gc.collect()
    record["rss_after_build_mib"] = status("VmRSS")
    views = getattr(nl, "sparse_views", {})
    record["views"] = {}
    for name, view in views.items():
        record["views"][name] = {
            "shape": list(view.matrix.shape), "nnz": int(view.matrix.nnz),
            "nbytes": nbytes(view),
        }
    gc.collect()
    gc.freeze()
    for name, view in views.items():
        record["views"][name]["fork_reads"] = fork_reads(view)
        record["views"][name]["round_trip"] = round_trip(nl, name, view)
    print(json.dumps(record, indent=1), flush=True)

    if args.coactivation:
        record["coactivation"] = coactivation(nl, args.coactivation)
        print(record["coactivation"], flush=True)

    queries = {Path(q).stem: Path(q).read_text() for q in args.queries}
    if args.examples:
        cfg = engine_registry.get_engine_config(args.engine)
        for number, query in enumerate(
            (cfg.get("examples") or {}).get("datalog", [])
        ):
            queries[f"example{number}"] = query
    record["queries"] = {}
    if queries:
        answers = Path(args.answers)
        answers.mkdir(parents=True, exist_ok=True)
    for name, query in queries.items():
        began = time.perf_counter()
        try:
            with nl.scope:
                frame = answer_frame(nl.execute_datalog_program(query))
            frame.to_pickle(answers / f"{name}.pkl")
            outcome = {"rows": len(frame)}
        except Exception as error:  # the same failure on both is compared
            outcome = {"error": f"{type(error).__name__}: {error}"[:300]}
        outcome["seconds"] = round(time.perf_counter() - began, 1)
        record["queries"][name] = outcome
        print(name, outcome, flush=True)
    Path(args.out).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
