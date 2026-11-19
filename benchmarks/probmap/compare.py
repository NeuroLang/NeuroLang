"""
Compare the answers of runs of examples_run.py: the same rows, sorted,
the floats within a relative tolerance of 1e-9 (absolute 1e-12). Print
the time and the solvers of each program whose solvers differ, and the
programs whose answers differ.

    python compare.py BASE_DIR BRANCH_DIR [BASE_DIR2 BRANCH_DIR2 ...]

Measurement only.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-9, 1e-12


def same(a, b):
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        return repr(a) == repr(b)
    if a.shape != b.shape:
        return False
    if a.empty:
        return True
    a, b = a.copy(), b.copy()
    a.columns = b.columns = range(a.shape[1])
    floats = [c for c in a.columns if pd.api.types.is_float_dtype(a[c])]
    keys = [c for c in a.columns if c not in floats]
    for df in (a, b):
        for c in floats:
            df[f"r{c}"] = df[c].round(9)
    order = keys + [f"r{c}" for c in floats]
    a = a.sort_values(order, kind="stable").reset_index(drop=True)
    b = b.sort_values(order, kind="stable").reset_index(drop=True)
    if keys and not a[keys].astype(str).equals(b[keys].astype(str)):
        return False
    return all(
        np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                    rtol=RTOL, atol=ATOL, equal_nan=True)
        for c in floats
    )


def load(directory):
    directory = Path(directory)
    with open(directory / "answers.pkl", "rb") as f:
        answers = pickle.load(f)
    with open(directory / "summary.json") as f:
        return answers, json.load(f)


def main():
    runs = [load(d) for d in sys.argv[1:]]
    bases, branches = runs[0::2], runs[1::2]
    names = list(bases[0][1])
    compared, differing = 0, []
    for name in names:
        reference = bases[0][0][name]
        for answers, _ in bases[1:] + branches:
            other = answers.get(name)
            if reference is None or other is None:
                if (reference is None) != (other is None):
                    differing.append((name, "error in one run only"))
                continue
            if set(reference) != set(other) or not all(
                same(reference[k], other[k]) for k in reference
            ):
                differing.append((name, "answers differ"))
        if reference is not None:
            compared += 1
        base_s = [s[name]["seconds"] for _, s in bases]
        branch_s = [s[name]["seconds"] for _, s in branches]
        base_solvers = bases[0][1][name]["solvers"]
        branch_solvers = branches[0][1][name]["solvers"]
        if base_solvers != branch_solvers:
            print(f"{name[:70]}\n  base   {base_solvers} "
                  f"{min(base_s):.2f} s (min of {base_s})"
                  f" peak {bases[0][1][name]['peak_rss_delta_mb']} MB"
                  f"\n  branch {branch_solvers} "
                  f"{min(branch_s):.2f} s (min of {branch_s})"
                  f" peak {branches[0][1][name]['peak_rss_delta_mb']} MB")
    print(f"{compared} programs with answers compared, "
          f"{len(names) - compared} with an error in the base run")
    print("differing:", differing or "none")
    for name in ("query pain_map", "query decode_mask"):
        if name in bases[0][1]:
            print(name, "base", [s[name]["seconds"] for _, s in bases],
                  "branch", [s[name]["seconds"] for _, s in branches])


if __name__ == "__main__":
    main()
