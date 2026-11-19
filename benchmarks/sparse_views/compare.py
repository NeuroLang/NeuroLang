"""
Compare the answers two runs of measure.py wrote (sorted tables).

    python compare.py MASTER_DIR BRANCH_DIR

Floats are compared with numpy.isclose(rtol=1e-12, atol=0); other values
exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def same(a, b):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for column in a.columns:
        x, y = a[column], b[column]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            if not np.allclose(x, y, rtol=1e-12, atol=0, equal_nan=True):
                return False
        elif not x.astype(str).equals(y.astype(str)):
            return False
    return True


def main():
    master, branch = map(Path, sys.argv[1:3])
    names = sorted({p.name for p in master.glob("*.pkl")} |
                   {p.name for p in branch.glob("*.pkl")})
    ok = bool(names)
    for name in names:
        if not (master / name).exists() or not (branch / name).exists():
            print(name, "missing on one side")
            ok = False
            continue
        a, b = pd.read_pickle(master / name), pd.read_pickle(branch / name)
        result = same(a, b)
        ok &= result
        print(name, a.shape, "identical" if result else "DIFFERENT")
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
