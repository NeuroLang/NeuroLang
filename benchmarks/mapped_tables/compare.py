"""Compare the reports and query results of bench.py runs.

    python compare.py BASE_DIR OTHER_DIR [--rtol 1e-9]

Tables: same rows, dtypes, row type, duplicates flag and order-free hash of
the rows. Queries: same status; results equal as sorted rows, floats within
a relative tolerance of --rtol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sorted_frame(frame):
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str)
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


def equal(a, b, rtol):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    a, b = sorted_frame(a), sorted_frame(b)
    for column in a.columns:
        x, y = a[column], b[column]
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            if not np.allclose(x.to_numpy(float), y.to_numpy(float),
                               rtol=rtol, atol=0, equal_nan=True):
                return False
        elif not (x.astype(str).to_numpy() == y.astype(str).to_numpy()).all():
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    base, other = Path(args.base), Path(args.other)
    a = json.loads((base / "report.json").read_text())
    b = json.loads((other / "report.json").read_text())
    differ = 0
    sa, sb = a.get("signatures", {}), b.get("signatures", {})
    for name in sorted(set(sa) | set(sb)):
        if sa.get(name) != sb.get(name):
            differ += 1
            print("TABLE DIFFERS", name, sa.get(name), sb.get(name))
    print(f"tables: {len(sa)} vs {len(sb)}, {differ} differ")
    qa, qb = a.get("queries", {}), b.get("queries", {})
    same = 0
    for name in sorted(set(qa) | set(qb)):
        x, y = qa.get(name, {}), qb.get(name, {})
        if x.get("status") != y.get("status"):
            print("QUERY STATUS DIFFERS", name, x, y)
            continue
        if x.get("status") != "ok":
            print("query", name, "both", x.get("status"))
            same += 1
            continue
        fa = pd.read_parquet(base / f"q-{name}.parquet")
        fb = pd.read_parquet(other / f"q-{name}.parquet")
        if equal(fa, fb, args.rtol):
            same += 1
        else:
            print("QUERY RESULT DIFFERS", name, fa.shape, fb.shape)
    print(f"queries: {same} of {len(set(qa) | set(qb))} identical "
          f"(rtol {args.rtol})")


if __name__ == "__main__":
    main()
