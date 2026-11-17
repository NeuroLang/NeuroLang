"""Compare the results first_screen.py saved, and summarise its times.

    python benchmarks/result_cache/compare.py REFERENCE OTHER [OTHER ...]

For each query of each session of each OTHER directory, compare its frames
with those of session 1 of REFERENCE: the same columns and dtypes, and the
same rows once sorted, exactly or within a relative tolerance of 1e-9 for
the floats. Then print the median time of each query per directory and
session, and whether the reply said the result was kept.
"""
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _sorted(frame):
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame
    try:
        return frame.sort_values(list(frame.columns)).reset_index(drop=True)
    except TypeError:
        return frame.iloc[frame.astype(str).apply(tuple, axis=1).argsort()] \
            .reset_index(drop=True)


def same(a, b) -> str:
    if len(a) != len(b):
        return f"{len(a)} vs {len(b)} frames"
    for (ta, fa), (tb, fb) in zip(a, b):
        if ta != tb or list(fa.columns) != list(fb.columns):
            return "columns differ"
        if list(fa.dtypes) != list(fb.dtypes):
            return f"dtypes {list(fa.dtypes)} vs {list(fb.dtypes)}"
        fa, fb = _sorted(fa), _sorted(fb)
        if fa.equals(fb):
            continue
        for column in fa.columns:
            x, y = fa[column], fb[column]
            if x.dtype.kind == "f":
                if not np.allclose(x, y, rtol=1e-9, atol=0, equal_nan=True):
                    return f"{column} differs beyond 1e-9"
            elif not x.equals(y):
                return f"{column} differs"
        return "equal within 1e-9"
    return "identical"


def main():
    reference, others = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    for other in others:
        for path in sorted(other.glob("s*-q*.pkl")):
            ref = reference / ("s1-" + path.name.split("-", 1)[1])
            a, b = (pickle.load(open(p, "rb")) for p in (ref, path))
            verdict = same(a["frames"], b["frames"])
            if a["failed"] != b["failed"]:
                verdict = "failed differs"
            print(f"{other.name}/{path.stem}: {verdict}")
    for directory in [reference] + others:
        lines = [json.loads(line) for line in
                 open(directory / "times.jsonl")]
        build = json.load(open(directory / "build.json"))
        print(f"\n{directory.name}: build {build['seconds']:.1f} s, "
              f"engine digest {build['engine_digest_seconds']}")
        for line in lines:
            print(f"  s{line['session']} q{line['index']:2d} "
                  f"{line['kind']:8s} {line['seconds']:7.2f} s "
                  f"rows {line['rows']:7d} kept {line['kept']!s:5s} "
                  f"failed {line['failed']!s:5s} "
                  f"private {line['private_mb_before']:.0f}->"
                  f"{line['private_mb_after']:.0f} MB")
        for number in sorted({line["session"] for line in lines}):
            times = [line["seconds"] for line in lines
                     if line["session"] == number]
            print(f"  session {number}: total {sum(times):.2f} s, median "
                  f"{statistics.median(times):.2f} s")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
