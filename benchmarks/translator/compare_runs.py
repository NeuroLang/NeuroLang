"""Compare the answers and times of two runs of ``compare_modes.py``.

    python benchmarks/translator/compare_runs.py REFERENCE OTHER
        [--reference-mode off]

For each query answered in both output directories, compares the answer
of OTHER in each mode (on, off) with that of REFERENCE in
``--reference-mode`` (by default ``off``, relational algebra): same rows,
floats within rtol 1e-9, as ``compare_modes.py`` compares two modes. A
change that applies in both modes of a run (a rewrite before the solver)
is checked against another code this way. Then the median times and peaks
of each mode of each run, for the queries answered (not failing) that
differ by more than 20%.
"""
import argparse
import collections
import json
import pickle
import statistics
from pathlib import Path

from compare_modes import same


def runs(directory):
    times = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(Path(directory) / "times.jsonl"):
        row = json.loads(line)
        if not row.get("profiled") and not row.get("error"):
            times[row["name"]][row["mode"]].append(
                (row["seconds"], row["peak_mb"])
            )
    return times


def answer(directory, name, mode):
    path = Path(directory) / f"{name}-{mode}.pkl"
    return pickle.load(open(path, "rb")) if path.exists() else None


def median(rows, index):
    return statistics.median(row[index] for row in rows) if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("other")
    parser.add_argument("--reference-mode", default="off")
    args = parser.parse_args()
    reference, other = runs(args.reference), runs(args.other)
    counts = collections.Counter()
    for name in sorted(set(reference) & set(other)):
        expected = answer(args.reference, name, args.reference_mode)
        for mode in ("on", "off"):
            found = answer(args.other, name, mode)
            if expected is None or found is None:
                continue
            ok, note = same(found, expected)
            counts[ok] += 1
            if not ok:
                print(f"{name} [{mode}]: DIFFERENT ({note})")
    print(f"answers: {counts[True]} same, {counts[False]} different")
    for name in sorted(set(reference) & set(other)):
        for mode in ("on", "off"):
            a = median(reference[name][mode], 0)
            b = median(other[name][mode], 0)
            if a and b and abs(b - a) > max(0.3, 0.2 * a):
                print(
                    f"{name} [{mode}]: {a:.1f} s -> {b:.1f} s, "
                    f"{median(reference[name][mode], 1):.0f} -> "
                    f"{median(other[name][mode], 1):.0f} MB"
                )


if __name__ == "__main__":
    main()
