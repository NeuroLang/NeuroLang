"""Classify each S2 final answer as declarative, hybrid, hand-coded or no
answer (PROTOCOL.md, amendment 5).

    python classify_queries.py RESULTS.jsonl [RESULTS.jsonl ...]

* declarative: one probability construct (PROB[...], EXPECT[...],
  EXPECT_NORM[...]) answers the question, and no arithmetic or aggregation
  is applied to a variable it binds, and no probability is written as a
  numeric constant;
* hybrid: probability constructs whose values are combined by arithmetic or
  aggregation, or by several constructs feeding the answer;
* hand-grounded: otherwise declarative, but quoting a constant (a gene, a
  dataset...) that the question does not mention: entities found by
  inspecting the data and written into the query;
* hand-coded: no probability construct, or the answer is a constant;
* none: no answer.

The rule is mechanical (regular expressions over the final program), so it
can be applied identically to every run.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

QUESTIONS = {
    q["id"]: (q.get("text") or q.get("question") or "")
    for q in (lambda d: d if isinstance(d, list) else d["questions"])(
        json.load(open(Path(__file__).resolve().parent / "questions.json")))
}

CONSTRUCT = re.compile(r"\b(PROB|EXPECT_NORM|EXPECT)\s*\[")
BOUND = re.compile(r"\b(?:PROB|EXPECT_NORM|EXPECT)\s*\[.*?\]\s*=\s*([A-Za-z_]\w*)")
ARITH = re.compile(r"[+\-*/]|\blog\s*\(|\bexp\s*\(")
FLOAT = re.compile(r"(?<![\w'\"])(?:0?\.\d+|1\.0+|0\.0+)(?![\w'\"])")
AGG = re.compile(r"AGGREGATE")


def _comparisons(program):
    """The parenthesised builtin expressions of the program, e.g. (p == a * b)."""
    return re.findall(r"\(([^()]*(?:\([^()]*\)[^()]*)*==[^()]*(?:\([^()]*\)[^()]*)*)\)", program)


def classify(record):
    if record.get("answer") is None:
        return "none"
    program = record.get("final_query") or ""
    constructs = CONSTRUCT.findall(program)
    if not constructs:
        return "hand-coded"
    bound = set(BOUND.findall(program))
    derived = set(bound)
    # variables computed from probability variables, through the rules
    for _ in range(5):
        for comparison in _comparisons(program):
            left, _, right = comparison.partition("==")
            if set(re.findall(r"[A-Za-z_]\w*", right)) & derived:
                derived |= set(re.findall(r"[A-Za-z_]\w*", left))
    for comparison in _comparisons(program):
        names = set(re.findall(r"[A-Za-z_]\w*", comparison))
        if names & derived and ARITH.search(comparison.replace("==", " ")):
            return "hybrid"
        if FLOAT.search(comparison.split("==", 1)[-1]) and names & derived:
            return "hybrid"
    for line in program.splitlines():
        if AGG.search(line) and set(re.findall(r"[A-Za-z_]\w*", line)) & derived:
            return "hybrid"
    if len(constructs) > 1:
        # several probability constructs reaching the answer
        return "hybrid"
    question = QUESTIONS.get(record.get("id"), "").lower()
    quoted = re.findall(r"'([^']*)'", program)
    if any(c.lower() not in question for c in quoted):
        return "hand-grounded"
    return "declarative"


def main(paths):
    for path in paths:
        rows = [json.loads(line) for line in open(path)]
        counts = Counter(classify(r) for r in rows)
        n = len(rows)
        print(f"{path}: {n} answers  " + "  ".join(
            f"{k} {counts[k]} ({counts[k] / n:.0%})"
            for k in ("declarative", "hand-grounded", "hybrid", "hand-coded", "none")))


if __name__ == "__main__":
    main(sys.argv[1:])
