"""Shared pieces of the BiomedSQL benchmark: items, gold answers, canonical values."""

import ast
import csv
import hashlib
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)


def load_biomedsql(path):
    """The 68,227 question / SQL / answer / execution-result items."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(value):
    """A value as a comparable string: None and NaN are empty, floats keep
    6 significant digits, booleans are true/false, text is stripped."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text in ("True", "False"):
        return text.lower()
    if text.lower() == "nan":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:.6g}" if "." in text or "e" in text.lower() else text


def gold_table(item):
    """(columns, rows, truncated) of BiomedSQL's stored result, without UUID.

    ``truncated`` is True when the result has as many rows as the query's
    LIMIT, so that it is only the first rows of a longer answer.
    """
    results = ast.literal_eval(item["execution_results"])
    if not results:
        columns, rows = [], []
    else:
        columns = [c for c in results[0] if c != "UUID"]
        rows = dedupe([tuple(canon(r[c]) for c in columns) for r in results])
    limit = re.search(
        r"LIMIT\s+(\d+)\s*$", item["benchmark_query"].strip(), re.I
    )
    # A list cut by its LIMIT; LIMIT 1 is the one best row (or one group), an exact answer.
    truncated = (
        bool(limit)
        and int(limit.group(1)) > 1
        and len(results) >= int(limit.group(1))
    )
    return columns, rows, truncated


def dedupe(rows):
    """The distinct rows, in order of first appearance."""
    seen = set()
    return [r for r in rows if not (r in seen or seen.add(r))]


def table_text(columns, rows, limit=25):
    """A table as text for a prompt: its size, then up to *limit* distinct rows."""
    rows = dedupe(rows)
    head = " | ".join(columns) if columns else "(no columns)"
    lines = [f"{len(rows)} rows", head]
    lines += [" | ".join(r) for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)
