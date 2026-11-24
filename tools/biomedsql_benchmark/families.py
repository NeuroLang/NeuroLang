"""Which test templates have a close development template (a 'twin').

A twin shares the table (or tables) and at least half of the columns of the WHERE
clause (Jaccard >= 0.5) with a development template: the few-shot bank holds an
example of its shape. Computed from BiomedSQL's SQL alone.
"""

import json
import re

from common import HERE, load_biomedsql


def where_columns(sql):
    sql = re.sub(r"\s+", " ", sql)
    parts = [
        m.group(1)
        for m in re.finditer(
            r"WHERE (.*?)(?: GROUP BY| ORDER BY| LIMIT|$)", sql, re.I
        )
    ]
    cols = set()
    for part in parts:
        cols |= set(
            re.findall(r"(?:LOWER\()?(\w+)\)?\s*(?:=|<|>|LIKE)", part, re.I)
        )
    return {c for c in cols if not c.isdigit()}


def twins(csv_path):
    rows = load_biomedsql(csv_path)
    first = {}
    for r in rows:
        first.setdefault(r["template_uuid"], r["benchmark_query"])
    spec = json.load(open(HERE / "items.json"))["items"]
    split = {i["template"]: i["split"] for i in spec}
    table = {i["template"]: i["table"] for i in spec}
    out = {}
    for t in split:
        if split[t] != "test":
            continue
        a = where_columns(first[t])
        best = 0.0
        for d in split:
            if split[d] == "dev" and table[d] == table[t]:
                b = where_columns(first[d])
                best = max(best, len(a & b) / len(a | b) if a | b else 1.0)
        out[t] = best >= 0.5
    return out


if __name__ == "__main__":
    import sys

    result = twins(sys.argv[1])
    print(
        "with a development twin:",
        sorted((t for t in result if result[t]), key=lambda t: int(t[1:])),
    )
    print(
        "without:",
        sorted((t for t in result if not result[t]), key=lambda t: int(t[1:])),
    )
