"""HPO disease-phenotype annotations as P(phenotype | disease)."""
import collections, re
from pathlib import Path

HPOA = Path(__file__).resolve().parents[2] / "neurolang_data/probe/sci/phenotype.hpoa"
CLASS_P = {  # HPO frequency terms; midpoints of the stated ranges
    "HP:0040280": 0.99, "HP:0040281": 0.895, "HP:0040282": 0.545,
    "HP:0040283": 0.17, "HP:0040284": 0.025,
}
EXCLUDED = "HP:0040285"
DEFAULT_P = 0.5   # annotation without a frequency: an assumption


def freq(s):
    if not s:
        return None, "missing"
    if s in CLASS_P:
        return CLASS_P[s], "class"
    if s == EXCLUDED:
        return 0.0, "excluded"
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return min(max(n / d, 0.01), 0.99), "fraction"
    m = re.fullmatch(r"([\d.]+)%", s)
    if m:
        return min(max(float(m.group(1)) / 100, 0.01), 0.99), "percent"
    return None, "other"


def load():
    """{(disease, phenotype): probability}, plus counts of frequency kinds."""
    out, kinds = {}, collections.Counter()
    for line in open(HPOA, encoding="utf-8"):
        if line.startswith("#") or line.startswith("database_id"):
            continue
        f = line.rstrip("\n").split("\t")
        disease, qualifier, hp, fr, aspect = f[0], f[2], f[3], f[7], f[10]
        if aspect != "P" or qualifier == "NOT":
            continue
        p, kind = freq(fr)
        kinds[kind] += 1
        if kind == "excluded":
            continue
        if p is None:
            p = DEFAULT_P
        out[(disease, hp)] = max(p, out.get((disease, hp), 0.0))
    return out, kinds
