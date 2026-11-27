"""HPO ontology: is_a graph from hp.json, with ancestor and descendant sets."""
import json
from collections import defaultdict
from pathlib import Path

HPJSON = Path(__file__).resolve().parents[2] / "neurolang_data/probe/sci/hp.json"
OBO = "http://purl.obolibrary.org/obo/HP_"


def _id(iri):
    return "HP:" + iri.rsplit("HP_", 1)[1] if "HP_" in iri else None


def load():
    """(parents, obsolete, replaced_by): parents maps a term to its direct is_a parents."""
    g = json.load(open(HPJSON))["graphs"][0]
    parents, obsolete, replaced = defaultdict(set), set(), {}
    for n in g["nodes"]:
        i = _id(n["id"])
        if i is None:
            continue
        m = n.get("meta", {})
        if m.get("deprecated"):
            obsolete.add(i)
        for b in m.get("basicPropertyValues", []):
            if b["pred"].endswith("IAO_0100001"):
                replaced[i] = _id(b["val"]) or b["val"]
    for e in g["edges"]:
        s, o = _id(e["sub"]), _id(e["obj"])
        if s and o and e["pred"].endswith("is_a"):
            parents[s].add(o)
    return parents, obsolete, replaced


def closure(parents):
    """ancestors[t]: all proper is_a ancestors; with the distance in is_a steps."""
    memo = {}

    def anc(t):
        if t in memo:
            return memo[t]
        out = {}
        for p in parents.get(t, ()):
            out[p] = min(out.get(p, 99), 1)
            for a, d in anc(p).items():
                out[a] = min(out.get(a, 99), d + 1)
        memo[t] = out
        return out

    terms = set(parents) | {p for ps in parents.values() for p in ps}
    return {t: anc(t) for t in terms}
